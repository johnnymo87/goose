// Modified for the goose fork (github.com/johnnymo87/goose, branch patched/v1.54.0):
// owner-token HTTP mailbox (takeover) and per-stream replay ring. See ../VENDORED.md.
// Licensed Apache-2.0, like the rest of this crate.

//! Outbound mailbox for HTTP SSE streams. WebSocket connections keep using the
//! original `connection::OutboundMailbox`, unchanged.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex as StdMutex, MutexGuard},
};

use tokio::sync::{Notify, mpsc::error::TryRecvError};

/// Per-stream replay ring caps (frames dequeued for writing, kept for Last-Event-ID).
pub(crate) const REPLAY_RING_MAX_FRAMES: usize = 4096;
pub(crate) const REPLAY_RING_MAX_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Replay {
    /// Every frame after the client's last id is being re-sent or is still queued.
    Complete,
    /// Some frame after the client's last id is gone; the client must reconcile.
    Gap,
}

struct MailboxState {
    queue: VecDeque<(u64, String)>,
    ring: VecDeque<(u64, String)>,
    ring_bytes: usize,
    max_ring_frames: usize,
    max_ring_bytes: usize,
    /// Highest seq removed from the ring (evicted or acknowledged).
    trimmed_through: u64,
    next_seq: u64,
    owner: Option<u64>,
    next_lease_id: u64,
}

impl MailboxState {
    /// Under the lock, at the moment a frame is dequeued for writing.
    fn record_dequeued(&mut self, seq: u64, msg: &str) {
        self.ring.push_back((seq, msg.to_owned()));
        self.ring_bytes += msg.len();
        // Always keep the newest frame: a single frame larger than the byte budget must
        // not evict itself (that would turn every resume after it into a gap).
        while self.ring.len() > 1
            && (self.ring.len() > self.max_ring_frames || self.ring_bytes > self.max_ring_bytes)
        {
            let Some((evicted, old)) = self.ring.pop_front() else {
                break;
            };
            self.ring_bytes -= old.len();
            self.trimmed_through = evicted;
        }
    }
}

struct Shared {
    state: StdMutex<MailboxState>,
    notify: Notify,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, MailboxState> {
        self.state.lock().expect("http mailbox lock poisoned")
    }
}

pub(crate) struct HttpMailbox {
    shared: Arc<Shared>,
}

pub(crate) struct HttpLease {
    shared: Arc<Shared>,
    id: u64,
    replay: VecDeque<(u64, String)>,
}

impl HttpMailbox {
    pub(crate) fn new() -> Self {
        Self::with_ring_limits(REPLAY_RING_MAX_FRAMES, REPLAY_RING_MAX_BYTES)
    }

    pub(crate) fn with_ring_limits(max_frames: usize, max_bytes: usize) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: StdMutex::new(MailboxState {
                    queue: VecDeque::new(),
                    ring: VecDeque::new(),
                    ring_bytes: 0,
                    max_ring_frames: max_frames,
                    max_ring_bytes: max_bytes,
                    trimmed_through: 0,
                    next_seq: 1,
                    owner: None,
                    next_lease_id: 0,
                }),
                notify: Notify::new(),
            }),
        }
    }

    #[allow(clippy::unnecessary_wraps)] // same Result contract as OutboundMailbox::push
    pub(crate) fn push(&self, msg: String) -> Result<(), &'static str> {
        {
            let mut state = self.shared.lock();
            let seq = state.next_seq;
            state.next_seq += 1;
            state.queue.push_back((seq, msg));
        }
        self.shared.notify.notify_waiters();
        Ok(())
    }

    pub(crate) fn try_acquire(&self) -> Option<HttpLease> {
        let mut state = self.shared.lock();
        if state.owner.is_some() {
            return None;
        }
        Some(self.lease(&mut state))
    }

    /// Takes the stream from its current owner, if any. Works even if the old
    /// subscriber's SSE body is never polled again (a half-open socket).
    pub(crate) fn take_over(&self) -> HttpLease {
        let lease = {
            let mut state = self.shared.lock();
            self.lease(&mut state)
        };
        // Wake a revoked lease parked in `recv` so a still-polled old stream ends.
        self.shared.notify.notify_waiters();
        lease
    }

    fn lease(&self, state: &mut MailboxState) -> HttpLease {
        let id = state.next_lease_id;
        state.next_lease_id += 1;
        state.owner = Some(id);
        HttpLease {
            shared: self.shared.clone(),
            id,
            replay: VecDeque::new(),
        }
    }
}

fn take_next(
    shared: &Shared,
    id: u64,
    replay: &mut VecDeque<(u64, String)>,
) -> Result<(u64, String), TryRecvError> {
    let mut state = shared.lock();
    if state.owner != Some(id) {
        return Err(TryRecvError::Disconnected);
    }
    if let Some(frame) = replay.pop_front() {
        return Ok(frame); // already in the ring; do not record twice
    }
    let (seq, msg) = state.queue.pop_front().ok_or(TryRecvError::Empty)?;
    state.record_dequeued(seq, &msg);
    Ok((seq, msg))
}

impl HttpLease {
    /// Call once, before the first recv. Acknowledges everything up to `last_seen`
    /// and queues the retained frames after it ahead of live frames.
    pub(crate) fn resume_after(&mut self, last_seen: u64) -> Replay {
        let mut state = self.shared.lock();
        let gap = last_seen < state.trimmed_through || last_seen >= state.next_seq;
        while state.ring.front().is_some_and(|(seq, _)| *seq <= last_seen) {
            let (seq, msg) = state.ring.pop_front().expect("front checked");
            state.ring_bytes -= msg.len();
            state.trimmed_through = state.trimmed_through.max(seq);
        }
        self.replay = state.ring.iter().cloned().collect();
        if gap { Replay::Gap } else { Replay::Complete }
    }

    pub(crate) async fn recv_frame(&mut self) -> Option<(u64, String)> {
        let shared = self.shared.clone();
        loop {
            // Register for a wakeup before checking, so a push in between is not missed.
            let mut notified = std::pin::pin!(shared.notify.notified());
            notified.as_mut().enable();
            match take_next(&shared, self.id, &mut self.replay) {
                Ok(frame) => return Some(frame),
                Err(TryRecvError::Disconnected) => return None,
                Err(TryRecvError::Empty) => notified.await,
            }
        }
    }

    pub(crate) fn try_recv_frame(&mut self) -> Result<(u64, String), TryRecvError> {
        take_next(&self.shared, self.id, &mut self.replay)
    }

    #[allow(dead_code)]
    pub(crate) async fn recv(&mut self) -> Option<String> {
        self.recv_frame().await.map(|(_, msg)| msg)
    }

    #[allow(dead_code)]
    pub(crate) fn try_recv(&mut self) -> Result<String, TryRecvError> {
        self.try_recv_frame().map(|(_, msg)| msg)
    }
}

impl Drop for HttpLease {
    fn drop(&mut self) {
        // A revoked lease must not release its successor's ownership.
        if let Ok(mut state) = self.shared.state.lock()
            && state.owner == Some(self.id)
        {
            state.owner = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::time::{Duration, sleep, timeout};

    use super::*;

    const ISSUE_288_BURST: usize = 1_025;

    #[tokio::test]
    async fn http_mailbox_buffers_bursts_before_subscription() {
        let mailbox = HttpMailbox::new();
        for index in 0..ISSUE_288_BURST {
            mailbox.push(format!("message-{index}")).unwrap();
        }
        let mut lease = mailbox.try_acquire().unwrap();
        for index in 0..ISSUE_288_BURST {
            assert_eq!(lease.recv().await, Some(format!("message-{index}")));
        }
    }

    #[tokio::test]
    async fn http_mailbox_has_one_active_owner_and_preserves_queued_frames() {
        let mailbox = HttpMailbox::new();
        let lease = mailbox.try_acquire().unwrap();
        assert!(mailbox.try_acquire().is_none());
        mailbox
            .push("queued before disconnect".to_string())
            .unwrap();
        drop(lease);
        mailbox.push("queued after disconnect".to_string()).unwrap();
        let mut resumed = mailbox.try_acquire().unwrap();
        assert_eq!(
            resumed.recv().await.as_deref(),
            Some("queued before disconnect")
        );
        assert_eq!(
            resumed.recv().await.as_deref(),
            Some("queued after disconnect")
        );
    }

    #[tokio::test]
    async fn take_over_revokes_the_old_lease_without_polling_it() {
        let mailbox = HttpMailbox::new();
        let mut old = mailbox.try_acquire().unwrap();
        mailbox.push("first".to_string()).unwrap();
        assert!(
            mailbox.try_acquire().is_none(),
            "plain acquire still refuses a held stream"
        );

        let mut new = mailbox.take_over();
        mailbox.push("second".to_string()).unwrap();

        assert_eq!(new.recv().await.as_deref(), Some("first"));
        assert_eq!(new.recv().await.as_deref(), Some("second"));
        assert_eq!(
            timeout(Duration::from_secs(1), old.recv()).await.unwrap(),
            None,
            "the revoked lease ends instead of competing for frames"
        );
    }

    #[tokio::test]
    async fn dropping_a_revoked_lease_keeps_the_new_owner() {
        let mailbox = HttpMailbox::new();
        let old = mailbox.try_acquire().unwrap();
        let _new = mailbox.take_over();
        drop(old);
        assert!(mailbox.try_acquire().is_none());
    }

    #[tokio::test]
    async fn take_over_wakes_a_revoked_lease_parked_in_recv() {
        let mailbox = HttpMailbox::new();
        let mut old = mailbox.try_acquire().unwrap();
        let parked = tokio::spawn(async move { old.recv().await });
        sleep(Duration::from_millis(20)).await;
        let _new = mailbox.take_over();
        assert_eq!(
            timeout(Duration::from_secs(1), parked)
                .await
                .unwrap()
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn resume_replays_dequeued_frames_after_last_seen() {
        let mailbox = HttpMailbox::new();
        for n in 1..=3 {
            mailbox.push(format!("m{n}")).unwrap();
        }
        let mut first = mailbox.try_acquire().unwrap();
        for n in 1..=3u64 {
            assert_eq!(first.recv_frame().await, Some((n, format!("m{n}"))));
        }
        drop(first); // written to a socket, but the client only processed m1
        mailbox.push("m4".to_string()).unwrap();

        let mut resumed = mailbox.try_acquire().unwrap();
        assert_eq!(resumed.resume_after(1), Replay::Complete);
        for n in 2..=4u64 {
            assert_eq!(resumed.recv_frame().await, Some((n, format!("m{n}"))));
        }
        assert!(matches!(resumed.try_recv_frame(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn resume_below_the_ring_floor_is_a_gap() {
        let mailbox = HttpMailbox::with_ring_limits(2, usize::MAX);
        for n in 1..=4 {
            mailbox.push(format!("m{n}")).unwrap();
        }
        let mut first = mailbox.try_acquire().unwrap();
        for _ in 0..4 {
            first.recv_frame().await.unwrap();
        }
        drop(first);

        let mut resumed = mailbox.try_acquire().unwrap();
        assert_eq!(resumed.resume_after(0), Replay::Gap);
        assert_eq!(resumed.recv_frame().await, Some((3, "m3".to_string())));
        assert_eq!(resumed.recv_frame().await, Some((4, "m4".to_string())));
    }

    #[tokio::test]
    async fn ring_respects_the_byte_budget() {
        let mailbox = HttpMailbox::with_ring_limits(usize::MAX, 10);
        mailbox.push("aaaaaa".to_string()).unwrap();
        mailbox.push("bbbbbb".to_string()).unwrap();
        let mut lease = mailbox.try_acquire().unwrap();
        lease.recv_frame().await.unwrap();
        lease.recv_frame().await.unwrap(); // 12 bytes > 10: seq 1 is evicted at this dequeue
        drop(lease);

        let mut resumed = mailbox.try_acquire().unwrap();
        assert_eq!(resumed.resume_after(0), Replay::Gap);
    }

    #[tokio::test]
    async fn ring_keeps_a_single_frame_larger_than_the_byte_budget() {
        let mailbox = HttpMailbox::with_ring_limits(usize::MAX, 10);
        mailbox.push("x".repeat(64)).unwrap();
        let mut lease = mailbox.try_acquire().unwrap();
        lease.recv_frame().await.unwrap();
        drop(lease);

        let mut resumed = mailbox.try_acquire().unwrap();
        assert_eq!(
            resumed.resume_after(0),
            Replay::Complete,
            "the oversize frame is retained"
        );
        assert_eq!(resumed.recv_frame().await, Some((1, "x".repeat(64))));
    }

    #[tokio::test]
    async fn resume_acknowledges_so_an_older_resume_is_a_gap() {
        let mailbox = HttpMailbox::new();
        for n in 1..=3 {
            mailbox.push(format!("m{n}")).unwrap();
        }
        let mut lease = mailbox.try_acquire().unwrap();
        for _ in 0..3 {
            lease.recv_frame().await.unwrap();
        }
        drop(lease);
        let mut acked = mailbox.try_acquire().unwrap();
        assert_eq!(acked.resume_after(2), Replay::Complete);
        drop(acked);
        let mut stale = mailbox.try_acquire().unwrap();
        assert_eq!(stale.resume_after(1), Replay::Gap);
    }

    #[tokio::test]
    async fn resume_past_the_last_issued_seq_is_a_gap() {
        let mailbox = HttpMailbox::new();
        mailbox.push("m1".to_string()).unwrap();
        let mut lease = mailbox.try_acquire().unwrap();
        assert_eq!(lease.resume_after(9), Replay::Gap);
    }

    #[tokio::test]
    async fn take_over_then_resume_recovers_what_the_revoked_lease_dequeued() {
        let mailbox = HttpMailbox::new();
        mailbox.push("m1".to_string()).unwrap();
        mailbox.push("m2".to_string()).unwrap();
        let mut stuck = mailbox.try_acquire().unwrap();
        stuck.recv_frame().await.unwrap();
        stuck.recv_frame().await.unwrap(); // dequeued into a socket nobody reads
        let mut taken = mailbox.take_over();
        assert_eq!(taken.resume_after(0), Replay::Complete);
        assert_eq!(taken.recv_frame().await, Some((1, "m1".to_string())));
        assert_eq!(taken.recv_frame().await, Some((2, "m2".to_string())));
        drop(stuck);
    }
}
