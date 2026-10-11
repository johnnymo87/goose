// Modified for the goose fork (github.com/johnnymo87/goose, branch patched/v1.54.0):
// new file, not part of agent-client-protocol-http 2.2.0. See ../VENDORED.md.
// Licensed Apache-2.0, like the rest of this crate.

//! Outbound mailbox for HTTP SSE streams. WebSocket connections keep using the
//! original `connection::OutboundMailbox`, unchanged.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex as StdMutex, MutexGuard},
};

use tokio::sync::{Notify, mpsc::error::TryRecvError};

struct MailboxState {
    queue: VecDeque<String>,
    /// Lease that currently owns the stream. Takeover replaces it; a revoked lease's
    /// next dequeue sees the mismatch and ends, so it never has to be polled.
    owner: Option<u64>,
    next_lease_id: u64,
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
}

impl HttpMailbox {
    pub(crate) fn new() -> Self {
        Self {
            shared: Arc::new(Shared {
                state: StdMutex::new(MailboxState {
                    queue: VecDeque::new(),
                    owner: None,
                    next_lease_id: 0,
                }),
                notify: Notify::new(),
            }),
        }
    }

    #[allow(clippy::unnecessary_wraps)] // same Result contract as OutboundMailbox::push
    pub(crate) fn push(&self, msg: String) -> Result<(), &'static str> {
        self.shared.lock().queue.push_back(msg);
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
        }
    }
}

fn take_next(shared: &Shared, id: u64) -> Result<String, TryRecvError> {
    let mut state = shared.lock();
    if state.owner != Some(id) {
        return Err(TryRecvError::Disconnected);
    }
    state.queue.pop_front().ok_or(TryRecvError::Empty)
}

impl HttpLease {
    /// Next frame, or `None` once this lease has been taken over.
    pub(crate) async fn recv(&mut self) -> Option<String> {
        let shared = self.shared.clone();
        loop {
            // Register for a wakeup before checking, so a push in between is not missed.
            let mut notified = std::pin::pin!(shared.notify.notified());
            notified.as_mut().enable();
            match take_next(&shared, self.id) {
                Ok(msg) => return Some(msg),
                Err(TryRecvError::Disconnected) => return None,
                Err(TryRecvError::Empty) => notified.await,
            }
        }
    }

    pub(crate) fn try_recv(&mut self) -> Result<String, TryRecvError> {
        take_next(&self.shared, self.id)
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
}
