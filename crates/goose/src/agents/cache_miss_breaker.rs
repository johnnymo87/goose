// Modified by Jonathan Mohrbacher, 2026: cache-miss breaker for Gemini delegates.
//! Stop a Gemini delegate whose prompt cache has stopped hitting.
//!
//! Vertex sometimes stops serving a single session from its prompt cache: every
//! call re-reads the whole context at full price, while concurrent sessions on
//! the same model hit 97-99% (workstation-ke9f; about $117 of one day's $170).
//! A stuck session usually stays stuck, and a fresh session gets a fresh chance.
//!
//! The rule (same as opencode's cache-miss-breaker plugin): count consecutive
//! model calls with at least `min_context` prompt tokens and zero cache reads.
//! A call with any cache read resets the count; a smaller call neither counts nor
//! resets. At `trip_steps` in a row the delegate is stopped, but only for a
//! Gemini model that has reported a cache hit in this process before (a model
//! that never reports cache reads would otherwise always trip). Only delegates
//! feed that set, and it starts empty at each goose-serve start, so a delegate
//! stuck from its first call is counted only from the first hit any Gemini
//! delegate in the process reports (opencode seeds it from every session).
//!
//! It is a step observer over the delegate's `AgentEvent` stream, so it needs
//! nothing from the agent loop beyond `AgentEvent::Usage` (one per provider
//! call) and `AgentEvent::Message`. Delegates are the only callers, so a
//! top-level session is never stopped.
//!
//! Env: `GOOSE_CACHE_MISS_BREAKER=off` disables it;
//! `GOOSE_CACHE_MISS_BREAKER_STEPS` and `GOOSE_CACHE_MISS_BREAKER_MIN_CONTEXT`
//! override the thresholds.

use crate::agents::final_output_tool::FINAL_OUTPUT_TOOL_NAME;
use crate::agents::AgentEvent;
use crate::conversation::message::{Message, MessageContent};
use goose_providers::conversation::token_usage::ProviderUsage;
use rmcp::model::Role;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

pub const TRIP_STEPS: u32 = 10;
pub const MIN_CONTEXT: i64 = 50_000;

/// One delegate's run of uncached calls.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Streak {
    pub count: u32,
    pub cost: f64,
    pub context: i64,
}

/// The pure trip condition, for one session.
#[derive(Debug)]
pub struct Detector {
    trip_steps: u32,
    min_context: i64,
    streak: Streak,
    tripped: bool,
}

impl Detector {
    pub fn new(trip_steps: u32, min_context: i64) -> Self {
        Self {
            trip_steps: trip_steps.max(1),
            min_context,
            streak: Streak::default(),
            tripped: false,
        }
    }

    /// Record one model call. `context` is all prompt tokens, cache reads included
    /// (goose's `input_tokens`). True exactly once, on the call that trips.
    pub fn step(&mut self, context: i64, cache_read: i64, cost: f64) -> bool {
        if cache_read > 0 {
            // A hit re-arms it: a delegate that tripped on its final answer and kept
            // going (it was never stopped) gets counted afresh.
            self.streak = Streak::default();
            self.tripped = false;
            return false;
        }
        if self.tripped {
            return false;
        }
        if context < self.min_context {
            return false;
        }
        self.streak.count += 1;
        self.streak.cost += cost;
        self.streak.context = context;
        if self.streak.count < self.trip_steps {
            return false;
        }
        self.tripped = true;
        true
    }

    pub fn streak(&self) -> &Streak {
        &self.streak
    }
}

/// Models that have reported a cache read in this process.
fn models_with_cache() -> &'static Mutex<HashSet<String>> {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn is_gemini(model: &str) -> bool {
    model
        .rsplit('/')
        .next()
        .is_some_and(|m| m.to_ascii_lowercase().starts_with("gemini"))
}

fn env_number<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok())
}

/// What the delegate's event loop should do after an event.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Continue,
    /// Cancel the delegate now and fail it with [`Breaker::stop_message`].
    Stop,
}

/// Which a provider reports first within one model call. Gemini (google format)
/// streams the message parts and then the usage; OpenAI-style providers send the
/// usage with or before the message, so agent.rs yields it first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    MessageFirst,
    UsageFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Watching,
    /// Tripped; the trip step's response has not asked for more tools (yet).
    Tripped,
    /// Tripped on a step that asked for tools; stop once their results are in.
    AwaitingToolResults,
    Stopped,
}

/// Per-delegate observer. Feed it every event of the delegate's stream.
#[derive(Debug)]
pub struct Breaker {
    session_id: String,
    detector: Option<Detector>,
    model: String,
    order: Option<Order>,
    /// Message-first only: the current step asked for a tool other than final output.
    step_wants_tools: bool,
    /// Wait for the trip step's tool results before stopping (the state machine
    /// runs a step's tools concurrently with reporting its usage; the legacy loop
    /// has already finished them by the time it yields the usage).
    defer_to_tool_results: bool,
    state: State,
}

impl Breaker {
    pub fn from_env(session_id: &str) -> Self {
        let off = std::env::var("GOOSE_CACHE_MISS_BREAKER")
            .map(|v| matches!(v.trim(), "off" | "0" | "false"))
            .unwrap_or(false);
        let detector = (!off).then(|| {
            Detector::new(
                env_number("GOOSE_CACHE_MISS_BREAKER_STEPS").unwrap_or(TRIP_STEPS),
                env_number("GOOSE_CACHE_MISS_BREAKER_MIN_CONTEXT").unwrap_or(MIN_CONTEXT),
            )
        });
        Self::with_detector(session_id, detector)
    }

    pub fn with_detector(session_id: &str, detector: Option<Detector>) -> Self {
        Self {
            session_id: session_id.to_string(),
            detector,
            model: String::new(),
            order: None,
            step_wants_tools: false,
            defer_to_tool_results: false,
            state: State::Watching,
        }
    }

    /// For the state-machine loop: let the trip step's tools finish first.
    pub fn deferring_to_tool_results(mut self) -> Self {
        self.defer_to_tool_results = true;
        self
    }

    pub fn observe(&mut self, event: &AgentEvent) -> Verdict {
        if self.state == State::Stopped || self.detector.is_none() {
            return Verdict::Continue;
        }
        match event {
            AgentEvent::Usage(usage) => {
                self.order.get_or_insert(Order::UsageFirst);
                self.observe_usage(usage)
            }
            AgentEvent::Message(message) => {
                if message.role == Role::Assistant {
                    self.order.get_or_insert(Order::MessageFirst);
                }
                self.observe_message(message)
            }
            _ => Verdict::Continue,
        }
    }

    fn observe_usage(&mut self, usage: &ProviderUsage) -> Verdict {
        // Message-first: the tool requests seen since the last usage are this step's.
        let step_wants_tools =
            std::mem::take(&mut self.step_wants_tools) && self.order == Some(Order::MessageFirst);
        if self.state == State::AwaitingToolResults {
            // Results never showed up as a message; do not let another call through.
            return self.stop();
        }
        if !is_gemini(&usage.model) {
            return Verdict::Continue;
        }
        let read = i64::from(usage.usage.cache_read_input_tokens.unwrap_or(0));
        let context = i64::from(usage.usage.input_tokens.unwrap_or(0));
        {
            let mut seen = models_with_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if read > 0 {
                seen.insert(usage.model.clone());
            } else if !seen.contains(&usage.model) {
                // A model that has never reported a cache hit here may just not report cache.
                return Verdict::Continue;
            }
        }
        if read > 0 && self.state == State::Tripped {
            // The cache recovered before the delegate asked for more tools.
            self.state = State::Watching;
        }
        let Some(detector) = self.detector.as_mut() else {
            return Verdict::Continue;
        };
        if !detector.step(context, read, usage.cost.unwrap_or(0.0)) {
            return Verdict::Continue;
        }
        self.model = usage.model.clone();
        let s = detector.streak();
        tracing::warn!(
            session_id = %self.session_id,
            model = %self.model,
            "cache-miss-breaker: tripped: {} consecutive uncached calls at ~{}K context, ${:.2}",
            s.count,
            s.context / 1000,
            s.cost
        );
        if !step_wants_tools {
            self.state = State::Tripped;
            return Verdict::Continue;
        }
        if self.defer_to_tool_results {
            self.state = State::AwaitingToolResults;
            return Verdict::Continue;
        }
        self.stop()
    }

    fn stop(&mut self) -> Verdict {
        self.state = State::Stopped;
        tracing::warn!(session_id = %self.session_id, "cache-miss-breaker: stopping delegate");
        Verdict::Stop
    }

    /// The trip step's own response decides: a request for more tools stops the
    /// delegate before its next model call; a final answer (text, or the
    /// final-output tool) is let through.
    fn observe_message(&mut self, message: &Message) -> Verdict {
        if self.state == State::AwaitingToolResults {
            let has_results = message
                .content
                .iter()
                .any(|c| matches!(c, MessageContent::ToolResponse(_)));
            return if has_results {
                self.stop()
            } else {
                Verdict::Continue
            };
        }
        if message.role != Role::Assistant {
            return Verdict::Continue;
        }
        let wants_more = message.content.iter().any(|c| match c {
            MessageContent::ToolRequest(req) => req
                .tool_call
                .as_ref()
                .map_or(true, |call| call.name != FINAL_OUTPUT_TOOL_NAME),
            _ => false,
        });
        if !wants_more {
            return Verdict::Continue;
        }
        match self.state {
            State::Tripped if self.defer_to_tool_results => {
                self.state = State::AwaitingToolResults;
                Verdict::Continue
            }
            State::Tripped => self.stop(),
            _ => {
                self.step_wants_tools = true;
                Verdict::Continue
            }
        }
    }

    pub fn stopped(&self) -> bool {
        self.state == State::Stopped
    }

    /// The failure the lead sees. `last_response` is the delegate's last assistant text.
    pub fn stop_message(&self, last_response: Option<&str>) -> String {
        let s = self
            .detector
            .as_ref()
            .map(|d| d.streak().clone())
            .unwrap_or_default();
        stop_message(&self.session_id, &self.model, &s, last_response)
    }
}

pub fn stop_message(
    session_id: &str,
    model: &str,
    streak: &Streak,
    last_response: Option<&str>,
) -> String {
    let mut lines = vec![
        format!(
            "STOPPED by the cache-miss breaker: delegate {session_id} ({model}) made {} consecutive model calls with no prompt-cache hits at ~{}K context (${:.2} for those calls).",
            streak.count,
            streak.context / 1000,
            streak.cost
        ),
        "This is almost certainly a Vertex-side fault that sticks to one session (workstation-ke9f), not something wrong with the task. Its work is UNFINISHED; check its working directory for partial edits or commits before continuing.".to_string(),
        "Delegate it again ONCE as a NEW delegate (do not resume this one): the same instructions plus what it already did, with \"re-dispatch #1 after cache-miss breaker\" in them. If that delegate is stopped too, do not dispatch a third time: report it.".to_string(),
    ];
    if let Some(text) = last_response.map(str::trim).filter(|t| !t.is_empty()) {
        lines.push("--- its last output ---".to_string());
        lines.push(text.to_string());
    }
    lines.join("\n")
}

/// The delegate's last non-empty assistant text, for the stop message.
pub fn last_assistant_text(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .filter(|m| m.role == Role::Assistant)
        .map(Message::as_concat_text)
        .find(|t| !t.trim().is_empty())
}

/// Typed error so the foreground runner can recognise a breaker stop.
#[derive(Debug)]
pub struct Stopped(pub String);

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Stopped {}

#[cfg(test)]
mod tests {
    use super::*;
    use goose_providers::conversation::token_usage::Usage;

    fn usage(model: &str, input: i32, read: i32) -> AgentEvent {
        let mut provider_usage = ProviderUsage::new(
            model.to_string(),
            Usage {
                input_tokens: Some(input),
                output_tokens: Some(100),
                total_tokens: Some(input + 100),
                cache_read_input_tokens: Some(read),
                cache_write_input_tokens: None,
            },
        );
        provider_usage.cost = Some(0.05);
        AgentEvent::Usage(provider_usage)
    }

    fn tool_call(name: &str) -> AgentEvent {
        AgentEvent::Message(Message::assistant().with_tool_request(
            "call-1",
            Ok(rmcp::model::CallToolRequestParams::new(name.to_string())),
        ))
    }

    #[test]
    fn trips_on_the_tenth_consecutive_uncached_large_call() {
        let mut d = Detector::new(TRIP_STEPS, MIN_CONTEXT);
        for _ in 0..9 {
            assert!(!d.step(60_000, 0, 0.1));
        }
        assert!(d.step(60_000, 0, 0.1));
        assert_eq!(d.streak().count, 10);
        assert!((d.streak().cost - 1.0).abs() < 1e-9);
        assert!(!d.step(60_000, 0, 0.1), "trips only once");
    }

    #[test]
    fn a_cache_read_resets_and_small_calls_neither_count_nor_reset() {
        let mut d = Detector::new(TRIP_STEPS, MIN_CONTEXT);
        for _ in 0..9 {
            d.step(60_000, 0, 0.0);
        }
        assert!(!d.step(60_000, 1, 0.0), "a hit resets");
        assert_eq!(d.streak().count, 0);
        for _ in 0..9 {
            d.step(60_000, 0, 0.0);
        }
        assert!(!d.step(49_999, 0, 0.0), "below the floor does not count");
        assert_eq!(d.streak().count, 9, "and does not reset");
        assert!(d.step(50_000, 0, 0.0));
    }

    #[test]
    fn breaker_stops_at_the_trip_steps_next_tool_call() {
        let model = "gemini-test-trips";
        let mut b = Breaker::with_detector("child", Some(Detector::new(3, 1000)));
        // The process must have seen this model hit the cache.
        assert_eq!(b.observe(&usage(model, 2000, 1500)), Verdict::Continue);
        // Usage reported before the step's message (OpenAI-style): the previous
        // step's tool request must not stop the trip step, only its own does.
        for _ in 0..2 {
            assert_eq!(b.observe(&usage(model, 2000, 0)), Verdict::Continue);
            assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Continue);
        }
        assert_eq!(b.observe(&usage(model, 2000, 0)), Verdict::Continue);
        assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Stop);
        assert!(b.stopped());
        assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Continue);
        let msg = b.stop_message(Some("edited a.rs"));
        assert!(msg.contains("STOPPED by the cache-miss breaker"), "{msg}");
        assert!(msg.contains("3 consecutive model calls"), "{msg}");
        assert!(
            msg.contains("re-dispatch #1 after cache-miss breaker"),
            "{msg}"
        );
        assert!(msg.ends_with("edited a.rs"), "{msg}");
    }

    #[test]
    fn breaker_stops_when_usage_follows_the_tool_request() {
        // Gemini streams the step's message first and its usage last.
        let model = "gemini-test-usage-last";
        let mut b = Breaker::with_detector("child", Some(Detector::new(3, 1000)));
        b.observe(&tool_call("developer__shell"));
        b.observe(&usage(model, 2000, 1500));
        for _ in 0..2 {
            assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Continue);
            assert_eq!(b.observe(&usage(model, 2000, 0)), Verdict::Continue);
        }
        assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Continue);
        assert_eq!(b.observe(&usage(model, 2000, 0)), Verdict::Stop);
    }

    #[test]
    fn usage_first_trip_step_with_a_final_answer_is_let_through() {
        let model = "gemini-test-usage-first-final";
        let mut b = Breaker::with_detector("child", Some(Detector::new(2, 1000)));
        b.observe(&usage(model, 2000, 1500));
        b.observe(&tool_call("developer__shell"));
        b.observe(&usage(model, 2000, 0));
        b.observe(&tool_call("developer__shell"));
        assert_eq!(b.observe(&usage(model, 2000, 0)), Verdict::Continue);
        assert_eq!(
            b.observe(&AgentEvent::Message(Message::assistant().with_text("done"))),
            Verdict::Continue
        );
        assert!(!b.stopped());
    }

    #[test]
    fn deferred_breaker_waits_for_the_trip_steps_tool_results() {
        let model = "gemini-test-deferred";
        let mut b = Breaker::with_detector("child", Some(Detector::new(1, 1000)))
            .deferring_to_tool_results();
        b.observe(&tool_call("developer__shell"));
        b.observe(&usage(model, 2000, 1500));
        assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Continue);
        assert_eq!(b.observe(&usage(model, 2000, 0)), Verdict::Continue);
        let results = Message::user()
            .with_tool_response("call-1", Ok(rmcp::model::CallToolResult::success(vec![])));
        assert_eq!(b.observe(&AgentEvent::Message(results)), Verdict::Stop);
    }

    #[test]
    fn deferred_breaker_never_lets_another_call_through() {
        let model = "gemini-test-deferred-fallback";
        let mut b = Breaker::with_detector("child", Some(Detector::new(1, 1000)))
            .deferring_to_tool_results();
        b.observe(&tool_call("developer__shell"));
        b.observe(&usage(model, 2000, 1500));
        b.observe(&tool_call("developer__shell"));
        b.observe(&usage(model, 2000, 0));
        assert_eq!(b.observe(&usage(model, 2000, 0)), Verdict::Stop);
    }

    #[test]
    fn a_cache_hit_after_a_text_only_trip_rearms_the_breaker() {
        let model = "gemini-test-rearm";
        let mut b = Breaker::with_detector("child", Some(Detector::new(1, 1000)));
        b.observe(&AgentEvent::Message(
            Message::assistant().with_text("thinking"),
        ));
        b.observe(&usage(model, 2000, 1500));
        b.observe(&AgentEvent::Message(
            Message::assistant().with_text("text only"),
        ));
        assert_eq!(b.observe(&usage(model, 2000, 0)), Verdict::Continue);
        b.observe(&usage(model, 2000, 1500));
        assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Continue);
        assert_eq!(
            b.observe(&usage(model, 2000, 0)),
            Verdict::Stop,
            "trips again"
        );
    }

    #[test]
    fn breaker_lets_a_final_answer_through() {
        let model = "gemini-test-final";
        let mut b = Breaker::with_detector("child", Some(Detector::new(1, 1000)));
        b.observe(&usage(model, 2000, 1500));
        b.observe(&usage(model, 2000, 0));
        assert_eq!(
            b.observe(&tool_call(FINAL_OUTPUT_TOOL_NAME)),
            Verdict::Continue
        );
        assert_eq!(
            b.observe(&AgentEvent::Message(Message::assistant().with_text("done"))),
            Verdict::Continue
        );
        assert!(!b.stopped());
    }

    #[test]
    fn breaker_ignores_non_gemini_and_models_never_seen_caching() {
        let mut b = Breaker::with_detector("child", Some(Detector::new(1, 1000)));
        b.observe(&usage("claude-opus-5-5", 2000, 0));
        assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Continue);
        b.observe(&usage("gemini-test-never-cached", 2000, 0));
        assert_eq!(b.observe(&tool_call("developer__shell")), Verdict::Continue);
        let mut off = Breaker::with_detector("child", None);
        off.observe(&usage("gemini-test-trips", 2000, 0));
        assert_eq!(
            off.observe(&tool_call("developer__shell")),
            Verdict::Continue
        );
    }
}
