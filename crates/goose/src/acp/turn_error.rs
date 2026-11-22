// Modified by Jonathan Mohrbacher, 2026: session/prompt turn errors carry a kind and a bounded message in error.data; max turns is recognised as a normal stop.
//! Why a turn failed, for clients that only see the `session/prompt` error.
//!
//! Upstream fails the prompt with `data: {reason: "turn_failed"}` and the full
//! error text as the RPC message. Clients that branch on the cause (stall-watch's
//! circuit breaker, honk's turn log) also need a stable kind, so the data
//! becomes `{reason, kind, message}`. The kind is recovered from the persisted
//! error message, which is all the server has once the turn has ended.

use crate::agents::state_machine::{is_compaction_failure_message, is_retry_failure_message};
use crate::conversation::message::{ErrorContent, Message, MessageContent, MessageErrorKind};
use goose_agent::max_turns::MAX_TURNS_MESSAGE;
use goose_providers::errors::ProviderError;
use serde::Serialize;

pub(crate) const TURN_ERROR_MESSAGE_MAX_BYTES: usize = 2048;

const TRUNCATION_MARKER: &str = "…";
const REFUSAL_PREFIX: &str = "The provider refused this request.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TurnErrorKind {
    Authentication,
    Compaction,
    ContextLength,
    CreditsExhausted,
    MaxTurns,
    Network,
    Provider,
    Refusal,
    ResourceExhausted,
    RetryExhausted,
}

pub(crate) fn turn_error_kind(error: &ErrorContent) -> TurnErrorKind {
    match error.kind {
        MessageErrorKind::Authentication => TurnErrorKind::Authentication,
        MessageErrorKind::ContextLengthExceeded => TurnErrorKind::ContextLength,
        MessageErrorKind::CreditsExhausted => TurnErrorKind::CreditsExhausted,
        MessageErrorKind::ResourceExhausted if error.message == MAX_TURNS_MESSAGE => {
            TurnErrorKind::MaxTurns
        }
        MessageErrorKind::ResourceExhausted => TurnErrorKind::ResourceExhausted,
        MessageErrorKind::Other => kind_from_message(&error.message),
    }
}

fn kind_from_message(message: &str) -> TurnErrorKind {
    if is_compaction_failure_message(message) {
        TurnErrorKind::Compaction
    } else if is_retry_failure_message(message) {
        TurnErrorKind::RetryExhausted
    } else if message.starts_with(&ProviderError::NetworkError(String::new()).to_string()) {
        TurnErrorKind::Network
    } else if message.starts_with(REFUSAL_PREFIX) {
        TurnErrorKind::Refusal
    } else {
        TurnErrorKind::Provider
    }
}

/// Reaching the turn limit is a normal stop (`stopReason: max_turn_requests`),
/// not a failed turn, even though the notice is persisted as an error.
pub(crate) fn is_max_turns_message(message: &Message) -> bool {
    message.content.iter().any(|content| {
        matches!(
            content,
            MessageContent::Error(error) if turn_error_kind(error) == TurnErrorKind::MaxTurns
        )
    })
}

pub(crate) fn turn_failed_data(error: &ErrorContent) -> serde_json::Value {
    serde_json::json!({
        "reason": crate::acp::TURN_FAILED_REASON,
        "kind": turn_error_kind(error),
        "message": truncate_message(&error.message),
    })
}

fn truncate_message(message: &str) -> String {
    if message.len() <= TURN_ERROR_MESSAGE_MAX_BYTES {
        return message.to_string();
    }
    let budget = TURN_ERROR_MESSAGE_MAX_BYTES - TRUNCATION_MARKER.len();
    let mut kept = String::with_capacity(TURN_ERROR_MESSAGE_MAX_BYTES);
    for c in message.chars() {
        if kept.len() + c.len_utf8() > budget {
            break;
        }
        kept.push(c);
    }
    kept.push_str(TRUNCATION_MARKER);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind_of(message: Message) -> TurnErrorKind {
        turn_error_kind(message.error().expect("message carries an error"))
    }

    fn provider_error_kind(err: ProviderError) -> TurnErrorKind {
        kind_of(Message::from_provider_error(&err))
    }

    #[test]
    fn only_the_max_turns_error_is_a_normal_stop() {
        let error =
            |text: &str| Message::assistant().with_error(MessageErrorKind::ResourceExhausted, text);
        assert!(is_max_turns_message(&error(MAX_TURNS_MESSAGE)));
        assert!(!is_max_turns_message(&error("Rate limit exceeded")));
        assert!(!is_max_turns_message(
            &Message::assistant().with_text(MAX_TURNS_MESSAGE)
        ));
    }

    #[test]
    fn provider_errors_map_to_their_kind() {
        assert_eq!(
            provider_error_kind(ProviderError::NetworkError("connection reset".into())),
            TurnErrorKind::Network
        );
        assert_eq!(
            provider_error_kind(ProviderError::Refusal {
                details: "no".into(),
                category: Some("cyber".into()),
            }),
            TurnErrorKind::Refusal
        );
        assert_eq!(
            provider_error_kind(ProviderError::ContextLengthExceeded("too long".into())),
            TurnErrorKind::ContextLength
        );
        assert_eq!(
            provider_error_kind(ProviderError::Authentication("expired".into())),
            TurnErrorKind::Authentication
        );
        assert_eq!(
            provider_error_kind(ProviderError::ServerError("HTTP 503".into())),
            TurnErrorKind::Provider
        );
        assert_eq!(
            provider_error_kind(ProviderError::RateLimitExceeded {
                details: "slow down".into(),
                retry_delay: None,
            }),
            TurnErrorKind::Provider
        );
    }

    #[test]
    fn loop_errors_map_to_their_kind() {
        let other = |text: &str| Message::assistant().with_error(MessageErrorKind::Other, text);
        assert_eq!(
            kind_of(other(
                "Maximum retry attempts (2) exceeded. Unable to complete the task successfully."
            )),
            TurnErrorKind::RetryExhausted
        );
        assert_eq!(
            kind_of(other("Retry logic encountered an error: boom")),
            TurnErrorKind::RetryExhausted
        );
        assert_eq!(
            kind_of(other(
                "Ran into this error trying to compact: refused.\n\nPlease try again"
            )),
            TurnErrorKind::Compaction
        );
        let exhausted =
            |text: &str| Message::assistant().with_error(MessageErrorKind::ResourceExhausted, text);
        assert_eq!(
            kind_of(exhausted(MAX_TURNS_MESSAGE)),
            TurnErrorKind::MaxTurns
        );
        assert_eq!(
            kind_of(exhausted("turn budget exhausted")),
            TurnErrorKind::ResourceExhausted
        );
        assert_eq!(
            kind_of(Message::assistant().with_error(MessageErrorKind::CreditsExhausted, "top up")),
            TurnErrorKind::CreditsExhausted
        );
    }

    #[test]
    fn turn_failed_data_carries_reason_kind_and_message() {
        let message = Message::from_provider_error(&ProviderError::NetworkError("reset".into()));
        let error = message.error().unwrap();
        assert_eq!(
            turn_failed_data(error),
            serde_json::json!({
                "reason": "turn_failed",
                "kind": "network",
                "message": error.message,
            })
        );
    }

    #[test]
    fn turn_error_message_under_the_cap_is_kept_whole() {
        assert_eq!(truncate_message("connection reset"), "connection reset");
    }

    #[test]
    fn turn_error_message_over_the_cap_is_cut_on_a_char_boundary() {
        let message = truncate_message(&"é".repeat(TURN_ERROR_MESSAGE_MAX_BYTES));
        assert!(message.len() <= TURN_ERROR_MESSAGE_MAX_BYTES);
        assert!(message.ends_with(TRUNCATION_MARKER));
        assert!(message
            .trim_end_matches(TRUNCATION_MARKER)
            .chars()
            .all(|c| c == 'é'));
    }
}
