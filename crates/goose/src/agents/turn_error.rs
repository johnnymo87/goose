// Modified by Jonathan Mohrbacher, 2026: turn-error kinds reported in the ACP session/prompt response.
//! Why a turn ended in failure, for clients that only see the turn's stop reason.
//!
//! Most provider failures end the turn with an ordinary assistant message and
//! `end_turn`, so a client cannot tell a failed turn from a finished one. The
//! agent records the failure here and the ACP server reports it as
//! `_meta.goose.turnError` on the `session/prompt` response.

use crate::conversation::message::MessageErrorKind;
use crate::conversation::Conversation;
use serde::Serialize;

pub const TURN_ERROR_MESSAGE_MAX_BYTES: usize = 2048;

const TRUNCATION_MARKER: &str = "…";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnErrorKind {
    ContextLength,
    Refusal,
    Network,
    Provider,
    SelfCompact,
    RetryExhausted,
    EmptyResponse,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TurnError {
    pub kind: TurnErrorKind,
    pub message: String,
}

impl TurnError {
    pub fn new(kind: TurnErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: truncate_message(message.into()),
        }
    }

    pub fn to_meta_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

fn truncate_message(message: String) -> String {
    if message.len() <= TURN_ERROR_MESSAGE_MAX_BYTES {
        return message;
    }
    let mut end = TURN_ERROR_MESSAGE_MAX_BYTES - TRUNCATION_MARKER.len();
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{TRUNCATION_MARKER}", &message[..end])
}

/// The state machine ends a failed turn on a trailing error message; recipe
/// retry failures are told apart by the messages `ops_retry` writes.
pub(crate) fn turn_error_from_trailing_error(conversation: &Conversation) -> Option<TurnError> {
    let error = conversation
        .last()?
        .content
        .iter()
        .find_map(|content| content.as_error())?;
    let kind = match error.kind {
        MessageErrorKind::ContextLengthExceeded => TurnErrorKind::ContextLength,
        _ if crate::agents::state_machine::is_retry_failure_message(&error.message) => {
            TurnErrorKind::RetryExhausted
        }
        _ => TurnErrorKind::Provider,
    };
    Some(TurnError::new(kind, error.message.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::Message;

    #[test]
    fn turn_error_message_under_the_cap_is_kept_whole() {
        let error = TurnError::new(TurnErrorKind::Network, "connection reset");
        assert_eq!(error.message, "connection reset");
    }

    #[test]
    fn turn_error_message_over_the_cap_is_cut_on_a_char_boundary() {
        let message = "é".repeat(TURN_ERROR_MESSAGE_MAX_BYTES);
        let error = TurnError::new(TurnErrorKind::Provider, message);
        assert!(error.message.len() <= TURN_ERROR_MESSAGE_MAX_BYTES);
        assert!(error.message.ends_with(TRUNCATION_MARKER));
        assert!(error
            .message
            .trim_end_matches(TRUNCATION_MARKER)
            .chars()
            .all(|c| c == 'é'));
    }

    #[test]
    fn turn_error_serializes_kind_in_snake_case() {
        let value = TurnError::new(TurnErrorKind::RetryExhausted, "gave up").to_meta_value();
        assert_eq!(
            value,
            serde_json::json!({ "kind": "retry_exhausted", "message": "gave up" })
        );
    }

    #[test]
    fn turn_error_from_trailing_error_maps_kinds() {
        let conversation = |message: Message| Conversation::new_unvalidated(vec![message]);

        let context = conversation(
            Message::assistant().with_error(MessageErrorKind::ContextLengthExceeded, "too long"),
        );
        assert_eq!(
            turn_error_from_trailing_error(&context).map(|e| e.kind),
            Some(TurnErrorKind::ContextLength)
        );

        let retry = conversation(Message::assistant().with_error(
            MessageErrorKind::Other,
            "Maximum retry attempts (2) exceeded. Unable to complete the task successfully.",
        ));
        assert_eq!(
            turn_error_from_trailing_error(&retry).map(|e| e.kind),
            Some(TurnErrorKind::RetryExhausted)
        );

        let provider =
            conversation(Message::assistant().with_error(MessageErrorKind::Other, "HTTP 500"));
        assert_eq!(
            turn_error_from_trailing_error(&provider),
            Some(TurnError::new(TurnErrorKind::Provider, "HTTP 500"))
        );

        let clean = conversation(Message::assistant().with_text("done"));
        assert_eq!(turn_error_from_trailing_error(&clean), None);
    }
}
