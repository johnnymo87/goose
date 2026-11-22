// Modified by Jonathan Mohrbacher, 2026: client-chosen prompt message ids and truncate_from_message (honk /undo).
//! A client can choose the id goose stores a prompt's user message under
//! (`_meta.goose.messageId` on `session/prompt`) and later delete history from that
//! message onward with `_goose/unstable/session/conversation/truncate_from_message`.
//! Upstream's truncate cuts by timestamp, which also deletes a message stored in the
//! same second before the boundary.

use super::*;

pub(super) const TRUNCATE_FROM_MESSAGE_CAPABILITY: &str = "truncateFromMessage";

/// The caller owns uniqueness; truncation cuts from the first row with the id.
pub(super) fn message_id_from_meta(
    meta: Option<&Meta>,
) -> Result<Option<String>, agent_client_protocol::Error> {
    let Some(value) = meta
        .and_then(|meta| meta.get("goose"))
        .and_then(|goose| goose.get("messageId"))
    else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    match value.as_str().map(str::trim) {
        Some(id) if !id.is_empty() => Ok(Some(id.to_string())),
        _ => Err(agent_client_protocol::Error::invalid_params()
            .data("_meta.goose.messageId must be a non-empty string")),
    }
}

impl AcpConnection {
    pub(super) async fn on_truncate_session_conversation_from_message(
        &self,
        req: TruncateSessionConversationFromMessageRequest,
    ) -> Result<TruncateSessionConversationFromMessageResponse, agent_client_protocol::Error> {
        let session_id = req.session_id.trim();
        if session_id.is_empty() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("sessionId cannot be empty")
            );
        }
        let message_id = req.message_id.trim();
        if message_id.is_empty() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("messageId cannot be empty")
            );
        }
        // A running turn (or a live voice interaction, which writes messages) would write
        // its tail after the cut. The check is not atomic with a prompt arriving next;
        // callers that need that (honkd) serialize prompts and truncation themselves.
        if self.runs().is_active(session_id) {
            return Err(agent_client_protocol::Error::invalid_request().data(format!(
                "session {session_id} has an active run or live voice interaction; truncate after it ends"
            )));
        }

        self.services
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|_| {
                agent_client_protocol::Error::resource_not_found(Some(session_id.to_string()))
                    .data(format!("Session not found: {session_id}"))
            })?;
        let found = self
            .services
            .session_manager
            .truncate_conversation_from_message(session_id, message_id)
            .await
            .internal_err()?;
        Ok(TruncateSessionConversationFromMessageResponse { found })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(value: serde_json::Value) -> Meta {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn message_id_from_meta_reads_goose_message_id() {
        assert_eq!(message_id_from_meta(None).unwrap(), None);
        for absent in [
            serde_json::json!({ "other": 1 }),
            serde_json::json!({ "goose": {} }),
            serde_json::json!({ "goose": { "messageId": null } }),
        ] {
            assert_eq!(message_id_from_meta(Some(&meta(absent))).unwrap(), None);
        }
        assert_eq!(
            message_id_from_meta(Some(&meta(
                serde_json::json!({ "goose": { "messageId": " msg_honk_1 " } })
            )))
            .unwrap()
            .as_deref(),
            Some("msg_honk_1")
        );
    }

    #[test]
    fn message_id_from_meta_rejects_empty_or_non_string() {
        for bad in [
            serde_json::json!(""),
            serde_json::json!("   "),
            serde_json::json!(7),
        ] {
            let error = message_id_from_meta(Some(&meta(
                serde_json::json!({ "goose": { "messageId": bad } }),
            )))
            .unwrap_err();
            assert_eq!(
                error.code,
                agent_client_protocol::schema::v1::ErrorCode::InvalidParams
            );
        }
    }

    #[test]
    fn agent_capabilities_advertise_truncate_from_message() {
        assert_eq!(
            agent_capabilities_meta()
                .and_then(|meta| meta.get("goose").cloned())
                .and_then(|goose| goose.get(TRUNCATE_FROM_MESSAGE_CAPABILITY).cloned()),
            Some(serde_json::json!(true))
        );
    }
}
