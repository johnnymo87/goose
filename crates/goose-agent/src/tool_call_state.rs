// Modified by Jonathan Mohrbacher, 2026: track tool calls sequentially so repeated tool call IDs across steps within a turn remain distinct.
use goose_provider_types::conversation::message::{
    ActionRequiredData, Message, MessageContent, ToolRequest, ToolResponse,
};
use goose_provider_types::permission::Permission;
use rmcp::model::{CallToolResult, ContentBlock};

use crate::operation::ConversationEffect;

pub const EXECUTABLE_KEY: &str = "goose.executable";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToolCallState<'a> {
    Responded(&'a ToolResponse),
    Unchecked,
    Executable,
    Blocked,
    AwaitingUser,
    UserAnswered(&'a Permission),
}

impl ToolCallState<'_> {
    pub fn is_open(&self) -> bool {
        !matches!(self, ToolCallState::Responded(_))
    }
}

pub fn tool_calls(messages: &[Message]) -> Vec<(&ToolRequest, ToolCallState<'_>)> {
    struct Tracked<'a> {
        request: &'a ToolRequest,
        response: Option<&'a ToolResponse>,
        confirmation_requested: bool,
        confirmation: Option<&'a Permission>,
    }

    let mut tracked: Vec<Tracked<'_>> = Vec::new();

    for content in messages.iter().flat_map(|message| &message.content) {
        match content {
            MessageContent::ToolRequest(request) => {
                if !tracked
                    .iter()
                    .any(|t| t.response.is_none() && t.request.id == request.id)
                {
                    tracked.push(Tracked {
                        request,
                        response: None,
                        confirmation_requested: false,
                        confirmation: None,
                    });
                }
            }
            MessageContent::ToolResponse(response) => {
                if let Some(target) = tracked
                    .iter_mut()
                    .find(|t| t.response.is_none() && t.request.id == response.id)
                {
                    target.response = Some(response);
                }
            }
            MessageContent::ActionRequired(action) => match &action.data {
                ActionRequiredData::ToolConfirmation { id, .. } => {
                    if let Some(target) = tracked
                        .iter_mut()
                        .rfind(|t| t.response.is_none() && t.request.id == *id)
                    {
                        target.confirmation_requested = true;
                    }
                }
                ActionRequiredData::ToolConfirmationResponse { id, permission } => {
                    if let Some(target) = tracked
                        .iter_mut()
                        .rfind(|t| t.response.is_none() && t.request.id == *id)
                    {
                        target.confirmation = Some(permission);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    tracked
        .into_iter()
        .map(|item| {
            let state = match (item.response, request_executable(item.request)) {
                (Some(response), _) => ToolCallState::Responded(response),
                (None, Some(true)) => ToolCallState::Executable,
                (None, None) => ToolCallState::Unchecked,
                (None, Some(false)) if !item.confirmation_requested => ToolCallState::Blocked,
                (None, Some(false)) => match item.confirmation {
                    Some(permission) => ToolCallState::UserAnswered(permission),
                    None => ToolCallState::AwaitingUser,
                },
            };
            (item.request, state)
        })
        .collect()
}

pub fn request_executable(request: &ToolRequest) -> Option<bool> {
    request
        .tool_meta
        .as_ref()
        .and_then(|meta| meta.get(EXECUTABLE_KEY))
        .and_then(serde_json::Value::as_bool)
}

pub(crate) fn mark_executable(tool_call_id: &str, executable: bool) -> ConversationEffect {
    ConversationEffect::PatchToolRequestMeta {
        tool_call_id: tool_call_id.to_string(),
        patch: serde_json::json!({ EXECUTABLE_KEY: executable }),
    }
}

pub(crate) fn add_error_response(tool_responses: &mut Message, request: &ToolRequest, text: &str) {
    tool_responses.add_tool_response_with_metadata(
        request.id.clone(),
        Ok(CallToolResult::error(vec![ContentBlock::text(text)])),
        request.metadata.as_ref(),
    );
}
