// Added by Jonathan Mohrbacher, 2026: self-compaction operation on the state machine.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::{CallToolResult, ContentBlock, JsonObject, Role, Tool, ToolAnnotations};
use tracing_futures::Instrument;

use crate::agents::state_machine::ops_llm::{chat_span, record_chat_usage};
use crate::agents::state_machine::ops_toolcalling::request_was_advertised;
use crate::agents::state_machine::{
    applied, assistant_turn_count, ends_turn, messages_since_kickoff, not_applicable, yielded,
    yielded_with, Emitter, GooseEffect, Operation, OperationResult,
};
use crate::config::Config;
use crate::context_mgmt::{compact_messages_with_kind, CompactionKind};
use crate::conversation::message::{
    Message, MessageContent, MessageErrorKind, SystemNotificationType, ToolRequest,
};
use crate::conversation::Conversation;
use crate::providers::base::Provider;
use crate::session::session_manager::SessionType;
use crate::session::Session;
use goose_agent::max_turns::MAX_TURNS_MESSAGE;
use goose_providers::model::ModelConfig;

pub const SELF_COMPACT_TOOL_NAME: &str = "self_compact_and_resume";
pub const SELF_COMPACT_PROMPT_MAX_CHARS: usize = 16_000;
pub const SELF_COMPACT_TOOL_DESCRIPTION: &str =
    "Compact your own context and continue with a resumption prompt you write. Use it only when your context has grown long enough that earlier detail is being crowded out, or when the user asks you to; it is not a routine step between tasks, and works at most once per turn. Before calling it, persist anything durable (beads, plan files, commits) that a summary may lose. After this step the conversation so far is replaced by a summary and `prompt` is delivered to you as the next user message; you keep working in the same turn. Call it alone, as the last tool call of the step. The prompt must stand on its own: restate the goal, the current state, and the first concrete action.";
pub const SELF_COMPACT_SUCCESS_MESSAGE: &str =
    "Compaction runs right after this step; your prompt will be the next user message. Do not call other tools in this step.";
pub const SELF_COMPACT_ALREADY_COMPACTED_MESSAGE: &str =
    "Already self-compacted in this turn; continue from the resume prompt. Context overflow is still handled automatically.";
pub const SELF_COMPACTED_NOTE_TEXT: &str = "Context self-compacted.";

pub const SELF_COMPACT_OP_NAME: &str = "self_compact";
pub const NOTE_ARMED: &str = "armed";
pub const NOTE_OUTCOME: &str = "outcome";
pub const NOTE_TURNS_BEFORE: &str = "turns_before";

pub const OUTCOME_FAILED: &str = "failed";
pub const OUTCOME_SKIPPED: &str = "skipped";

pub fn self_compact_tool() -> Tool {
    let schema: JsonObject = serde_json::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The resumption prompt: it becomes your next user message after compaction. Make it self-contained."
            }
        },
        "required": ["prompt"]
    }))
    .expect("valid self_compact schema");

    Tool::new(
        SELF_COMPACT_TOOL_NAME.to_string(),
        SELF_COMPACT_TOOL_DESCRIPTION.to_string(),
        schema,
    )
    .annotate(
        ToolAnnotations::with_title("Self Compact and Resume".to_string())
            .read_only(true)
            .destructive(false)
            .idempotent(false)
            .open_world(false),
    )
}

pub fn validate_self_compact_args(args: Option<&JsonObject>) -> Result<String, String> {
    let args = args.ok_or_else(|| "Missing required argument: 'prompt'".to_string())?;
    let prompt_val = args
        .get("prompt")
        .ok_or_else(|| "Missing required argument: 'prompt'".to_string())?;
    let prompt_str = prompt_val
        .as_str()
        .ok_or_else(|| "Argument 'prompt' must be a string".to_string())?;
    if prompt_str.trim().is_empty() {
        return Err("Argument 'prompt' cannot be empty".to_string());
    }
    if prompt_str.chars().count() > SELF_COMPACT_PROMPT_MAX_CHARS {
        return Err(format!(
            "Argument 'prompt' exceeds maximum length of {} characters",
            SELF_COMPACT_PROMPT_MAX_CHARS
        ));
    }
    Ok(prompt_str.to_string())
}

pub fn self_compact_skipped_notice(prompt: &str) -> Message {
    let mut msg = Message::assistant()
        .with_text(format!(
            "Self-compaction skipped: the turn limit was reached. Resume prompt (not delivered):\n\n{prompt}"
        ))
        .with_generated_id();
    msg.metadata.set_operation_note(
        SELF_COMPACT_OP_NAME,
        NOTE_OUTCOME,
        serde_json::json!(OUTCOME_SKIPPED),
    );
    msg
}

pub fn self_compact_progress_notice() -> Message {
    Message::assistant().with_system_notification(
        SystemNotificationType::InlineMessage,
        "Compacting context (self-requested)...",
    )
}

pub fn self_compacted_note_message(created: i64) -> Message {
    let mut msg = Message::assistant()
        .with_text(SELF_COMPACTED_NOTE_TEXT)
        .user_only()
        .with_generated_id();
    if msg.created < created {
        msg.created = created;
    }
    msg
}

pub fn self_compact_resume_message(prompt: &str, created: i64, turns_before: u32) -> Message {
    let mut msg = Message::user()
        .with_text(prompt)
        .with_steer()
        .with_generated_id();
    if msg.created < created {
        msg.created = created;
    }
    msg.metadata.set_operation_note(
        SELF_COMPACT_OP_NAME,
        NOTE_TURNS_BEFORE,
        serde_json::json!(turns_before),
    );
    msg
}

pub fn self_compact_failed_notice(error: &str, prompt: &str) -> Message {
    let mut msg = Message::assistant()
        .with_text(format!(
            "Self-compaction failed: {error}. Context unchanged; the resume prompt was not delivered:\n\n{prompt}"
        ))
        .with_generated_id();
    msg.metadata.set_operation_note(
        SELF_COMPACT_OP_NAME,
        NOTE_OUTCOME,
        serde_json::json!(OUTCOME_FAILED),
    );
    msg
}

pub fn armed_prompt(messages: &[Message]) -> Option<String> {
    for msg in messages.iter().rev() {
        if msg
            .metadata
            .operation_note(SELF_COMPACT_OP_NAME, NOTE_OUTCOME)
            .is_some()
        {
            return None;
        }
        if let Some(val) = msg
            .metadata
            .operation_note(SELF_COMPACT_OP_NAME, NOTE_ARMED)
        {
            return val.as_str().map(|s| s.to_string());
        }
    }
    None
}

pub fn compacted_turns_before(messages: &[Message]) -> Option<u32> {
    for msg in messages.iter().rev() {
        if let Some(val) = msg
            .metadata
            .operation_note(SELF_COMPACT_OP_NAME, NOTE_TURNS_BEFORE)
        {
            return val.as_u64().map(|v| v as u32);
        }
        if msg.role == Role::User
            && !msg.is_tool_response()
            && msg.is_user_visible()
            && !msg.metadata.steer
        {
            return None;
        }
    }
    None
}

pub fn self_compact_available(session: &Session, provider: &dyn Provider) -> bool {
    if session.session_type == SessionType::SubAgent {
        return false;
    }
    if session
        .recipe
        .as_ref()
        .is_some_and(|r| r.retry.is_some() || r.response.is_some())
    {
        return false;
    }
    let enabled = Config::global()
        .get_param::<bool>("GOOSE_SELF_COMPACT")
        .unwrap_or(true);
    if !enabled {
        return false;
    }
    if provider.manages_own_context() {
        return false;
    }
    true
}

fn has_unanswered_tool_requests(messages: &[Message]) -> bool {
    let answered: std::collections::HashSet<&str> = messages
        .iter()
        .flat_map(Message::get_tool_response_ids)
        .collect();

    messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .flat_map(|message| &message.content)
        .filter_map(MessageContent::as_tool_request)
        .any(|request| {
            !answered.contains(request.id.as_str()) && !request.was_executed_externally()
        })
}

pub struct SelfCompactOperation {
    provider: Arc<dyn Provider>,
    model_config: ModelConfig,
    max_turns: u32,
    outcome_yielded: AtomicBool,
}

impl SelfCompactOperation {
    pub fn new(provider: Arc<dyn Provider>, model_config: ModelConfig, max_turns: u32) -> Self {
        Self {
            provider,
            model_config,
            max_turns,
            outcome_yielded: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for SelfCompactOperation {
    fn name(&self) -> &'static str {
        SELF_COMPACT_OP_NAME
    }

    async fn inference_tools(&self, session: &Session) -> Result<Vec<Tool>> {
        if self_compact_available(session, self.provider.as_ref()) {
            Ok(vec![self_compact_tool()])
        } else {
            Ok(Vec::new())
        }
    }

    async fn finalize_cancellation(
        &self,
        _session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Vec<GooseEffect> {
        if self.outcome_yielded.load(Ordering::SeqCst) {
            return Vec::new();
        }
        let Ok(messages) = messages_since_kickoff(conversation) else {
            return Vec::new();
        };
        if let Some(prompt) = armed_prompt(messages) {
            self.outcome_yielded.store(true, Ordering::SeqCst);
            let notice = self_compact_failed_notice("cancelled", &prompt);
            let notice = emit.message(notice);
            vec![notice.into()]
        } else {
            Vec::new()
        }
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let messages = messages_since_kickoff(conversation)?;

        if let Some(prompt) = armed_prompt(messages) {
            if !has_unanswered_tool_requests(messages) {
                if !self_compact_available(session, self.provider.as_ref()) {
                    self.outcome_yielded.store(true, Ordering::SeqCst);
                    let notice = self_compact_failed_notice(
                        "self-compaction is no longer available in this session",
                        &prompt,
                    );
                    let notice = emit.message(notice);
                    return yielded_with([notice.into()]);
                }

                if assistant_turn_count(messages) >= self.max_turns {
                    self.outcome_yielded.store(true, Ordering::SeqCst);
                    let notice = emit.message(self_compact_skipped_notice(&prompt));
                    let max_turns = emit.message(
                        Message::assistant()
                            .with_error(MessageErrorKind::ResourceExhausted, MAX_TURNS_MESSAGE),
                    );
                    return yielded_with([notice.into(), max_turns.into()]);
                }

                emit.message(self_compact_progress_notice());

                let span = chat_span(
                    self.provider.as_ref(),
                    &self.model_config,
                    &session.id,
                    "compaction",
                );

                let compact_result = tokio::select! {
                    biased;
                    _ = emit.cancelled() => {
                        Err(anyhow::anyhow!("cancelled"))
                    }
                    res = compact_messages_with_kind(
                        self.provider.as_ref(),
                        &self.model_config,
                        &session.id,
                        conversation,
                        CompactionKind::SelfRequested,
                    ).instrument(span.clone()) => res,
                };

                match compact_result {
                    Ok(compaction) => {
                        record_chat_usage(&span, &compaction.usage);

                        let turns = assistant_turn_count(messages);
                        let max_created = compaction
                            .conversation
                            .messages()
                            .iter()
                            .map(|m| m.created)
                            .max()
                            .unwrap_or_else(|| chrono::Utc::now().timestamp());

                        let note = emit.message(self_compacted_note_message(max_created));
                        let resume =
                            emit.message(self_compact_resume_message(&prompt, note.created, turns));

                        let mut new_messages = compaction.conversation.messages().clone();
                        new_messages.push(note);
                        new_messages.push(resume);
                        let new_conversation = Conversation::new_unvalidated(new_messages);

                        return applied([GooseEffect::CompactConversation {
                            conversation: new_conversation,
                            usage: Some(compaction.usage),
                        }]);
                    }
                    Err(err) => {
                        span.record("error.type", "compaction_error");
                        let is_cancelled = emit.cancel_token().is_cancelled();
                        let reason = if is_cancelled {
                            "cancelled"
                        } else {
                            &err.to_string()
                        };
                        self.outcome_yielded.store(true, Ordering::SeqCst);
                        let notice = self_compact_failed_notice(reason, &prompt);
                        let notice = emit.message(notice);
                        return yielded_with([notice.into()]);
                    }
                }
            }
        }

        let answered: std::collections::HashSet<&str> = messages
            .iter()
            .flat_map(Message::get_tool_response_ids)
            .collect();

        let self_compact_requests: Vec<ToolRequest> = messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .flat_map(|message| &message.content)
            .filter_map(MessageContent::as_tool_request)
            .filter(|request| {
                !answered.contains(request.id.as_str())
                    && !request.was_executed_externally()
                    && request
                        .tool_call
                        .as_ref()
                        .is_ok_and(|call| call.name == SELF_COMPACT_TOOL_NAME)
                    && request_was_advertised(messages, request)
            })
            .cloned()
            .collect();

        if !self_compact_requests.is_empty() {
            let mut response = Message::user();
            let already_compacted = compacted_turns_before(conversation.messages()).is_some();
            let mut last_valid_prompt = None;

            for request in self_compact_requests {
                let result = if !self_compact_available(session, self.provider.as_ref()) {
                    Ok(CallToolResult::error(vec![ContentBlock::text(
                        "self_compact_and_resume is not available in this session",
                    )]))
                } else if already_compacted {
                    Ok(CallToolResult::error(vec![ContentBlock::text(
                        SELF_COMPACT_ALREADY_COMPACTED_MESSAGE,
                    )]))
                } else {
                    let tool_call = request.tool_call.as_ref().unwrap();
                    match validate_self_compact_args(tool_call.arguments.as_ref()) {
                        Ok(prompt) => {
                            last_valid_prompt = Some(prompt);
                            Ok(CallToolResult::success(vec![ContentBlock::text(
                                SELF_COMPACT_SUCCESS_MESSAGE,
                            )]))
                        }
                        Err(err) => Ok(CallToolResult::error(vec![ContentBlock::text(err)])),
                    }
                };
                response.add_tool_response_with_metadata(
                    request.id,
                    result,
                    request.metadata.as_ref(),
                );
            }

            if let Some(prompt) = last_valid_prompt {
                response.metadata.set_operation_note(
                    SELF_COMPACT_OP_NAME,
                    NOTE_ARMED,
                    serde_json::json!(prompt),
                );
            }

            let response = emit.message(response);
            return applied([response.into()]);
        }

        not_applicable()
    }
}

pub struct SelfCompactTurnLimitOperation {
    max_turns: u32,
}

impl SelfCompactTurnLimitOperation {
    pub fn new(max_turns: u32) -> Self {
        Self { max_turns }
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for SelfCompactTurnLimitOperation {
    fn name(&self) -> &'static str {
        "self_compact_turn_limit"
    }

    async fn run(
        &self,
        _session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let messages = messages_since_kickoff(conversation)?;
        if messages.last().and_then(Message::error_kind)
            == Some(MessageErrorKind::ResourceExhausted)
        {
            return yielded();
        }

        let Some(turns_before) = compacted_turns_before(conversation.messages()) else {
            return not_applicable();
        };

        let turns = turns_before + assistant_turn_count(messages);
        if ends_turn(messages) || turns < self.max_turns {
            return not_applicable();
        }

        let message =
            Message::assistant().with_error(MessageErrorKind::ResourceExhausted, MAX_TURNS_MESSAGE);
        let message = emit.message(message);
        yielded_with([message.into()])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_self_compact_tool_schema_and_annotations() {
        let tool = self_compact_tool();
        assert_eq!(tool.name, SELF_COMPACT_TOOL_NAME);
        assert_eq!(
            tool.description.as_deref(),
            Some(SELF_COMPACT_TOOL_DESCRIPTION)
        );
        let schema = &tool.input_schema;
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
        let properties = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("properties object");
        let prompt_prop = properties
            .get("prompt")
            .and_then(|v| v.as_object())
            .expect("prompt property");
        assert_eq!(
            prompt_prop.get("type").and_then(|v| v.as_str()),
            Some("string")
        );
        assert!(prompt_prop.get("description").is_some());
        let required = schema
            .get("required")
            .and_then(|v| v.as_array())
            .expect("required array");
        assert!(required.iter().any(|v| v.as_str() == Some("prompt")));

        let annotations = tool.annotations.expect("annotations present");
        assert_eq!(annotations.read_only_hint, Some(true));
        assert_eq!(annotations.destructive_hint, Some(false));
        assert_eq!(annotations.idempotent_hint, Some(false));
        assert_eq!(annotations.open_world_hint, Some(false));
        assert_eq!(
            annotations.title.as_deref(),
            Some("Self Compact and Resume")
        );
    }

    #[test]
    fn test_validate_self_compact_args_valid() {
        let args: JsonObject = serde_json::from_value(json!({
            "prompt": "Continue with implementing feature X"
        }))
        .unwrap();
        let res = validate_self_compact_args(Some(&args));
        assert_eq!(res.unwrap(), "Continue with implementing feature X");
    }

    #[test]
    fn test_validate_self_compact_args_missing() {
        assert!(validate_self_compact_args(None).is_err());
        let empty: JsonObject = serde_json::from_value(json!({})).unwrap();
        assert!(validate_self_compact_args(Some(&empty)).is_err());
    }

    #[test]
    fn test_validate_self_compact_args_not_a_string() {
        let args: JsonObject = serde_json::from_value(json!({
            "prompt": 12345
        }))
        .unwrap();
        assert!(validate_self_compact_args(Some(&args)).is_err());
    }

    #[test]
    fn test_validate_self_compact_args_empty_after_trim() {
        let args: JsonObject = serde_json::from_value(json!({
            "prompt": "   \n\t  "
        }))
        .unwrap();
        assert!(validate_self_compact_args(Some(&args)).is_err());
    }

    #[test]
    fn test_validate_self_compact_args_oversized() {
        let long_str = "a".repeat(SELF_COMPACT_PROMPT_MAX_CHARS + 1);
        let args: JsonObject = serde_json::from_value(json!({
            "prompt": long_str
        }))
        .unwrap();
        assert!(validate_self_compact_args(Some(&args)).is_err());
    }

    #[test]
    fn test_self_compact_messages_properties_self_compact() {
        let skipped = self_compact_skipped_notice("Resume after turns");
        assert_eq!(skipped.role, Role::Assistant);
        assert!(skipped.is_user_visible());
        assert!(skipped.is_agent_visible());
        assert!(skipped.id.is_some());
        assert_eq!(
            skipped
                .metadata
                .operation_note(SELF_COMPACT_OP_NAME, NOTE_OUTCOME)
                .and_then(|v| v.as_str()),
            Some(OUTCOME_SKIPPED)
        );
        assert!(skipped.as_concat_text().contains(
            "Self-compaction skipped: the turn limit was reached. Resume prompt (not delivered):\n\nResume after turns"
        ));

        let progress = self_compact_progress_notice();
        assert_eq!(progress.role, Role::Assistant);
        assert!(progress.is_user_visible());
        assert!(!progress.is_agent_visible());
        assert!(progress.content.iter().any(|c| match c {
            MessageContent::SystemNotification(n) => {
                n.notification_type == SystemNotificationType::InlineMessage
                    && n.msg.contains("Compacting context (self-requested)...")
            }
            _ => false,
        }));

        let note = self_compacted_note_message(1_700_000_000);
        assert_eq!(note.role, Role::Assistant);
        assert!(note.is_user_visible());
        assert!(!note.is_agent_visible());
        assert_eq!(note.as_concat_text(), SELF_COMPACTED_NOTE_TEXT);
        assert!(note.created >= 1_700_000_000);
        assert!(note.id.is_some());

        let resume = self_compact_resume_message("Resume now", 1_700_000_000, 3);
        assert_eq!(resume.role, Role::User);
        assert!(resume.is_user_visible());
        assert!(resume.is_agent_visible());
        assert!(resume.metadata.steer);
        assert_eq!(
            resume
                .metadata
                .operation_note(SELF_COMPACT_OP_NAME, NOTE_TURNS_BEFORE)
                .and_then(|v| v.as_u64()),
            Some(3)
        );
        assert_eq!(resume.as_concat_text(), "Resume now");
        assert!(resume.created >= 1_700_000_000);
        assert!(resume.id.is_some());

        let failed = self_compact_failed_notice("network error", "Resume prompt");
        assert_eq!(failed.role, Role::Assistant);
        assert!(failed.is_user_visible());
        assert!(failed.is_agent_visible());
        assert!(failed.id.is_some());
        assert_eq!(
            failed
                .metadata
                .operation_note(SELF_COMPACT_OP_NAME, NOTE_OUTCOME)
                .and_then(|v| v.as_str()),
            Some(OUTCOME_FAILED)
        );
        assert!(failed.as_concat_text().contains(
            "Self-compaction failed: network error. Context unchanged; the resume prompt was not delivered:\n\nResume prompt"
        ));
    }

    #[test]
    fn test_armed_prompt_derivation() {
        assert_eq!(armed_prompt(&[]), None);

        let mut msg1 = Message::user().with_text("kickoff");
        msg1.metadata
            .set_operation_note(SELF_COMPACT_OP_NAME, NOTE_ARMED, json!("prompt 1"));
        assert_eq!(armed_prompt(&[msg1.clone()]), Some("prompt 1".to_string()));

        let mut msg2 = Message::assistant().with_text("outcome");
        msg2.metadata
            .set_operation_note(SELF_COMPACT_OP_NAME, NOTE_OUTCOME, json!(OUTCOME_FAILED));
        assert_eq!(armed_prompt(&[msg1.clone(), msg2.clone()]), None);

        let mut msg3 = Message::user().with_text("tool resp");
        msg3.metadata
            .set_operation_note(SELF_COMPACT_OP_NAME, NOTE_ARMED, json!("prompt 2"));
        assert_eq!(
            armed_prompt(&[msg1.clone(), msg3.clone()]),
            Some("prompt 2".to_string())
        );

        assert_eq!(
            armed_prompt(&[msg1, msg2, msg3]),
            Some("prompt 2".to_string())
        );
    }

    #[test]
    fn test_compacted_turns_before_derivation() {
        assert_eq!(compacted_turns_before(&[]), None);

        let resume = self_compact_resume_message("Resume", 1_700_000_000, 4);
        assert_eq!(
            compacted_turns_before(std::slice::from_ref(&resume)),
            Some(4)
        );

        let mut steer = Message::user().with_text("steer message");
        steer.metadata.steer = true;
        assert_eq!(compacted_turns_before(&[resume.clone(), steer]), Some(4));

        let assistant = Message::assistant().with_text("working");
        assert_eq!(
            compacted_turns_before(&[resume.clone(), assistant]),
            Some(4)
        );

        let genuine_user = Message::user().with_text("genuine user message");
        assert_eq!(compacted_turns_before(&[resume, genuine_user]), None);
    }

    struct DummyTestProvider;
    #[async_trait]
    impl Provider for DummyTestProvider {
        fn get_name(&self) -> &str {
            "dummy"
        }
        async fn stream(
            &self,
            _model_config: &ModelConfig,
            _system_prompt: &str,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> Result<crate::providers::base::MessageStream, goose_providers::errors::ProviderError>
        {
            unimplemented!()
        }
    }

    fn test_emitter() -> Emitter {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        Emitter::new(tx, tokio_util::sync::CancellationToken::new())
    }

    #[tokio::test]
    async fn test_self_compact_turn_limit_operation() {
        let op = SelfCompactTurnLimitOperation::new(3);
        let session = Session::default();
        let emit = test_emitter();

        let conv1 = Conversation::new_unvalidated(vec![
            Message::user().with_text("start"),
            Message::assistant().with_text("done"),
        ]);
        let res1 = op.run(&session, &conv1, &emit).await.unwrap();
        assert!(matches!(res1, OperationResult::NotApplicable));

        let resume = self_compact_resume_message("Resume", 1_700_000_000, 2);
        let asst = Message::assistant();
        let mut tool_resp = Message::user();
        tool_resp.add_tool_response_with_metadata(
            "call_id",
            Ok(rmcp::model::CallToolResult::success(vec![
                ContentBlock::text("ok"),
            ])),
            None,
        );
        let conv2 = Conversation::new_unvalidated(vec![resume, asst, tool_resp]);
        let res2 = op.run(&session, &conv2, &emit).await.unwrap();
        assert!(matches!(res2, OperationResult::Applied(_)));

        let err_msg =
            Message::assistant().with_error(MessageErrorKind::ResourceExhausted, MAX_TURNS_MESSAGE);
        let conv3 = Conversation::new_unvalidated(vec![
            self_compact_resume_message("Resume", 1_700_000_000, 2),
            err_msg,
        ]);
        let res3 = op.run(&session, &conv3, &emit).await.unwrap();
        assert!(
            matches!(res3, OperationResult::Applied(ref a) if a.status == crate::agents::state_machine::RunStatus::Yielded && a.effects.is_empty())
        );
    }

    #[tokio::test]
    async fn test_finalize_cancellation_guards() {
        let provider = Arc::new(DummyTestProvider);
        let op = SelfCompactOperation::new(provider, ModelConfig::new("mock"), 5);
        let session = Session::default();
        let emit = test_emitter();

        let mut tool_resp = Message::user();
        tool_resp.metadata.set_operation_note(
            SELF_COMPACT_OP_NAME,
            NOTE_ARMED,
            json!("cancel test prompt"),
        );
        let conv =
            Conversation::new_unvalidated(vec![Message::user().with_text("kickoff"), tool_resp]);

        let effects = op.finalize_cancellation(&session, &conv, &emit).await;
        assert_eq!(effects.len(), 1);

        let effects2 = op.finalize_cancellation(&session, &conv, &emit).await;
        assert!(effects2.is_empty());
    }

    #[tokio::test]
    async fn test_finalize_cancellation_skips_prompt_with_persisted_outcome() {
        let provider = Arc::new(DummyTestProvider);
        let op = SelfCompactOperation::new(provider, ModelConfig::new("mock"), 5);
        let session = Session::default();
        let emit = test_emitter();

        let mut tool_resp = Message::user();
        tool_resp.metadata.set_operation_note(
            SELF_COMPACT_OP_NAME,
            NOTE_ARMED,
            json!("cancel test prompt"),
        );
        let conv = Conversation::new_unvalidated(vec![
            Message::user().with_text("kickoff"),
            tool_resp,
            self_compact_failed_notice("cancelled", "cancel test prompt"),
        ]);

        let effects = op.finalize_cancellation(&session, &conv, &emit).await;
        assert!(effects.is_empty());
    }
}
