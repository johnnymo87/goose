use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::StreamExt;
use goose::agents::self_compact::{
    SELF_COMPACTED_NOTE_TEXT, SELF_COMPACT_ALREADY_COMPACTED_MESSAGE, SELF_COMPACT_TOOL_NAME,
};
use goose::agents::{Agent, AgentConfig, AgentEvent, GoosePlatform, SessionConfig};
use goose::config::permission::PermissionManager;
use goose::config::GooseMode;
use goose::conversation::message::Message;
use goose::conversation::Conversation;
use goose::providers::base::{stream_from_single_message, MessageStream, Provider};
use goose::session::session_manager::{Session, SessionManager, SessionType};
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::{CallToolRequestParams, Tool};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct RecordedCall {
    system_prompt: String,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    is_summary: bool,
}

type SummaryHook = Box<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>;

struct ScriptedSelfCompactProvider {
    normal_responses: Arc<Mutex<Vec<Message>>>,
    summary_response: Arc<Mutex<Option<Result<Message, ProviderError>>>>,
    recorded_calls: Arc<Mutex<Vec<RecordedCall>>>,
    on_summary: Arc<Mutex<Option<SummaryHook>>>,
}

impl ScriptedSelfCompactProvider {
    fn new(normal_responses: Vec<Message>) -> Self {
        Self {
            normal_responses: Arc::new(Mutex::new(normal_responses)),
            summary_response: Arc::new(Mutex::new(None)),
            recorded_calls: Arc::new(Mutex::new(Vec::new())),
            on_summary: Arc::new(Mutex::new(None)),
        }
    }

    fn with_summary_error(self, err: ProviderError) -> Self {
        Self {
            normal_responses: self.normal_responses,
            summary_response: Arc::new(Mutex::new(Some(Err(err)))),
            recorded_calls: self.recorded_calls,
            on_summary: self.on_summary,
        }
    }

    async fn set_on_summary<F>(&self, hook: F)
    where
        F: Fn() -> BoxFuture<'static, ()> + Send + Sync + 'static,
    {
        *self.on_summary.lock().await = Some(Box::new(hook));
    }
}

#[async_trait]
impl Provider for ScriptedSelfCompactProvider {
    async fn stream(
        &self,
        _model_config: &ModelConfig,
        system_prompt: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let is_summary = tools.is_empty();
        self.recorded_calls.lock().await.push(RecordedCall {
            system_prompt: system_prompt.to_string(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
            is_summary,
        });

        if is_summary {
            let hook = self.on_summary.lock().await.as_ref().map(|h| h());
            if let Some(fut) = hook {
                fut.await;
            }
            let resp = self.summary_response.lock().await.clone();
            match resp {
                Some(Ok(msg)) => Ok(stream_from_single_message(
                    msg,
                    ProviderUsage::new("mock-model".to_string(), Usage::default()),
                )),
                Some(Err(err)) => Err(err),
                None => Ok(stream_from_single_message(
                    Message::assistant().with_text("<mock summary of conversation>"),
                    ProviderUsage::new("mock-model".to_string(), Usage::default()),
                )),
            }
        } else {
            let mut guard = self.normal_responses.lock().await;
            let msg = if guard.is_empty() {
                Message::assistant().with_text("default fallback")
            } else {
                guard.remove(0)
            };
            Ok(stream_from_single_message(
                msg,
                ProviderUsage::new("mock-model".to_string(), Usage::default()),
            ))
        }
    }

    async fn get_context_limit(&self, _model: &str, override_limit: Option<usize>) -> usize {
        override_limit.unwrap_or(100_000)
    }

    fn get_name(&self) -> &str {
        "scripted-self-compact"
    }
}

async fn setup_agent(temp_dir: &TempDir, mode: GooseMode) -> Result<(Agent, Arc<SessionManager>)> {
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("data")));
    let permission_manager = Arc::new(PermissionManager::new(temp_dir.path().join("data")));
    let config = AgentConfig::new(
        session_manager.clone(),
        permission_manager,
        None,
        mode,
        true,
        GoosePlatform::GooseCli,
    );
    let agent = Agent::with_config(config);
    Ok((agent, session_manager))
}

struct TestContext {
    _temp_dir: TempDir,
    agent: Arc<Agent>,
    session_manager: Arc<SessionManager>,
    session: Session,
}

impl TestContext {
    async fn new() -> Result<Self> {
        Self::with_mode(GooseMode::Auto).await
    }

    async fn with_mode(mode: GooseMode) -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let (agent, session_manager) = setup_agent(&temp_dir, mode).await?;
        let session = session_manager
            .create_session(
                temp_dir.path().to_path_buf(),
                "test".to_string(),
                SessionType::User,
                mode,
            )
            .await?;
        Ok(Self {
            _temp_dir: temp_dir,
            agent: Arc::new(agent),
            session_manager,
            session,
        })
    }

    fn session_config(&self, max_turns: Option<u32>) -> SessionConfig {
        SessionConfig {
            id: self.session.id.clone(),
            schedule_id: None,
            max_turns,
            retry_config: None,
        }
    }
}

#[tokio::test]
async fn test_happy_path_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    // Script:
    // 1. Tool call self_compact_and_resume{prompt: "R"}
    // 2. Text "done"
    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .agent
        .reply(user_msg, ctx.session_config(Some(5)), false, None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    // Assert provider calls are exactly turn, summary, turn
    let calls = recorded_calls.lock().await.clone();
    assert_eq!(
        calls.len(),
        3,
        "Expected exactly 3 calls: turn, summary, turn"
    );
    assert!(!calls[0].is_summary, "Call 0 should be normal turn");
    assert!(calls[1].is_summary, "Call 1 should be summary call");
    assert!(!calls[2].is_summary, "Call 2 should be normal turn");

    // The last turn's request messages contain summary, continuation, resume "R", turn context (no pre-compaction content)
    for m in &calls[2].messages {
        let text = m.as_concat_text();
        assert!(
            !text.contains("start work"),
            "Pre-compaction message found in last turn: {text}"
        );
        assert!(
            !text.contains("call_1"),
            "Pre-compaction tool request found in last turn: {text}"
        );
    }

    // Assert exact request order in final request: summary, continuation, resume, turn context
    let summary_pos = calls[2]
        .messages
        .iter()
        .position(|m| m.as_concat_text().contains("mock summary"))
        .expect("summary in final request");
    let cont_pos = calls[2]
        .messages
        .iter()
        .position(|m| {
            m.as_concat_text()
                .contains("You compacted your own context")
        })
        .expect("continuation in final request");
    let resume_pos = calls[2]
        .messages
        .iter()
        .position(|m| m.as_concat_text().contains("R"))
        .expect("resume in final request");
    let tc_pos = calls[2]
        .messages
        .iter()
        .position(|m| m.as_concat_text().contains("<turn-context>"))
        .expect("turn context in final request");

    assert!(summary_pos < cont_pos, "summary must precede continuation");
    assert!(cont_pos < resume_pos, "continuation must precede resume");
    assert!(
        resume_pos <= tc_pos,
        "resume must precede or accompany turn context"
    );

    // Events include Usage event for compaction (retained tokens), HistoryReplaced, and resume user message
    let usage_idx = events
        .iter()
        .position(|e| match e {
            AgentEvent::Usage(u) => {
                u.usage.output_tokens.is_none()
                    && u.usage.total_tokens.is_some_and(|t| t > 0)
                    && u.usage.input_tokens == u.usage.total_tokens
            }
            _ => false,
        })
        .expect("Usage event following compaction missing");

    let history_idx = events
        .iter()
        .position(|e| matches!(e, AgentEvent::HistoryReplaced(_)))
        .expect("HistoryReplaced event missing");

    let resume_idx = events
        .iter()
        .position(|e| match e {
            AgentEvent::Message(m) => {
                m.role == rmcp::model::Role::User && m.metadata.steer && m.as_concat_text() == "R"
            }
            _ => false,
        })
        .expect("Resume user message with steer metadata missing from events");

    assert!(
        usage_idx <= history_idx,
        "Usage event must precede or accompany HistoryReplaced"
    );
    assert!(
        history_idx < resume_idx,
        "HistoryReplaced must precede resume message"
    );

    // HistoryReplaced carries conversation WITHOUT trailing note + resume
    let history_replaced_conv = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::HistoryReplaced(c) => Some(c),
            _ => None,
        })
        .expect("HistoryReplaced conversation");
    assert!(
        !history_replaced_conv
            .messages()
            .iter()
            .any(|m| m.as_concat_text() == SELF_COMPACTED_NOTE_TEXT),
        "HistoryReplaced must not contain compaction note"
    );
    assert!(
        !history_replaced_conv
            .messages()
            .iter()
            .any(|m| m.as_concat_text() == "R"),
        "HistoryReplaced must not contain resume prompt"
    );

    // Simulate CLI/scheduler state: replacing conversation and then appending messages results in exactly 1 note and 1 resume
    let mut client_messages: Vec<Message> = Vec::new();
    for event in &events {
        match event {
            AgentEvent::HistoryReplaced(c) => {
                client_messages = c.messages().clone();
            }
            AgentEvent::Message(m) => {
                client_messages.push(m.clone());
            }
            _ => {}
        }
    }
    let note_count = client_messages
        .iter()
        .filter(|m| m.as_concat_text() == SELF_COMPACTED_NOTE_TEXT)
        .count();
    let resume_count = client_messages
        .iter()
        .filter(|m| m.as_concat_text() == "R")
        .count();
    assert_eq!(note_count, 1, "Client must have exactly 1 note message");
    assert_eq!(resume_count, 1, "Client must have exactly 1 resume prompt");

    Ok(())
}

#[tokio::test]
async fn test_reload_order_and_metadata_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("original task");
    let reply_stream = ctx
        .agent
        .reply(user_msg, ctx.session_config(Some(5)), false, None)
        .await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    // Reload conversation from DB: get_session(id, true)
    let reloaded = ctx
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conversation = reloaded.conversation.expect("conversation exists");
    let messages = conversation.messages();

    // 1. Originals: agent-invisible / user-visible
    assert_eq!(messages[0].as_concat_text(), "original task");
    assert!(!messages[0].is_agent_visible());
    assert!(messages[0].is_user_visible());

    // Assistant tool request "call_1"
    assert!(!messages[1].is_agent_visible());

    // User tool response "call_1"
    assert!(!messages[2].is_agent_visible());

    // 2. Summary: agent-only
    let summary_idx = messages
        .iter()
        .position(|m| m.as_concat_text().contains("mock summary"))
        .expect("summary message");
    assert!(messages[summary_idx].is_agent_visible());
    assert!(!messages[summary_idx].is_user_visible());

    // 3. Continuation: agent-only
    let cont_idx = messages
        .iter()
        .position(|m| {
            m.as_concat_text()
                .contains("You compacted your own context")
        })
        .expect("continuation message");
    assert!(messages[cont_idx].is_agent_visible());
    assert!(!messages[cont_idx].is_user_visible());

    // 4. Note: "Context self-compacted." user-only
    let note_idx = messages
        .iter()
        .position(|m| m.as_concat_text() == SELF_COMPACTED_NOTE_TEXT)
        .expect("note message");
    assert!(!messages[note_idx].is_agent_visible());
    assert!(messages[note_idx].is_user_visible());

    // 5. Resume: user + agent visible with steer=true
    let resume_idx = messages
        .iter()
        .position(|m| m.as_concat_text() == "Resume prompt")
        .expect("resume message");
    assert!(messages[resume_idx].is_agent_visible());
    assert!(messages[resume_idx].is_user_visible());
    assert!(messages[resume_idx].metadata.steer);

    // 6. Final assistant: "done"
    let done_idx = messages
        .iter()
        .position(|m| m.as_concat_text() == "done")
        .expect("done message");
    assert!(messages[done_idx].is_agent_visible());
    assert!(messages[done_idx].is_user_visible());

    // Verify ordering
    assert!(summary_idx < cont_idx);
    assert!(cont_idx < note_idx);
    assert!(note_idx < resume_idx);
    assert!(resume_idx < done_idx);

    // Timestamps monotonically non-decreasing
    let timestamps: Vec<i64> = messages.iter().map(|m| m.created).collect();
    for w in timestamps.windows(2) {
        assert!(w[0] <= w[1], "Timestamps inverted: {} > {}", w[0], w[1]);
    }

    Ok(())
}

#[tokio::test]
async fn test_turns_taken_not_reset_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tc1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R1" }));
    let other_tc = CallToolRequestParams::new("nonexistent_tool");
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tc1));
    let turn2_msg = Message::assistant().with_tool_request("call_2", Ok(other_tc));

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .agent
        .reply(user_msg, ctx.session_config(Some(2)), false, None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let calls = recorded_calls.lock().await.clone();
    assert_eq!(calls.len(), 3, "Expected 3 calls: turn 1, summary, turn 2");
    assert!(!calls[0].is_summary);
    assert!(calls[1].is_summary);
    assert!(!calls[2].is_summary);

    // Max turns message was yielded because turns_taken was not reset on compaction and reached 2
    let has_max_turns = events.iter().any(|e| match e {
        AgentEvent::Message(m) => m
            .as_concat_text()
            .contains("reached the maximum number of actions"),
        _ => false,
    });
    assert!(has_max_turns, "Max turns message missing");

    Ok(())
}

#[tokio::test]
async fn test_second_call_in_same_run_returns_error_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tc1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R1" }));
    let tc2 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R2" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tc1));
    let turn2_msg = Message::assistant().with_tool_request("call_2", Ok(tc2));
    let turn3_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![
        turn1_msg, turn2_msg, turn3_msg,
    ]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .agent
        .reply(user_msg, ctx.session_config(Some(5)), false, None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    // Provider calls: Turn 1 (calls R1) -> Summary 1 -> Turn 2 (calls R2, gets error) -> Turn 3 ("done")
    // Exactly ONE summary call occurred, because the second call returned an error and did not arm!
    let calls = recorded_calls.lock().await.clone();
    let summary_calls = calls.iter().filter(|c| c.is_summary).count();
    assert_eq!(
        summary_calls, 1,
        "Exactly 1 summary call expected; second call must not compact"
    );

    // Verify DB conversation has the error response for call_2
    let reloaded = ctx
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.unwrap();
    let error_found = conv.messages().iter().any(|m| {
        m.content.iter().any(|c| match c {
            goose::conversation::message::MessageContentBlock::ToolResponse(r) => match &r
                .tool_result
            {
                Ok(res) => res.content.iter().any(|b| {
                    b.as_text()
                        .is_some_and(|t| t.text.contains(SELF_COMPACT_ALREADY_COMPACTED_MESSAGE))
                }),
                _ => false,
            },
            _ => false,
        })
    });
    assert!(
        error_found,
        "Error message for second self_compact_and_resume call missing from history"
    );

    Ok(())
}

#[tokio::test]
async fn test_new_run_can_compact_again_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tc1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R1" }));
    let turn1_1 = Message::assistant().with_tool_request("call_1", Ok(tc1));
    let turn1_2 = Message::assistant().with_text("done 1");

    let tc2 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R2" }));
    let turn2_1 = Message::assistant().with_tool_request("call_2", Ok(tc2));
    let turn2_2 = Message::assistant().with_text("done 2");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![
        turn1_1, turn1_2, turn2_1, turn2_2,
    ]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    // Run 1
    let reply_stream1 = ctx
        .agent
        .reply(
            Message::user().with_text("run 1 prompt"),
            ctx.session_config(Some(5)),
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream1);
    while let Some(res) = reply_stream1.next().await {
        res?;
    }

    // Run 2 (new user turn)
    let reply_stream2 = ctx
        .agent
        .reply(
            Message::user().with_text("run 2 prompt"),
            ctx.session_config(Some(5)),
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream2);
    while let Some(res) = reply_stream2.next().await {
        res?;
    }

    // Assert that both runs were able to self-compact (2 summary calls)
    let calls = recorded_calls.lock().await.clone();
    let summary_calls = calls.iter().filter(|c| c.is_summary).count();
    assert_eq!(
        summary_calls, 2,
        "Both runs must be able to self-compact once"
    );

    Ok(())
}

#[tokio::test]
async fn test_failed_compaction_drains_pending_steers_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt R" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("handled steer after compaction failure");

    let provider = Arc::new(
        ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg])
            .with_summary_error(ProviderError::ExecutionError("provider broke".to_string())),
    );
    let recorded_calls = provider.recorded_calls.clone();

    let agent_clone = ctx.agent.clone();
    let session_id_clone = ctx.session.id.clone();
    provider
        .set_on_summary(move || {
            let ag = agent_clone.clone();
            let sid = session_id_clone.clone();
            Box::pin(async move {
                ag.steer(&sid, Message::user().with_text("urgent steer on failure"))
                    .await;
            })
        })
        .await;

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let reply_stream = ctx
        .agent
        .reply(
            Message::user().with_text("start work"),
            ctx.session_config(Some(5)),
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    // Failure notice was yielded
    let has_fail_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text().contains("Self-compaction failed:")
                && m.as_concat_text().contains("Resume prompt R")
        }
        _ => false,
    });
    assert!(has_fail_notice, "Failure notice missing from events");

    // Steer message was drained and received by provider on the next iteration
    let calls = recorded_calls.lock().await.clone();
    assert_eq!(
        calls.len(),
        3,
        "Calls: turn 1, failed summary, turn 2 handling steer"
    );
    let turn2_call = &calls[2];
    assert!(
        turn2_call
            .messages
            .iter()
            .any(|m| m.as_concat_text().contains("urgent steer on failure")),
        "Steer message must be drained and delivered to model after compaction failure"
    );

    Ok(())
}

#[tokio::test]
async fn test_chat_mode_does_not_arm_self_compact() -> Result<()> {
    let ctx = TestContext::with_mode(GooseMode::Chat).await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_chat", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("chat completed");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let reply_stream = ctx
        .agent
        .reply(
            Message::user().with_text("chat message"),
            ctx.session_config(Some(5)),
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    // In Chat mode, tool calls are skipped and do NOT arm self-compaction!
    let calls = recorded_calls.lock().await.clone();
    assert_eq!(
        calls.iter().filter(|c| c.is_summary).count(),
        0,
        "Chat mode must never arm self-compaction"
    );

    Ok(())
}

#[tokio::test]
async fn test_last_call_wins_and_failed_call_does_not_arm_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    // Scenario A: First call fails (empty prompt -> is_error), second call succeeds ("Winner Prompt").
    // Failed call does not arm; successful call arms ("Winner Prompt").
    let tc1_invalid = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "   " }));
    let tc2 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Winner Prompt" }));
    let turn1_msg = Message::assistant()
        .with_tool_request("call_1", Ok(tc1_invalid))
        .with_tool_request("call_2", Ok(tc2));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .agent
        .reply(user_msg, ctx.session_config(Some(5)), false, None)
        .await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    let reloaded = ctx
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.unwrap();
    assert!(conv
        .messages()
        .iter()
        .any(|m| m.as_concat_text() == "Winner Prompt"));

    // Scenario B: Failed call (empty prompt -> is_error) alone does NOT arm.
    let ctx_b = TestContext::new().await?;

    let tc_invalid = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "   " }));
    let turn1_b = Message::assistant().with_tool_request("call_err", Ok(tc_invalid));
    let turn2_b = Message::assistant().with_text("finished after err");

    let provider_b = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_b, turn2_b]));
    let recorded_b = provider_b.recorded_calls.clone();

    ctx_b
        .agent
        .update_provider(
            provider_b,
            ModelConfig::new("mock-model"),
            &ctx_b.session.id,
        )
        .await?;

    let reply_stream_b = ctx_b
        .agent
        .reply(
            Message::user().with_text("test_b"),
            ctx_b.session_config(Some(5)),
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream_b);
    while let Some(res) = reply_stream_b.next().await {
        res?;
    }

    let calls_b = recorded_b.lock().await.clone();
    assert_eq!(
        calls_b.len(),
        2,
        "No summary call should have occurred for failed call"
    );
    assert!(!calls_b[0].is_summary);
    assert!(!calls_b[1].is_summary);

    Ok(())
}

#[tokio::test]
async fn test_summary_provider_error_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt R" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));

    let provider = Arc::new(
        ScriptedSelfCompactProvider::new(vec![turn1_msg])
            .with_summary_error(ProviderError::ExecutionError("provider broke".to_string())),
    );

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .agent
        .reply(user_msg, ctx.session_config(Some(5)), false, None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    // Failure notice was yielded
    let has_fail_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text().contains("Self-compaction failed:")
                && m.as_concat_text().contains("provider broke")
                && m.as_concat_text().contains("Resume prompt R")
        }
        _ => false,
    });
    assert!(has_fail_notice, "Failure notice missing from events");

    // DB conversation unchanged
    let reloaded = ctx
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.expect("conversation exists");
    let msgs = conv.messages();

    assert_eq!(msgs[0].as_concat_text(), "start work");
    assert!(msgs[0].is_agent_visible());

    assert!(msgs.iter().any(|m| {
        m.as_concat_text().contains("Self-compaction failed:")
            && m.as_concat_text().contains("provider broke")
    }));

    assert!(!msgs.iter().any(|m| m.as_concat_text() == "Resume prompt R"));
    assert!(!msgs
        .iter()
        .any(|m| m.as_concat_text().contains("mock summary")));

    Ok(())
}

#[tokio::test]
async fn test_cancel_during_summary_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Cancel prompt R" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg]));

    let cancel_token = CancellationToken::new();
    let token_clone = cancel_token.clone();
    provider
        .set_on_summary(move || {
            let t = token_clone.clone();
            Box::pin(async move {
                t.cancel();
                std::future::pending::<()>().await;
            })
        })
        .await;

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .agent
        .reply(
            user_msg,
            ctx.session_config(Some(5)),
            false,
            Some(cancel_token),
        )
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let has_cancel_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text().contains("cancelled")
                && m.as_concat_text().contains("Cancel prompt R")
        }
        _ => false,
    });
    assert!(has_cancel_notice, "Cancel notice missing from events");

    let reloaded = ctx
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.unwrap();
    assert!(!conv
        .messages()
        .iter()
        .any(|m| m.as_concat_text() == "Cancel prompt R"));

    Ok(())
}

#[tokio::test]
async fn test_max_turns_skips_compaction_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R on last turn" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .agent
        .reply(user_msg, ctx.session_config(Some(1)), false, None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let calls = recorded_calls.lock().await.clone();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].is_summary);

    let has_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text()
                .contains("Self-compaction skipped: the turn limit was reached")
                && m.as_concat_text().contains("R on last turn")
        }
        _ => false,
    });
    assert!(has_notice, "Skipped notice missing");

    let reloaded = ctx
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.unwrap();
    assert!(!conv
        .messages()
        .iter()
        .any(|m| m.as_concat_text() == "R on last turn"));

    Ok(())
}

#[tokio::test]
async fn test_steer_queued_during_compaction_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    let agent_clone = ctx.agent.clone();
    let session_id_clone = ctx.session.id.clone();
    provider
        .set_on_summary(move || {
            let ag = agent_clone.clone();
            let sid = session_id_clone.clone();
            Box::pin(async move {
                ag.steer(&sid, Message::user().with_text("steer during compaction"))
                    .await;
            })
        })
        .await;

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .agent
        .reply(user_msg, ctx.session_config(Some(5)), false, None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let resume_pos = events
        .iter()
        .position(|e| match e {
            AgentEvent::Message(m) => m.as_concat_text() == "Resume prompt",
            _ => false,
        })
        .expect("resume event");

    let steer_pos = events
        .iter()
        .position(|e| match e {
            AgentEvent::Message(m) => m.as_concat_text() == "steer during compaction",
            _ => false,
        })
        .expect("steer event");

    assert!(
        resume_pos < steer_pos,
        "Resume prompt should arrive before steer message in events"
    );

    let calls = recorded_calls.lock().await.clone();
    let last_call_text = calls[2]
        .messages
        .iter()
        .map(|m| m.as_concat_text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(last_call_text.contains("Resume prompt"));
    assert!(last_call_text.contains("steer during compaction"));

    Ok(())
}

#[tokio::test]
async fn test_turn_context_after_compaction_not_stale_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    // ModelConfig uses "mock-model" whose context limit is 100,000 (0.8 threshold -> 80,000 compaction threshold).
    // Seed session usage at 72,000 tokens (90% of 80,000), so compaction_remaining_line generates "~8k tokens remaining".
    ctx.session_manager
        .update(&ctx.session.id)
        .usage(Usage::new(Some(72_000), Some(0), Some(72_000)))
        .apply()
        .await?;

    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let reply_stream = ctx
        .agent
        .reply(
            Message::user().with_text("start work"),
            ctx.session_config(Some(5)),
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    let calls = recorded_calls.lock().await.clone();
    assert_eq!(calls.len(), 3);

    // Turn 1 (before compaction): turn context contains "~8k tokens remaining"
    let turn1_tc = calls[0]
        .messages
        .iter()
        .find(|m| m.as_concat_text().contains("<turn-context>"))
        .expect("turn context in turn 1");
    assert!(
        turn1_tc.as_concat_text().contains("tokens remaining"),
        "Turn 1 turn context must show tokens remaining before compaction: {}",
        turn1_tc.as_concat_text()
    );

    // Turn 2 (after compaction): turn context was recomputed with retained tokens and does NOT show stale "~8k tokens remaining"
    let turn2_tc = calls[2]
        .messages
        .iter()
        .find(|m| m.as_concat_text().contains("<turn-context>"))
        .expect("turn context in turn 2");
    assert!(
        !turn2_tc.as_concat_text().contains("tokens remaining"),
        "Turn 2 turn context must not show stale tokens remaining after compaction: {}",
        turn2_tc.as_concat_text()
    );

    let reloaded = ctx
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.unwrap();
    let turn_context_msgs: Vec<_> = conv
        .messages()
        .iter()
        .filter(|m| m.is_turn_context())
        .collect();
    assert!(
        !turn_context_msgs.is_empty(),
        "Turn context message must be in DB"
    );
    assert!(turn_context_msgs.last().unwrap().is_agent_visible());
    assert!(
        !turn_context_msgs
            .last()
            .unwrap()
            .as_concat_text()
            .contains("tokens remaining"),
        "Latest DB turn context must not contain stale tokens remaining"
    );

    Ok(())
}

#[tokio::test]
async fn test_rolling_memory_does_not_fold_after_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    // Enable rolling memory with tight budget
    let settings = goose::context_mgmt::rolling::RollingMemorySettings {
        budget_tokens: 1_000,
        tail_tokens: 400,
    };
    let mut session = ctx
        .session_manager
        .get_session(&ctx.session.id, false)
        .await?;
    goose::context_mgmt::rolling::write_settings(&mut session.extension_data, Some(settings));
    ctx.session_manager
        .update(&ctx.session.id)
        .extension_data(session.extension_data.clone())
        .apply()
        .await?;

    // Seed conversation with history that exceeds 1,000 budget
    let long_text = "lorem ipsum dolor sit amet consectetur adipiscing elit ".repeat(30);
    let mut initial_msgs = Vec::new();
    for i in 0..6 {
        initial_msgs.push(Message::user().with_text(format!("user {i}: {long_text}")));
        initial_msgs.push(Message::assistant().with_text(format!("assistant {i}: {long_text}")));
    }
    let pre_conv = Conversation::new_unvalidated(initial_msgs);
    ctx.session_manager
        .replace_conversation(&ctx.session.id, &pre_conv)
        .await?;

    // Verify that prior to compaction, rolling check would fold
    let pre_check = goose::context_mgmt::rolling::check(&session, &settings, &pre_conv).await?;
    assert!(
        pre_check.is_some(),
        "Pre-compaction conversation must exceed budget and trigger rolling memory check"
    );

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    ctx.agent
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    // Run Turn 1: performs self-compaction
    let reply_stream1 = ctx
        .agent
        .reply(
            Message::user().with_text("start work"),
            ctx.session_config(Some(5)),
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream1);
    while let Some(res) = reply_stream1.next().await {
        res?;
    }

    // Verify session and conversation after Turn 1: under budget
    let reloaded = ctx
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conversation = reloaded.conversation.clone().expect("conversation exists");

    let post_check =
        goose::context_mgmt::rolling::check(&reloaded, &settings, &conversation).await?;
    assert!(
        post_check.is_none(),
        "Post-compaction conversation must not exceed budget"
    );

    // Run Turn 2 (new turn): verify it does NOT trigger rolling memory fold
    let turn3_msg = Message::assistant().with_text("turn 2 finished");
    let provider2 = Arc::new(ScriptedSelfCompactProvider::new(vec![turn3_msg]));
    ctx.agent
        .update_provider(provider2, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let reply_stream2 = ctx
        .agent
        .reply(
            Message::user().with_text("next prompt"),
            ctx.session_config(Some(5)),
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream2);
    let mut turn2_events = Vec::new();
    while let Some(res) = reply_stream2.next().await {
        turn2_events.push(res?);
    }

    let folded_event = turn2_events.iter().any(|e| match e {
        AgentEvent::Message(m) => m
            .as_concat_text()
            .contains("Folding older history into memory"),
        _ => false,
    });
    assert!(
        !folded_event,
        "Turn 2 must not fold history because self-compaction kept context under budget"
    );

    Ok(())
}
