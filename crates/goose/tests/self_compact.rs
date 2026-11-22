// Added by Jonathan Mohrbacher, 2026: self-compaction integration tests on the state machine.
use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::StreamExt;
use goose::agents::state_machine::ops_self_compact::{
    SELF_COMPACTED_NOTE_TEXT, SELF_COMPACT_ALREADY_COMPACTED_MESSAGE, SELF_COMPACT_SUCCESS_MESSAGE,
    SELF_COMPACT_TOOL_NAME,
};
use goose::agents::{
    AgentEvent, GoosePlatform, SessionConfig, StateMachineServices, StateMachineServicesConfig,
};
use goose::config::permission::PermissionManager;
use goose::config::GooseMode;
use goose::context_mgmt::SELF_COMPACT_CONTINUATION_TEXT;
use goose::conversation::message::{Message, MessageContent};
use goose::providers::base::{stream_from_single_message, MessageStream, Provider};
use goose::recipe::Recipe;
use goose::session::extension_data::{EnabledExtensionsState, ExtensionState};
use goose::session::session_manager::{Session, SessionManager, SessionType};
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::{CallToolRequestParams, Role, Tool};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

static PROCESS_ROOT: std::sync::LazyLock<TempDir> =
    std::sync::LazyLock::new(|| TempDir::new().unwrap());

fn isolate_process() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::env::set_var("GOOSE_PATH_ROOT", PROCESS_ROOT.path());
        std::env::set_var("GOOSE_TELEMETRY_OFF", "1");
        std::env::set_var("GOOSE_DISABLE_KEYRING", "1");
    });
}

fn isolated_services(temp_dir: &TempDir) -> Arc<StateMachineServices> {
    isolate_process();
    StateMachineServices::with_config(StateMachineServicesConfig::new(
        Arc::new(SessionManager::new(temp_dir.path().join("data"))),
        Arc::new(PermissionManager::new(temp_dir.path().join("config"))),
        None,
        true,
        GoosePlatform::GooseCli,
    ))
}

static ENV_MUTEX: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct EnvVarGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, val: &str) -> Self {
        let prev = std::env::var(key).ok();
        std::env::set_var(key, val);
        Self { key, prev }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(val) => std::env::set_var(self.key, val),
            None => std::env::remove_var(self.key),
        }
    }
}

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
    summary_responses: Arc<Mutex<Vec<Result<Message, ProviderError>>>>,
    recorded_calls: Arc<Mutex<Vec<RecordedCall>>>,
    on_summary: Arc<Mutex<Option<SummaryHook>>>,
    normal_usage: Arc<Mutex<Option<Usage>>>,
    manages_own_context: bool,
}

impl ScriptedSelfCompactProvider {
    fn new(normal_responses: Vec<Message>) -> Self {
        Self {
            normal_responses: Arc::new(Mutex::new(normal_responses)),
            summary_responses: Arc::new(Mutex::new(Vec::new())),
            recorded_calls: Arc::new(Mutex::new(Vec::new())),
            on_summary: Arc::new(Mutex::new(None)),
            normal_usage: Arc::new(Mutex::new(None)),
            manages_own_context: false,
        }
    }

    fn with_summary_error(self, err: ProviderError) -> Self {
        Self {
            normal_responses: self.normal_responses,
            summary_responses: Arc::new(Mutex::new(vec![Err(err)])),
            recorded_calls: self.recorded_calls,
            on_summary: self.on_summary,
            normal_usage: self.normal_usage,
            manages_own_context: self.manages_own_context,
        }
    }

    fn with_manages_own_context(mut self, val: bool) -> Self {
        self.manages_own_context = val;
        self
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
    fn get_name(&self) -> &str {
        "scripted-self-compact"
    }

    fn manages_own_context(&self) -> bool {
        self.manages_own_context
    }

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
            let mut guard = self.summary_responses.lock().await;
            let resp = if guard.is_empty() {
                Ok(Message::assistant().with_text("<mock summary of conversation>"))
            } else {
                guard.remove(0)
            };
            match resp {
                Ok(msg) => {
                    let usage = ProviderUsage::new(
                        "mock-model".to_string(),
                        Usage::new(Some(500), Some(100), Some(600)),
                    );
                    Ok(stream_from_single_message(msg, usage))
                }
                Err(e) => Err(e),
            }
        } else {
            let mut guard = self.normal_responses.lock().await;
            let msg = if guard.is_empty() {
                Message::assistant().with_text("default response")
            } else {
                guard.remove(0)
            };
            let usage = (*self.normal_usage.lock().await)
                .unwrap_or_else(|| Usage::new(Some(100), Some(20), Some(120)));
            Ok(stream_from_single_message(
                msg,
                ProviderUsage::new("mock-model".to_string(), usage),
            ))
        }
    }
}

struct TestContext {
    services: Arc<StateMachineServices>,
    session: Session,
    _temp_dir: TempDir,
    _env_lock: tokio::sync::MutexGuard<'static, ()>,
}

impl TestContext {
    async fn new() -> Result<Self> {
        Self::with_type_and_mode(SessionType::User, GooseMode::Auto).await
    }

    async fn with_type_and_mode(session_type: SessionType, mode: GooseMode) -> Result<Self> {
        let env_lock = ENV_MUTEX.lock().await;
        let temp_dir = TempDir::new()?;
        let services = isolated_services(&temp_dir);
        let session = services
            .session_manager
            .create_session(
                temp_dir.path().to_path_buf(),
                "test-session".to_string(),
                session_type,
                mode,
            )
            .await?;
        Ok(Self {
            services,
            session,
            _temp_dir: temp_dir,
            _env_lock: env_lock,
        })
    }

    fn session_config(&self, max_turns: Option<u32>) -> SessionConfig {
        SessionConfig {
            id: self.session.id.clone(),
            schedule_id: None,
            max_turns,
        }
    }
}

#[tokio::test]
async fn test_happy_path_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(5)), None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let has_tool_success = events.iter().any(|e| match e {
        AgentEvent::Message(m) => m.content.iter().any(|c| match c {
            MessageContent::ToolResponse(r) => {
                if let Ok(res) = &r.tool_result {
                    res.content.iter().any(|b| {
                        b.as_text()
                            .is_some_and(|t| t.text == SELF_COMPACT_SUCCESS_MESSAGE)
                    })
                } else {
                    false
                }
            }
            _ => false,
        }),
        _ => false,
    });
    assert!(has_tool_success, "Tool response success message missing");

    let calls = recorded_calls.lock().await.clone();
    assert_eq!(calls.len(), 3, "Expected 3 calls: turn 1, summary, turn 2");
    assert!(!calls[0].is_summary, "Call 0 should be normal turn");
    assert!(calls[1].is_summary, "Call 1 should be summary call");
    assert!(!calls[2].is_summary, "Call 2 should be normal turn");

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

    let summary_pos = calls[2]
        .messages
        .iter()
        .position(|m| m.as_concat_text().contains("mock summary"))
        .expect("summary in final request");
    let cont_pos = calls[2]
        .messages
        .iter()
        .position(|m| m.as_concat_text().contains(SELF_COMPACT_CONTINUATION_TEXT))
        .expect("continuation in final request");
    let resume_pos = calls[2]
        .messages
        .iter()
        .position(|m| m.role == Role::User && m.as_concat_text().starts_with("R"))
        .expect("resume in final request");

    assert!(summary_pos < cont_pos, "summary must precede continuation");
    assert!(cont_pos < resume_pos, "continuation must precede resume");

    let last_user_idx = calls[2]
        .messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .expect("user message in final request");
    assert_eq!(last_user_idx, resume_pos);

    let reloaded = ctx
        .services
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.expect("conversation in db");
    let db_msgs = conv.messages();

    let db_summary = db_msgs
        .iter()
        .position(|m| m.as_concat_text().contains("mock summary"))
        .expect("summary in db");
    let db_cont = db_msgs
        .iter()
        .position(|m| m.as_concat_text().contains(SELF_COMPACT_CONTINUATION_TEXT))
        .expect("continuation in db");
    let db_note = db_msgs
        .iter()
        .position(|m| m.as_concat_text() == SELF_COMPACTED_NOTE_TEXT)
        .expect("note in db");
    let db_resume = db_msgs
        .iter()
        .position(|m| m.as_concat_text() == "R")
        .expect("resume in db");
    let db_done = db_msgs
        .iter()
        .position(|m| m.as_concat_text() == "done")
        .expect("done in db");

    assert!(db_summary < db_cont);
    assert!(db_cont < db_note);
    assert!(db_note < db_resume);
    assert!(db_resume < db_done);

    assert!(db_msgs[db_note].metadata.user_visible);
    assert!(!db_msgs[db_note].metadata.agent_visible);
    assert!(db_msgs[db_resume].metadata.user_visible);
    assert!(db_msgs[db_resume].metadata.agent_visible);
    assert!(db_msgs[db_resume].metadata.steer);

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
    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("original task");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(5)), None)
        .await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    let reloaded = ctx
        .services
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conversation = reloaded.conversation.expect("conversation exists");
    let messages = conversation.messages();

    assert_eq!(messages[0].as_concat_text(), "original task");
    assert!(!messages[0].is_agent_visible());
    assert!(messages[0].is_user_visible());
    assert!(!messages[1].is_agent_visible());
    assert!(!messages[2].is_agent_visible());

    let summary_idx = messages
        .iter()
        .position(|m| m.as_concat_text().contains("mock summary"))
        .expect("summary message");
    assert!(messages[summary_idx].is_agent_visible());
    assert!(!messages[summary_idx].is_user_visible());

    let cont_idx = messages
        .iter()
        .position(|m| m.as_concat_text().contains(SELF_COMPACT_CONTINUATION_TEXT))
        .expect("continuation message");
    assert!(messages[cont_idx].is_agent_visible());
    assert!(!messages[cont_idx].is_user_visible());

    let note_idx = messages
        .iter()
        .position(|m| m.as_concat_text() == SELF_COMPACTED_NOTE_TEXT)
        .expect("note message");
    assert!(!messages[note_idx].is_agent_visible());
    assert!(messages[note_idx].is_user_visible());

    let resume_idx = messages
        .iter()
        .position(|m| m.as_concat_text() == "Resume prompt")
        .expect("resume message");
    assert!(messages[resume_idx].is_agent_visible());
    assert!(messages[resume_idx].is_user_visible());
    assert!(messages[resume_idx].metadata.steer);

    let done_idx = messages
        .iter()
        .position(|m| m.as_concat_text() == "done")
        .expect("done message");
    assert!(messages[done_idx].is_agent_visible());
    assert!(messages[done_idx].is_user_visible());

    assert!(summary_idx < cont_idx);
    assert!(cont_idx < note_idx);
    assert!(note_idx < resume_idx);
    assert!(resume_idx < done_idx);

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

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(2)), None)
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

    let has_max_turns = events.iter().any(|e| match e {
        AgentEvent::Message(m) => m.content.iter().any(|c| match c {
            MessageContent::Error(e) => e.message.contains("reached the maximum number of actions"),
            _ => false,
        }),
        _ => false,
    });
    assert!(has_max_turns, "Max turns message missing");

    let reloaded = ctx
        .services
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.expect("conversation in db");
    let msgs = conv.messages();

    let call2_idx = msgs
        .iter()
        .position(|m| m.get_tool_response_ids().contains(&"call_2"))
        .expect("tool response for call_2 must be persisted");

    let max_turns_idx = msgs
        .iter()
        .position(|m| {
            m.content.iter().any(|c| match c {
                MessageContent::Error(e) => {
                    e.message.contains("reached the maximum number of actions")
                }
                _ => false,
            })
        })
        .expect("max turns error message must be persisted");

    assert!(
        call2_idx < max_turns_idx,
        "tool response for call_2 must be persisted before max-turns error"
    );

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

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(5)), None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let calls = recorded_calls.lock().await.clone();
    assert_eq!(
        calls.len(),
        4,
        "Expected 4 calls: turn 1, summary, turn 2, turn 3"
    );
    assert!(!calls[0].is_summary);
    assert!(calls[1].is_summary);
    assert!(!calls[2].is_summary);
    assert!(!calls[3].is_summary);

    let has_already_error = events.iter().any(|e| match e {
        AgentEvent::Message(m) => m.content.iter().any(|c| match c {
            MessageContent::ToolResponse(r) => {
                if let Ok(res) = &r.tool_result {
                    res.is_error == Some(true)
                        && res.content.iter().any(|b| {
                            b.as_text()
                                .is_some_and(|t| t.text == SELF_COMPACT_ALREADY_COMPACTED_MESSAGE)
                        })
                } else {
                    false
                }
            }
            _ => false,
        }),
        _ => false,
    });
    assert!(
        has_already_error,
        "ALREADY error message missing for second call in run"
    );

    Ok(())
}

#[tokio::test]
async fn test_new_run_can_compact_again_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tc1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R1" }));
    let tc2 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R2" }));

    let turn1_run1 = Message::assistant().with_tool_request("call_1", Ok(tc1));
    let turn2_run1 = Message::assistant().with_text("done 1");
    let turn1_run2 = Message::assistant().with_tool_request("call_2", Ok(tc2));
    let turn2_run2 = Message::assistant().with_text("done 2");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![
        turn1_run1, turn2_run1, turn1_run2, turn2_run2,
    ]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    // Run 1
    let stream1 = ctx
        .services
        .reply(
            Message::user().with_text("start run 1"),
            ctx.session_config(Some(5)),
            None,
        )
        .await?;
    tokio::pin!(stream1);
    while let Some(res) = stream1.next().await {
        res?;
    }

    let stream2 = ctx
        .services
        .reply(
            Message::user().with_text("start run 2"),
            ctx.session_config(Some(5)),
            None,
        )
        .await?;
    tokio::pin!(stream2);
    while let Some(res) = stream2.next().await {
        res?;
    }

    let calls = recorded_calls.lock().await.clone();
    let summary_count = calls.iter().filter(|c| c.is_summary).count();
    assert_eq!(summary_count, 2, "Expected 2 summary calls (one per run)");

    Ok(())
}

#[tokio::test]
async fn test_last_call_wins_and_failed_call_does_not_arm_self_compact() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tc1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "first" }));
    let tc2 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "second" }));
    let tc_invalid = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "   " }));

    let turn1_msg = Message::assistant()
        .with_tool_request("call_1", Ok(tc1))
        .with_tool_request("call_2", Ok(tc2))
        .with_tool_request("call_3", Ok(tc_invalid));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(5)), None)
        .await?;
    tokio::pin!(reply_stream);

    while let Some(res) = reply_stream.next().await {
        res?;
    }

    let calls = recorded_calls.lock().await.clone();
    assert_eq!(calls.len(), 3);
    assert!(!calls[0].is_summary);
    assert!(calls[1].is_summary);
    assert!(!calls[2].is_summary);

    let resume_msg = calls[2]
        .messages
        .iter()
        .find(|m| m.role == Role::User && m.as_concat_text().starts_with("second"));
    assert!(resume_msg.is_some(), "Resume prompt must be 'second'");

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

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(5)), None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let has_fail_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text().contains("Self-compaction failed:")
                && m.as_concat_text().contains("provider broke")
                && m.as_concat_text().contains("Resume prompt R")
        }
        _ => false,
    });
    assert!(has_fail_notice, "Failure notice missing from events");

    let reloaded = ctx
        .services
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.expect("conversation exists");
    let msgs = conv.messages();

    assert_eq!(msgs[0].as_concat_text(), "start work");
    assert!(msgs[0].is_agent_visible());
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

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(5)), Some(cancel_token))
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let cancel_count = events
        .iter()
        .filter(|e| match e {
            AgentEvent::Message(m) => {
                m.as_concat_text()
                    .contains("Self-compaction failed: cancelled")
                    && m.as_concat_text().contains("Cancel prompt R")
            }
            _ => false,
        })
        .count();
    assert_eq!(
        cancel_count, 1,
        "Exactly one cancel notice must be emitted in events"
    );

    let reloaded = ctx
        .services
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.unwrap();
    let db_cancel_count = conv
        .messages()
        .iter()
        .filter(|m| {
            m.as_concat_text()
                .contains("Self-compaction failed: cancelled")
                && m.as_concat_text().contains("Cancel prompt R")
        })
        .count();
    assert_eq!(
        db_cancel_count, 1,
        "Exactly one cancel notice must be persisted in DB"
    );
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

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(1)), None)
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    let calls = recorded_calls.lock().await.clone();
    assert_eq!(calls.len(), 1, "Only 1 turn should run (no summary call)");
    assert!(!calls[0].is_summary);

    let has_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text()
                .contains("Self-compaction skipped: the turn limit was reached")
                && m.as_concat_text().contains("R on last turn")
        }
        _ => false,
    });
    assert!(has_notice, "Skipped notice missing from events");

    let reloaded = ctx
        .services
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.unwrap();
    let notice_in_db = conv.messages().iter().any(|m| {
        m.as_concat_text()
            .contains("Self-compaction skipped: the turn limit was reached")
            && m.as_concat_text().contains("R on last turn")
    });
    assert!(notice_in_db, "Skipped notice must be persisted in DB");
    let error = conv
        .messages()
        .last()
        .and_then(|message| message.error())
        .expect("max turns error ends the turn");
    assert_eq!(
        error.kind,
        goose::conversation::message::MessageErrorKind::ResourceExhausted
    );
    assert!(error.message.contains("maximum number of actions"));

    assert!(!conv
        .messages()
        .iter()
        .any(|m| m.as_concat_text() == "R on last turn"));

    Ok(())
}

#[tokio::test]
async fn test_tool_not_advertised_when_self_compact_disabled_by_config() -> Result<()> {
    let ctx = TestContext::new().await?;
    let _guard = EnvVarGuard::set("GOOSE_SELF_COMPACT", "false");
    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![
        Message::assistant().with_text("done")
    ]));
    let recorded = provider.recorded_calls.clone();
    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let stream = ctx
        .services
        .reply(
            Message::user().with_text("hi"),
            ctx.session_config(Some(1)),
            None,
        )
        .await?;
    tokio::pin!(stream);
    while let Some(res) = stream.next().await {
        res?;
    }

    let calls = recorded.lock().await.clone();
    assert_eq!(calls.len(), 1);
    assert!(
        !calls[0]
            .tools
            .iter()
            .any(|t| t.name == SELF_COMPACT_TOOL_NAME),
        "tool must not be advertised when disabled by config"
    );
    Ok(())
}

#[tokio::test]
async fn test_tool_not_advertised_for_subagent_session() -> Result<()> {
    isolate_process();
    let ctx = TestContext::with_type_and_mode(SessionType::SubAgent, GooseMode::Auto).await?;
    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![
        Message::assistant().with_text("done")
    ]));
    let recorded = provider.recorded_calls.clone();
    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let stream = ctx
        .services
        .reply(
            Message::user().with_text("hi"),
            ctx.session_config(Some(1)),
            None,
        )
        .await?;
    tokio::pin!(stream);
    while let Some(res) = stream.next().await {
        res?;
    }

    let calls = recorded.lock().await.clone();
    assert_eq!(calls.len(), 1);
    assert!(
        !calls[0]
            .tools
            .iter()
            .any(|t| t.name == SELF_COMPACT_TOOL_NAME),
        "tool must not be advertised for subagent session"
    );
    Ok(())
}

#[tokio::test]
async fn test_tool_not_advertised_when_recipe_retry_configured() -> Result<()> {
    isolate_process();
    let ctx = TestContext::new().await?;
    let mut recipe = Recipe::builder()
        .title("retry-recipe")
        .description("test")
        .instructions("test")
        .build()
        .unwrap();
    recipe.retry = Some(goose::agents::types::RetryConfig {
        max_retries: 3,
        timeout_seconds: None,
        on_failure: None,
        on_failure_timeout_seconds: None,
        checks: vec![],
    });
    ctx.services
        .session_manager
        .update(&ctx.session.id)
        .recipe(Some(recipe))
        .apply()
        .await?;

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![
        Message::assistant().with_text("done")
    ]));
    let recorded = provider.recorded_calls.clone();
    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let stream = ctx
        .services
        .reply(
            Message::user().with_text("hi"),
            ctx.session_config(Some(1)),
            None,
        )
        .await?;
    tokio::pin!(stream);
    while let Some(res) = stream.next().await {
        res?;
    }

    let calls = recorded.lock().await.clone();
    assert_eq!(calls.len(), 1);
    assert!(
        !calls[0]
            .tools
            .iter()
            .any(|t| t.name == SELF_COMPACT_TOOL_NAME),
        "tool must not be advertised when recipe retry is configured"
    );
    Ok(())
}

#[tokio::test]
async fn test_tool_not_advertised_when_recipe_response_configured() -> Result<()> {
    isolate_process();
    let ctx = TestContext::new().await?;
    let mut recipe = Recipe::builder()
        .title("response-recipe")
        .description("test")
        .instructions("test")
        .build()
        .unwrap();
    recipe.response = Some(goose::recipe::Response {
        json_schema: Some(serde_json::json!({ "type": "object" })),
    });
    ctx.services
        .session_manager
        .update(&ctx.session.id)
        .recipe(Some(recipe))
        .apply()
        .await?;

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![
        Message::assistant().with_text("done")
    ]));
    let recorded = provider.recorded_calls.clone();
    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let stream = ctx
        .services
        .reply(
            Message::user().with_text("hi"),
            ctx.session_config(Some(1)),
            None,
        )
        .await?;
    tokio::pin!(stream);
    while let Some(res) = stream.next().await {
        res?;
    }

    let calls = recorded.lock().await.clone();
    assert_eq!(calls.len(), 1);
    assert!(
        !calls[0]
            .tools
            .iter()
            .any(|t| t.name == SELF_COMPACT_TOOL_NAME),
        "tool must not be advertised when recipe response is configured"
    );
    Ok(())
}

#[tokio::test]
async fn test_tool_not_advertised_when_provider_manages_own_context() -> Result<()> {
    isolate_process();
    let ctx = TestContext::new().await?;
    let provider = Arc::new(
        ScriptedSelfCompactProvider::new(vec![Message::assistant().with_text("done")])
            .with_manages_own_context(true),
    );
    let recorded = provider.recorded_calls.clone();
    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let stream = ctx
        .services
        .reply(
            Message::user().with_text("hi"),
            ctx.session_config(Some(1)),
            None,
        )
        .await?;
    tokio::pin!(stream);
    while let Some(res) = stream.next().await {
        res?;
    }

    let calls = recorded.lock().await.clone();
    assert_eq!(calls.len(), 1);
    assert!(
        !calls[0]
            .tools
            .iter()
            .any(|t| t.name == SELF_COMPACT_TOOL_NAME),
        "tool must not be advertised when provider manages own context"
    );
    Ok(())
}

#[tokio::test]
async fn test_tools_not_advertised_when_session_has_zero_extensions() -> Result<()> {
    let ctx = TestContext::new().await?;

    let mut extension_data = ctx.session.extension_data.clone();
    EnabledExtensionsState::new(Vec::new()).to_extension_data(&mut extension_data)?;
    ctx.services
        .session_manager
        .update(&ctx.session.id)
        .extension_data(extension_data)
        .apply()
        .await?;

    let turn1_msg = Message::assistant().with_text("done");
    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg]));
    let recorded = provider.recorded_calls.clone();

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let stream = ctx
        .services
        .reply(
            Message::user().with_text("hi"),
            ctx.session_config(Some(1)),
            None,
        )
        .await?;
    tokio::pin!(stream);
    while let Some(res) = stream.next().await {
        res?;
    }

    let calls = recorded.lock().await.clone();
    assert_eq!(calls.len(), 1);
    assert!(
        calls[0].tools.is_empty(),
        "no tools must be offered when session has zero extensions: {:?}",
        calls[0].tools
    );
    assert!(
        !calls[0].system_prompt.contains("# Skills"),
        "skills prompt must not be included when session has zero extensions"
    );
    Ok(())
}

#[tokio::test]
async fn test_sibling_tool_call_compaction_waits_for_siblings() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tc_sibling = CallToolRequestParams::new("dummy_sibling_tool");
    let tc_self_compact = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Sibling resume prompt" }));

    let turn1_msg = Message::assistant()
        .with_tool_request("call_sibling", Ok(tc_sibling))
        .with_tool_request("call_self_compact", Ok(tc_self_compact));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    ctx.services
        .update_provider(provider, ModelConfig::new("mock-model"), &ctx.session.id)
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(5)), None)
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

    // Verify summary call system prompt included tool responses for BOTH call_sibling and call_self_compact
    let summary_call = &calls[1];
    let has_sibling_response = summary_call.system_prompt.contains("dummy_sibling_tool")
        || summary_call.system_prompt.contains("call_sibling");
    assert!(
        has_sibling_response,
        "Summary call must include answered sibling tool response"
    );

    let has_self_compact_response = summary_call
        .system_prompt
        .contains("self_compact_and_resume")
        || summary_call.system_prompt.contains("call_self_compact");
    assert!(
        has_self_compact_response,
        "Summary call must include answered self_compact tool response"
    );

    let reloaded = ctx
        .services
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let conv = reloaded.conversation.expect("conversation in db");
    let db_msgs = conv.messages();

    let sibling_resp_idx = db_msgs
        .iter()
        .position(|m| m.get_tool_response_ids().contains(&"call_sibling"))
        .expect("call_sibling tool response must be persisted in db");

    let summary_idx = db_msgs
        .iter()
        .position(|m| {
            m.as_concat_text()
                .contains("<mock summary of conversation>")
        })
        .expect("summary message must be persisted in db");

    assert!(
        sibling_resp_idx < summary_idx,
        "call_sibling tool response must be persisted before compaction summary message"
    );

    Ok(())
}

#[tokio::test]
async fn test_rebuilt_machine_mid_turn_at_most_once_per_turn() -> Result<()> {
    let ctx = TestContext::new().await?;

    let tc1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "First compaction" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tc1));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    ctx.services
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &ctx.session.id,
        )
        .await?;

    let user_msg = Message::user().with_text("start work");
    let reply_stream = ctx
        .services
        .reply(user_msg, ctx.session_config(Some(5)), None)
        .await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    let reloaded = ctx
        .services
        .session_manager
        .get_session(&ctx.session.id, true)
        .await?;
    let mut conv = reloaded.conversation.expect("conversation in db");

    let op = goose::agents::state_machine::ops_self_compact::SelfCompactOperation::new(
        provider.clone(),
        ModelConfig::new("mock-model"),
        5,
    );
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let emit =
        goose::agents::state_machine::Emitter::new(tx, tokio_util::sync::CancellationToken::new());

    let second_tc = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Second compaction" }));
    let mut messages = conv.messages().to_vec();
    messages.push(Message::assistant().with_tool_request("call_2", Ok(second_tc)));
    conv = goose::conversation::Conversation::new_unvalidated(messages);

    let res = goose::agents::state_machine::Operation::run(&op, &ctx.session, &conv, &emit).await?;
    match res {
        goose::agents::state_machine::OperationResult::Applied(applied) => {
            let effect_msg = applied.effects.first().expect("response effect");
            match effect_msg {
                goose::agents::state_machine::GooseEffect::Conversation(
                    goose::agents::state_machine::ConversationEffect::AppendMessage(msg),
                ) => {
                    let has_err = msg.content.iter().any(|c| match c {
                        MessageContent::ToolResponse(r) => {
                            if let Ok(res) = &r.tool_result {
                                res.content.iter().any(|b| {
                                    b.as_text().is_some_and(|t| {
                                        t.text == SELF_COMPACT_ALREADY_COMPACTED_MESSAGE
                                    })
                                })
                            } else {
                                false
                            }
                        }
                        _ => false,
                    });
                    assert!(
                        has_err,
                        "Expected already compacted message in tool response"
                    );
                }
                _ => panic!("Unexpected effect"),
            }
        }
        _ => panic!("Expected Applied result"),
    }

    Ok(())
}

#[tokio::test]
async fn test_recheck_availability_when_about_to_compact_self_compact() -> Result<()> {
    isolate_process();
    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![]));
    let op = goose::agents::state_machine::ops_self_compact::SelfCompactOperation::new(
        provider.clone(),
        ModelConfig::new("mock-model"),
        5,
    );
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let emit =
        goose::agents::state_machine::Emitter::new(tx, tokio_util::sync::CancellationToken::new());

    let mut tool_resp = Message::user();
    tool_resp.metadata.set_operation_note(
        "self_compact",
        "armed",
        serde_json::json!("Armed prompt"),
    );
    let conv = goose::conversation::Conversation::new_unvalidated(vec![
        Message::user().with_text("kickoff"),
        tool_resp,
    ]);

    let session = Session {
        session_type: SessionType::SubAgent,
        ..Default::default()
    };

    let res = goose::agents::state_machine::Operation::run(&op, &session, &conv, &emit).await?;
    match res {
        goose::agents::state_machine::OperationResult::Applied(applied) => {
            assert_eq!(
                applied.status,
                goose::agents::state_machine::RunStatus::Yielded
            );
            let effect_msg = applied.effects.first().expect("failure notice effect");
            match effect_msg {
                goose::agents::state_machine::GooseEffect::Conversation(
                    goose::agents::state_machine::ConversationEffect::AppendMessage(msg),
                ) => {
                    let text = msg.as_concat_text();
                    assert!(
                        text.contains("Self-compaction failed: self-compaction is no longer available in this session"),
                        "Expected availability failure notice, got: {text}"
                    );
                    assert_eq!(
                        msg.metadata
                            .operation_note("self_compact", "outcome")
                            .and_then(|v| v.as_str()),
                        Some("failed")
                    );
                }
                _ => panic!("Unexpected effect"),
            }
        }
        _ => panic!("Expected Applied result"),
    }

    Ok(())
}
