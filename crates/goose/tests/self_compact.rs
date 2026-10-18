use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::StreamExt;
use goose::agents::self_compact::{SELF_COMPACTED_NOTE_TEXT, SELF_COMPACT_TOOL_NAME};
use goose::agents::{Agent, AgentConfig, AgentEvent, GoosePlatform, SessionConfig};
use goose::config::permission::PermissionManager;
use goose::config::GooseMode;
use goose::conversation::message::Message;
use goose::providers::base::{stream_from_single_message, MessageStream, Provider};
use goose::session::session_manager::{SessionManager, SessionType};
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

    fn get_name(&self) -> &str {
        "scripted-self-compact"
    }
}

async fn setup_agent(temp_dir: &TempDir) -> Result<(Agent, Arc<SessionManager>)> {
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("data")));
    let permission_manager = Arc::new(PermissionManager::new(temp_dir.path().join("data")));
    let config = AgentConfig::new(
        session_manager.clone(),
        permission_manager,
        None,
        GooseMode::Auto,
        true,
        GoosePlatform::GooseCli,
    );
    let agent = Agent::with_config(config);
    Ok((agent, session_manager))
}

#[tokio::test]
async fn test_happy_path_self_compact() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    // Script:
    // 1. Tool call self_compact_and_resume{prompt: "R"}
    // 2. Text "done"
    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let user_msg = Message::user().with_text("start work");
    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
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

    // Check summary message is present (agent-only)
    assert!(calls[2]
        .messages
        .iter()
        .any(|m| m.as_concat_text().contains("mock summary")));

    // Check continuation message is present (agent-only)
    assert!(calls[2].messages.iter().any(|m| m
        .as_concat_text()
        .contains("You compacted your own context")));

    // Check resume prompt "R" and turn context are present (in request messages)
    assert!(calls[2]
        .messages
        .iter()
        .any(|m| m.as_concat_text().contains("R")));
    assert!(calls[2]
        .messages
        .iter()
        .any(|m| m.as_concat_text().contains("<turn-context>")));

    // Events include HistoryReplaced and a user message "R" with steer metadata
    let has_history_replaced = events
        .iter()
        .any(|e| matches!(e, AgentEvent::HistoryReplaced(_)));
    assert!(has_history_replaced, "HistoryReplaced event missing");

    let has_steer_resume = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.role == rmcp::model::Role::User && m.metadata.steer && m.as_concat_text() == "R"
        }
        _ => false,
    });
    assert!(
        has_steer_resume,
        "Resume user message with steer metadata missing from events"
    );

    Ok(())
}

#[tokio::test]
async fn test_reload_order_and_metadata_self_compact() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let user_msg = Message::user().with_text("original task");
    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    // Reload conversation from DB: get_session(id, true)
    let reloaded = session_manager.get_session(&session.id, true).await?;
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
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    // max_turns: 2.
    // Turn 1: tool call self_compact_and_resume{prompt: "R1"} -> turns_taken becomes 1. Compaction runs (turns_taken stays 1).
    // Turn 2: tool call self_compact_and_resume{prompt: "R2"} -> turns_taken becomes 2.
    // Turn 3: top of loop has turns_taken (2) >= max_turns (2) -> skips compaction, emits notice with R2, hits max turns break.
    let tc1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R1" }));
    let tc2 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R2" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tc1));
    let turn2_msg = Message::assistant().with_tool_request("call_2", Ok(tc2));

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };

    let user_msg = Message::user().with_text("start work");
    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    // Verify calls:
    // Turn 1 (normal) + Summary (compaction 1) + Turn 2 (normal) = 3 calls total.
    // No second summary call occurred because turns_taken was not reset!
    let calls = recorded_calls.lock().await.clone();
    assert_eq!(calls.len(), 3);
    assert!(!calls[0].is_summary);
    assert!(calls[1].is_summary);
    assert!(!calls[2].is_summary);

    // Event checking: notice that self-compaction skipped due to turn limit was yielded
    let has_skipped_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text()
                .contains("Self-compaction skipped: the turn limit was reached")
                && m.as_concat_text().contains("R2")
        }
        _ => false,
    });
    assert!(has_skipped_notice, "Skipped notice for R2 missing");

    // Max turns message was also yielded
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
async fn test_last_call_wins_and_failed_call_does_not_arm_self_compact() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    // Scenario A: Two calls in one step. Last call wins ("Winner Prompt").
    let tc1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "First Prompt" }));
    let tc2 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Winner Prompt" }));
    let turn1_msg = Message::assistant()
        .with_tool_request("call_1", Ok(tc1))
        .with_tool_request("call_2", Ok(tc2));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let user_msg = Message::user().with_text("start work");
    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    let reloaded = session_manager.get_session(&session.id, true).await?;
    let conv = reloaded.conversation.unwrap();
    // The resume prompt should be "Winner Prompt", not "First Prompt"
    assert!(conv
        .messages()
        .iter()
        .any(|m| m.as_concat_text() == "Winner Prompt"));
    assert!(!conv
        .messages()
        .iter()
        .any(|m| m.as_concat_text() == "First Prompt"));

    // Scenario B: Failed call (empty prompt -> is_error) does NOT arm.
    let session_b = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test_b".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    let tc_invalid = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "   " })); // empty after trim -> validation error
    let turn1_b = Message::assistant().with_tool_request("call_err", Ok(tc_invalid));
    let turn2_b = Message::assistant().with_text("finished after err");

    let provider_b = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_b, turn2_b]));
    let recorded_b = provider_b.recorded_calls.clone();

    agent
        .update_provider(provider_b, ModelConfig::new("mock-model"), &session_b.id)
        .await?;

    let session_config_b = SessionConfig {
        id: session_b.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let reply_stream_b = agent
        .reply(
            Message::user().with_text("test_b"),
            session_config_b,
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream_b);
    while let Some(res) = reply_stream_b.next().await {
        res?;
    }

    // Since the call failed, it did NOT arm, so no summary call occurred!
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
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt R" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));

    let provider = Arc::new(
        ScriptedSelfCompactProvider::new(vec![turn1_msg])
            .with_summary_error(ProviderError::ExecutionError("provider broke".to_string())),
    );

    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let user_msg = Message::user().with_text("start work");
    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
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

    // DB conversation unchanged (original messages remain agent-visible, no summary message, failure notice persisted)
    let reloaded = session_manager.get_session(&session.id, true).await?;
    let conv = reloaded.conversation.expect("conversation exists");
    let msgs = conv.messages();

    // Original user message is still agent-visible (not compacted into invisible!)
    assert_eq!(msgs[0].as_concat_text(), "start work");
    assert!(msgs[0].is_agent_visible());

    // Failure notice is in DB
    assert!(msgs.iter().any(|m| {
        m.as_concat_text().contains("Self-compaction failed:")
            && m.as_concat_text().contains("provider broke")
    }));

    // No resume message in DB
    assert!(!msgs.iter().any(|m| m.as_concat_text() == "Resume prompt R"));

    // No summary message in DB
    assert!(!msgs
        .iter()
        .any(|m| m.as_concat_text().contains("mock summary")));

    Ok(())
}

#[tokio::test]
async fn test_cancel_during_summary_self_compact() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

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
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            })
        })
        .await;

    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let user_msg = Message::user().with_text("start work");
    let reply_stream = agent
        .reply(user_msg, session_config, false, Some(cancel_token))
        .await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    // Failure notice was yielded with "cancelled"
    let has_cancel_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text().contains("cancelled")
                && m.as_concat_text().contains("Cancel prompt R")
        }
        _ => false,
    });
    assert!(has_cancel_notice, "Cancel notice missing from events");

    // DB has no resume message
    let reloaded = session_manager.get_session(&session.id, true).await?;
    let conv = reloaded.conversation.unwrap();
    assert!(!conv
        .messages()
        .iter()
        .any(|m| m.as_concat_text() == "Cancel prompt R"));

    Ok(())
}

#[tokio::test]
async fn test_max_turns_skips_compaction_self_compact() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    // max_turns: 1.
    // Turn 1 calls self_compact_and_resume{prompt: "R"}.
    // At top of loop 2, turns_taken >= max_turns (1 >= 1) -> skips compaction, yields notice.
    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "R on last turn" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(1),
        retry_config: None,
    };

    let user_msg = Message::user().with_text("start work");
    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    // No summary call made!
    let calls = recorded_calls.lock().await.clone();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].is_summary);

    // Notice yielded
    let has_notice = events.iter().any(|e| match e {
        AgentEvent::Message(m) => {
            m.as_concat_text()
                .contains("Self-compaction skipped: the turn limit was reached")
                && m.as_concat_text().contains("R on last turn")
        }
        _ => false,
    });
    assert!(has_notice, "Skipped notice missing");

    // No resume message delivered
    let reloaded = session_manager.get_session(&session.id, true).await?;
    let conv = reloaded.conversation.unwrap();
    assert!(!conv
        .messages()
        .iter()
        .any(|m| m.as_concat_text() == "R on last turn"));

    Ok(())
}

#[tokio::test]
async fn test_steer_queued_during_compaction_self_compact() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;
    let agent = Arc::new(agent);

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    // During compaction summary call, steer a message
    let agent_clone = agent.clone();
    let session_id_clone = session.id.clone();
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

    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let user_msg = Message::user().with_text("start work");
    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
    tokio::pin!(reply_stream);

    let mut events = Vec::new();
    while let Some(res) = reply_stream.next().await {
        events.push(res?);
    }

    // In events: resume message comes BEFORE steer message
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

    // In call 2 (last turn) request messages:
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
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    let recorded_calls = provider.recorded_calls.clone();

    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let reply_stream = agent
        .reply(
            Message::user().with_text("start work"),
            session_config,
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    // Turn context is present in the final turn request
    let calls = recorded_calls.lock().await.clone();
    let last_call = &calls[2];
    assert!(last_call
        .messages
        .iter()
        .any(|m| m.as_concat_text().contains("<turn-context>")));

    // And in the DB
    let reloaded = session_manager.get_session(&session.id, true).await?;
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
    // The latest turn context in DB is agent-visible
    assert!(turn_context_msgs.last().unwrap().is_agent_visible());

    Ok(())
}

#[tokio::test]
async fn test_rolling_memory_does_not_fold_after_self_compact() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (agent, session_manager) = setup_agent(&temp_dir).await?;

    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "test".to_string(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;

    let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
        .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }));
    let turn1_msg = Message::assistant().with_tool_request("call_1", Ok(tool_call));
    let turn2_msg = Message::assistant().with_text("done");

    let provider = Arc::new(ScriptedSelfCompactProvider::new(vec![turn1_msg, turn2_msg]));
    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(5),
        retry_config: None,
    };

    let reply_stream = agent
        .reply(
            Message::user().with_text("start work"),
            session_config,
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream);
    while let Some(res) = reply_stream.next().await {
        res?;
    }

    // Load session and conversation after self-compaction
    let reloaded = session_manager.get_session(&session.id, true).await?;
    let conversation = reloaded.conversation.clone().expect("conversation exists");

    let settings = goose::context_mgmt::rolling::RollingMemorySettings {
        budget_tokens: 100_000,
        tail_tokens: 20_000,
    };

    let check_result =
        goose::context_mgmt::rolling::check(&reloaded, &settings, &conversation).await?;
    assert!(
        check_result.is_none(),
        "Rolling memory should not fold after self-compaction"
    );

    Ok(())
}
