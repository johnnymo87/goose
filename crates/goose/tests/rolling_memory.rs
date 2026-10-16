use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::{Agent, AgentEvent, SessionConfig};
use goose::config::GooseMode;
use goose::context_mgmt::rolling::{self, RollingMemorySettings};
use goose::conversation::message::{Message, MessageUsage};
use goose::conversation::Conversation;
use goose::providers::base::{
    stream_from_single_message, MessageStream, Provider, ProviderDef, ProviderMetadata,
};
use goose::session::session_manager::SessionType;
use goose::session::Session;
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CapturedRequest {
    is_memory: bool,
    system: String,
    messages: Vec<Message>,
}

type ResponseQueue = Arc<Mutex<VecDeque<Result<(Message, ProviderUsage), ProviderError>>>>;

struct MockRollingProvider {
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    memory_responses: ResponseQueue,
    main_responses: ResponseQueue,
    manages_own_context: bool,
    session_manager: Option<Arc<goose::session::SessionManager>>,
    session_id: Option<String>,
    pub main_call_saw_retained: Arc<std::sync::atomic::AtomicBool>,
    default_main_tokens: Arc<Mutex<Option<i32>>>,
    pub block_memory_call: Arc<std::sync::atomic::AtomicBool>,
    pub memory_call_entered: Arc<tokio::sync::Notify>,
    pub memory_call_gate: Arc<tokio::sync::Notify>,
}

impl MockRollingProvider {
    fn new() -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            memory_responses: Arc::new(Mutex::new(VecDeque::new())),
            main_responses: Arc::new(Mutex::new(VecDeque::new())),
            manages_own_context: false,
            session_manager: None,
            session_id: None,
            main_call_saw_retained: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            default_main_tokens: Arc::new(Mutex::new(None)),
            block_memory_call: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            memory_call_entered: Arc::new(tokio::sync::Notify::new()),
            memory_call_gate: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn with_session_tracker(
        session_manager: Arc<goose::session::SessionManager>,
        session_id: String,
    ) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            memory_responses: Arc::new(Mutex::new(VecDeque::new())),
            main_responses: Arc::new(Mutex::new(VecDeque::new())),
            manages_own_context: false,
            session_manager: Some(session_manager),
            session_id: Some(session_id),
            main_call_saw_retained: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            default_main_tokens: Arc::new(Mutex::new(None)),
            block_memory_call: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            memory_call_entered: Arc::new(tokio::sync::Notify::new()),
            memory_call_gate: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn set_default_main_tokens(&self, tokens: i32) {
        *self.default_main_tokens.lock().unwrap() = Some(tokens);
    }

    fn queue_memory_response(&self, res: Result<(Message, ProviderUsage), ProviderError>) {
        self.memory_responses.lock().unwrap().push_back(res);
    }

    #[allow(dead_code)]
    fn queue_main_response(&self, res: Result<(Message, ProviderUsage), ProviderError>) {
        self.main_responses.lock().unwrap().push_back(res);
    }

    fn captured_requests(&self) -> Vec<CapturedRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn memory_call_count(&self) -> usize {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.is_memory)
            .count()
    }

    fn handle_request(
        &self,
        system: &str,
        messages: &[Message],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let is_memory = system.contains("## Your memory")
            || messages.iter().any(|m| {
                m.as_concat_text()
                    .contains("Write your memory of the history in the system prompt.")
            });

        self.requests.lock().unwrap().push(CapturedRequest {
            is_memory,
            system: system.to_string(),
            messages: messages.to_vec(),
        });

        if is_memory {
            let mut queue = self.memory_responses.lock().unwrap();
            if let Some(res) = queue.pop_front() {
                return res;
            }
            let msg = Message::assistant().with_text("This is the folded memory.");
            let usage = ProviderUsage::new(
                "mock-model".to_string(),
                Usage::new(Some(1000), Some(200), Some(1200)),
            )
            .with_finish_reasons(vec!["end_turn".to_string()]);
            Ok((msg, usage))
        } else {
            let mut queue = self.main_responses.lock().unwrap();
            if let Some(res) = queue.pop_front() {
                return res;
            }
            let main_tokens = self.default_main_tokens.lock().unwrap().unwrap_or(1500);
            let msg = Message::assistant().with_text("Assistant answer.");
            let usage = ProviderUsage::new(
                "mock-model".to_string(),
                Usage::new(Some(main_tokens), Some(100), Some(main_tokens + 100)),
            )
            .with_finish_reasons(vec!["end_turn".to_string()]);
            Ok((msg, usage))
        }
    }
}

#[async_trait]
impl Provider for MockRollingProvider {
    fn get_name(&self) -> &str {
        "mock-rolling"
    }

    fn manages_own_context(&self) -> bool {
        self.manages_own_context
    }

    async fn complete(
        &self,
        _model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let is_memory = system.contains("## Your memory")
            || messages.iter().any(|m| {
                m.as_concat_text()
                    .contains("Write your memory of the history in the system prompt.")
            });
        if is_memory
            && self
                .block_memory_call
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            self.memory_call_entered.notify_waiters();
            self.memory_call_gate.notified().await;
        }
        if !is_memory {
            if let (Some(sm), Some(sid)) = (&self.session_manager, &self.session_id) {
                if let Ok(s) = sm.get_session(sid, false).await {
                    if rolling::is_retained(&s.usage) {
                        self.main_call_saw_retained
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
        }
        self.handle_request(system, messages)
    }

    async fn stream(
        &self,
        _model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let is_memory = system.contains("## Your memory")
            || messages.iter().any(|m| {
                m.as_concat_text()
                    .contains("Write your memory of the history in the system prompt.")
            });
        if is_memory
            && self
                .block_memory_call
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            self.memory_call_entered.notify_waiters();
            self.memory_call_gate.notified().await;
        }
        if !is_memory {
            if let (Some(sm), Some(sid)) = (&self.session_manager, &self.session_id) {
                if let Ok(s) = sm.get_session(sid, false).await {
                    if rolling::is_retained(&s.usage) {
                        self.main_call_saw_retained
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
        }
        let (msg, usage) = self.handle_request(system, messages)?;
        Ok(stream_from_single_message(msg, usage))
    }
}

impl goose::providers::base::ProviderDescriptor for MockRollingProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata {
            name: "mock-rolling".to_string(),
            display_name: "Mock Rolling Provider".to_string(),
            description: "Mock provider for rolling memory tests".to_string(),
            default_model: "mock-model".to_string(),
            known_models: vec![],
            model_doc_link: "".to_string(),
            config_keys: vec![],
            setup_steps: vec![],
            setup: None,
            deprecated: None,
        }
    }
}

impl ProviderDef for MockRollingProvider {
    type Provider = Self;

    fn from_env(
        _extensions: Vec<goose::config::ExtensionConfig>,
        _tls_config: Option<goose::providers::api_client::TlsConfig>,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<Self>> {
        Box::pin(async { Ok(Self::new()) })
    }
}

async fn setup_test_session(
    agent: &Agent,
    temp_dir: &TempDir,
    session_name: &str,
    messages: Vec<Message>,
    initial_usage: Option<Usage>,
) -> Result<Session> {
    let session = agent
        .config
        .session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            session_name.to_string(),
            SessionType::Hidden,
            GooseMode::default(),
        )
        .await?;

    let conversation = Conversation::new_unvalidated(messages);
    agent
        .config
        .session_manager
        .replace_conversation(&session.id, &conversation)
        .await?;

    let usage = initial_usage.unwrap_or_else(|| Usage::new(Some(3500), Some(200), Some(3700)));
    agent
        .config
        .session_manager
        .update(&session.id)
        .usage(usage)
        .accumulated_usage(usage)
        .apply()
        .await?;

    let reloaded = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    Ok(reloaded)
}

fn build_long_message(role: rmcp::model::Role, text_prefix: &str, repeat_words: usize) -> Message {
    let text = format!("{} {}", text_prefix, "content ".repeat(repeat_words));
    if role == rmcp::model::Role::User {
        Message::user().with_text(text)
    } else {
        Message::assistant().with_text(text)
    }
}

#[tokio::test]
async fn rolling_fold_happens_and_next_call_sees_memory_plus_tail() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    // Build 6 messages that exceed a 12000-token budget with a 3000-token tail
    // Messages 0, 1: Turn 1 (will be folded)
    // Messages 2, 3: Turn 2 (will be folded)
    // Messages 4, 5: Turn 3 (tail)
    let mut messages = vec![
        build_long_message(rmcp::model::Role::User, "turn 1 prompt", 1500),
        build_long_message(rmcp::model::Role::Assistant, "turn 1 reply", 1500),
        build_long_message(rmcp::model::Role::User, "turn 2 prompt", 1500),
        build_long_message(rmcp::model::Role::Assistant, "turn 2 reply", 1500),
        build_long_message(rmcp::model::Role::User, "turn 3 prompt", 1500),
        build_long_message(rmcp::model::Role::Assistant, "turn 3 reply", 1500),
    ];
    // Attach usage to turn 3 assistant so it serves as an anchor
    messages[5].metadata.usage = Some(Box::new(MessageUsage {
        input_tokens: Some(24_000),
        output_tokens: Some(1000),
        total_tokens: Some(25_000),
        ..Default::default()
    }));

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-fold-happens",
        messages,
        Some(Usage::new(Some(24_000), Some(1000), Some(25_000))),
    )
    .await?;

    // Enable rolling memory: budget 12000, tail 3000
    let mut s = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    rolling::write_settings(
        &mut s.extension_data,
        Some(RollingMemorySettings {
            budget_tokens: 12_000,
            tail_tokens: 3_000,
        }),
    );
    agent
        .config
        .session_manager
        .update(&session.id)
        .extension_data(s.extension_data)
        .apply()
        .await?;

    let provider = Arc::new(MockRollingProvider::new());
    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let new_user_prompt = Message::user().with_text("Turn 4 new user prompt");
    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let reply_stream = agent
        .reply(new_user_prompt, session_config, false, None)
        .await?;
    tokio::pin!(reply_stream);

    let mut inline_notifications = Vec::new();
    while let Some(event_res) = reply_stream.next().await {
        if let Ok(AgentEvent::Message(msg)) = event_res {
            for content in &msg.content {
                if let Some(note) = content.as_system_notification() {
                    inline_notifications.push(note.msg.clone());
                }
            }
        }
    }

    assert!(
        inline_notifications
            .iter()
            .any(|n| n.contains("Folding older history into memory")),
        "should emit folding progress notification"
    );
    assert!(
        inline_notifications
            .iter()
            .any(|n| n.contains("Memory updated")),
        "should emit memory updated notification"
    );

    // Verify DB has the memory pair
    let reloaded_session = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;
    let conv = reloaded_session.conversation.unwrap();
    let raw_messages = conv.messages();

    let recall_count = raw_messages
        .iter()
        .filter(|m| rolling::fold_part(m) == Some("recall"))
        .count();
    let memory_count = raw_messages
        .iter()
        .filter(|m| rolling::fold_part(m) == Some("memory"))
        .count();
    assert_eq!(recall_count, 1, "DB must have exactly one recall message");
    assert_eq!(memory_count, 1, "DB must have exactly one memory message");

    // Verify the main provider call saw memory + tail
    let captured = provider.captured_requests();
    assert_eq!(provider.memory_call_count(), 1, "memory call was made once");

    let main_call = captured
        .iter()
        .find(|r| !r.is_memory)
        .expect("main provider call must exist");

    let main_msgs = &main_call.messages;
    assert!(
        main_msgs.len() >= 3,
        "main call must see memory pair + tail"
    );
    assert_eq!(
        rolling::fold_part(&main_msgs[0]),
        Some("recall"),
        "first message seen by main call must be recall"
    );
    assert_eq!(
        rolling::fold_part(&main_msgs[1]),
        Some("memory"),
        "second message seen by main call must be memory"
    );
    assert!(
        main_msgs
            .last()
            .unwrap()
            .as_concat_text()
            .starts_with("Turn 4 new user prompt"),
        "last message seen by main call must be the new prompt"
    );

    let pool = agent.config.session_manager.pool().await?;
    let compaction_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM usage_ledger WHERE session_id = ? AND is_compaction = 1",
    )
    .bind(&session.id)
    .fetch_one(pool)
    .await?;
    assert_eq!(
        compaction_rows, 1,
        "ledger must have exactly one is_compaction row"
    );

    Ok(())
}

#[tokio::test]
async fn rolling_off_by_default_no_memory_call() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    let messages = vec![
        build_long_message(rmcp::model::Role::User, "prompt 1", 300),
        build_long_message(rmcp::model::Role::Assistant, "reply 1", 300),
    ];

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-rolling-off",
        messages,
        Some(Usage::new(Some(3500), Some(200), Some(3700))),
    )
    .await?;

    let provider = Arc::new(MockRollingProvider::new());
    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let user_msg = Message::user().with_text("Hello off by default");
    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
    tokio::pin!(reply_stream);
    while reply_stream.next().await.is_some() {}

    assert_eq!(
        provider.memory_call_count(),
        0,
        "rolling off by default must not make memory calls"
    );

    Ok(())
}

#[tokio::test]
async fn rolling_skips_threshold_auto_compaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let _guard = env_lock::lock_env([("GOOSE_AUTO_COMPACT_THRESHOLD", Some("0.001"))]);
    let agent = Agent::new();

    let messages = vec![
        build_long_message(rmcp::model::Role::User, "prompt 1", 100),
        build_long_message(rmcp::model::Role::Assistant, "reply 1", 100),
    ];

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-skip-auto-compact",
        messages,
        Some(Usage::new(Some(500), Some(100), Some(600))),
    )
    .await?;

    // Rolling memory budget is high (100k), so rolling won't fold, but threshold is 0.1%
    let mut s = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    rolling::write_settings(
        &mut s.extension_data,
        Some(RollingMemorySettings {
            budget_tokens: 100_000,
            tail_tokens: 30_000,
        }),
    );
    agent
        .config
        .session_manager
        .update(&session.id)
        .extension_data(s.extension_data)
        .apply()
        .await?;

    let provider = Arc::new(MockRollingProvider::new());
    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let user_msg = Message::user().with_text("Testing threshold skipped");
    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let reply_stream = agent.reply(user_msg, session_config, false, None).await?;
    tokio::pin!(reply_stream);

    let mut had_history_replaced = false;
    let mut had_exceeded_notification = false;
    while let Some(event_res) = reply_stream.next().await {
        match event_res {
            Ok(AgentEvent::HistoryReplaced(_)) => had_history_replaced = true,
            Ok(AgentEvent::Message(m)) => {
                for content in &m.content {
                    if let Some(note) = content.as_system_notification() {
                        if note.msg.contains("Exceeded auto-compact") {
                            had_exceeded_notification = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    assert!(
        !had_history_replaced,
        "rolling on must not yield HistoryReplaced"
    );
    assert!(
        !had_exceeded_notification,
        "rolling on must skip threshold auto-compaction"
    );

    Ok(())
}

#[tokio::test]
async fn rolling_failure_proceeds_and_counts() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    let mut messages = vec![
        build_long_message(rmcp::model::Role::User, "prompt 1", 12_000),
        build_long_message(rmcp::model::Role::Assistant, "reply 1", 12_000),
        build_long_message(rmcp::model::Role::User, "prompt 2", 5_000),
        build_long_message(rmcp::model::Role::Assistant, "reply 2", 5_000),
    ];
    messages[3].metadata.usage = Some(Box::new(MessageUsage {
        input_tokens: Some(52_000),
        output_tokens: Some(500),
        total_tokens: Some(52_500),
        ..Default::default()
    }));

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-failure-counts",
        messages,
        Some(Usage::new(Some(52_000), Some(500), Some(52_500))),
    )
    .await?;

    // Enable rolling memory: budget 50000, tail 10000
    let mut s = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    rolling::write_settings(
        &mut s.extension_data,
        Some(RollingMemorySettings {
            budget_tokens: 50_000,
            tail_tokens: 10_000,
        }),
    );
    agent
        .config
        .session_manager
        .update(&session.id)
        .extension_data(s.extension_data)
        .apply()
        .await?;

    let provider = Arc::new(MockRollingProvider::new());
    provider.set_default_main_tokens(52_000);
    // Queue two refusals: initial call and retry will both refuse
    provider.queue_memory_response(Err(ProviderError::Refusal {
        details: "refusal attempt 1".to_string(),
        category: None,
    }));
    provider.queue_memory_response(Err(ProviderError::Refusal {
        details: "refusal attempt 2".to_string(),
        category: None,
    }));

    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    // Turn 1: failure 1
    let stream1 = agent
        .reply(
            Message::user().with_text("Turn 1 user prompt"),
            session_config.clone(),
            false,
            None,
        )
        .await?;
    tokio::pin!(stream1);
    while stream1.next().await.is_some() {}

    let s1 = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    let state1 = rolling::read_state(&s1.extension_data);
    assert_eq!(
        state1.consecutive_failures, 1,
        "consecutive_failures must be 1 after first failure"
    );

    // Turn 2: failure 2
    provider.queue_memory_response(Err(ProviderError::Refusal {
        details: "refusal turn 2 call 1".to_string(),
        category: None,
    }));
    provider.queue_memory_response(Err(ProviderError::Refusal {
        details: "refusal turn 2 call 2".to_string(),
        category: None,
    }));

    let stream2 = agent
        .reply(
            Message::user().with_text("Turn 2 user prompt"),
            session_config.clone(),
            false,
            None,
        )
        .await?;
    tokio::pin!(stream2);
    while stream2.next().await.is_some() {}

    let s2 = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    let state2 = rolling::read_state(&s2.extension_data);
    assert_eq!(
        state2.consecutive_failures, 2,
        "consecutive_failures must be 2 after second failure"
    );

    // Turn 3: small growth -> back-off kicks in!
    let mem_calls_before = provider.memory_call_count();
    let stream3 = agent
        .reply(
            Message::user().with_text("Turn 3 small prompt"),
            session_config.clone(),
            false,
            None,
        )
        .await?;
    tokio::pin!(stream3);
    while stream3.next().await.is_some() {}

    let mem_calls_after = provider.memory_call_count();
    assert_eq!(
        mem_calls_after, mem_calls_before,
        "backed off turn must make no memory calls"
    );

    // Turn 4: Large prompt that passes backoff threshold (or hard cap), memory call succeeds
    // Queue success for memory call
    let large_prompt = "large turn 4 ".repeat(5000);
    let stream4 = agent
        .reply(
            Message::user().with_text(large_prompt),
            session_config.clone(),
            false,
            None,
        )
        .await?;
    tokio::pin!(stream4);
    while stream4.next().await.is_some() {}

    let s4 = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    let state4 = rolling::read_state(&s4.extension_data);
    assert_eq!(
        state4.consecutive_failures, 0,
        "successful fold must reset consecutive_failures to 0"
    );
    assert_eq!(state4.failed_at_estimate, None);

    Ok(())
}

#[tokio::test]
async fn rolling_refuses_unrolled_agent_loop() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-refuse-unrolled",
        vec![Message::user().with_text("hello")],
        None,
    )
    .await?;

    let mut s = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    rolling::write_settings(
        &mut s.extension_data,
        Some(RollingMemorySettings {
            budget_tokens: 12_000,
            tail_tokens: 3_000,
        }),
    );
    agent
        .config
        .session_manager
        .update(&session.id)
        .extension_data(s.extension_data)
        .apply()
        .await?;

    let provider = Arc::new(MockRollingProvider::new());
    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let user_msg = Message::user().with_text("run state machine");
    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let res = agent.reply(user_msg, session_config, true, None).await;
    let err_msg = match res {
        Err(e) => e.to_string(),
        Ok(_) => panic!("unrolled agent loop must be refused"),
    };
    assert!(
        err_msg.contains(
            "rolling memory is on for this session; the unrolled agent loop is not supported"
        ),
        "error must contain expected message: {err_msg}"
    );

    Ok(())
}

#[tokio::test]
async fn rolling_metrics_use_retained_form() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    let mut messages = vec![
        build_long_message(rmcp::model::Role::User, "prompt 1", 1500),
        build_long_message(rmcp::model::Role::Assistant, "reply 1", 1500),
        build_long_message(rmcp::model::Role::User, "prompt 2", 1500),
        build_long_message(rmcp::model::Role::Assistant, "reply 2", 1500),
    ];
    messages[3].metadata.usage = Some(Box::new(MessageUsage {
        input_tokens: Some(24_000),
        output_tokens: Some(1000),
        total_tokens: Some(25_000),
        ..Default::default()
    }));

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-metrics-retained",
        messages,
        Some(Usage::new(Some(24_000), Some(1000), Some(25_000))),
    )
    .await?;

    let mut s = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    rolling::write_settings(
        &mut s.extension_data,
        Some(RollingMemorySettings {
            budget_tokens: 12_000,
            tail_tokens: 3_000,
        }),
    );
    agent
        .config
        .session_manager
        .update(&session.id)
        .extension_data(s.extension_data)
        .apply()
        .await?;

    let initial_accumulated = session.accumulated_usage.total_tokens.unwrap_or(0);

    let provider = Arc::new(MockRollingProvider::with_session_tracker(
        agent.config.session_manager.clone(),
        session.id.clone(),
    ));
    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let reply_stream = agent
        .reply(
            Message::user().with_text("New prompt"),
            session_config,
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream);
    while reply_stream.next().await.is_some() {}

    let updated_session = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;

    assert_eq!(
        provider.memory_call_count(),
        1,
        "a rolling memory fold must have been performed"
    );

    let updated_accumulated = updated_session.accumulated_usage.total_tokens.unwrap_or(0);
    // Memory call total: 1200. Main call total: 1600.
    // Total increase should be at least memory call (1200) + main call (1600) = 2800.
    assert!(
        updated_accumulated >= initial_accumulated + 2800,
        "accumulated tokens must include both memory call and main call: initial={}, updated={}",
        initial_accumulated,
        updated_accumulated
    );

    assert!(
        provider
            .main_call_saw_retained
            .load(std::sync::atomic::Ordering::SeqCst),
        "main call must see retained session usage with NULL cache fields before main call overwrites it"
    );

    Ok(())
}

#[tokio::test]
async fn rolling_cancel_during_memory_call_drops_cleanly() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    let mut messages = vec![
        build_long_message(rmcp::model::Role::User, "prompt 1", 1500),
        build_long_message(rmcp::model::Role::Assistant, "reply 1", 1500),
        build_long_message(rmcp::model::Role::User, "prompt 2", 1500),
        build_long_message(rmcp::model::Role::Assistant, "reply 2", 1500),
    ];
    messages[3].metadata.usage = Some(Box::new(MessageUsage {
        input_tokens: Some(24_000),
        output_tokens: Some(1000),
        total_tokens: Some(25_000),
        ..Default::default()
    }));

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-cancel-cleanly",
        messages,
        Some(Usage::new(Some(24_000), Some(1000), Some(25_000))),
    )
    .await?;

    let mut s = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    rolling::write_settings(
        &mut s.extension_data,
        Some(RollingMemorySettings {
            budget_tokens: 12_000,
            tail_tokens: 3_000,
        }),
    );
    // Set initial consecutive_failures to 1
    rolling::write_state(
        &mut s.extension_data,
        &rolling::RollingState {
            ratio: 1.45,
            consecutive_failures: 1,
            failed_at_estimate: Some(25_000),
        },
    );
    agent
        .config
        .session_manager
        .update(&session.id)
        .extension_data(s.extension_data)
        .apply()
        .await?;

    let initial_accumulated = session.accumulated_usage.total_tokens.unwrap_or(0);

    let provider = Arc::new(MockRollingProvider::new());
    provider
        .block_memory_call
        .store(true, std::sync::atomic::Ordering::SeqCst);

    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let cancel_token = tokio_util::sync::CancellationToken::new();

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let entered = provider.memory_call_entered.clone();
    let cancel = cancel_token.clone();
    tokio::spawn(async move {
        entered.notified().await;
        cancel.cancel();
    });

    let reply_stream = agent
        .reply(
            Message::user().with_text("Prompt while memory call is cancelled"),
            session_config,
            false,
            Some(cancel_token),
        )
        .await?;
    tokio::pin!(reply_stream);

    let mut turn_events = Vec::new();
    while let Some(event_res) = reply_stream.next().await {
        turn_events.push(event_res?);
    }
    assert!(
        !turn_events.is_empty(),
        "turn must complete and yield events"
    );

    let reloaded = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;
    let conv = reloaded.conversation.unwrap();

    // 1. Assert no pair rows saved
    let pair_count = conv
        .messages()
        .iter()
        .filter(|m| rolling::fold_part(m).is_some())
        .count();
    assert_eq!(pair_count, 0, "no pair rows saved after cancellation");

    // 2. Assert consecutive_failures unchanged (still 1)
    let state = rolling::read_state(&reloaded.extension_data);
    assert_eq!(
        state.consecutive_failures, 1,
        "consecutive_failures must remain unchanged after cancellation"
    );

    // 3. Assert no usage recorded for memory call (accumulated usage unchanged)
    let accumulated_after = reloaded.accumulated_usage.total_tokens.unwrap_or(0);
    assert_eq!(
        accumulated_after, initial_accumulated,
        "no memory call usage recorded"
    );

    Ok(())
}

#[tokio::test]
async fn rolling_decide_skip_yields_no_folding_notice_and_no_memory_call() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    // Budget: 50_000, Tail: 10_000.
    // Pre-cut: 100 words (~150 tokens) -> ratio * new_material < 30_000 MIN_PROGRESS_TOKENS.
    // Tail: 10_000 words (~13_000 tokens).
    // Total tokens: ~55_000 > 50_000 budget, but <= 100_000 (hard cap 2B).
    // Decide will skip with "too little progress".
    let mut messages = vec![
        build_long_message(rmcp::model::Role::User, "prompt 1", 100),
        build_long_message(rmcp::model::Role::Assistant, "reply 1", 100),
        build_long_message(rmcp::model::Role::User, "prompt 2", 5000),
        build_long_message(rmcp::model::Role::Assistant, "reply 2", 5000),
    ];
    messages[3].metadata.usage = Some(Box::new(MessageUsage {
        input_tokens: Some(54_000),
        output_tokens: Some(1000),
        total_tokens: Some(55_000),
        ..Default::default()
    }));

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-decide-skip",
        messages,
        Some(Usage::new(Some(54_000), Some(1000), Some(55_000))),
    )
    .await?;

    let mut s = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    rolling::write_settings(
        &mut s.extension_data,
        Some(RollingMemorySettings {
            budget_tokens: 50_000,
            tail_tokens: 10_000,
        }),
    );
    agent
        .config
        .session_manager
        .update(&session.id)
        .extension_data(s.extension_data)
        .apply()
        .await?;

    let provider = Arc::new(MockRollingProvider::new());
    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let reply_stream = agent
        .reply(
            Message::user().with_text("New user prompt"),
            session_config,
            false,
            None,
        )
        .await?;
    tokio::pin!(reply_stream);

    let mut inline_notifications = Vec::new();
    while let Some(event_res) = reply_stream.next().await {
        if let Ok(AgentEvent::Message(msg)) = event_res {
            for content in &msg.content {
                if let Some(note) = content.as_system_notification() {
                    inline_notifications.push(note.msg.clone());
                }
            }
        }
    }

    assert_eq!(
        provider.memory_call_count(),
        0,
        "no memory call when decide skips"
    );
    assert!(
        !inline_notifications
            .iter()
            .any(|n| n.contains("Folding older history into memory")),
        "must not emit folding notice when decide skips"
    );
    assert!(
        !inline_notifications
            .iter()
            .any(|n| n.contains("compacting the conversation")),
        "must not emit progress notice when decide skips"
    );

    Ok(())
}

#[tokio::test]
async fn rolling_fires_only_once_across_turns() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    // Budget: 12_000, Tail: 3_000.
    // Turn 1 setup: over 24_000 tokens (hard cap triggers fold).
    let mut messages = vec![
        build_long_message(rmcp::model::Role::User, "prompt 1", 1500),
        build_long_message(rmcp::model::Role::Assistant, "reply 1", 1500),
        build_long_message(rmcp::model::Role::User, "prompt 2", 1500),
        build_long_message(rmcp::model::Role::Assistant, "reply 2", 1500),
    ];
    messages[3].metadata.usage = Some(Box::new(MessageUsage {
        input_tokens: Some(24_000),
        output_tokens: Some(1000),
        total_tokens: Some(25_000),
        ..Default::default()
    }));

    let session = setup_test_session(
        &agent,
        &temp_dir,
        "test-fires-only-once",
        messages,
        Some(Usage::new(Some(24_000), Some(1000), Some(25_000))),
    )
    .await?;

    let mut s = agent
        .config
        .session_manager
        .get_session(&session.id, false)
        .await?;
    rolling::write_settings(
        &mut s.extension_data,
        Some(RollingMemorySettings {
            budget_tokens: 12_000,
            tail_tokens: 3_000,
        }),
    );
    agent
        .config
        .session_manager
        .update(&session.id)
        .extension_data(s.extension_data)
        .apply()
        .await?;

    let provider = Arc::new(MockRollingProvider::new());
    // On Turn 1 main call, report realistic post-fold usage (input ~ 4000, with cache fields set so is_retained is false)
    let post_fold_usage = ProviderUsage::new(
        "mock-model".to_string(),
        Usage {
            input_tokens: Some(4000),
            output_tokens: Some(100),
            total_tokens: Some(4100),
            cache_read_input_tokens: Some(3500),
            cache_write_input_tokens: Some(500),
        },
    )
    .with_finish_reasons(vec!["end_turn".to_string()]);
    provider.queue_main_response(Ok((
        Message::assistant().with_text("Turn 1 assistant answer."),
        post_fold_usage,
    )));

    agent
        .update_provider(
            provider.clone(),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    // Turn 1: fold happens!
    let stream1 = agent
        .reply(
            Message::user().with_text("Turn 1 user prompt"),
            session_config.clone(),
            false,
            None,
        )
        .await?;
    tokio::pin!(stream1);
    while stream1.next().await.is_some() {}

    assert_eq!(
        provider.memory_call_count(),
        1,
        "turn 1 must trigger exactly one memory call"
    );

    // Turn 2: small user prompt. Context is well under budget (total ~ 4100 < 12_000).
    let stream2 = agent
        .reply(
            Message::user().with_text("Turn 2 small prompt"),
            session_config.clone(),
            false,
            None,
        )
        .await?;
    tokio::pin!(stream2);
    while stream2.next().await.is_some() {}

    // Assert exactly one memory call in total (turn 2 did not fold!)
    assert_eq!(
        provider.memory_call_count(),
        1,
        "turn 2 must not trigger a second memory call"
    );

    // Assert exactly one is_compaction row in usage_ledger
    let pool = agent.config.session_manager.pool().await?;
    let compaction_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM usage_ledger WHERE session_id = ? AND is_compaction = 1",
    )
    .bind(&session.id)
    .fetch_one(pool)
    .await?;
    assert_eq!(
        compaction_rows, 1,
        "must have exactly one is_compaction row in ledger across both turns"
    );

    Ok(())
}
