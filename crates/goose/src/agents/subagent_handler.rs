// Modified by Jonathan Mohrbacher, 2026: subagent policy: with GOOSE_SUBAGENT_DEFAULT_RECIPE set, an extension that fails to start fails the delegate; cache-miss breaker stops a Gemini delegate whose prompt cache stopped hitting.
use crate::{
    agents::{
        cache_miss_breaker,
        final_output_tool::FinalOutputTool,
        state_machine::{trailing_error, MAX_TURNS_MESSAGE},
        subagent_task_config::TaskConfig,
        Agent, AgentConfig, AgentEvent, ExtensionConfig, GoosePlatform, SessionConfig,
    },
    config::permission::PermissionManager,
    conversation::{
        message::{Message, MessageContent},
        Conversation,
    },
    prompt_template::render_template,
    recipe::Recipe,
    session::extension_data::{EnabledExtensionsState, ExtensionState},
    session::{Session, SessionManager, SessionType},
};
use anyhow::{anyhow, Result};
use futures::future::BoxFuture;
use futures::StreamExt;
use rmcp::model::{ErrorCode, ErrorData, Notification, Role, ServerNotification};
#[expect(deprecated)]
use rmcp::model::{LoggingLevel, LoggingMessageNotificationParam};
use serde::Serialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

pub type OnMessageCallback = Arc<dyn Fn(&Message) + Send + Sync>;

#[derive(Serialize)]
pub struct SubagentPromptContext {
    pub max_turns: usize,
    pub task_instructions: String,
    pub tool_count: usize,
    pub available_tools: String,
}

type AgentMessagesFuture =
    Pin<Box<dyn Future<Output = Result<(Conversation, Option<String>)>> + Send>>;

pub struct SubagentRunParams {
    pub config: AgentConfig,
    pub recipe: Recipe,
    pub task_config: TaskConfig,
    pub return_last_only: bool,
    pub session_id: String,
    pub cancellation_token: Option<CancellationToken>,
    pub on_message: Option<OnMessageCallback>,
    pub notification_tx: Option<tokio::sync::mpsc::UnboundedSender<ServerNotification>>,
}

pub async fn run_subagent_task(params: SubagentRunParams) -> Result<String, anyhow::Error> {
    let return_last_only = params.return_last_only;
    let (messages, final_output) = get_agent_messages(params).await.map_err(|e| {
        ErrorData::new(
            ErrorCode::INTERNAL_ERROR,
            format!("Failed to execute task: {}", e),
            None,
        )
    })?;

    if let Some(output) = final_output {
        return Ok(output);
    }

    Ok(extract_response_text(&messages, return_last_only))
}

pub(crate) async fn from_foreground_subagent_session(
    session_manager: Arc<SessionManager>,
    session: &Session,
    use_login_shell_path: bool,
) -> Result<(Agent, SessionConfig)> {
    let session_id = &session.id;
    if session.session_type != SessionType::SubAgent {
        return Err(anyhow!("Session {session_id} is not a subagent"));
    }
    let recipe = session
        .recipe
        .as_ref()
        .ok_or_else(|| anyhow!("Subagent {session_id} has no saved recipe"))?;
    let max_turns = recipe
        .settings
        .as_ref()
        .and_then(|settings| settings.max_turns)
        .ok_or_else(|| anyhow!("Subagent {session_id} has no saved turn limit"))?;
    let provider_name = session
        .provider_name
        .as_deref()
        .ok_or_else(|| anyhow!("Subagent {session_id} has no saved provider"))?;
    let model_config = session
        .model_config
        .as_ref()
        .ok_or_else(|| anyhow!("Subagent {session_id} has no saved model"))?;
    let saved_extensions = session
        .extension_data
        .get_extension_state(
            EnabledExtensionsState::EXTENSION_NAME,
            EnabledExtensionsState::VERSION,
        )
        .ok_or_else(|| anyhow!("Subagent {session_id} has no saved extension selection"))?;
    let extensions = EnabledExtensionsState::from_value(saved_extensions)?.extensions;

    let provider = crate::providers::create_with_working_dir(
        provider_name,
        extensions.clone(),
        session.working_dir.clone(),
    )
    .await?;
    provider.apply_model_selection(model_config).await?;

    let mut config = AgentConfig::new(
        session_manager,
        PermissionManager::instance(),
        None,
        session.goose_mode,
        true,
        GoosePlatform::GooseCli,
    )
    .with_use_login_shell_path(use_login_shell_path);
    config.is_subagent = true;
    let agent = Agent::with_config(config);
    *agent.provider.lock().await = Some(provider);
    let strict =
        crate::agents::platform_extensions::summon::subagent_default_recipe_path().is_some();
    add_subagent_extensions(extensions, strict, |ext| {
        agent.add_extension_inner(ext, session_id)
    })
    .await?;

    let subagent_prompt = build_subagent_prompt(
        &agent,
        max_turns,
        session_id,
        recipe.instructions.clone().unwrap_or_default(),
    )
    .await?;
    agent.override_system_prompt(subagent_prompt).await;
    let session_config = SessionConfig {
        id: session_id.to_string(),
        schedule_id: None,
        max_turns: Some(max_turns as u32),
        retry_config: recipe.retry.clone(),
    };
    Ok((agent, session_config))
}

/// Fork patch: with GOOSE_SUBAGENT_DEFAULT_RECIPE set (the subagent policy), an
/// extension the subagent was given but could not start is an error -- a reviewer
/// whose ro-shell failed must not run on with no shell and report anyway.
async fn add_subagent_extensions<F, Fut, E>(
    extensions: impl IntoIterator<Item = ExtensionConfig>,
    strict: bool,
    mut add_fn: F,
) -> Result<()>
where
    F: FnMut(ExtensionConfig) -> Fut,
    Fut: Future<Output = std::result::Result<(), E>>,
    E: std::fmt::Display,
{
    for extension in extensions {
        let name = extension.name();
        if let Err(e) = add_fn(extension).await {
            if strict {
                return Err(anyhow!(
                    "Failed to start extension '{name}' for subagent: {e}"
                ));
            }
            debug!("Failed to add extension '{}' to subagent: {}", name, e);
        }
    }
    Ok(())
}

pub(crate) enum SubagentOutcome {
    Completed(String),
    Failed(String),
}

pub(crate) enum SubagentStart {
    HasOutcome(SubagentOutcome),
    Started {
        task: Option<String>,
        run: BoxFuture<'static, SubagentOutcome>,
    },
}

#[derive(Clone)]
pub(crate) struct ForegroundSubagentRunner {
    session_manager: Arc<SessionManager>,
    use_login_shell_path: bool,
}

impl ForegroundSubagentRunner {
    pub(crate) fn new(session_manager: Arc<SessionManager>, use_login_shell_path: bool) -> Self {
        Self {
            session_manager,
            use_login_shell_path,
        }
    }

    pub(crate) async fn start(
        &self,
        parent_id: &str,
        subagent_id: &str,
        cancel: CancellationToken,
    ) -> SubagentStart {
        let subagent = match self.session_manager.get_session(subagent_id, true).await {
            Ok(subagent) => subagent,
            Err(error) => {
                return SubagentStart::HasOutcome(SubagentOutcome::Failed(error.to_string()))
            }
        };
        if subagent.session_type != SessionType::SubAgent
            || subagent.parent_session_id.as_deref() != Some(parent_id)
        {
            return SubagentStart::HasOutcome(SubagentOutcome::Failed(
                "it does not belong to this session".to_string(),
            ));
        }
        if let Some(output) = subagent
            .conversation
            .as_ref()
            .and_then(|conversation| FinalOutputTool::successful_output(conversation.messages()))
        {
            return SubagentStart::HasOutcome(SubagentOutcome::Completed(output));
        }
        SubagentStart::Started {
            task: subagent
                .recipe
                .as_ref()
                .and_then(|recipe| recipe.prompt.clone()),
            run: Box::pin(self.clone().run(subagent, cancel)),
        }
    }

    async fn run_to_end(&self, subagent: &Session, cancel: CancellationToken) -> Result<()> {
        let (agent, session_config) = from_foreground_subagent_session(
            self.session_manager.clone(),
            subagent,
            self.use_login_shell_path,
        )
        .await?;
        // A child token, so the breaker can stop this delegate without touching its parent.
        let cancel = cancel.child_token();
        let mut breaker =
            cache_miss_breaker::Breaker::from_env(&subagent.id).deferring_to_tool_results();
        let mut events = agent
            .stream_state_machine_session(session_config, cancel.clone())
            .await?;
        while let Some(event) = events.next().await {
            if breaker.observe(&event?) == cache_miss_breaker::Verdict::Stop {
                cancel.cancel();
                drain_after_stop(&mut events).await;
                return Err(cache_miss_breaker::Stopped(breaker.stop_message(None)).into());
            }
        }
        Ok(())
    }

    async fn run(self, subagent: Session, cancel: CancellationToken) -> SubagentOutcome {
        let run_result = self.run_to_end(&subagent, cancel).await;
        let subagent = match self.session_manager.get_session(&subagent.id, true).await {
            Ok(subagent) => subagent,
            Err(error) => return SubagentOutcome::Failed(error.to_string()),
        };
        let messages = subagent.conversation.as_ref().map(Conversation::messages);
        if let Some(output) =
            messages.and_then(|messages| FinalOutputTool::successful_output(messages))
        {
            return SubagentOutcome::Completed(output);
        }
        if let Err(error) = run_result {
            if error.is::<cache_miss_breaker::Stopped>() {
                let last = messages.and_then(|m| cache_miss_breaker::last_assistant_text(m));
                return SubagentOutcome::Failed(with_last_output(error.to_string(), last));
            }
            return SubagentOutcome::Failed(error.to_string());
        }
        if let Some(error) = subagent.conversation.as_ref().and_then(trailing_error) {
            return SubagentOutcome::Failed(format!("{error:?}"));
        }
        SubagentOutcome::Failed(failure_reason(
            messages.map(Vec::as_slice).unwrap_or_default(),
        ))
    }
}

fn failure_reason(messages: &[Message]) -> String {
    let Some(last) = messages.last() else {
        return "stopped without final output".to_string();
    };
    let last_text = last.as_concat_text();
    if last_text == MAX_TURNS_MESSAGE {
        let last_response = messages
            .iter()
            .rev()
            .filter(|message| message.role == Role::Assistant)
            .map(Message::as_concat_text)
            .find(|text| !text.is_empty() && text != MAX_TURNS_MESSAGE);
        return match last_response {
            Some(text) => format!("max turns reached; last response: {text}"),
            None => "max turns reached".to_string(),
        };
    }
    if last_text.is_empty() {
        "stopped without final output".to_string()
    } else {
        last_text
    }
}

fn extract_response_text(messages: &Conversation, return_last_only: bool) -> String {
    if return_last_only {
        messages
            .messages()
            .last()
            .and_then(|message| {
                message.content.iter().find_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    _ => None,
                })
            })
            .unwrap_or_else(|| String::from("No text content in last message"))
    } else {
        let all_text_content: Vec<String> = messages
            .iter()
            .flat_map(|message| {
                message.content.iter().filter_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    crate::conversation::message::MessageContent::ToolResponse(tool_response) => {
                        if let Ok(result) = &tool_response.tool_result {
                            let texts: Vec<String> = result
                                .content
                                .iter()
                                .filter_map(|content| {
                                    if let rmcp::model::ContentBlock::Text(raw_text_content) =
                                        content
                                    {
                                        Some(raw_text_content.text.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if !texts.is_empty() {
                                Some(format!("Tool result: {}", texts.join("\n")))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
            })
            .collect();

        all_text_content.join("\n")
    }
}

pub const SUBAGENT_TOOL_REQUEST_TYPE: &str = "subagent_tool_request";

fn get_agent_messages(params: SubagentRunParams) -> AgentMessagesFuture {
    Box::pin(async move {
        let SubagentRunParams {
            config,
            recipe,
            task_config,
            session_id,
            cancellation_token,
            on_message,
            notification_tx,
            ..
        } = params;

        let system_instructions = recipe.instructions.clone().unwrap_or_default();
        let user_task = recipe
            .prompt
            .clone()
            .unwrap_or_else(|| "Begin.".to_string());

        let agent = Arc::new(Agent::with_config(config));

        agent
            .update_provider(
                task_config.provider.clone(),
                task_config.model_config.clone(),
                &session_id,
            )
            .await
            .map_err(|e| anyhow!("Failed to set provider on sub agent: {}", e))?;

        let strict =
            crate::agents::platform_extensions::summon::subagent_default_recipe_path().is_some();
        add_subagent_extensions(task_config.extensions.clone(), strict, |ext| {
            agent.add_extension(ext, &session_id)
        })
        .await?;

        let has_response_schema = recipe.response.is_some();
        agent
            .apply_recipe_components(recipe.response.clone(), true)
            .await?;

        let max_turns = task_config
            .max_turns
            .expect("TaskConfig always sets max_turns");
        let subagent_prompt =
            build_subagent_prompt(&agent, max_turns, &session_id, system_instructions).await?;
        agent.override_system_prompt(subagent_prompt).await;

        let user_message =
            Message::user().with_text(format!("Subagent ID: {session_id}\n\n{user_task}"));
        let mut conversation = Conversation::new_unvalidated(vec![user_message.clone()]);

        agent
            .config
            .session_manager
            .update(&session_id)
            .recipe(Some(recipe.clone()))
            .apply()
            .await?;

        if let Some(activities) = recipe.activities {
            for activity in activities {
                info!("Recipe activity: {}", activity);
            }
        }
        let session_config = SessionConfig {
            id: session_id.clone(),
            schedule_id: None,
            max_turns: task_config.max_turns.map(|v| v as u32),
            retry_config: recipe.retry,
        };

        // A child token, so the breaker can stop this delegate without touching its parent.
        let cancel = cancellation_token
            .map(|token| token.child_token())
            .unwrap_or_default();
        let mut breaker = cache_miss_breaker::Breaker::from_env(&session_id);
        let mut stream =
            crate::session_context::with_session_id(Some(session_id.to_string()), async {
                agent
                    .reply(
                        user_message,
                        session_config,
                        crate::agents::state_machine::enabled(),
                        Some(cancel.clone()),
                    )
                    .await
            })
            .await
            .map_err(|e| anyhow!("Failed to get reply from agent: {}", e))?;

        while let Some(message_result) = stream.next().await {
            if let Ok(event) = &message_result {
                if breaker.observe(event) == cache_miss_breaker::Verdict::Stop {
                    cancel.cancel();
                    drain_after_stop(&mut stream).await;
                    // The trip step may have delivered the final output alongside other tools.
                    if let Some(output) = get_final_output(&agent, has_response_schema).await {
                        return Ok((conversation, Some(output)));
                    }
                    let last = cache_miss_breaker::last_assistant_text(conversation.messages());
                    return Err(anyhow!(breaker.stop_message(last.as_deref())));
                }
            }
            match message_result {
                Ok(AgentEvent::Message(msg)) => {
                    if let Some(ref callback) = on_message {
                        callback(&msg);
                    }
                    if let Some(ref tx) = notification_tx {
                        for content in &msg.content {
                            if let Some(notif) = create_tool_notification(content, &session_id) {
                                if tx.send(notif).is_err() {
                                    debug!(
                                        "Notification receiver dropped for subagent {}",
                                        session_id
                                    );
                                }
                            }
                        }
                    }
                    conversation.push(msg);
                }
                Ok(AgentEvent::Usage(_)) => {}
                Ok(AgentEvent::MessageUsage { .. }) => {}
                Ok(AgentEvent::McpNotification(_)) => {}
                Ok(AgentEvent::HistoryReplaced(updated_conversation)) => {
                    conversation = updated_conversation;
                }
                Err(e) => {
                    tracing::error!("Error receiving message from subagent: {}", e);
                    break;
                }
            }
        }

        let final_output = get_final_output(&agent, has_response_schema).await;

        Ok((conversation, final_output))
    })
}

/// Let a cancelled delegate's stream wind down so its session is saved cleanly, but
/// never wait long: a stopped delegate must not hold its parent's tool call.
async fn drain_after_stop<S, T>(stream: &mut S)
where
    S: futures::Stream<Item = T> + Unpin,
{
    let drain = async { while stream.next().await.is_some() {} };
    if tokio::time::timeout(std::time::Duration::from_secs(30), drain)
        .await
        .is_err()
    {
        debug!("cache-miss-breaker: delegate stream did not end within 30s of cancel");
    }
}

fn with_last_output(message: String, last: Option<String>) -> String {
    match last.map(|t| t.trim().to_string()).filter(|t| !t.is_empty()) {
        Some(text) => format!("{message}\n--- its last output ---\n{text}"),
        None => message,
    }
}

async fn build_subagent_prompt(
    agent: &Agent,
    max_turns: usize,
    session_id: &str,
    system_instructions: String,
) -> Result<String> {
    let mut tool_names: Vec<_> = agent
        .list_tools(session_id, None)
        .await
        .into_iter()
        .filter(super::reply_parts::is_tool_visible_to_model)
        .map(|t| t.name.to_string())
        .collect();
    tool_names.sort_unstable();
    render_template(
        "subagent_system.md",
        &SubagentPromptContext {
            max_turns,
            task_instructions: system_instructions,
            tool_count: tool_names.len(),
            available_tools: tool_names.join(", "),
        },
    )
    .map_err(|e| anyhow!("Failed to render subagent system prompt: {}", e))
}

async fn get_final_output(agent: &Agent, has_response_schema: bool) -> Option<String> {
    if has_response_schema {
        agent
            .final_output_tool
            .lock()
            .await
            .as_ref()
            .and_then(|tool| tool.final_output.clone())
    } else {
        None
    }
}

#[expect(deprecated)]
pub fn create_tool_notification(
    content: &MessageContent,
    subagent_id: &str,
) -> Option<ServerNotification> {
    if let MessageContent::ToolRequest(req) = content {
        let tool_call = req.tool_call.as_ref().ok()?;

        Some(ServerNotification::LoggingMessageNotification(
            Notification::new(
                LoggingMessageNotificationParam::new(
                    LoggingLevel::Info,
                    serde_json::json!({
                        "type": SUBAGENT_TOOL_REQUEST_TYPE,
                        "subagent_id": subagent_id,
                        "tool_call": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments
                        }
                    }),
                )
                .with_logger(format!("subagent:{}", subagent_id)),
            ),
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{create_tool_notification, failure_reason, SUBAGENT_TOOL_REQUEST_TYPE};
    use crate::agents::state_machine::MAX_TURNS_MESSAGE;
    use crate::conversation::message::{Message, MessageContent};
    use rmcp::model::{CallToolRequestParams, ServerNotification};
    use serde_json::json;
    use serial_test::serial;

    #[test]
    #[expect(deprecated)]
    fn create_tool_notification_for_tool_request() {
        let tool_call = CallToolRequestParams::new("developer__shell".to_string())
            .with_arguments(json!({"command": "ls"}).as_object().unwrap().clone());
        let content = MessageContent::tool_request("req1", Ok(tool_call));
        let notification =
            create_tool_notification(&content, "session_1").expect("expected notification");

        let ServerNotification::LoggingMessageNotification(log_notif) = notification else {
            panic!("expected logging notification");
        };
        let data = log_notif
            .params
            .data
            .as_object()
            .expect("expected object data");
        assert_eq!(
            data.get("type").and_then(|v| v.as_str()),
            Some(SUBAGENT_TOOL_REQUEST_TYPE)
        );
        assert_eq!(
            data.get("subagent_id").and_then(|v| v.as_str()),
            Some("session_1")
        );
        let tool_call = data
            .get("tool_call")
            .and_then(|v| v.as_object())
            .expect("expected tool_call object");
        assert_eq!(
            tool_call.get("name").and_then(|v| v.as_str()),
            Some("developer__shell")
        );
    }

    #[test]
    fn create_tool_notification_ignores_non_tool_request() {
        let content = MessageContent::text("hello");
        assert!(create_tool_notification(&content, "session_1").is_none());
    }

    #[test]
    fn failure_reason_describes_how_the_subagent_stopped() {
        let max_turns = Message::assistant().with_text(MAX_TURNS_MESSAGE);
        let cases = [
            (vec![], "stopped without final output".to_string()),
            (
                vec![Message::assistant().with_text("")],
                "stopped without final output".to_string(),
            ),
            (
                vec![Message::assistant().with_text("I gave up")],
                "I gave up".to_string(),
            ),
            (
                vec![Message::user().with_text("go"), max_turns.clone()],
                "max turns reached".to_string(),
            ),
            (
                vec![
                    Message::assistant().with_text("halfway there"),
                    Message::user().with_text("tool result"),
                    max_turns,
                ],
                "max turns reached; last response: halfway there".to_string(),
            ),
        ];
        for (messages, expected) in cases {
            assert_eq!(failure_reason(&messages), expected);
        }
    }

    #[tokio::test]
    async fn subagent_policy_strict_extension_failure_is_an_error() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
        let session = session_manager
            .create_session(
                temp_dir.path().to_path_buf(),
                "test-session".to_string(),
                SessionType::SubAgent,
                crate::config::GooseMode::Auto,
            )
            .await
            .unwrap();
        let agent = Agent::with_config(AgentConfig::new(
            session_manager,
            PermissionManager::instance(),
            None,
            crate::config::GooseMode::default(),
            true,
            GoosePlatform::GooseCli,
        ));
        let good = crate::config::ExtensionConfig::Platform {
            name: "developer".into(),
            description: String::new(),
            display_name: None,
            bundled: None,
            available_tools: vec![],
        };
        assert!(
            add_subagent_extensions(vec![good], true, |ext| agent
                .add_extension_inner(ext, &session.id))
            .await
            .is_ok(),
            "valid extension should succeed in strict mode"
        );
        let bad = crate::config::ExtensionConfig::Stdio {
            name: "ro-shell".into(),
            description: String::new(),
            cmd: "/nonexistent/ro-shell".into(),
            args: vec![],
            envs: Default::default(),
            env_keys: vec![],
            timeout: Some(5),
            cwd: None,
            bundled: None,
            available_tools: vec![],
        };
        assert!(add_subagent_extensions(vec![bad.clone()], true, |ext| agent
            .add_extension_inner(ext, &session.id))
        .await
        .is_err());
        assert!(
            add_subagent_extensions(vec![bad.clone()], false, |ext| agent
                .add_extension_inner(ext, &session.id))
            .await
            .is_ok()
        );
    }

    #[tokio::test]
    #[serial]
    async fn subagent_policy_get_agent_messages_strict_extension_failure() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
        let session = session_manager
            .create_session(
                temp_dir.path().to_path_buf(),
                "test-session".to_string(),
                SessionType::SubAgent,
                crate::config::GooseMode::Auto,
            )
            .await
            .unwrap();
        let parent_provider = crate::providers::create("openai", Vec::new())
            .await
            .unwrap();
        let model_config = goose_providers::model::ModelConfig::new("test-model");
        let bad = crate::config::ExtensionConfig::Stdio {
            name: "ro-shell".into(),
            description: String::new(),
            cmd: "/nonexistent/ro-shell".into(),
            args: vec![],
            envs: Default::default(),
            env_keys: vec![],
            timeout: Some(5),
            cwd: None,
            bundled: None,
            available_tools: vec![],
        };
        let make_params = || SubagentRunParams {
            config: AgentConfig::new(
                session_manager.clone(),
                PermissionManager::instance(),
                None,
                crate::config::GooseMode::Auto,
                true,
                GoosePlatform::GooseCli,
            ),
            recipe: Recipe::builder()
                .version("1.0.0")
                .title("T")
                .description("T")
                .instructions("test instructions")
                .build()
                .unwrap(),
            task_config: TaskConfig::new(
                parent_provider.clone(),
                model_config.clone(),
                &session.id,
                temp_dir.path(),
                vec![bad.clone()],
            ),
            return_last_only: true,
            session_id: session.id.clone(),
            cancellation_token: None,
            on_message: None,
            notification_tx: None,
        };

        // 1. With GOOSE_SUBAGENT_DEFAULT_RECIPE set (strict mode): fails mentioning "Failed to start extension"
        {
            let d = tempfile::TempDir::new().unwrap();
            let p = d.path().join("unscoped.yaml");
            std::fs::write(
                &p,
                "version: 1.0.0\ntitle: U\ndescription: U\nextensions:\n  - type: platform\n    name: developer\n",
            )
            .unwrap();
            let p_str: &'static str = Box::leak(p.to_str().unwrap().to_string().into_boxed_str());
            let _env = env_lock::lock_env([("GOOSE_SUBAGENT_DEFAULT_RECIPE", Some(p_str))]);

            let err = get_agent_messages(make_params()).await.unwrap_err();
            assert!(
                err.to_string()
                    .contains("Failed to start extension 'ro-shell' for subagent"),
                "strict mode should fail with extension start error, got: {err}"
            );
        }

        // 2. Unset: no error from that cause
        {
            let _env = env_lock::lock_env([("GOOSE_SUBAGENT_DEFAULT_RECIPE", None::<&str>)]);

            let res = get_agent_messages(make_params()).await;
            if let Err(err) = res {
                assert!(
                    !err.to_string().contains("Failed to start extension"),
                    "non-strict mode should not fail with extension start error, got: {err}"
                );
            }
        }
    }

    /// Replays a stuck delegate: every call after the first comes back at 60K
    /// context with no cache read, and every response asks for another tool.
    #[derive(Debug)]
    struct UncachedGeminiProvider {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl crate::providers::base::Provider for UncachedGeminiProvider {
        fn get_name(&self) -> &str {
            "gcp_vertex_ai"
        }

        async fn stream(
            &self,
            _model_config: &goose_providers::model::ModelConfig,
            _system: &str,
            _messages: &[Message],
            _tools: &[rmcp::model::Tool],
        ) -> std::result::Result<
            crate::providers::base::MessageStream,
            goose_providers::errors::ProviderError,
        > {
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let read = if n == 0 { 40_000 } else { 0 };
            let message = Message::assistant()
                .with_text(format!("step {n}"))
                .with_tool_request(
                    format!("call-{n}"),
                    Ok(CallToolRequestParams::new("nosuch__tool".to_string())),
                );
            let mut usage = goose_providers::conversation::token_usage::Usage::new(
                Some(60_000),
                Some(10),
                None,
            );
            usage.cache_read_input_tokens = Some(read);
            let usage = goose_providers::conversation::token_usage::ProviderUsage::new(
                "gemini-test-replay".to_string(),
                usage,
            );
            Ok(Box::pin(futures::stream::iter(vec![
                Ok((Some(message), None)),
                Ok((None, Some(usage))),
            ])))
        }
    }

    #[tokio::test]
    #[serial]
    async fn cache_miss_breaker_stops_a_stuck_gemini_delegate_and_tells_the_lead() {
        let _env = env_lock::lock_env([
            ("GOOSE_CACHE_MISS_BREAKER", None::<&str>),
            ("GOOSE_CACHE_MISS_BREAKER_STEPS", Some("3")),
            ("GOOSE_CACHE_MISS_BREAKER_MIN_CONTEXT", None),
            ("GOOSE_SUBAGENT_DEFAULT_RECIPE", None),
        ]);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
        let session = session_manager
            .create_session(
                temp_dir.path().to_path_buf(),
                "stuck-child".to_string(),
                SessionType::SubAgent,
                crate::config::GooseMode::Auto,
            )
            .await
            .unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let provider = Arc::new(UncachedGeminiProvider {
            calls: calls.clone(),
        });
        let parent_cancel = CancellationToken::new();
        let params = SubagentRunParams {
            config: AgentConfig::new(
                session_manager.clone(),
                PermissionManager::instance(),
                None,
                crate::config::GooseMode::Auto,
                true,
                GoosePlatform::GooseCli,
            ),
            recipe: Recipe::builder()
                .version("1.0.0")
                .title("T")
                .description("T")
                .instructions("work forever")
                .build()
                .unwrap(),
            task_config: TaskConfig::new(
                provider,
                goose_providers::model::ModelConfig::new("gemini-test-replay"),
                &session.id,
                temp_dir.path(),
                vec![],
            )
            .with_max_turns(Some(50)),
            return_last_only: true,
            session_id: session.id.clone(),
            cancellation_token: Some(parent_cancel.clone()),
            on_message: None,
            notification_tx: None,
        };

        let err = run_subagent_task(params).await.unwrap_err().to_string();

        assert!(err.contains("STOPPED by the cache-miss breaker"), "{err}");
        assert!(err.contains("3 consecutive model calls"), "{err}");
        assert!(err.contains("~60K context"), "{err}");
        assert!(
            err.contains("re-dispatch #1 after cache-miss breaker"),
            "{err}"
        );
        assert!(err.contains("--- its last output ---\nstep 3"), "{err}");
        // One seeding call with a cache hit, then exactly three uncached ones.
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 4);
        assert!(
            !parent_cancel.is_cancelled(),
            "the parent's token is untouched"
        );
    }
}
