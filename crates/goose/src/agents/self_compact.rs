// Modified by Jonathan Mohrbacher, 2026: self-compaction agent tool definition and validation; loop execution helpers, outcome and messages; per-run cap message, updated tool description, and error resilience.
use crate::conversation::message::{
    Message, MessageContentBlock, SystemNotificationType, ToolRequest,
};
use crate::conversation::Conversation;
use crate::providers::base::Provider;
use crate::session::session_manager::SessionManager;
use goose_providers::conversation::token_usage::ProviderUsage;
use goose_providers::model::ModelConfig;
use rmcp::model::{JsonObject, Tool, ToolAnnotations};
use tokio_util::sync::CancellationToken;

pub const SELF_COMPACT_TOOL_NAME: &str = "self_compact_and_resume";
pub const SELF_COMPACT_PROMPT_MAX_CHARS: usize = 16_000;
pub const SELF_COMPACT_TOOL_DESCRIPTION: &str =
    "Compact your own context and continue with a resumption prompt you write. Use it only when your context has grown long enough that earlier detail is being crowded out, or when the user asks you to; it is not a routine step between tasks, and works at most once per turn. Before calling it, persist anything durable (beads, plan files, commits) that a summary may lose. After this step the conversation so far is replaced by a summary and `prompt` is delivered to you as the next user message; you keep working in the same turn. Call it alone, as the last tool call of the step. The prompt must stand on its own: restate the goal, the current state, and the first concrete action.";
pub const SELF_COMPACT_SUCCESS_MESSAGE: &str =
    "Compaction runs right after this step; your prompt will be the next user message. Do not call other tools in this step.";
pub const SELF_COMPACT_ALREADY_COMPACTED_MESSAGE: &str =
    "Already self-compacted in this turn; continue from the resume prompt. Context overflow is still handled automatically.";

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

pub fn check_self_compact_call(request: &ToolRequest, final_response: &Message) -> Option<String> {
    let Ok(call) = &request.tool_call else {
        return None;
    };
    if call.name != SELF_COMPACT_TOOL_NAME {
        return None;
    }
    let is_success = final_response.content.iter().any(|c| {
        if let MessageContentBlock::ToolResponse(r) = c {
            if r.id == request.id {
                if let Ok(res) = &r.tool_result {
                    if res.is_error == Some(true) {
                        return false;
                    }
                    let text = res
                        .content
                        .iter()
                        .filter_map(|b| b.as_text().map(|t| t.text.as_str()))
                        .collect::<Vec<_>>()
                        .join("");
                    return text == SELF_COMPACT_SUCCESS_MESSAGE;
                }
            }
        }
        false
    });
    if is_success {
        validate_self_compact_args(call.arguments.as_ref()).ok()
    } else {
        None
    }
}

pub const SELF_COMPACTED_NOTE_TEXT: &str = "Context self-compacted.";

pub fn self_compact_skipped_notice(prompt: &str) -> Message {
    Message::assistant()
        .with_text(format!(
            "Self-compaction skipped: the turn limit was reached. Resume prompt (not delivered):\n\n{prompt}"
        ))
        .with_generated_id()
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

pub fn self_compact_resume_message(prompt: &str, created: i64) -> Message {
    let mut msg = Message::user()
        .with_text(prompt)
        .with_steer()
        .with_generated_id();
    if msg.created < created {
        msg.created = created;
    }
    msg
}

pub fn self_compact_failed_notice(error: &str, prompt: &str) -> Message {
    Message::assistant()
        .with_text(format!(
            "Self-compaction failed: {error}. Context unchanged; the resume prompt was not delivered:\n\n{prompt}"
        ))
        .with_generated_id()
}

pub async fn handle_leftover_pending(
    session_manager: &SessionManager,
    session_id: &str,
    cancel_token: &Option<CancellationToken>,
    prompt: &str,
) -> Message {
    let is_cancelled = crate::utils::is_token_cancelled(cancel_token);
    let reason = if is_cancelled {
        "cancelled"
    } else {
        "run ended before compaction"
    };
    let notice = self_compact_failed_notice(reason, prompt);
    if let Err(e) = session_manager.add_message(session_id, &notice).await {
        tracing::warn!("Failed to persist self-compaction leftover notice: {e}");
    }
    notice
}

#[derive(Debug)]
pub struct SelfCompactSuccess {
    pub new_conversation: Conversation,
    pub history_replaced_conversation: Conversation,
    pub note_message: Message,
    pub resume_message: Message,
    pub retained_context_tokens: i32,
    pub usage: ProviderUsage,
}

#[derive(Debug)]
pub enum SelfCompactOutcome {
    Success(Box<SelfCompactSuccess>),
    Failed { notice: Message, is_cancelled: bool },
}

pub async fn execute_self_compact(
    session_manager: &SessionManager,
    session_id: &str,
    provider: &dyn Provider,
    model_config: &ModelConfig,
    cancel_token: &Option<CancellationToken>,
    prompt: &str,
) -> SelfCompactOutcome {
    let session = match session_manager.get_session(session_id, true).await {
        Ok(s) => s,
        Err(e) => {
            let error_str = format!("failed to load session: {e}");
            let notice = self_compact_failed_notice(&error_str, prompt);
            if let Err(err) = session_manager.add_message(session_id, &notice).await {
                tracing::warn!("Failed to persist self-compaction failure notice: {err}");
            }
            return SelfCompactOutcome::Failed {
                notice,
                is_cancelled: false,
            };
        }
    };
    let conversation = session.conversation.unwrap_or_default();

    let compact_result = if let Some(token) = cancel_token.as_ref() {
        tokio::select! {
            biased;
            _ = token.cancelled() => {
                Err(anyhow::anyhow!("cancelled"))
            }
            res = crate::context_mgmt::compact_messages(
                provider,
                model_config,
                session_id,
                &conversation,
                crate::context_mgmt::CompactionKind::SelfRequested,
            ) => res,
        }
    } else {
        crate::context_mgmt::compact_messages(
            provider,
            model_config,
            session_id,
            &conversation,
            crate::context_mgmt::CompactionKind::SelfRequested,
        )
        .await
    };

    match compact_result {
        Ok(compaction) => {
            let max_created = compaction
                .conversation
                .messages()
                .iter()
                .map(|m| m.created)
                .max()
                .unwrap_or_else(|| chrono::Utc::now().timestamp());

            let note = self_compacted_note_message(max_created);
            let resume = self_compact_resume_message(prompt, note.created);

            let history_replaced_conversation = compaction.conversation.clone();

            let mut new_messages = compaction.conversation.messages().clone();
            new_messages.push(note.clone());
            new_messages.push(resume.clone());
            let new_conversation = Conversation::new_unvalidated(new_messages);

            if let Err(e) = session_manager
                .replace_conversation(session_id, &new_conversation)
                .await
            {
                tracing::error!(
                    session_id,
                    error = %e,
                    "Failed to replace conversation during self-compaction"
                );
                let error_str = format!("failed to save conversation: {e}");
                let notice = self_compact_failed_notice(&error_str, prompt);
                if let Err(err) = session_manager.add_message(session_id, &notice).await {
                    tracing::warn!("Failed to persist self-compaction failure notice: {err}");
                }
                return SelfCompactOutcome::Failed {
                    notice,
                    is_cancelled: false,
                };
            }

            SelfCompactOutcome::Success(Box::new(SelfCompactSuccess {
                new_conversation,
                history_replaced_conversation,
                note_message: note,
                resume_message: resume,
                retained_context_tokens: compaction.retained_context_tokens,
                usage: compaction.usage,
            }))
        }
        Err(err) => {
            let is_cancelled = cancel_token.as_ref().is_some_and(|t| t.is_cancelled())
                || err.to_string().contains("cancelled");
            let error_str = if is_cancelled {
                "cancelled"
            } else {
                &err.to_string()
            };
            let notice = self_compact_failed_notice(error_str, prompt);
            if let Err(e) = session_manager.add_message(session_id, &notice).await {
                tracing::warn!("Failed to persist self-compaction failure notice: {e}");
            }
            SelfCompactOutcome::Failed {
                notice,
                is_cancelled,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
    use goose_providers::errors::ProviderError;
    use goose_providers::model::ModelConfig;
    use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, ListToolsResult};
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio_util::sync::CancellationToken;

    use crate::agents::extension::ExtensionConfig;
    use crate::agents::mcp_client::McpClientTrait;
    use crate::agents::tool_execution::ToolCallContext;
    use crate::agents::{Agent, AgentConfig, GoosePlatform};
    use crate::config::permission::PermissionManager;
    use crate::config::GooseMode;
    use crate::conversation::message::Message;
    use crate::providers::base::{stream_from_single_message, MessageStream, Provider};
    use crate::recipe::Response;
    use crate::session::session_manager::{Session, SessionManager, SessionType};

    struct MockTextProvider;
    #[async_trait::async_trait]
    impl Provider for MockTextProvider {
        async fn stream(
            &self,
            _model_config: &ModelConfig,
            _system_prompt: &str,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> std::result::Result<MessageStream, ProviderError> {
            Ok(stream_from_single_message(
                Message::assistant().with_text("mock response"),
                ProviderUsage::new("mock-model".to_string(), Usage::default()),
            ))
        }

        fn get_name(&self) -> &str {
            "mock-text"
        }
    }

    struct ContextOwningProvider;
    #[async_trait::async_trait]
    impl Provider for ContextOwningProvider {
        async fn stream(
            &self,
            _model_config: &ModelConfig,
            _system_prompt: &str,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> std::result::Result<MessageStream, ProviderError> {
            Ok(stream_from_single_message(
                Message::assistant().with_text("mock"),
                ProviderUsage::new("mock-model".to_string(), Usage::default()),
            ))
        }

        fn get_name(&self) -> &str {
            "context-owning"
        }

        fn manages_own_context(&self) -> bool {
            true
        }
    }

    struct ShadowingClient;
    #[async_trait::async_trait]
    impl McpClientTrait for ShadowingClient {
        async fn list_tools(
            &self,
            _session_id: &str,
            _next_cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> std::result::Result<ListToolsResult, crate::agents::mcp_client::Error> {
            Ok(ListToolsResult::with_all_items(vec![Tool::new(
                SELF_COMPACT_TOOL_NAME.to_string(),
                "Shadowing extension tool".to_string(),
                Arc::new(serde_json::Map::new()),
            )]))
        }

        async fn call_tool(
            &self,
            _ctx: &ToolCallContext,
            _name: &str,
            _arguments: Option<JsonObject>,
            _cancel_token: CancellationToken,
        ) -> std::result::Result<CallToolResult, crate::agents::mcp_client::Error> {
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "extension output",
            )]))
        }

        fn get_info(&self) -> Option<&rmcp::model::InitializeResult> {
            None
        }
    }

    fn persisted_builtin(name: &str) -> ExtensionConfig {
        ExtensionConfig::Builtin {
            name: name.to_string(),
            description: String::new(),
            display_name: None,
            timeout: None,
            bundled: None,
            available_tools: Vec::new(),
        }
    }

    struct TestSetup {
        agent: Agent,
        session: Session,
        provider: Arc<dyn Provider>,
    }

    async fn setup_self_compact_test(temp_dir: &TempDir) -> Result<TestSetup> {
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
        let session = session_manager
            .create_session(
                PathBuf::default(),
                "test".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?;
        let provider: Arc<dyn Provider> = Arc::new(MockTextProvider);
        agent
            .update_provider(
                provider.clone(),
                ModelConfig::new("mock-model"),
                &session.id,
            )
            .await?;
        Ok(TestSetup {
            agent,
            session,
            provider,
        })
    }

    async fn assert_self_compact_unavailable(agent: &Agent, session: &Session) -> Result<()> {
        assert!(!agent.self_compact_available(session).await);

        let tools = agent.list_tools(&session.id, None).await;
        assert!(!tools.iter().any(|t| t.name == SELF_COMPACT_TOOL_NAME));

        let (_, _, prep_tools, _, _, _) = agent.prepare_tools_and_prompt(session).await?;
        assert!(!prep_tools.iter().any(|t| t.name == SELF_COMPACT_TOOL_NAME));
        Ok(())
    }

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

    #[tokio::test]
    async fn test_self_compact_tool_listed_when_available() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        assert!(setup.agent.self_compact_available(&setup.session).await);

        let tools = setup.agent.list_tools(&setup.session.id, None).await;
        assert!(tools.iter().any(|t| t.name == SELF_COMPACT_TOOL_NAME));

        let (_, _, prep_tools, _, _, _) =
            setup.agent.prepare_tools_and_prompt(&setup.session).await?;
        assert!(prep_tools.iter().any(|t| t.name == SELF_COMPACT_TOOL_NAME));
        Ok(())
    }

    #[tokio::test]
    async fn test_self_compact_tool_not_listed_when_session_cannot_be_loaded() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        let tools = setup
            .agent
            .list_tools("non-existent-session-id", None)
            .await;
        assert!(!tools.iter().any(|t| t.name == SELF_COMPACT_TOOL_NAME));
        Ok(())
    }

    #[tokio::test]
    async fn test_self_compact_tool_not_listed_when_disabled_by_config() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", Some("false")),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        assert_self_compact_unavailable(&setup.agent, &setup.session).await
    }

    #[tokio::test]
    async fn test_self_compact_tool_not_listed_when_final_output_tool_set() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        let response = Response {
            json_schema: Some(serde_json::json!({
                "type": "object",
                "properties": { "result": { "type": "string" } }
            })),
        };
        setup.agent.add_final_output_tool(response).await?;
        assert_self_compact_unavailable(&setup.agent, &setup.session).await
    }

    #[tokio::test]
    async fn test_self_compact_tool_not_listed_when_provider_manages_own_context() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        let context_owning = Arc::new(ContextOwningProvider);
        setup
            .agent
            .update_provider(
                context_owning,
                ModelConfig::new("mock-model"),
                &setup.session.id,
            )
            .await?;
        assert_self_compact_unavailable(&setup.agent, &setup.session).await
    }

    #[tokio::test]
    async fn test_self_compact_tool_not_listed_for_subagent_session() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        let subagent_session = setup
            .agent
            .config
            .session_manager
            .create_session(
                PathBuf::default(),
                "subagent".to_string(),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await?;
        setup
            .agent
            .update_provider(
                setup.provider.clone(),
                ModelConfig::new("mock-model"),
                &subagent_session.id,
            )
            .await?;

        assert_self_compact_unavailable(&setup.agent, &subagent_session).await
    }

    #[tokio::test]
    async fn test_self_compact_tool_not_listed_when_recipe_retry_configured() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        let mut recipe = crate::recipe::Recipe::builder()
            .title("test-recipe")
            .description("test-desc")
            .instructions("test-instructions")
            .build()
            .unwrap();
        recipe.retry = Some(crate::agents::types::RetryConfig {
            max_retries: 3,
            timeout_seconds: None,
            on_failure: None,
            on_failure_timeout_seconds: None,
            checks: vec![],
        });
        let recipe_session = setup
            .agent
            .config
            .session_manager
            .create_session(
                PathBuf::default(),
                "recipe".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?;
        setup
            .agent
            .update_provider(
                setup.provider.clone(),
                ModelConfig::new("mock-model"),
                &recipe_session.id,
            )
            .await?;
        setup
            .agent
            .config
            .session_manager
            .update(&recipe_session.id)
            .recipe(Some(recipe))
            .apply()
            .await?;
        let recipe_session = setup
            .agent
            .config
            .session_manager
            .get_session(&recipe_session.id, false)
            .await?;

        assert_self_compact_unavailable(&setup.agent, &recipe_session).await
    }

    #[tokio::test]
    async fn test_self_compact_tool_not_listed_when_shadowed_by_extension() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        setup
            .agent
            .extension_manager
            .add_client(
                persisted_builtin("developer"),
                Arc::new(ShadowingClient),
                None,
            )
            .await;

        let tools = setup.agent.list_tools(&setup.session.id, None).await;
        let self_compact_tools: Vec<_> = tools
            .into_iter()
            .filter(|t| t.name == SELF_COMPACT_TOOL_NAME)
            .collect();
        assert_eq!(self_compact_tools.len(), 1);
        assert_eq!(
            self_compact_tools[0].description.as_deref(),
            Some("Shadowing extension tool")
        );

        let (_, _, prep_tools, _, _, _) =
            setup.agent.prepare_tools_and_prompt(&setup.session).await?;
        let self_compact_prep_tools: Vec<_> = prep_tools
            .into_iter()
            .filter(|t| t.name == SELF_COMPACT_TOOL_NAME)
            .collect();
        assert_eq!(self_compact_prep_tools.len(), 1);
        assert_eq!(
            self_compact_prep_tools[0].description.as_deref(),
            Some("Shadowing extension tool")
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_self_compact_tool_dispatch_success() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        let tool_call =
            CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME).with_arguments(rmcp::object!({
                "prompt": "Continue with task B"
            }));

        let (req_id, result) = setup
            .agent
            .dispatch_tool_call(tool_call, "call-1".to_string(), None, &setup.session)
            .await;
        assert_eq!(req_id, "call-1");
        let tool_call_result = result.expect("dispatch should return Ok");
        let output = tool_call_result
            .result
            .await
            .expect("tool call result success");
        assert_ne!(output.is_error, Some(true));
        assert_eq!(
            output.content[0].as_text().unwrap().text,
            SELF_COMPACT_SUCCESS_MESSAGE
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_self_compact_tool_dispatch_validation_errors() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;

        // Missing prompt
        let call_missing = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME);
        let (_, res) = setup
            .agent
            .dispatch_tool_call(
                call_missing,
                "call-missing".to_string(),
                None,
                &setup.session,
            )
            .await;
        let out = res.unwrap().result.await.unwrap();
        assert_eq!(out.is_error, Some(true));

        // Empty prompt
        let call_empty = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
            .with_arguments(rmcp::object!({ "prompt": "   " }));
        let (_, res) = setup
            .agent
            .dispatch_tool_call(call_empty, "call-empty".to_string(), None, &setup.session)
            .await;
        let out = res.unwrap().result.await.unwrap();
        assert_eq!(out.is_error, Some(true));

        // Oversized prompt
        let call_oversized = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME).with_arguments(
            rmcp::object!({ "prompt": "x".repeat(SELF_COMPACT_PROMPT_MAX_CHARS + 1) }),
        );
        let (_, res) = setup
            .agent
            .dispatch_tool_call(
                call_oversized,
                "call-over".to_string(),
                None,
                &setup.session,
            )
            .await;
        let out = res.unwrap().result.await.unwrap();
        assert_eq!(out.is_error, Some(true));
        Ok(())
    }

    #[tokio::test]
    async fn test_self_compact_tool_dispatch_unavailable() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let _env = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap())),
            ("GOOSE_SELF_COMPACT", None::<&str>),
        ]);
        let setup = setup_self_compact_test(&temp_dir).await?;
        setup
            .agent
            .add_final_output_tool(Response {
                json_schema: Some(serde_json::json!({ "type": "object" })),
            })
            .await?;

        let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
            .with_arguments(rmcp::object!({ "prompt": "Resume" }));

        let (_, res) = setup
            .agent
            .dispatch_tool_call(tool_call, "call-unavail".to_string(), None, &setup.session)
            .await;
        let out = res.unwrap().result.await.unwrap();
        assert_eq!(out.is_error, Some(true));
        assert!(out.content[0]
            .as_text()
            .unwrap()
            .text
            .contains("self_compact_and_resume is not available in this session"));
        Ok(())
    }

    #[tokio::test]
    async fn test_persist_turn_context_readds_when_prior_is_agent_invisible_self_compact(
    ) -> Result<()> {
        let temp_dir = TempDir::new()?;
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("data")));
        let session = session_manager
            .create_session(
                PathBuf::default(),
                "test".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?;

        let old_tc = Message::user()
            .with_text("tokens remaining: 10k")
            .with_metadata(
                crate::conversation::message::MessageMetadata::agent_only()
                    .with_turn_context()
                    .with_agent_invisible(),
            );
        let mut conversation = crate::conversation::Conversation::new_unvalidated(vec![old_tc]);

        let new_tc = Message::user()
            .with_text("tokens remaining: 10k")
            .with_metadata(
                crate::conversation::message::MessageMetadata::agent_only().with_turn_context(),
            );

        crate::agents::agent::persist_turn_context_if_changed(
            &session_manager,
            &session.id,
            &mut conversation,
            Some(new_tc),
        )
        .await?;

        assert_eq!(conversation.messages().len(), 2);
        assert!(conversation.messages()[1].is_agent_visible());

        // Subsequent call with matching text should not duplicate the agent-visible message
        crate::agents::agent::persist_turn_context_if_changed(
            &session_manager,
            &session.id,
            &mut conversation,
            Some(
                Message::user()
                    .with_text("tokens remaining: 10k")
                    .with_metadata(
                        crate::conversation::message::MessageMetadata::agent_only()
                            .with_turn_context(),
                    ),
            ),
        )
        .await?;
        assert_eq!(conversation.messages().len(), 2);
        Ok(())
    }

    #[test]
    fn test_check_self_compact_call_self_compact() {
        use rmcp::model::CallToolResult;

        // 1. Successful call with valid prompt -> Some("Resume here")
        let request = ToolRequest {
            id: "call-1".to_string(),
            tool_call: Ok(CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
                .with_arguments(rmcp::object!({ "prompt": "Resume here" }))),
            metadata: None,
            tool_meta: None,
        };
        let mut response = Message::user().with_generated_id();
        response.add_tool_response_with_metadata(
            "call-1",
            Ok(CallToolResult::success(vec![ContentBlock::text(
                SELF_COMPACT_SUCCESS_MESSAGE,
            )])),
            None,
        );
        assert_eq!(
            check_self_compact_call(&request, &response),
            Some("Resume here".to_string())
        );

        // 2. Failed call (is_error: true) -> None
        let mut err_response = Message::user().with_generated_id();
        err_response.add_tool_response_with_metadata(
            "call-1",
            Ok(CallToolResult::error(vec![ContentBlock::text(
                "some error",
            )])),
            None,
        );
        assert_eq!(check_self_compact_call(&request, &err_response), None);

        // 3. Tool result is Err -> None
        let mut err_result_response = Message::user().with_generated_id();
        err_result_response.add_tool_response_with_metadata(
            "call-1",
            Err(rmcp::model::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                "execution failed".to_string(),
                None,
            )),
            None,
        );
        assert_eq!(
            check_self_compact_call(&request, &err_result_response),
            None
        );

        // 4. Invalid arguments (empty prompt) -> None
        let req_empty_prompt = ToolRequest {
            id: "call-2".to_string(),
            tool_call: Ok(CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
                .with_arguments(rmcp::object!({ "prompt": "   " }))),
            metadata: None,
            tool_meta: None,
        };
        let mut resp_empty = Message::user().with_generated_id();
        resp_empty.add_tool_response_with_metadata(
            "call-2",
            Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
            None,
        );
        assert_eq!(
            check_self_compact_call(&req_empty_prompt, &resp_empty),
            None
        );

        // 5. Different tool name -> None
        let req_other_tool = ToolRequest {
            id: "call-3".to_string(),
            tool_call: Ok(CallToolRequestParams::new("other_tool")
                .with_arguments(rmcp::object!({ "prompt": "Resume here" }))),
            metadata: None,
            tool_meta: None,
        };
        let mut resp_other = Message::user().with_generated_id();
        resp_other.add_tool_response_with_metadata(
            "call-3",
            Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
            None,
        );
        assert_eq!(check_self_compact_call(&req_other_tool, &resp_other), None);
    }

    #[test]
    fn test_self_compact_messages_properties_self_compact() {
        // 1. Skipped notice
        let skipped = self_compact_skipped_notice("Resume after turns");
        assert_eq!(skipped.role, rmcp::model::Role::Assistant);
        assert!(skipped.is_user_visible());
        assert!(skipped.is_agent_visible());
        assert!(skipped.id.is_some());
        assert!(skipped.as_concat_text().contains("Self-compaction skipped: the turn limit was reached. Resume prompt (not delivered):\n\nResume after turns"));

        // 2. Progress notice
        let progress = self_compact_progress_notice();
        assert_eq!(progress.role, rmcp::model::Role::Assistant);
        assert!(progress.is_user_visible());
        assert!(!progress.is_agent_visible());
        assert!(progress.content.iter().any(|c| match c {
            MessageContentBlock::SystemNotification(n) => {
                n.notification_type == SystemNotificationType::InlineMessage
                    && n.msg.contains("Compacting context (self-requested)")
            }
            _ => false,
        }));

        // 3. Note message
        let note = self_compacted_note_message(1_700_000_000);
        assert_eq!(note.role, rmcp::model::Role::Assistant);
        assert!(note.is_user_visible());
        assert!(!note.is_agent_visible());
        assert_eq!(note.as_concat_text(), SELF_COMPACTED_NOTE_TEXT);
        assert!(note.created >= 1_700_000_000);
        assert!(note.id.is_some());

        // 4. Resume message
        let resume = self_compact_resume_message("Resume now", 1_700_000_000);
        assert_eq!(resume.role, rmcp::model::Role::User);
        assert!(resume.is_user_visible());
        assert!(resume.is_agent_visible());
        assert!(resume.metadata.steer);
        assert_eq!(resume.as_concat_text(), "Resume now");
        assert!(resume.created >= 1_700_000_000);
        assert!(resume.id.is_some());

        // 5. Failed notice
        let failed = self_compact_failed_notice("network error", "Resume prompt");
        assert_eq!(failed.role, rmcp::model::Role::Assistant);
        assert!(failed.is_user_visible());
        assert!(failed.is_agent_visible());
        assert!(failed.id.is_some());
        assert!(failed.as_concat_text().contains("Self-compaction failed: network error. Context unchanged; the resume prompt was not delivered:\n\nResume prompt"));
    }

    struct MockErrorProvider;
    #[async_trait::async_trait]
    impl Provider for MockErrorProvider {
        async fn stream(
            &self,
            _model_config: &ModelConfig,
            _system_prompt: &str,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> std::result::Result<MessageStream, ProviderError> {
            Err(ProviderError::ExecutionError(
                "summarization failed".to_string(),
            ))
        }

        fn get_name(&self) -> &str {
            "mock-error"
        }
    }

    #[tokio::test]
    async fn test_execute_self_compact_success_self_compact() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("data")));
        let session = session_manager
            .create_session(
                PathBuf::default(),
                "test".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?;

        let user_msg = Message::user().with_text("original user msg");
        let assistant_msg = Message::assistant().with_text("original assistant msg");
        session_manager.add_message(&session.id, &user_msg).await?;
        session_manager
            .add_message(&session.id, &assistant_msg)
            .await?;

        let provider: Arc<dyn Provider> = Arc::new(MockTextProvider);
        let model_config = ModelConfig::new("mock-model");
        let cancel_token = None;

        let outcome = execute_self_compact(
            &session_manager,
            &session.id,
            provider.as_ref(),
            &model_config,
            &cancel_token,
            "Resume task after compaction",
        )
        .await;

        match outcome {
            SelfCompactOutcome::Success(success) => {
                let SelfCompactSuccess {
                    new_conversation,
                    history_replaced_conversation,
                    note_message,
                    resume_message,
                    ..
                } = *success;
                // Check in-memory conversation matches DB
                let reloaded = session_manager.get_session(&session.id, true).await?;
                let reloaded_conv = reloaded.conversation.expect("conversation exists");
                assert_eq!(
                    reloaded_conv.messages().len(),
                    new_conversation.messages().len()
                );

                // HistoryReplaced conversation does NOT contain trailing note or resume
                assert!(!history_replaced_conversation
                    .messages()
                    .iter()
                    .any(|m| m.as_concat_text() == SELF_COMPACTED_NOTE_TEXT));
                assert!(!history_replaced_conversation
                    .messages()
                    .iter()
                    .any(|m| m.as_concat_text() == "Resume task after compaction"));

                // Note message checks
                assert_eq!(note_message.as_concat_text(), SELF_COMPACTED_NOTE_TEXT);
                assert!(note_message.is_user_visible());
                assert!(!note_message.is_agent_visible());

                // Resume message checks
                assert_eq!(
                    resume_message.as_concat_text(),
                    "Resume task after compaction"
                );
                assert!(resume_message.is_user_visible());
                assert!(resume_message.is_agent_visible());
                assert!(resume_message.metadata.steer);

                // Timestamps non-decreasing
                let timestamps: Vec<i64> = new_conversation
                    .messages()
                    .iter()
                    .map(|m| m.created)
                    .collect();
                for w in timestamps.windows(2) {
                    assert!(
                        w[0] <= w[1],
                        "Timestamps not non-decreasing: {} > {}",
                        w[0],
                        w[1]
                    );
                }

                // Check that original messages are agent-invisible
                assert!(!new_conversation.messages()[0].is_agent_visible());
                assert!(!new_conversation.messages()[1].is_agent_visible());
            }
            SelfCompactOutcome::Failed { .. } => panic!("Expected Success outcome"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_execute_self_compact_provider_error_self_compact() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("data")));
        let session = session_manager
            .create_session(
                PathBuf::default(),
                "test".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?;

        let user_msg = Message::user().with_text("original user msg");
        session_manager.add_message(&session.id, &user_msg).await?;

        let provider: Arc<dyn Provider> = Arc::new(MockErrorProvider);
        let model_config = ModelConfig::new("mock-model");
        let cancel_token = None;

        let outcome = execute_self_compact(
            &session_manager,
            &session.id,
            provider.as_ref(),
            &model_config,
            &cancel_token,
            "Resume prompt",
        )
        .await;

        match outcome {
            SelfCompactOutcome::Failed {
                notice,
                is_cancelled,
            } => {
                assert!(!is_cancelled);
                assert!(notice.as_concat_text().contains("Self-compaction failed"));
                assert!(notice.as_concat_text().contains("Resume prompt"));

                // DB conversation was not compacted; has original + notice
                let reloaded = session_manager.get_session(&session.id, true).await?;
                let msgs = reloaded.conversation.unwrap().messages().clone();
                assert_eq!(msgs.len(), 2);
                assert_eq!(msgs[0].as_concat_text(), "original user msg");
                assert!(msgs[0].is_agent_visible()); // still agent-visible!
                assert!(msgs[1].as_concat_text().contains("Self-compaction failed"));
            }
            SelfCompactOutcome::Success { .. } => panic!("Expected Failed outcome"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_execute_self_compact_cancelled_self_compact() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("data")));
        let session = session_manager
            .create_session(
                PathBuf::default(),
                "test".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?;

        let user_msg = Message::user().with_text("original user msg");
        session_manager.add_message(&session.id, &user_msg).await?;

        let provider: Arc<dyn Provider> = Arc::new(MockTextProvider);
        let model_config = ModelConfig::new("mock-model");
        let token = CancellationToken::new();
        token.cancel(); // Pre-cancel
        let cancel_token = Some(token);

        let outcome = execute_self_compact(
            &session_manager,
            &session.id,
            provider.as_ref(),
            &model_config,
            &cancel_token,
            "Resume prompt",
        )
        .await;

        match outcome {
            SelfCompactOutcome::Failed {
                notice,
                is_cancelled,
            } => {
                assert!(is_cancelled);
                assert!(notice.as_concat_text().contains("cancelled"));
                assert!(notice.as_concat_text().contains("Resume prompt"));
            }
            SelfCompactOutcome::Success { .. } => panic!("Expected Failed outcome"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_execute_self_compact_session_load_error_self_compact() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("data")));
        let provider: Arc<dyn Provider> = Arc::new(MockTextProvider);
        let model_config = ModelConfig::new("mock-model");

        let outcome = execute_self_compact(
            &session_manager,
            "non-existent-session-id",
            provider.as_ref(),
            &model_config,
            &None,
            "Resume prompt",
        )
        .await;

        match outcome {
            SelfCompactOutcome::Failed {
                notice,
                is_cancelled,
            } => {
                assert!(!is_cancelled);
                assert!(notice.as_concat_text().contains("failed to load session"));
                assert!(notice.as_concat_text().contains("Resume prompt"));
            }
            SelfCompactOutcome::Success { .. } => panic!("Expected Failed outcome"),
        }
        Ok(())
    }

    #[test]
    fn test_check_self_compact_call_chat_mode_skipped_does_not_arm_self_compact() {
        let request = ToolRequest {
            id: "call-chat".to_string(),
            tool_call: Ok(CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
                .with_arguments(rmcp::object!({ "prompt": "Resume prompt" }))),
            metadata: None,
            tool_meta: None,
        };
        let mut response = Message::user().with_generated_id();
        response.add_tool_response_with_metadata(
            "call-chat",
            Ok(rmcp::model::CallToolResult::success(vec![
                rmcp::model::ContentBlock::text(
                    crate::agents::tool_execution::CHAT_MODE_TOOL_SKIPPED_RESPONSE,
                ),
            ])),
            None,
        );

        assert_eq!(check_self_compact_call(&request, &response), None);
    }

    #[tokio::test]
    async fn test_self_compact_tool_dispatch_second_call_in_same_run_returns_error_self_compact(
    ) -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let setup = setup_self_compact_test(&temp_dir).await?;

        let tool_call1 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
            .with_arguments(rmcp::object!({ "prompt": "First call" }));
        let (_, res1) = setup
            .agent
            .dispatch_tool_call(tool_call1, "call-1".to_string(), None, &setup.session)
            .await;
        let out1 = res1.unwrap().result.await.unwrap();
        assert_ne!(out1.is_error, Some(true));
        assert_eq!(
            out1.content[0].as_text().unwrap().text,
            SELF_COMPACT_SUCCESS_MESSAGE
        );

        // Second call in the same run must return error
        let tool_call2 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
            .with_arguments(rmcp::object!({ "prompt": "Second call" }));
        let (_, res2) = setup
            .agent
            .dispatch_tool_call(tool_call2, "call-2".to_string(), None, &setup.session)
            .await;
        let out2 = res2.unwrap().result.await.unwrap();
        assert_eq!(out2.is_error, Some(true));
        assert!(out2.content[0]
            .as_text()
            .unwrap()
            .text
            .contains(SELF_COMPACT_ALREADY_COMPACTED_MESSAGE));

        // Resetting run allows compacting again
        setup.agent.reset_self_compact_run(&setup.session.id).await;
        let tool_call3 = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
            .with_arguments(rmcp::object!({ "prompt": "Third call in new run" }));
        let (_, res3) = setup
            .agent
            .dispatch_tool_call(tool_call3, "call-3".to_string(), None, &setup.session)
            .await;
        let out3 = res3.unwrap().result.await.unwrap();
        assert_ne!(out3.is_error, Some(true));
        assert_eq!(
            out3.content[0].as_text().unwrap().text,
            SELF_COMPACT_SUCCESS_MESSAGE
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_self_compact_tool_dispatch_shadowed_by_extension_routes_to_extension_self_compact(
    ) -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let setup = setup_self_compact_test(&temp_dir).await?;

        setup
            .agent
            .extension_manager
            .add_client(
                persisted_builtin("developer"),
                Arc::new(ShadowingClient),
                None,
            )
            .await;

        let tool_call = CallToolRequestParams::new(SELF_COMPACT_TOOL_NAME)
            .with_arguments(rmcp::object!({ "prompt": "Extension prompt" }));
        let (_, res) = setup
            .agent
            .dispatch_tool_call(tool_call, "call-shadow".to_string(), None, &setup.session)
            .await;
        let out = res.unwrap().result.await.unwrap();
        assert_ne!(out.is_error, Some(true));
        // Shadowing extension output, NOT the built-in SELF_COMPACT_SUCCESS_MESSAGE!
        assert_eq!(out.content[0].as_text().unwrap().text, "extension output");
        Ok(())
    }

    #[tokio::test]
    async fn test_self_compact_tool_not_listed_when_recipe_response_schema_set_self_compact(
    ) -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let setup = setup_self_compact_test(&temp_dir).await?;

        let recipe_session = setup
            .agent
            .config
            .session_manager
            .create_session(
                PathBuf::default(),
                "recipe_response_test".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?;

        setup
            .agent
            .update_provider(
                setup.provider.clone(),
                ModelConfig::new("mock-model"),
                &recipe_session.id,
            )
            .await?;

        let mut recipe = crate::recipe::Recipe::builder()
            .title("Response Schema Recipe")
            .description("Recipe with response schema")
            .instructions("Instructions")
            .build()
            .unwrap();
        recipe.response = Some(Response {
            json_schema: Some(serde_json::json!({ "type": "object" })),
        });
        setup
            .agent
            .config
            .session_manager
            .update(&recipe_session.id)
            .recipe(Some(recipe))
            .apply()
            .await?;

        let recipe_session = setup
            .agent
            .config
            .session_manager
            .get_session(&recipe_session.id, false)
            .await?;

        assert!(
            setup.agent.final_output_tool.lock().await.is_none(),
            "final_output_tool must not be installed in memory"
        );
        assert_self_compact_unavailable(&setup.agent, &recipe_session).await
    }
}
