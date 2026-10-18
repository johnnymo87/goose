// Modified by Jonathan Mohrbacher, 2026: self-compaction agent tool definition and validation.
use rmcp::model::{JsonObject, Tool, ToolAnnotations};

pub const SELF_COMPACT_TOOL_NAME: &str = "self_compact_and_resume";
pub const SELF_COMPACT_PROMPT_MAX_CHARS: usize = 16_000;
pub const SELF_COMPACT_TOOL_DESCRIPTION: &str =
    "Compact your own context now and continue with a resumption prompt you write. After this step the conversation so far is replaced by a summary, then `prompt` is delivered to you as the next user message and you keep working in the same turn. Use it at a juncture (a chunk of work done, before the next), after persisting anything durable (beads, plan files, commits) that the summary may lose. Call it alone, as the last tool call of the step. The prompt must stand on its own: restate the goal, current state, and the first concrete action.";
pub const SELF_COMPACT_SUCCESS_MESSAGE: &str =
    "Compaction runs right after this step; your prompt will be the next user message. Do not call other tools in this step.";

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
}
