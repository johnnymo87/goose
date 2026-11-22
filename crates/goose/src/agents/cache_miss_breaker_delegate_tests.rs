// Copyright 2026 Jonathan Mohrbacher. Licensed under the Apache License, Version 2.0.
// Fork-only tests: the cache-miss breaker stops a real state-machine delegate.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use goose_agent::subagent::{SubagentOutcome, SubagentUpdate};
use goose_providers::base::{MessageStream, Provider};
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::{CallToolRequestParams, Tool};
use serial_test::serial;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use crate::agents::subagent_handler::ForegroundSubagentRunner;
use crate::agents::StateMachineServices;
use crate::conversation::message::Message;
use crate::recipe::{Recipe, Settings};
use crate::session::extension_data::{EnabledExtensionsState, ExtensionData, ExtensionState};
use crate::session::SessionType;

const MODEL: &str = "gemini-test-replay";

/// Replays a stuck delegate: every call after the first comes back at 60K context
/// with no cache read, and every response asks for another tool.
struct UncachedGeminiProvider {
    calls: Arc<AtomicUsize>,
    final_answer_from: Option<usize>,
}

#[async_trait]
impl Provider for UncachedGeminiProvider {
    fn get_name(&self) -> &str {
        "gcp_vertex_ai"
    }

    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let tool = if self.final_answer_from.is_some_and(|from| n >= from) {
            CallToolRequestParams::new("recipe__final_output").with_arguments(
                serde_json::json!({"summary": "done"})
                    .as_object()
                    .unwrap()
                    .clone(),
            )
        } else {
            CallToolRequestParams::new("nosuch__tool")
        };
        let message = Message::assistant()
            .with_generated_id()
            .with_text(format!("step {n}"))
            .with_tool_request(format!("call-{n}"), Ok(tool));
        let mut usage = Usage::new(Some(60_000), Some(10), None);
        usage.cache_read_input_tokens = Some(if n == 0 { 40_000 } else { 0 });
        let usage = ProviderUsage::new(MODEL.to_string(), usage);
        Ok(Box::pin(futures::stream::iter(vec![
            Ok((Some(message), None)),
            Ok((None, Some(usage))),
        ])))
    }
}

struct Run {
    outcome: SubagentOutcome,
    calls: usize,
    parent_cancelled: bool,
}

async fn run_stuck_delegate(final_answer_from: Option<usize>) -> Run {
    let dir = TempDir::new().unwrap();
    let services = StateMachineServices::for_tool_tests(dir.path().to_path_buf());
    let manager = &services.session_manager;
    let parent = manager
        .create_session(
            dir.path().to_path_buf(),
            "parent".to_string(),
            SessionType::User,
            crate::config::GooseMode::Auto,
        )
        .await
        .unwrap();
    let child = manager
        .create_session(
            dir.path().to_path_buf(),
            "stuck-child".to_string(),
            SessionType::SubAgent,
            crate::config::GooseMode::Auto,
        )
        .await
        .unwrap();
    let mut extension_data = ExtensionData::default();
    EnabledExtensionsState::new(vec![])
        .to_extension_data(&mut extension_data)
        .unwrap();
    let recipe = Recipe {
        version: "1.0.0".to_string(),
        title: "T".to_string(),
        description: "T".to_string(),
        instructions: Some("work forever".to_string()),
        prompt: None,
        extensions: Some(vec![]),
        settings: Some(Settings {
            goose_provider: None,
            goose_model: None,
            temperature: None,
            max_turns: Some(50),
            thinking_effort: None,
        }),
        activities: None,
        author: None,
        parameters: None,
        response: Some(crate::recipe::Response {
            json_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"]
            })),
        }),
        sub_recipes: None,
        retry: None,
    };
    manager
        .update(&child.id)
        .recipe(Some(recipe))
        .provider_name("gcp_vertex_ai")
        .model_config(ModelConfig::new(MODEL))
        .extension_data(extension_data)
        .parent_session_id(Some(parent.id.clone()))
        .apply()
        .await
        .unwrap();
    manager
        .add_message(&child.id, &Message::user().with_text("Begin."))
        .await
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    services
        .update_provider(
            Arc::new(UncachedGeminiProvider {
                calls: Arc::clone(&calls),
                final_answer_from,
            }),
            ModelConfig::new(MODEL),
            &child.id,
        )
        .await
        .unwrap();

    let runner = ForegroundSubagentRunner::new(Arc::clone(&services));
    let parent_cancel = CancellationToken::new();
    let subagent = runner.load(&child.id).await.unwrap();
    let mut updates = runner.spawn(subagent, parent_cancel.clone());
    let mut outcome = None;
    while let Some(update) = updates.next().await {
        if let SubagentUpdate::Finished(finished) = update {
            outcome = Some(finished);
        }
    }
    Run {
        outcome: outcome.expect("delegate finished without an outcome"),
        calls: calls.load(Ordering::SeqCst),
        parent_cancelled: parent_cancel.is_cancelled(),
    }
}

fn breaker_env() -> [(&'static str, Option<&'static str>); 3] {
    [
        ("GOOSE_CACHE_MISS_BREAKER", None),
        ("GOOSE_CACHE_MISS_BREAKER_STEPS", Some("3")),
        ("GOOSE_CACHE_MISS_BREAKER_MIN_CONTEXT", None),
    ]
}

#[tokio::test]
#[serial]
async fn cache_miss_breaker_stops_a_stuck_gemini_delegate_and_tells_the_lead() {
    let _env = env_lock::lock_env(breaker_env());
    let run = run_stuck_delegate(None).await;

    let SubagentOutcome::Failed(err) = run.outcome else {
        panic!(
            "expected the breaker to fail the delegate, got {:?}",
            run.outcome
        );
    };
    assert!(err.contains("STOPPED by the cache-miss breaker"), "{err}");
    assert!(err.contains("3 consecutive model calls"), "{err}");
    assert!(err.contains("~60K context"), "{err}");
    assert!(
        err.contains("re-dispatch #1 after cache-miss breaker"),
        "{err}"
    );
    assert!(err.contains("--- its last output ---\nstep 3"), "{err}");
    assert_eq!(
        run.calls, 4,
        "one seeding call with a hit, then three uncached"
    );
    assert!(!run.parent_cancelled, "the parent's token is untouched");
}

#[tokio::test]
#[serial]
async fn cache_miss_breaker_lets_a_final_answer_through_on_the_trip_step() {
    let _env = env_lock::lock_env(breaker_env());
    let run = run_stuck_delegate(Some(3)).await;

    let SubagentOutcome::Completed(output) = run.outcome else {
        panic!("expected the final answer through, got {:?}", run.outcome);
    };
    assert!(output.contains("done"), "{output}");
    assert!(!run.parent_cancelled);
}

#[tokio::test]
#[serial]
async fn cache_miss_breaker_off_leaves_a_stuck_delegate_running() {
    let _env = env_lock::lock_env([
        ("GOOSE_CACHE_MISS_BREAKER", Some("off")),
        ("GOOSE_CACHE_MISS_BREAKER_STEPS", Some("3")),
        ("GOOSE_CACHE_MISS_BREAKER_MIN_CONTEXT", None),
    ]);
    let run = run_stuck_delegate(Some(6)).await;

    assert!(
        matches!(run.outcome, SubagentOutcome::Completed(_)),
        "{:?}",
        run.outcome
    );
    assert_eq!(run.calls, 7);
}
