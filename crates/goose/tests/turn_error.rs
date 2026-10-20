// Modified by Jonathan Mohrbacher, 2026: the agent records why a turn ended in failure.
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::turn_error::{TurnError, TurnErrorKind};
use goose::agents::types::{RetryConfig, SuccessCheck};
use goose::agents::{Agent, AgentEvent, SessionConfig};
use goose::config::GooseMode;
use goose::conversation::message::{Message, MessageContent};
use goose::providers::base::{stream_from_single_message, MessageStream, Provider};
use goose::session::session_manager::SessionType;
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

enum Step {
    Text(&'static str),
    Empty,
    Fail(ProviderError),
    TextThenFail(&'static str, ProviderError),
}

/// Answers each normal provider call with the next scripted step and each
/// compaction call with the next scripted summary result. Unscripted calls
/// get `fallback`.
struct ScriptedProvider {
    steps: Mutex<VecDeque<Step>>,
    summaries: Mutex<VecDeque<Result<&'static str, ProviderError>>>,
    fallback_empty: bool,
    own_context: bool,
}

impl ScriptedProvider {
    fn new(steps: Vec<Step>) -> Self {
        Self {
            steps: Mutex::new(steps.into()),
            summaries: Mutex::new(VecDeque::new()),
            fallback_empty: false,
            own_context: false,
        }
    }

    fn with_summaries(mut self, summaries: Vec<Result<&'static str, ProviderError>>) -> Self {
        self.summaries = Mutex::new(summaries.into());
        self
    }

    fn always_empty() -> Self {
        Self {
            fallback_empty: true,
            ..Self::new(Vec::new())
        }
    }

    fn with_own_context(mut self) -> Self {
        self.own_context = true;
        self
    }
}

fn usage() -> ProviderUsage {
    ProviderUsage::new(
        "mock-model".to_string(),
        Usage::new(Some(10), Some(5), Some(15)),
    )
}

/// A compaction request is a single message asking for a summary; later
/// turns also mention the summary, so the message count matters.
fn is_compaction_call(messages: &[Message]) -> bool {
    messages.len() == 1
        && messages.iter().any(|message| {
            message.content.iter().any(|content| match content {
                MessageContent::Text(text) => text.text.to_lowercase().contains("summarize"),
                _ => false,
            })
        })
}

#[async_trait]
impl Provider for ScriptedProvider {
    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system_prompt: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        if is_compaction_call(messages) {
            let summary = self
                .summaries
                .lock()
                .await
                .pop_front()
                .unwrap_or(Ok("<summary>"));
            return summary.map(|text| {
                stream_from_single_message(Message::assistant().with_text(text), usage())
            });
        }

        let step = self.steps.lock().await.pop_front();
        let step = match step {
            Some(step) => step,
            None if self.fallback_empty => Step::Empty,
            None => Step::Text("default reply"),
        };
        match step {
            Step::Text(text) => Ok(stream_from_single_message(
                Message::assistant().with_text(text),
                usage(),
            )),
            Step::Empty => Ok(stream_from_single_message(Message::assistant(), usage())),
            Step::Fail(error) => Err(error),
            Step::TextThenFail(text, error) => {
                let chunks = vec![
                    Ok((Some(Message::assistant().with_text(text)), Some(usage()))),
                    Err(error),
                ];
                Ok(Box::pin(futures::stream::iter(chunks)))
            }
        }
    }

    fn get_name(&self) -> &str {
        "turn-error-mock"
    }

    fn manages_own_context(&self) -> bool {
        self.own_context
    }
}

struct Harness {
    agent: Agent,
    session_id: String,
}

impl Harness {
    async fn new(provider: ScriptedProvider) -> Result<Self> {
        let agent = Agent::new();
        let session = agent
            .config
            .session_manager
            .create_session(
                PathBuf::default(),
                "turn-error".to_string(),
                SessionType::Hidden,
                GooseMode::default(),
            )
            .await?;
        // A user-set name keeps session naming from consuming scripted steps.
        agent
            .config
            .session_manager
            .update(&session.id)
            .user_provided_name("turn-error")
            .apply()
            .await?;
        agent
            .update_provider(
                Arc::new(provider),
                ModelConfig::new("mock-model"),
                &session.id,
            )
            .await?;
        Ok(Self {
            agent,
            session_id: session.id,
        })
    }

    fn config(&self) -> SessionConfig {
        SessionConfig {
            id: self.session_id.clone(),
            schedule_id: None,
            max_turns: Some(5),
            retry_config: None,
        }
    }

    async fn run(&self, text: &str, config: SessionConfig, state_machine: bool) -> Result<()> {
        let stream = self
            .agent
            .reply(Message::user().with_text(text), config, state_machine, None)
            .await?;
        tokio::pin!(stream);
        while let Some(event) = stream.next().await {
            let _: AgentEvent = event?;
        }
        Ok(())
    }

    async fn turn(&self, text: &str) -> Result<Option<TurnError>> {
        self.run(text, self.config(), false).await?;
        Ok(self.agent.take_turn_error(&self.session_id).await)
    }
}

fn kind(error: &Option<TurnError>) -> Option<TurnErrorKind> {
    error.as_ref().map(|error| error.kind)
}

#[tokio::test]
async fn turn_error_none_when_the_turn_succeeds() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![Step::Text("hello")])).await?;
    assert_eq!(harness.turn("hi").await?, None);
    Ok(())
}

#[tokio::test]
async fn turn_error_network() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![Step::Fail(
        ProviderError::NetworkError("connection reset".into()),
    )]))
    .await?;
    let error = harness.turn("hi").await?;
    assert_eq!(kind(&error), Some(TurnErrorKind::Network));
    assert!(
        error.unwrap().message.contains("connection reset"),
        "message should carry the provider error"
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_refusal() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![Step::Fail(
        ProviderError::Refusal {
            details: "no".into(),
            category: None,
        },
    )]))
    .await?;
    assert_eq!(
        kind(&harness.turn("hi").await?),
        Some(TurnErrorKind::Refusal)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_generic_provider_error() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![Step::Fail(
        ProviderError::RequestFailed("HTTP 400".into()),
    )]))
    .await?;
    assert_eq!(
        kind(&harness.turn("hi").await?),
        Some(TurnErrorKind::Provider)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_text_then_error_in_one_response_ends_errored() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![Step::TextThenFail(
        "partial",
        ProviderError::NetworkError("dropped".into()),
    )]))
    .await?;
    assert_eq!(
        kind(&harness.turn("hi").await?),
        Some(TurnErrorKind::Network)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_context_length_when_recovery_compaction_fails() -> Result<()> {
    let provider = ScriptedProvider::new(vec![Step::Fail(ProviderError::ContextLengthExceeded(
        "too long".into(),
    ))])
    .with_summaries(vec![Err(ProviderError::ServerError(
        "summary failed".into(),
    ))]);
    let harness = Harness::new(provider).await?;
    assert_eq!(
        kind(&harness.turn("hi").await?),
        Some(TurnErrorKind::ContextLength)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_none_when_recovery_compaction_succeeds() -> Result<()> {
    let provider = ScriptedProvider::new(vec![
        Step::Fail(ProviderError::ContextLengthExceeded("too long".into())),
        Step::Text("after compaction"),
    ]);
    let harness = Harness::new(provider).await?;
    assert_eq!(harness.turn("hi").await?, None);
    Ok(())
}

#[tokio::test]
async fn turn_error_context_length_still_exceeded_after_compaction() -> Result<()> {
    let provider = ScriptedProvider::new(vec![
        Step::Fail(ProviderError::ContextLengthExceeded("too long".into())),
        Step::Fail(ProviderError::ContextLengthExceeded(
            "still too long".into(),
        )),
    ]);
    let harness = Harness::new(provider).await?;
    assert_eq!(
        kind(&harness.turn("hi").await?),
        Some(TurnErrorKind::ContextLength)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_context_length_when_pre_turn_auto_compaction_fails() -> Result<()> {
    let provider =
        ScriptedProvider::new(vec![Step::Text("never reached")]).with_summaries(vec![Err(
            ProviderError::ServerError("summary failed".into()),
        )]);
    let harness = Harness::new(provider).await?;
    harness
        .agent
        .config
        .session_manager
        .add_message(
            &harness.session_id,
            &Message::user().with_text("earlier message"),
        )
        .await?;
    harness
        .agent
        .config
        .session_manager
        .update(&harness.session_id)
        .usage(Usage::new(Some(10_000_000), Some(10), Some(10_000_010)))
        .apply()
        .await?;
    assert_eq!(
        kind(&harness.turn("hi").await?),
        Some(TurnErrorKind::ContextLength)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_empty_response_after_retries() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::always_empty()).await?;
    assert_eq!(
        kind(&harness.turn("hi").await?),
        Some(TurnErrorKind::EmptyResponse)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_retry_exhausted_for_recipe_retries() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(Vec::new())).await?;
    let mut config = harness.config();
    config.retry_config = Some(RetryConfig {
        max_retries: 1,
        checks: vec![SuccessCheck::Shell {
            command: "false".to_string(),
        }],
        on_failure: None,
        timeout_seconds: Some(30),
        on_failure_timeout_seconds: None,
    });
    harness.run("hi", config, false).await?;
    assert_eq!(
        kind(&harness.agent.take_turn_error(&harness.session_id).await),
        Some(TurnErrorKind::RetryExhausted)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_unsupported_structured_output() -> Result<()> {
    use goose::recipe::Response;

    let harness = Harness::new(ScriptedProvider::new(Vec::new()).with_own_context()).await?;
    harness
        .agent
        .add_final_output_tool(Response {
            json_schema: Some(serde_json::json!({
                "type": "object",
                "properties": { "result": { "type": "string" } }
            })),
        })
        .await?;
    assert_eq!(
        kind(&harness.turn("hi").await?),
        Some(TurnErrorKind::Unsupported)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_cleared_when_a_steer_recovers_the_turn() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![
        Step::Fail(ProviderError::NetworkError("connection reset".into())),
        Step::Text("recovered"),
    ]))
    .await?;
    harness
        .agent
        .steer(
            &harness.session_id,
            Message::user().with_text("try again please"),
        )
        .await;
    assert_eq!(harness.turn("hi").await?, None);
    Ok(())
}

#[tokio::test]
async fn turn_error_from_an_earlier_turn_is_reset_by_the_next_turn() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![
        Step::Fail(ProviderError::NetworkError("connection reset".into())),
        Step::Text("fine now"),
    ]))
    .await?;
    harness.run("first", harness.config(), false).await?;
    assert_eq!(harness.turn("second").await?, None);
    Ok(())
}

#[tokio::test]
async fn turn_error_reset_by_a_slash_command_turn() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![Step::Fail(
        ProviderError::NetworkError("connection reset".into()),
    )]))
    .await?;
    harness.run("first", harness.config(), false).await?;
    assert_eq!(harness.turn("/goal").await?, None);
    Ok(())
}

#[tokio::test]
async fn turn_error_state_machine_trailing_error() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![Step::Fail(
        ProviderError::RequestFailed("HTTP 400".into()),
    )]))
    .await?;
    harness.run("hi", harness.config(), true).await?;
    assert_eq!(
        kind(&harness.agent.take_turn_error(&harness.session_id).await),
        Some(TurnErrorKind::Provider)
    );
    Ok(())
}

#[tokio::test]
async fn turn_error_state_machine_none_when_the_turn_succeeds() -> Result<()> {
    let harness = Harness::new(ScriptedProvider::new(vec![Step::Text("hello")])).await?;
    harness.run("hi", harness.config(), true).await?;
    assert_eq!(
        harness.agent.take_turn_error(&harness.session_id).await,
        None
    );
    Ok(())
}
