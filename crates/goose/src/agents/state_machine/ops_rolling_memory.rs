// Modified by Jonathan Mohrbacher, 2026: RollingMemoryOperation folds older history into memory at turn start.
//! Folds older conversation history into a first-person memory at turn start when over token budget.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::state_machine::{
    applied, messages_since_kickoff, not_applicable, ConversationEffect, Emitter, GooseEffect,
    Operation, OperationResult,
};
use crate::agents::PromptManager;
use crate::agents::StateMachineServices;
use crate::context_mgmt::rolling::{self, FoldOutcome, PlanOutcome};
use crate::conversation::message::{Message, MessageUsage, SystemNotificationType};
use crate::conversation::Conversation;
use crate::providers::base::Provider;
use crate::session::{Session, SessionManager};
use goose_providers::model::ModelConfig;
use rmcp::model::Role;

pub struct RollingMemoryOperation {
    session_manager: Arc<SessionManager>,
    provider: Arc<dyn Provider>,
    model_config: ModelConfig,
    services: Arc<StateMachineServices>,
    folded_turns: Mutex<HashSet<String>>,
}

impl RollingMemoryOperation {
    pub fn new(
        session_manager: Arc<SessionManager>,
        provider: Arc<dyn Provider>,
        model_config: ModelConfig,
        services: Arc<StateMachineServices>,
    ) -> Self {
        Self {
            session_manager,
            provider,
            model_config,
            services,
            folded_turns: Mutex::new(HashSet::new()),
        }
    }

    /// Prepare system prompt and tools for plan_fold's base_o200k token estimation.
    ///
    /// Approximates the inference system prompt using PromptManager with session recipe instructions
    /// and project instructions, and tools resolved via services.current_session_snapshot.
    async fn prepare_system_and_tools(
        &self,
        session: &Session,
    ) -> (String, Vec<rmcp::model::Tool>) {
        let (snapshot_session, tools) = match self.services.current_session_snapshot(session).await
        {
            Ok((s, lease)) => {
                let tools = lease.tools().await;
                (s, tools)
            }
            Err(_) => (session.clone(), Vec::new()),
        };

        let mut prompt_parts = Vec::new();
        if let Some(project_id) = snapshot_session.project_id.as_deref() {
            if let Ok(entry) = crate::sources::read_project(project_id) {
                let mut parts = vec![format!("# Project: {}", entry.name)];
                if !entry.description.is_empty() {
                    parts.push(entry.description);
                }
                if !entry.content.is_empty() {
                    parts.push(entry.content);
                }
                prompt_parts.push(("project".to_string(), parts.join("\n\n")));
            }
        }
        if let Some(instructions) = snapshot_session
            .recipe
            .as_ref()
            .and_then(|r| r.instructions.clone())
        {
            prompt_parts.push(("recipe".to_string(), instructions));
        }

        let system_prompt = PromptManager::new()
            .build_system_prompt(&snapshot_session, prompt_parts, snapshot_session.goose_mode)
            .unwrap_or_default();

        (system_prompt, tools)
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for RollingMemoryOperation {
    fn name(&self) -> &'static str {
        "rolling_memory"
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let Some(settings) =
            rolling::resolve_settings(&session.extension_data, |k| std::env::var(k).ok())
        else {
            return not_applicable();
        };

        // Turn start: nothing but user messages since the kickoff. Slash commands
        // append an agent-only prompt before the first inference, so a length
        // check alone would never fold on those turns.
        let messages = messages_since_kickoff(conversation)?;
        if messages.is_empty() || messages[1..].iter().any(|m| m.role != Role::User) {
            return not_applicable();
        }

        let kickoff = &messages[0];
        let kickoff_id = kickoff
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        if kickoff
            .metadata
            .operation_note("rolling_memory", "folded")
            .is_some()
        {
            return not_applicable();
        }

        {
            let folded = self.folded_turns.lock().unwrap();
            if folded.contains(&kickoff_id) {
                return not_applicable();
            }
        }

        let pending = match rolling::check(session, &settings, conversation).await {
            Ok(Some(p)) => p,
            Ok(None) => {
                self.folded_turns.lock().unwrap().insert(kickoff_id);
                return not_applicable();
            }
            Err(e) => {
                tracing::warn!(session_id = %session.id, %e, "rolling memory check failed");
                self.folded_turns.lock().unwrap().insert(kickoff_id);
                return not_applicable();
            }
        };

        let (system_prompt, tools) = self.prepare_system_and_tools(session).await;

        let plan_outcome = rolling::plan_fold(
            &self.session_manager,
            &session.id,
            pending,
            conversation,
            &system_prompt,
            &tools,
        )
        .await;

        let fold_plan = match plan_outcome {
            PlanOutcome::NotNeeded => {
                self.folded_turns.lock().unwrap().insert(kickoff_id);
                return not_applicable();
            }
            PlanOutcome::Failed { reason } => {
                self.folded_turns.lock().unwrap().insert(kickoff_id);
                emit.message(Message::assistant().with_system_notification(
                    SystemNotificationType::InlineMessage,
                    format!("Rolling memory fold skipped: {reason}"),
                ));
                return not_applicable();
            }
            PlanOutcome::Ready(plan) => plan,
        };

        emit.message(Message::assistant().with_system_notification(
            SystemNotificationType::InlineMessage,
            "Context is over the rolling memory budget. Folding older history into memory...",
        ));
        emit.message(Message::assistant().with_system_notification(
            SystemNotificationType::ProgressMessage,
            "goose is compacting the conversation...",
        ));

        let cancel = emit.cancel_token();
        let fold_outcome = rolling::execute_fold(
            &self.session_manager,
            self.provider.as_ref(),
            &self.model_config,
            &session.id,
            fold_plan,
            conversation,
            Some(cancel),
        )
        .await;

        self.folded_turns.lock().unwrap().insert(kickoff_id.clone());

        match fold_outcome {
            FoldOutcome::Folded {
                conversation: folded_conv,
                usage,
                ..
            } => {
                emit.message(Message::assistant().with_system_notification(
                    SystemNotificationType::InlineMessage,
                    "Memory updated",
                ));

                let effects = vec![
                    GooseEffect::CompactConversation {
                        conversation: folded_conv,
                        usage: Some(*usage),
                    },
                    ConversationEffect::SetMessageOperationNote {
                        message_id: kickoff_id,
                        operation: "rolling_memory".to_string(),
                        key: "folded".to_string(),
                        value: serde_json::json!(true),
                    }
                    .into(),
                ];
                applied(effects)
            }
            FoldOutcome::Failed { reason, usage } => {
                tracing::warn!(
                    session_id = %session.id,
                    %reason,
                    "rolling memory fold skipped due to failure"
                );
                emit.message(Message::assistant().with_system_notification(
                    SystemNotificationType::InlineMessage,
                    format!("Rolling memory fold skipped: {reason}"),
                ));

                // Ledger only: the failed memory call must not replace the
                // session's current usage, which anchors the next estimate.
                for usage in &usage {
                    let ledger = MessageUsage::from_provider_usage(usage, false);
                    if let Err(e) = self
                        .session_manager
                        .record_usage_metrics(
                            &session.id,
                            session.schedule_id.clone(),
                            session.usage,
                            &usage.model,
                            &ledger,
                        )
                        .await
                    {
                        tracing::warn!(session_id = %session.id, %e, "failed to record rolling memory usage");
                    }
                }
                applied([ConversationEffect::SetMessageOperationNote {
                    message_id: kickoff_id,
                    operation: "rolling_memory".to_string(),
                    key: "folded".to_string(),
                    value: serde_json::json!(true),
                }
                .into()])
            }
            FoldOutcome::NotNeeded | FoldOutcome::BackedOff => not_applicable(),
        }
    }
}
