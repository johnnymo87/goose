// Modified by Jonathan Mohrbacher, 2026: on_set_session_settings pins session cache TTL and recipe; rolling memory setting in settings/set.
use super::*;

impl GooseAcpAgent {
    pub(super) async fn on_update_working_dir(
        &self,
        req: UpdateWorkingDirRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        let working_dir = req.working_dir.trim().to_string();
        if working_dir.is_empty() {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("working directory cannot be empty"));
        }
        let path = std::path::PathBuf::from(&working_dir);
        validate_absolute_cwd(&path)?;
        let session_id = &req.session_id;

        let session = self
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|_| {
                agent_client_protocol::Error::resource_not_found(Some(session_id.to_string()))
                    .data(format!("Session not found: {}", session_id))
            })?;

        let agent = self.get_session_agent(session_id).await?;
        agent
            .update_extension_working_dir(&session.id, &path)
            .await
            .internal_err_ctx("Failed to update extension working directory")?;

        let session = self
            .session_manager
            .get_session(session_id, false)
            .await
            .internal_err_ctx("Failed to reload session")?;
        agent
            .restore_provider_from_session(&session)
            .await
            .internal_err_ctx("Failed to refresh provider from session")?;

        Ok(EmptyResponse {})
    }

    pub(super) async fn on_set_session_system_prompt(
        &self,
        req: SetSessionSystemPromptRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        let session_id = req.session_id.trim();
        if session_id.is_empty() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("sessionId cannot be empty")
            );
        }

        let agent = self.get_session_agent(session_id).await?;
        match req.mode {
            SessionSystemPromptMode::Set => {
                if req.text.trim().is_empty() {
                    agent.clear_system_prompt_override().await;
                } else {
                    agent.override_system_prompt(req.text).await;
                }
            }
            SessionSystemPromptMode::Append => {
                let key = req
                    .key
                    .as_deref()
                    .map(str::trim)
                    .filter(|key| !key.is_empty())
                    .ok_or_else(|| {
                        agent_client_protocol::Error::invalid_params()
                            .data("key cannot be empty for append mode")
                    })?;
                if req.text.trim().is_empty() {
                    agent.remove_system_prompt_extra(key).await;
                } else {
                    agent.extend_system_prompt(key.to_string(), req.text).await;
                }
            }
        }

        Ok(EmptyResponse {})
    }

    pub(super) async fn on_set_session_settings(
        &self,
        req: SetSessionSettingsRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        let session_id = req.session_id.trim();
        if session_id.is_empty() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("sessionId cannot be empty")
            );
        }

        if let Some(ttl) = &req.cache_ttl {
            let normalized = ttl.trim().to_lowercase();
            if normalized != "5m" && normalized != "1h" {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data("invalid cacheTtl: must be '5m' or '1h'"));
            }
        }

        if let Some(Some(dto)) = &req.rolling_memory {
            if let Err(msg) = dto.validate() {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("invalid rollingMemory: {msg}")));
            }
        }

        let session = self
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|_| {
                agent_client_protocol::Error::resource_not_found(Some(session_id.to_string()))
                    .data(format!("Session not found: {}", session_id))
            })?;

        let (raw_recipe_to_persist, recipe_to_apply) = if let Some(recipe_dto) = req.recipe {
            let raw_recipe = crate::recipe::Recipe::try_from(recipe_dto).map_err(|e| {
                agent_client_protocol::Error::invalid_params().data(format!("invalid recipe: {e}"))
            })?;

            if raw_recipe.check_for_security_warnings() {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data("invalid recipe: contains hidden characters that could be malicious"));
            }

            let recipe_yaml = raw_recipe.to_yaml().map_err(|e| {
                agent_client_protocol::Error::invalid_params().data(format!("invalid recipe: {e}"))
            })?;
            crate::recipe::validate_recipe::validate_recipe_template_from_content(
                &recipe_yaml,
                None,
            )
            .map_err(|e| {
                let err_str = e.to_string();
                let stripped = err_str.strip_prefix("recipe: ").unwrap_or(&err_str);
                agent_client_protocol::Error::invalid_params()
                    .data(format!("invalid recipe: {stripped}"))
            })?;

            let recipe_to_apply = if session.session_type == SessionType::Scheduled {
                raw_recipe.clone()
            } else {
                let recipe_dir = crate::recipe::local_recipes::get_recipe_library_dir(true);
                let user_values = session.user_recipe_values.clone().unwrap_or_default();
                match self.render_recipe(&raw_recipe, &recipe_dir, user_values) {
                    Ok(Some(rendered)) => rendered,
                    Ok(None) => {
                        return Err(agent_client_protocol::Error::invalid_params()
                            .data("invalid recipe: missing required parameters"));
                    }
                    Err(e) => {
                        let msg = e
                            .data
                            .as_ref()
                            .and_then(|v| v.as_str())
                            .unwrap_or(&e.message);
                        let stripped = msg.strip_prefix("recipe: ").unwrap_or(msg);
                        return Err(agent_client_protocol::Error::invalid_params()
                            .data(format!("invalid recipe: {stripped}")));
                    }
                }
            };

            // Run the FinalOutputTool dry run on the version to apply
            if let Some(ref response) = recipe_to_apply.response {
                crate::agents::final_output_tool::FinalOutputTool::try_new(response.clone())
                    .map_err(|e| {
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("invalid recipe: {e}"))
                    })?;
            }

            (Some(raw_recipe), Some(recipe_to_apply))
        } else {
            (None, None)
        };

        let agent = self.get_session_agent(session_id).await?;

        if let Some(ttl) = req.cache_ttl.as_deref().map(str::trim) {
            let normalized = ttl.to_lowercase();
            let current_model_config = agent
                .model_config_for_session(session_id)
                .await
                .internal_err_ctx("Failed to resolve model config")?;
            let provider = agent
                .provider()
                .await
                .internal_err_ctx("Failed to get provider")?;
            let new_model_config = current_model_config.with_cache_ttl(&normalized);
            agent
                .update_provider(provider, new_model_config, session_id)
                .await
                .internal_err_ctx("Failed to update provider model config")?;
        }

        if let (Some(raw_recipe), Some(recipe_to_apply)) = (raw_recipe_to_persist, recipe_to_apply)
        {
            // Note: recipe.extensions is ignored by this call; session extensions are unchanged.
            // Apply the rendered recipe in memory first: instructions under "recipe" key and final_output tool from recipe.response.
            self.apply_recipe(&agent, &recipe_to_apply).await?;

            // Persist the raw recipe to session row.
            self.session_manager
                .update(session_id)
                .recipe(Some(raw_recipe))
                .apply()
                .await
                .internal_err_ctx("Failed to persist recipe to session")?;
        } else if session.recipe.is_some() {
            self.apply_session_recipe(&agent, &session).await?;
        }

        if let Some(v) = req.rolling_memory {
            let session = self
                .session_manager
                .get_session(session_id, false)
                .await
                .internal_err_ctx("Failed to re-read session")?;
            let mut data = session.extension_data;
            crate::context_mgmt::rolling::write_settings(&mut data, v.map(Into::into));
            self.session_manager
                .update(session_id)
                .extension_data(data)
                .apply()
                .await
                .internal_err_ctx("Failed to persist rolling memory settings to session")?;
        }

        Ok(EmptyResponse {})
    }

    pub(super) async fn on_delete_session(
        &self,
        req: DeleteSessionRequest,
    ) -> Result<DeleteSessionResponse, agent_client_protocol::Error> {
        let session_id = req.session_id.0.to_string();
        self.active_runs.cancel_agent_run(&session_id);
        self.live_voice.stop_session_interaction(&session_id).await;
        self.session_manager
            .delete_session(&session_id)
            .await
            .internal_err()?;
        self.sessions.lock().await.remove(&session_id);
        self.agent_manager
            .remove_session_if_loaded(&session_id)
            .await
            .internal_err_ctx("Failed to remove in-memory agent")?;
        Ok(DeleteSessionResponse::new())
    }

    pub(super) async fn on_export_session(
        &self,
        req: ExportSessionRequest,
    ) -> Result<ExportSessionResponse, agent_client_protocol::Error> {
        let data = match req.format {
            SessionExportFormat::Json => self.session_manager.export_session(&req.session_id).await,
            SessionExportFormat::Markdown => {
                self.session_manager
                    .export_session_markdown(&req.session_id)
                    .await
            }
        }
        .internal_err()?;
        Ok(ExportSessionResponse { data })
    }

    pub(super) async fn on_import_session(
        &self,
        req: ImportSessionRequest,
    ) -> Result<ImportSessionResponse, agent_client_protocol::Error> {
        let session = self
            .session_manager
            .import_session(&req.input, None)
            .await
            .internal_err()?;

        let msg_count = session.message_count as u64;

        Ok(ImportSessionResponse {
            session_id: session.id,
            title: Some(session.name),
            updated_at: Some(session.updated_at.to_rfc3339()),
            message_count: msg_count,
        })
    }

    pub(super) async fn on_get_session_info(
        &self,
        req: GetSessionInfoRequest,
    ) -> Result<GetSessionInfoResponse, agent_client_protocol::Error> {
        let session_id = req.session_id.trim();
        if session_id.is_empty() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("sessionId cannot be empty")
            );
        }

        let session = self
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|_| {
                agent_client_protocol::Error::resource_not_found(Some(session_id.to_string()))
                    .data(format!("Session not found: {}", session_id))
            })?;

        Ok(GetSessionInfoResponse {
            session: build_session_info(session),
        })
    }

    pub(super) async fn on_truncate_session_conversation(
        &self,
        req: TruncateSessionConversationRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        let session_id = req.session_id.trim();
        if session_id.is_empty() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("sessionId cannot be empty")
            );
        }

        self.session_manager
            .truncate_conversation(session_id, req.truncate_from)
            .await
            .internal_err()?;
        Ok(EmptyResponse {})
    }

    pub(super) async fn on_update_session_project(
        &self,
        req: UpdateSessionProjectRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        let session_id = &req.session_id;
        let session_not_found = || {
            agent_client_protocol::Error::resource_not_found(Some(session_id.to_string()))
                .data(format!("Session not found: {session_id}"))
        };
        let updated = self
            .session_manager
            .update_project_for_session_types(
                session_id,
                req.project_id,
                &ACP_VISIBLE_SESSION_TYPES,
            )
            .await
            .internal_err()?;
        if !updated {
            return Err(session_not_found());
        }
        Ok(EmptyResponse {})
    }

    pub(super) async fn on_rename_session(
        &self,
        req: RenameSessionRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        self.session_manager
            .update(&req.session_id)
            .user_provided_name(req.title)
            .apply()
            .await
            .map_err(|e| agent_client_protocol::Error::internal_error().data(e.to_string()))?;
        Ok(EmptyResponse {})
    }

    pub(super) async fn on_archive_session(
        &self,
        req: ArchiveSessionRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        self.session_manager
            .update(&req.session_id)
            .archived_at(Some(chrono::Utc::now()))
            .apply()
            .await
            .internal_err()?;
        self.sessions.lock().await.remove(&req.session_id);
        self.agent_manager
            .remove_session_if_loaded(&req.session_id)
            .await
            .internal_err_ctx("Failed to remove in-memory agent")?;
        Ok(EmptyResponse {})
    }

    pub(super) async fn on_unarchive_session(
        &self,
        req: UnarchiveSessionRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        self.session_manager
            .update(&req.session_id)
            .archived_at(None)
            .apply()
            .await
            .internal_err()?;
        Ok(EmptyResponse {})
    }
}
