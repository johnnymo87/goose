// Modified by Jonathan Mohrbacher, 2026: _goose/unstable/session/settings/set pins a session's cache TTL, recipe and rolling memory; session/prompt _meta.goose.maxTurns bounds the run.
use agent_client_protocol::schema::v1::Meta;

use super::{AcpConnection, ResultExt};
use crate::acp::custom_requests::{EmptyResponse, SetSessionSettingsRequest};
use crate::agents::final_output_tool::FinalOutputTool;
use crate::recipe::build_recipe::{build_recipe_from_template, RecipeError};
use crate::recipe::local_recipes::get_recipe_library_dir;
use crate::recipe::validate_recipe::validate_recipe_template_from_content;
use crate::recipe::Recipe;
use crate::session::{Session, SessionType};

const CACHE_TTLS: [&str; 2] = ["5m", "1h"];

fn invalid_recipe(detail: impl std::fmt::Display) -> agent_client_protocol::Error {
    let detail = detail.to_string();
    let detail = detail.strip_prefix("recipe: ").unwrap_or(&detail);
    agent_client_protocol::Error::invalid_params().data(format!("invalid recipe: {detail}"))
}

fn normalized_cache_ttl(ttl: &str) -> Result<String, agent_client_protocol::Error> {
    let ttl = ttl.trim().to_lowercase();
    if CACHE_TTLS.contains(&ttl.as_str()) {
        Ok(ttl)
    } else {
        Err(agent_client_protocol::Error::invalid_params()
            .data("invalid cacheTtl: must be '5m' or '1h'"))
    }
}

/// The recipe as the session row stores it: rendered with the session's
/// parameter values, except for scheduled sessions, which keep the template.
/// `retry` and `sub_recipes` are dropped: the state machine reads the stored
/// recipe every turn, and a retry block would let a failed check truncate a
/// long-lived session back to its kickoff message.
fn recipe_to_store(
    mut raw: Recipe,
    session: &Session,
) -> Result<Recipe, agent_client_protocol::Error> {
    raw.retry = None;
    raw.sub_recipes = None;
    if raw.check_for_security_warnings() {
        return Err(invalid_recipe(
            "contains hidden characters that could be malicious",
        ));
    }
    let yaml = raw.to_yaml().map_err(invalid_recipe)?;
    validate_recipe_template_from_content(&yaml, None).map_err(invalid_recipe)?;

    let recipe = if session.session_type == SessionType::Scheduled {
        raw
    } else {
        let values: Vec<(String, String)> = session
            .user_recipe_values
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect();
        match build_recipe_from_template(
            yaml,
            &get_recipe_library_dir(true),
            values,
            None::<fn(&str, &str) -> Result<String, anyhow::Error>>,
        ) {
            Ok(rendered) => rendered,
            Err(RecipeError::MissingParams { .. }) => {
                return Err(invalid_recipe("missing required parameters"))
            }
            Err(e) => return Err(invalid_recipe(e)),
        }
    };

    if let Some(response) = recipe.response.clone() {
        FinalOutputTool::try_new(response).map_err(invalid_recipe)?;
    }
    Ok(recipe)
}

/// `_meta.goose.maxTurns`: absent or null means the configured default; any
/// value other than a positive 32-bit integer is rejected.
pub(super) fn max_turns_from_meta(
    meta: Option<&Meta>,
) -> Result<Option<u32>, agent_client_protocol::Error> {
    let Some(value) = meta
        .and_then(|meta| meta.get("goose"))
        .and_then(|goose| goose.get("maxTurns"))
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    value
        .as_u64()
        .filter(|n| *n > 0)
        .and_then(|n| u32::try_from(n).ok())
        .map(Some)
        .ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data("_meta.goose.maxTurns must be a positive integer")
        })
}

impl AcpConnection {
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
        let cache_ttl = req
            .cache_ttl
            .as_deref()
            .map(normalized_cache_ttl)
            .transpose()?;

        if let Some(Some(dto)) = &req.rolling_memory {
            if let Err(msg) = dto.validate() {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("invalid rollingMemory: {msg}")));
            }
        }

        let mut session = self
            .services
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|_| {
                agent_client_protocol::Error::resource_not_found(Some(session_id.to_string()))
                    .data(format!("Session not found: {}", session_id))
            })?;

        let recipe = req
            .recipe
            .map(|dto| {
                let raw = Recipe::try_from(dto).map_err(invalid_recipe)?;
                recipe_to_store(raw, &session)
            })
            .transpose()?;

        self.open_session(session_id).await?;

        if let Some(ttl) = cache_ttl {
            let model_config = self
                .services
                .model_config_for_session(session_id)
                .await
                .internal_err_ctx("Failed to resolve model config")?;
            if model_config.cache_ttl().as_deref() != Some(ttl.as_str()) {
                let provider_name = self
                    .services
                    .provider(session_id)
                    .await
                    .internal_err_ctx("Failed to get provider")?
                    .get_name()
                    .to_string();
                self.services
                    .switch_provider(
                        session_id,
                        &provider_name,
                        model_config.with_cache_ttl(&ttl),
                    )
                    .await
                    .internal_err_ctx("Failed to pin cache TTL")?;
            }
        }

        if let Some(v) = req.rolling_memory {
            let session = self
                .services
                .session_manager
                .get_session(session_id, false)
                .await
                .internal_err_ctx("Failed to re-read session")?;
            let mut data = session.extension_data;
            crate::context_mgmt::rolling::write_settings(&mut data, v.map(Into::into));
            self.services
                .session_manager
                .update(session_id)
                .extension_data(data)
                .apply()
                .await
                .internal_err_ctx("Failed to persist rolling memory settings to session")?;
        }

        let Some(recipe) = recipe else {
            self.render_stored_recipe_template(&mut session).await?;
            return Ok(EmptyResponse {});
        };
        self.services
            .session_manager
            .update(session_id)
            .recipe(Some(recipe))
            .apply()
            .await
            .internal_err_ctx("Failed to persist recipe to session")?;

        Ok(EmptyResponse {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::ErrorCode;

    fn meta(value: serde_json::Value) -> Meta {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn max_turns_from_meta_absent_or_null_is_default() {
        assert_eq!(max_turns_from_meta(None).unwrap(), None);
        for value in [
            serde_json::json!({}),
            serde_json::json!({ "goose": {} }),
            serde_json::json!({ "goose": { "maxTurns": null } }),
        ] {
            assert_eq!(max_turns_from_meta(Some(&meta(value))).unwrap(), None);
        }
    }

    #[test]
    fn max_turns_from_meta_accepts_positive_integer() {
        let value = meta(serde_json::json!({ "goose": { "maxTurns": 4 } }));
        assert_eq!(max_turns_from_meta(Some(&value)).unwrap(), Some(4));
    }

    #[test]
    fn max_turns_from_meta_rejects_invalid_values() {
        for bad in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(2.5),
            serde_json::json!("two"),
            serde_json::json!(u64::from(u32::MAX) + 1),
        ] {
            let value = meta(serde_json::json!({ "goose": { "maxTurns": bad } }));
            assert_eq!(
                max_turns_from_meta(Some(&value)).unwrap_err().code,
                ErrorCode::InvalidParams,
                "maxTurns {bad}"
            );
        }
    }

    #[test]
    fn cache_ttl_is_normalized_and_validated() {
        assert_eq!(normalized_cache_ttl(" 1H ").unwrap(), "1h");
        assert_eq!(normalized_cache_ttl("5m").unwrap(), "5m");
        let err = normalized_cache_ttl("2h").unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidParams);
        assert!(err.data.unwrap().to_string().contains("invalid cacheTtl: "));
    }
}
