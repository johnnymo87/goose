// Modified by Jonathan Mohrbacher, 2026: SetSessionSettingsRequest pins a session's cache TTL and recipe.
use agent_client_protocol::JsonRpcRequest;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{EmptyResponse, RecipeDto};

/// Pin a session's prompt-cache TTL and recipe, persisted on the session row.
///
/// The recipe is stored on the session and read every turn: its `title`,
/// `instructions` and `response` take effect. Its `settings`, `extensions`,
/// `prompt` and `parameters` are ignored, `retry` and `sub_recipes` are
/// dropped, and the session keeps its model.
#[derive(Debug, Default, Clone, Serialize, Deserialize, JsonSchema, JsonRpcRequest)]
#[request(
    method = "_goose/unstable/session/settings/set",
    response = EmptyResponse
)]
#[serde(rename_all = "camelCase")]
pub struct SetSessionSettingsRequest {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_ttl: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<RecipeDto>,
}
