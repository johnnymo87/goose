// Modified by Jonathan Mohrbacher, 2026: SetSessionSettingsRequest pins a session's cache TTL, recipe and rolling memory.
use agent_client_protocol::JsonRpcRequest;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{EmptyResponse, RecipeDto};

/// Distinguishes an absent field (None) from an explicit null (Some(None)).
fn double_option<'de, D, T>(de: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    Option::<T>::deserialize(de).map(Some)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RollingMemoryDto {
    pub budget_tokens: u32,
    pub tail_tokens: u32,
}

impl RollingMemoryDto {
    /// 0 < tail < budget, budget >= 10_000, tail >= 1_000. Returns a message for invalid_params otherwise.
    pub fn validate(&self) -> Result<(), String> {
        if self.tail_tokens < 1_000 {
            return Err("tailTokens must be at least 1000".to_string());
        }
        if self.budget_tokens < 10_000 {
            return Err("budgetTokens must be at least 10000".to_string());
        }
        if self.tail_tokens >= self.budget_tokens {
            return Err("tailTokens must be less than budgetTokens".to_string());
        }
        Ok(())
    }
}

/// Pin a session's prompt-cache TTL, recipe and rolling memory, persisted on the session row.
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
    /// Rolling memory for this session. Absent: unchanged. null: off.
    #[serde(
        default,
        deserialize_with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "Option<RollingMemoryDto>")]
    pub rolling_memory: Option<Option<RollingMemoryDto>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_request_rolling_memory_absent() {
        let req: SetSessionSettingsRequest =
            serde_json::from_str(r#"{"sessionId":"abc"}"#).unwrap();
        assert_eq!(req.rolling_memory, None);
    }

    #[test]
    fn settings_request_rolling_memory_null() {
        let req: SetSessionSettingsRequest =
            serde_json::from_str(r#"{"sessionId":"abc","rollingMemory":null}"#).unwrap();
        assert_eq!(req.rolling_memory, Some(None));
    }

    #[test]
    fn settings_request_rolling_memory_object() {
        let req: SetSessionSettingsRequest = serde_json::from_str(
            r#"{"sessionId":"abc","rollingMemory":{"budgetTokens":10000,"tailTokens":1000}}"#,
        )
        .unwrap();
        assert_eq!(
            req.rolling_memory,
            Some(Some(RollingMemoryDto {
                budget_tokens: 10000,
                tail_tokens: 1000,
            }))
        );
    }

    #[test]
    fn rolling_memory_dto_validation() {
        assert!(RollingMemoryDto {
            budget_tokens: 10_000,
            tail_tokens: 1_000
        }
        .validate()
        .is_ok());
        assert!(RollingMemoryDto {
            budget_tokens: 9_999,
            tail_tokens: 1_000
        }
        .validate()
        .is_err());
        assert!(RollingMemoryDto {
            budget_tokens: 10_000,
            tail_tokens: 999
        }
        .validate()
        .is_err());
        assert!(RollingMemoryDto {
            budget_tokens: 10_000,
            tail_tokens: 10_000
        }
        .validate()
        .is_err());
        assert!(RollingMemoryDto {
            budget_tokens: 10_000,
            tail_tokens: 11_000
        }
        .validate()
        .is_err());
    }
}
