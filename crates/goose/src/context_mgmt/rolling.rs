use crate::session::ExtensionData;
use goose_sdk_types::custom_requests::RollingMemoryDto;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// extension_data key (name, version) of the per-session setting. Value: object, or JSON null = off.
pub const SETTINGS_KEY: (&str, &str) = ("rolling_memory", "v0");
/// extension_data key of the runtime state (calibration ratio, failure back-off).
pub const STATE_KEY: (&str, &str) = ("rolling_memory_state", "v0");
pub const ENV_BUDGET: &str = "GOOSE_ROLLING_MEMORY_BUDGET";
pub const ENV_TAIL: &str = "GOOSE_ROLLING_MEMORY_TAIL";
pub const DEFAULT_RATIO: f64 = 1.45;
pub const RATIO_MIN: f64 = 1.0;
pub const RATIO_MAX: f64 = 2.5;
pub const MEMORY_MAX_TOKENS: i32 = 16_000;
/// Claude tokens reserved for the memory pair when checking "gets under".
pub const MEMORY_ALLOWANCE: f64 = 6_000.0;
/// A memory above this (Claude tokens, estimated as r x o200k) is logged, never failed.
pub const MEMORY_SIZE_WARN: f64 = 6_000.0;
pub const GETS_UNDER_FRACTION: f64 = 0.8;
pub const MIN_PROGRESS_TOKENS: f64 = 30_000.0;
pub const MIN_PROGRESS_FRACTION: f64 = 0.2;
pub const HARD_CAP_FACTOR: f64 = 2.0;
pub const BACKOFF_AFTER_FAILURES: u32 = 2;
pub const MEMORY_REQUEST_TEXT: &str = "Write your memory of the history in the system prompt.";
/// Key under MessageMetadata.operations marking the rows a fold inserted.
pub const OPERATION_KEY: &str = "rolling_memory";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RollingMemorySettings {
    pub budget_tokens: u32,
    pub tail_tokens: u32,
}

impl RollingMemorySettings {
    /// 0 < tail < budget. Returns a message for invalid_params otherwise.
    pub fn validate(&self) -> Result<(), String> {
        if self.tail_tokens == 0 {
            return Err("tailTokens must be greater than 0".to_string());
        }
        if self.tail_tokens >= self.budget_tokens {
            return Err("tailTokens must be less than budgetTokens".to_string());
        }
        Ok(())
    }
}

impl From<RollingMemoryDto> for RollingMemorySettings {
    fn from(dto: RollingMemoryDto) -> Self {
        Self {
            budget_tokens: dto.budget_tokens,
            tail_tokens: dto.tail_tokens,
        }
    }
}

impl From<RollingMemorySettings> for RollingMemoryDto {
    fn from(settings: RollingMemorySettings) -> Self {
        Self {
            budget_tokens: settings.budget_tokens,
            tail_tokens: settings.tail_tokens,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RollingState {
    pub ratio: f64,                       // default DEFAULT_RATIO
    pub consecutive_failures: u32,        // default 0
    pub failed_at_estimate: Option<u64>,  // default None
}

impl Default for RollingState {
    fn default() -> Self {
        Self {
            ratio: DEFAULT_RATIO,
            consecutive_failures: 0,
            failed_at_estimate: None,
        }
    }
}

/// Tri-state read of SETTINGS_KEY:
/// - key present, object  -> Some(settings) (invalid object -> None + warn)
/// - key present, null    -> None (explicitly off; env is NOT consulted)
/// - key absent           -> settings_from_env(env)
pub fn resolve_settings(
    data: &ExtensionData,
    env: impl Fn(&str) -> Option<String>,
) -> Option<RollingMemorySettings> {
    match data.get_extension_state(SETTINGS_KEY.0, SETTINGS_KEY.1) {
        Some(Value::Null) => None,
        Some(val) => match serde_json::from_value::<RollingMemorySettings>(val.clone()) {
            Ok(settings) => match settings.validate() {
                Ok(()) => Some(settings),
                Err(err) => {
                    tracing::warn!("Invalid rolling memory settings in extension_data: {err}");
                    None
                }
            },
            Err(err) => {
                tracing::warn!(
                    "Failed to deserialize rolling memory settings from extension_data: {err}"
                );
                None
            }
        },
        None => settings_from_env(env),
    }
}

/// Both ENV_BUDGET and ENV_TAIL set and parse as u32 and validate -> Some; otherwise None
/// (warn when exactly one is set or a value fails to parse/validate). Read std::env, never Config::global.
fn settings_from_env(env: impl Fn(&str) -> Option<String>) -> Option<RollingMemorySettings> {
    let budget_str = env(ENV_BUDGET);
    let tail_str = env(ENV_TAIL);

    match (budget_str, tail_str) {
        (None, None) => None,
        (Some(_), None) => {
            tracing::warn!(
                "{ENV_BUDGET} is set but {ENV_TAIL} is not; rolling memory remains disabled"
            );
            None
        }
        (None, Some(_)) => {
            tracing::warn!(
                "{ENV_TAIL} is set but {ENV_BUDGET} is not; rolling memory remains disabled"
            );
            None
        }
        (Some(b_str), Some(t_str)) => {
            let budget = match b_str.trim().parse::<u32>() {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Failed to parse {ENV_BUDGET} ('{b_str}') as u32: {e}");
                    return None;
                }
            };
            let tail = match t_str.trim().parse::<u32>() {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Failed to parse {ENV_TAIL} ('{t_str}') as u32: {e}");
                    return None;
                }
            };
            let settings = RollingMemorySettings {
                budget_tokens: budget,
                tail_tokens: tail,
            };
            match settings.validate() {
                Ok(()) => Some(settings),
                Err(err) => {
                    tracing::warn!("Invalid rolling memory env settings: {err}");
                    None
                }
            }
        }
    }
}

pub fn read_state(data: &ExtensionData) -> RollingState {
    match data.get_extension_state(STATE_KEY.0, STATE_KEY.1) {
        Some(val) => serde_json::from_value(val.clone()).unwrap_or_default(),
        None => RollingState::default(),
    }
}

pub fn write_state(data: &mut ExtensionData, state: &RollingState) {
    let val = serde_json::to_value(state).expect("serialize RollingState");
    data.set_extension_state(STATE_KEY.0, STATE_KEY.1, val);
}

/// Write the setting: Some -> object, None -> JSON null.
pub fn write_settings(data: &mut ExtensionData, settings: Option<RollingMemorySettings>) {
    let val = match settings {
        Some(s) => serde_json::to_value(s).expect("serialize RollingMemorySettings"),
        None => Value::Null,
    };
    data.set_extension_state(SETTINGS_KEY.0, SETTINGS_KEY.1, val);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use goose_sdk_types::custom_requests::SetSessionSettingsRequest;

    fn make_env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn rolling_resolve_absent_no_env() {
        let data = ExtensionData::default();
        let env = make_env(&[]);
        assert_eq!(resolve_settings(&data, env), None);
    }

    #[test]
    fn rolling_resolve_absent_both_env() {
        let data = ExtensionData::default();
        let env = make_env(&[(ENV_BUDGET, "150000"), (ENV_TAIL, "60000")]);
        assert_eq!(
            resolve_settings(&data, env),
            Some(RollingMemorySettings {
                budget_tokens: 150_000,
                tail_tokens: 60_000,
            })
        );
    }

    #[test]
    fn rolling_resolve_absent_only_one_env() {
        let data = ExtensionData::default();
        let env = make_env(&[(ENV_BUDGET, "150000")]);
        assert_eq!(resolve_settings(&data, env), None);

        let env2 = make_env(&[(ENV_TAIL, "60000")]);
        assert_eq!(resolve_settings(&data, env2), None);
    }

    #[test]
    fn rolling_resolve_null_both_env() {
        let mut data = ExtensionData::default();
        data.set_extension_state(SETTINGS_KEY.0, SETTINGS_KEY.1, Value::Null);
        let env = make_env(&[(ENV_BUDGET, "150000"), (ENV_TAIL, "60000")]);
        // Explicit null in extension_data wins; env is not consulted
        assert_eq!(resolve_settings(&data, env), None);
    }

    #[test]
    fn rolling_resolve_object_key() {
        let mut data = ExtensionData::default();
        let settings = RollingMemorySettings {
            budget_tokens: 120_000,
            tail_tokens: 40_000,
        };
        data.set_extension_state(
            SETTINGS_KEY.0,
            SETTINGS_KEY.1,
            serde_json::to_value(&settings).unwrap(),
        );
        let env = make_env(&[]);
        assert_eq!(resolve_settings(&data, env), Some(settings));
    }

    #[test]
    fn rolling_resolve_invalid_object() {
        let mut data = ExtensionData::default();
        let invalid = serde_json::json!({
            "budgetTokens": 50000,
            "tailTokens": 60000
        });
        data.set_extension_state(SETTINGS_KEY.0, SETTINGS_KEY.1, invalid);
        let env = make_env(&[]);
        assert_eq!(resolve_settings(&data, env), None);
    }

    #[test]
    fn rolling_state_default_absent() {
        let data = ExtensionData::default();
        assert_eq!(read_state(&data), RollingState::default());
    }

    #[test]
    fn rolling_state_roundtrip() {
        let mut data = ExtensionData::default();
        let state = RollingState {
            ratio: 1.62,
            consecutive_failures: 1,
            failed_at_estimate: Some(85_000),
        };
        write_state(&mut data, &state);
        assert_eq!(read_state(&data), state);
    }

    #[test]
    fn rolling_settings_write_null() {
        let mut data = ExtensionData::default();
        write_settings(&mut data, None);
        assert_eq!(
            data.get_extension_state(SETTINGS_KEY.0, SETTINGS_KEY.1),
            Some(&Value::Null)
        );
    }

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
            r#"{"sessionId":"abc","rollingMemory":{"budgetTokens":1000,"tailTokens":500}}"#,
        )
        .unwrap();
        assert_eq!(
            req.rolling_memory,
            Some(Some(RollingMemoryDto {
                budget_tokens: 1000,
                tail_tokens: 500,
            }))
        );
    }
}
