// Modified by Jonathan Mohrbacher, 2026: rolling memory pure planning; memory call; materialize fold.
use crate::agents::final_output_tool::FINAL_OUTPUT_CONTINUATION_MESSAGE;
use crate::conversation::message::{Message, MessageContentBlock, MessageMetadata};
use crate::conversation::Conversation;
use crate::session::ExtensionData;
use goose_providers::base::Provider;
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use goose_sdk_types::custom_requests::RollingMemoryDto;
use rmcp::model::Role;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

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
        RollingMemoryDto::from(*self).validate()
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
        Some(val) => match serde_json::from_value(val.clone()) {
            Ok(state) => state,
            Err(err) => {
                tracing::warn!(
                    "Failed to deserialize rolling memory state from extension_data: {err}"
                );
                RollingState::default()
            }
        },
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

/// Role user, agent-visible AND user-visible (this excludes every agent-only nudge: goal/grind
/// kickoffs and nudges, stop-hook denials, the prompt copy compact_messages preserves, and a fold's
/// recall row), not turn context, not a steer, has text, has no tool request / tool response
/// content (after agent_visible_content() projection), and is not the final-output nudge
/// (text == crate::agents::final_output_tool::FINAL_OUTPUT_CONTINUATION_MESSAGE).
pub fn is_genuine_prompt(msg: &Message) -> bool {
    if msg.role != Role::User {
        return false;
    }
    if !msg.is_agent_visible() || !msg.is_user_visible() {
        return false;
    }
    if msg.is_turn_context() || msg.metadata.steer {
        return false;
    }
    let projected = msg.agent_visible_content();
    if projected.is_tool_call() || projected.is_tool_response() {
        return false;
    }
    let has_text = projected
        .content
        .iter()
        .any(|c| matches!(c, MessageContentBlock::Text(t) if !t.text.trim().is_empty()));
    if !has_text {
        return false;
    }
    let text = projected.as_concat_text();
    if text == FINAL_OUTPUT_CONTINUATION_MESSAGE {
        return false;
    }
    true
}

/// After a fold, recovery or /compact, update_session_metrics(.., Some(retained)) stores
/// Usage::new(Some(retained), None, Some(retained)): output and both cache fields are None.
pub fn is_retained(usage: &Usage) -> bool {
    usage.output_tokens.is_none()
        && usage.cache_read_input_tokens.is_none()
        && usage.cache_write_input_tokens.is_none()
}

/// Index of the last message carrying metadata.usage, but ONLY if that usage is the call now in
/// session.usage: its input_tokens, output_tokens and total_tokens all equal the session's. Usage is
/// recorded for every provider call (agent.rs ~2932) but attached to a message only when the call
/// left an assistant message (~3643); empty responses and retries record usage with no anchor, and
/// after a fold the newest anchor may predate the session's usage. A mismatch -> None.
pub fn anchor_index(messages: &[Message], session_usage: &Usage) -> Option<usize> {
    let last_usage_idx = messages
        .iter()
        .rposition(|m| m.metadata.usage.is_some())?;

    let msg_usage = messages[last_usage_idx].metadata.usage.as_ref()?;
    if msg_usage.input_tokens == session_usage.input_tokens
        && msg_usage.output_tokens == session_usage.output_tokens
        && msg_usage.total_tokens == session_usage.total_tokens
    {
        Some(last_usage_idx)
    } else {
        None
    }
}

/// Design §2 step 1, with one deliberate deviation (oracle review):
/// - total None                -> ratio * sum(counts)          (no usage yet)
/// - retained(usage)           -> ratio * sum(counts)          (deviation: the spec says "total as-is";
///   the retained value WAS this o200k sum at fold time, so recomputing it cannot double-count and,
///   unlike a frozen value, keeps growing if no later call overwrites it)
/// - anchor None               -> ratio * sum(counts)
/// - otherwise                 -> total + ratio * sum(counts[anchor+1..])
/// The fallbacks omit system + tools; Task 5 adds ratio * base_o200k to them only on the slow path.
pub fn estimate(usage: &Usage, messages: &[Message], counts: &[usize], ratio: f64) -> u64 {
    let fallback = || -> u64 {
        let sum_all: usize = counts.iter().sum();
        ((sum_all as f64) * ratio).round() as u64
    };

    let total = match usage.total_tokens {
        Some(t) => t,
        None => return fallback(),
    };

    if is_retained(usage) {
        return fallback();
    }

    match anchor_index(messages, usage) {
        Some(anchor) => {
            let sum_after: usize = counts.get(anchor + 1..).map_or(0, |s| s.iter().sum());
            ((total.max(0) as f64) + ratio * (sum_after as f64)).round() as u64
        }
        None => fallback(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutPlan {
    /// Index (into the full message vec) of the first tail message: a genuine prompt.
    pub cut: usize,
    /// Agent-visible indices < cut. All become agent-invisible.
    pub folded: Vec<usize>,
    /// Index of the previous memory's assistant row, if one is in `folded`.
    pub previous_memory: Option<usize>,
    /// o200k of folded messages EXCLUDING the previous memory pair and turn-context messages
    /// (the "new" material; used for the progress rule).
    pub new_material_o200k: usize,
    /// o200k of agent-visible messages >= cut.
    pub tail_o200k: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CutError {
    #[error("tool call {0} is split across the cut")]
    SplitToolPair(String),
}

/// Design §3. Walk agent-visible messages back from the end, accumulating ratio*counts[i], until
/// the sum >= tail_tokens (or the start is reached). From that position move further back to the
/// nearest genuine prompt (at or before it). Ok(None) when: there is no such prompt, or nothing
/// agent-visible lies before it apart from a previous memory pair (nothing new to fold).
/// Then the pairing check: collect ToolRequest ids and ToolResponse ids on each side (all content
/// items of all agent-visible messages; parallel calls share a message). If a request id before
/// the cut has its response at/after it, or the reverse -> Err(SplitToolPair(id)).
pub fn plan_cut(
    messages: &[Message],
    counts: &[usize],
    ratio: f64,
    tail_tokens: u32,
) -> Result<Option<CutPlan>, CutError> {
    let mut accumulated = 0.0;
    let mut walk_pos = None;

    for (i, msg) in messages.iter().enumerate().rev() {
        if !msg.is_agent_visible() {
            continue;
        }
        accumulated += ratio * (counts.get(i).copied().unwrap_or(0) as f64);
        walk_pos = Some(i);
        if accumulated >= tail_tokens as f64 {
            break;
        }
    }

    let walk_pos = match walk_pos {
        Some(pos) => pos,
        None => return Ok(None),
    };

    let mut cut = None;
    for i in (0..=walk_pos).rev() {
        if is_genuine_prompt(&messages[i]) {
            cut = Some(i);
            break;
        }
    }

    let cut = match cut {
        Some(c) => c,
        None => return Ok(None),
    };

    let folded: Vec<usize> = (0..cut)
        .filter(|&i| messages[i].is_agent_visible())
        .collect();

    let mut previous_memory = None;
    let mut has_new_material = false;
    let mut new_material_o200k = 0;

    for &idx in &folded {
        let msg = &messages[idx];
        let is_prev_mem = msg.role == Role::Assistant && fold_part(msg) == Some("memory");
        let is_prev_recall = fold_part(msg) == Some("recall");

        if is_prev_mem {
            previous_memory = Some(idx);
        }

        if !is_prev_mem && !is_prev_recall && !msg.is_turn_context() {
            has_new_material = true;
            new_material_o200k += counts.get(idx).copied().unwrap_or(0);
        }
    }

    if !has_new_material {
        return Ok(None);
    }

    check_tool_pairing(messages, cut)?;

    let tail_o200k: usize = (cut..messages.len())
        .filter(|&i| messages[i].is_agent_visible())
        .map(|i| counts.get(i).copied().unwrap_or(0))
        .sum();

    Ok(Some(CutPlan {
        cut,
        folded,
        previous_memory,
        new_material_o200k,
        tail_o200k,
    }))
}

/// Covers pairs the cut would split, not pre-existing orphans.
fn check_tool_pairing(messages: &[Message], cut: usize) -> Result<(), CutError> {
    let mut responses_before = HashSet::new();
    let mut responses_after = HashSet::new();

    for msg in &messages[..cut] {
        if msg.is_agent_visible() {
            responses_before.extend(msg.get_tool_response_ids());
        }
    }

    for msg in &messages[cut..] {
        if msg.is_agent_visible() {
            responses_after.extend(msg.get_tool_response_ids());
        }
    }

    // If a request id before the cut has its response at/after it
    for msg in &messages[..cut] {
        if msg.is_agent_visible() {
            for req_id in msg.get_tool_request_ids() {
                if responses_after.contains(req_id) {
                    return Err(CutError::SplitToolPair(req_id.to_string()));
                }
            }
        }
    }

    // Or the reverse: request at/after cut has its response before cut
    for msg in &messages[cut..] {
        if msg.is_agent_visible() {
            for req_id in msg.get_tool_request_ids() {
                if responses_before.contains(req_id) {
                    return Err(CutError::SplitToolPair(req_id.to_string()));
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Fold,
    Skip(&'static str),
}

/// Design §2 steps 2-3. `base_o200k` = o200k of system prompt + tools (count_chat_tokens(system, &[], tools)).
/// - estimate <= budget                                   -> Skip("under budget")
/// - plan None                                            -> Skip("nothing to fold")
/// - hard = estimate > HARD_CAP_FACTOR * budget; if hard  -> Fold
/// - planned = ratio*(base_o200k + tail_o200k) + MEMORY_ALLOWANCE
///   planned > GETS_UNDER_FRACTION * budget               -> Skip("would not get under")
/// - ratio*new_material_o200k < max(MIN_PROGRESS_TOKENS, MIN_PROGRESS_FRACTION*budget)
///                                                        -> Skip("too little progress")
/// - else Fold
pub fn decide(
    estimate: u64,
    plan: Option<&CutPlan>,
    base_o200k: usize,
    ratio: f64,
    settings: &RollingMemorySettings,
) -> Decision {
    if estimate <= settings.budget_tokens as u64 {
        return Decision::Skip("under budget");
    }

    let plan = match plan {
        Some(p) => p,
        None => return Decision::Skip("nothing to fold"),
    };

    let hard = (estimate as f64) > HARD_CAP_FACTOR * (settings.budget_tokens as f64);
    if hard {
        return Decision::Fold;
    }

    let planned = ratio * ((base_o200k + plan.tail_o200k) as f64) + MEMORY_ALLOWANCE;
    if planned > GETS_UNDER_FRACTION * (settings.budget_tokens as f64) {
        return Decision::Skip("would not get under");
    }

    let min_progress =
        MIN_PROGRESS_TOKENS.max(MIN_PROGRESS_FRACTION * (settings.budget_tokens as f64));
    if ratio * (plan.new_material_o200k as f64) < min_progress {
        return Decision::Skip("too little progress");
    }

    Decision::Fold
}

/// Design §6. true = skip this turn without calling the model:
/// consecutive_failures >= BACKOFF_AFTER_FAILURES && estimate <= HARD_CAP_FACTOR*budget
///   && estimate < failed_at_estimate.unwrap_or(0) + tail_tokens/2
pub fn backed_off(state: &RollingState, estimate: u64, settings: &RollingMemorySettings) -> bool {
    state.consecutive_failures >= BACKOFF_AFTER_FAILURES
        && (estimate as f64) <= HARD_CAP_FACTOR * (settings.budget_tokens as f64)
        && estimate
            < state
                .failed_at_estimate
                .unwrap_or(0)
                .saturating_add(settings.tail_tokens as u64 / 2)
}

/// New ratio from a calibration sample: the anchor call's billed INPUT tokens (anchor
/// metadata.usage.input_tokens; includes cache read/write; excludes output and thinking) over the
/// o200k of that call's request (system + tools + agent-visible messages BEFORE the anchor, anchor
/// excluded). None if not usable (o200k == 0 or input <= 0). Clamped to [RATIO_MIN, RATIO_MAX].
pub fn calibrated_ratio(anchor_input_tokens: i32, request_o200k: usize) -> Option<f64> {
    if request_o200k == 0 || anchor_input_tokens <= 0 {
        return None;
    }
    let raw = (anchor_input_tokens as f64) / (request_o200k as f64);
    Some(raw.clamp(RATIO_MIN, RATIO_MAX))
}

pub fn fold_part(msg: &Message) -> Option<&str> {
    msg.metadata
        .operations
        .as_ref()
        .and_then(|ops| ops.get(OPERATION_KEY))
        .and_then(|note| note.get("part"))
        .and_then(|val| val.as_str())
}

/// One history line: format_message_for_compacting(msg), with ONLY its leading "[user]: " or
/// "[assistant]: " replaced (strip_prefix, never replace-all: old summaries contain "[user]: "
/// lines) by "[user 2026-10-13 14:31 UTC]: " from msg.created (unix seconds, rendered in UTC with
/// chrono). Then neutralize: "</history" -> r"<\/history", "</previous-memory" -> r"<\/previous-memory".
pub fn render_history_line(msg: &Message) -> String {
    let formatted = crate::context_mgmt::format_message_for_compacting(msg);
    let role_str = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let old_prefix = format!("[{}]: ", role_str);
    let dt = chrono::DateTime::from_timestamp(msg.created, 0)
        .unwrap_or(chrono::DateTime::UNIX_EPOCH);
    let stamp = dt.format("%Y-%m-%d %H:%M UTC");
    let new_prefix = format!("[{} {}]: ", role_str, stamp);

    let stamped = match formatted.strip_prefix(&old_prefix) {
        Some(rest) => format!("{}{}", new_prefix, rest),
        None => formatted,
    };

    stamped
        .replace("</history", r"<\/history")
        .replace("</previous-memory", r"<\/previous-memory")
}

/// Messages to render: folded agent-visible rows, minus turn-context rows, minus the previous
/// memory pair (which goes in the previous_memory slot instead). With `drop_tool_responses`, drop
/// ToolResponse content items (and messages left empty) — used for the one refusal retry.
pub fn history_text(messages: &[Message], plan: &CutPlan, drop_tool_responses: bool) -> String {
    let mut lines = Vec::new();

    for &idx in &plan.folded {
        let Some(msg) = messages.get(idx) else {
            continue;
        };

        if !msg.is_agent_visible() || msg.is_turn_context() {
            continue;
        }

        let is_prev_mem = (msg.role == Role::Assistant && fold_part(msg) == Some("memory"))
            || plan.previous_memory == Some(idx);
        let is_prev_recall = fold_part(msg) == Some("recall")
            || (plan.previous_memory == Some(idx + 1) && msg.role == Role::User);

        if is_prev_mem || is_prev_recall {
            continue;
        }

        if drop_tool_responses {
            let mut filtered_content = msg.content.clone();
            filtered_content.retain(|c| !matches!(c, MessageContentBlock::ToolResponse(_)));
            if filtered_content.is_empty() {
                continue;
            }
            let mut filtered_msg = msg.clone();
            filtered_msg.content = filtered_content;
            lines.push(render_history_line(&filtered_msg));
        } else {
            lines.push(render_history_line(msg));
        }
    }

    lines.join("\n")
}

#[derive(Debug)]
pub struct Memory {
    pub text: String,
    pub usage: ProviderUsage,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("memory call refused: {0}")]
    Refused(String),
    #[error("memory hit the output token limit")]
    OutputLimit(ProviderUsage),
    #[error("memory call stopped with {0}")]
    StopReason(String, ProviderUsage),
    #[error("memory was empty")]
    Empty(ProviderUsage),
    #[error(transparent)]
    Provider(ProviderError),
    #[error(transparent)]
    Template(anyhow::Error),
}

/// Design §4. Renders compaction_rolling.md via crate::prompt_template::render_template with
/// {previous_memory: Option<String>, messages: String}; one user message MEMORY_REQUEST_TEXT; no tools.
/// Calls crate::model_config::complete_one_shot with model_config.clone().with_max_tokens(Some(MEMORY_MAX_TOKENS))
/// (NOT with_default_max_tokens; check one_shot_model_config in model_config.rs keeps max_tokens).
/// Failure: Err(ProviderError::Refusal) -> Refused; response.metadata.output_token_limit_reached ->
/// OutputLimit; usage.finish_reasons Some(non-empty) and none == "end_turn" or "stop"
/// (ASCII case-insensitive; other providers report "stop") -> StopReason(joined);
/// text (Text content only, joined, trimmed) empty -> Empty. Absent finish_reasons + text = success.
/// Success above MEMORY_SIZE_WARN (ratio * o200k of the text) logs tracing::warn with the size; it is
/// NOT an error.
pub async fn write_memory(
    provider: &dyn Provider,
    model_config: &ModelConfig,
    session_id: &str,
    previous_memory: Option<&str>,
    history: &str,
    ratio: f64,
) -> Result<Memory, MemoryError> {
    let neutralized_previous_memory = previous_memory.map(|prev| {
        prev.replace("</history", r"<\/history")
            .replace("</previous-memory", r"<\/previous-memory")
    });

    #[derive(Serialize)]
    struct Context<'a> {
        previous_memory: Option<&'a str>,
        messages: &'a str,
    }

    let context = Context {
        previous_memory: neutralized_previous_memory.as_deref(),
        messages: history,
    };

    let system_prompt = crate::prompt_template::render_template("compaction_rolling.md", &context)
        .map_err(|e| MemoryError::Template(anyhow::anyhow!(e)))?;

    let memory_model_config = model_config
        .clone()
        .with_max_tokens(Some(MEMORY_MAX_TOKENS));

    let user_msg = Message::user().with_text(MEMORY_REQUEST_TEXT);
    let (response, usage) = match crate::model_config::complete_one_shot(
        provider,
        &memory_model_config,
        session_id,
        &system_prompt,
        &[user_msg],
        &[],
    )
    .await
    {
        Ok(res) => res,
        Err(ProviderError::Refusal { details, .. }) => {
            return Err(MemoryError::Refused(details));
        }
        Err(e) => {
            return Err(MemoryError::Provider(e));
        }
    };

    if response.metadata.output_token_limit_reached {
        return Err(MemoryError::OutputLimit(usage));
    }

    if let Some(ref reasons) = usage.finish_reasons {
        // "end_turn" (Anthropic) or "stop" (OpenAI and other providers), ASCII case-insensitive.
        let is_normal_completion = reasons.iter().any(|r| {
            r.eq_ignore_ascii_case("end_turn") || r.eq_ignore_ascii_case("stop")
        });
        if !reasons.is_empty() && !is_normal_completion {
            return Err(MemoryError::StopReason(reasons.join(", "), usage));
        }
    }

    let text = response.as_concat_text().trim().to_string();
    if text.is_empty() {
        return Err(MemoryError::Empty(usage));
    }

    let o200k_tokens = match crate::token_counter::create_token_counter().await {
        Ok(counter) => counter.count_tokens(&text),
        Err(_) => text.len() / 4,
    };
    let estimated_tokens = ratio * (o200k_tokens as f64);
    if estimated_tokens > MEMORY_SIZE_WARN {
        tracing::warn!(
            estimated_tokens,
            MEMORY_SIZE_WARN,
            "rolling memory size above warning threshold"
        );
    }

    Ok(Memory { text, usage })
}

/// Refusal policy (§6): call write_memory; on Refused only, rebuild history with
/// drop_tool_responses = true and call once more. Returns the last result. OutputLimit, StopReason
/// and Empty come from a completed call and carry its usage so the caller records it.
/// ProviderError::Refusal carries no usage (goose-provider-types errors.rs ~57), so a refused call
/// cannot be accounted for; that is a known gap, not something to work around.
pub async fn write_memory_with_retry(
    provider: &dyn Provider,
    model_config: &ModelConfig,
    session_id: &str,
    previous_memory: Option<&str>,
    messages: &[Message],
    plan: &CutPlan,
    ratio: f64,
) -> Result<Memory, MemoryError> {
    let history = history_text(messages, plan, false);
    match write_memory(
        provider,
        model_config,
        session_id,
        previous_memory,
        &history,
        ratio,
    )
    .await
    {
        Err(MemoryError::Refused(_)) => {
            let retry_history = history_text(messages, plan, true);
            write_memory(
                provider,
                model_config,
                session_id,
                previous_memory,
                &retry_history,
                ratio,
            )
            .await
        }
        other => other,
    }
}

/// Design §5. Returns the conversation to hand to SessionManager::save_compacted_conversation:
/// - every index in plan.folded, cloned with metadata.with_agent_invisible() (ids unchanged);
/// - a recall row: Message::user().with_text(format!("What do you remember from {from} to {to}?"))
///   with MessageMetadata::agent_only(), a generated id, operations note part "recall";
/// - a memory row: Message::assistant().with_text(memory) with agent_only(), generated id,
///   operations note part "memory";
/// - both rows created = messages[plan.cut].created - 1, recall pushed before memory.
/// `from`: the previous memory row's note "from" if plan.previous_memory is Some, else the UTC date of
///   the oldest message before the cut (agent-visible or not). `to`: UTC date of the newest folded row.
/// Err if any folded row has no id (save_compacted_conversation would reject it anyway; fail early
/// with a clear message).
pub fn fold_rows(
    messages: &[Message],
    plan: &CutPlan,
    memory: &str,
) -> anyhow::Result<Conversation> {
    if plan.folded.is_empty() {
        anyhow::bail!("nothing to fold");
    }

    let cut_msg = messages.get(plan.cut).ok_or_else(|| {
        anyhow::anyhow!(
            "cut index {} out of bounds for {} messages",
            plan.cut,
            messages.len()
        )
    })?;

    for &idx in &plan.folded {
        if idx >= plan.cut {
            anyhow::bail!(
                "folded message index {idx} is at or after cut {}",
                plan.cut
            );
        }
        let msg = messages.get(idx).ok_or_else(|| {
            anyhow::anyhow!(
                "folded message index {idx} out of bounds for {} messages",
                messages.len()
            )
        })?;
        if msg.id.is_none() {
            anyhow::bail!("folded message at index {idx} has no id");
        }
    }

    let from = match plan
        .previous_memory
        .and_then(|idx| messages.get(idx))
        .and_then(|m| m.metadata.operation_note(OPERATION_KEY, "from"))
        .and_then(|v| v.as_str())
    {
        Some(prev_from) => prev_from.to_string(),
        None => {
            let oldest = messages[..plan.cut]
                .iter()
                .min_by_key(|m| m.created)
                .unwrap_or(cut_msg);
            chrono::DateTime::from_timestamp(oldest.created, 0)
                .unwrap_or(chrono::DateTime::UNIX_EPOCH)
                .format("%Y-%m-%d")
                .to_string()
        }
    };

    let newest_folded = plan
        .folded
        .iter()
        .filter_map(|&idx| messages.get(idx))
        .max_by_key(|m| m.created)
        .unwrap_or(cut_msg);

    let to = chrono::DateTime::from_timestamp(newest_folded.created, 0)
        .unwrap_or(chrono::DateTime::UNIX_EPOCH)
        .format("%Y-%m-%d")
        .to_string();

    let pair_created = cut_msg.created.saturating_sub(1);

    let mut recall_metadata = MessageMetadata::agent_only();
    recall_metadata.set_operation_note(OPERATION_KEY, "part", Value::String("recall".to_string()));
    recall_metadata.set_operation_note(OPERATION_KEY, "from", Value::String(from.clone()));
    recall_metadata.set_operation_note(OPERATION_KEY, "to", Value::String(to.clone()));

    let mut recall_msg = Message::user()
        .with_text(format!("What do you remember from {from} to {to}?"))
        .with_metadata(recall_metadata)
        .with_generated_id();
    recall_msg.created = pair_created;

    let mut memory_metadata = MessageMetadata::agent_only();
    memory_metadata.set_operation_note(OPERATION_KEY, "part", Value::String("memory".to_string()));
    memory_metadata.set_operation_note(OPERATION_KEY, "from", Value::String(from));
    memory_metadata.set_operation_note(OPERATION_KEY, "to", Value::String(to));

    let mut memory_msg = Message::assistant()
        .with_text(memory)
        .with_metadata(memory_metadata)
        .with_generated_id();
    memory_msg.created = pair_created;

    let mut out_messages = Vec::with_capacity(plan.folded.len() + 2);
    for &idx in &plan.folded {
        let msg = &messages[idx];
        let mut cloned = msg.clone();
        cloned.metadata = cloned.metadata.with_agent_invisible();
        out_messages.push(cloned);
    }
    out_messages.push(recall_msg);
    out_messages.push(memory_msg);

    Ok(Conversation::new_unvalidated(out_messages))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GooseMode;
    use crate::session::{SessionManager, SessionType};
    use std::collections::HashMap;

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
    fn rolling_settings_validate_zero_tail_invalid() {
        let settings = RollingMemorySettings {
            budget_tokens: 100_000,
            tail_tokens: 0,
        };
        assert!(settings.validate().is_err());
    }

    #[test]
    fn rolling_resolve_unparsable_env() {
        let data = ExtensionData::default();
        let env_bad_budget = make_env(&[(ENV_BUDGET, "not-a-number"), (ENV_TAIL, "60000")]);
        assert_eq!(resolve_settings(&data, env_bad_budget), None);

        let env_bad_tail = make_env(&[(ENV_BUDGET, "150000"), (ENV_TAIL, "invalid")]);
        assert_eq!(resolve_settings(&data, env_bad_tail), None);
    }

    #[test]
    fn rolling_state_invalid_defaults() {
        let mut data = ExtensionData::default();
        data.set_extension_state(
            STATE_KEY.0,
            STATE_KEY.1,
            serde_json::json!({"ratio": "not-a-float"}),
        );
        assert_eq!(read_state(&data), RollingState::default());
    }

    use crate::conversation::message::{MessageMetadata, MessageUsage};
    use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock};

    fn make_memory_pair(recall_text: &str, memory_text: &str) -> (Message, Message) {
        let mut recall_msg = Message::user()
            .with_text(recall_text)
            .with_visibility(false, true);
        let mut recall_ops = std::collections::BTreeMap::new();
        let mut recall_map = serde_json::Map::new();
        recall_map.insert("part".to_string(), serde_json::json!("recall"));
        recall_ops.insert(OPERATION_KEY.to_string(), recall_map);
        recall_msg.metadata.operations = Some(Box::new(recall_ops));

        let mut memory_msg = Message::assistant()
            .with_text(memory_text)
            .with_visibility(false, true);
        let mut memory_ops = std::collections::BTreeMap::new();
        let mut memory_map = serde_json::Map::new();
        memory_map.insert("part".to_string(), serde_json::json!("memory"));
        memory_ops.insert(OPERATION_KEY.to_string(), memory_map);
        memory_msg.metadata.operations = Some(Box::new(memory_ops));

        (recall_msg, memory_msg)
    }

    #[test]
    fn rolling_genuine_prompt_visibility_and_types() {
        // Normal user text is genuine
        let user_msg = Message::user().with_text("Hello there");
        assert!(is_genuine_prompt(&user_msg));

        // Agent-only user text is not genuine
        let agent_only_msg = Message::user()
            .with_text("nudge")
            .with_visibility(false, true);
        assert!(!is_genuine_prompt(&agent_only_msg));

        // User-only (invisible to agent) is not genuine
        let user_only_msg = Message::user()
            .with_text("hidden")
            .with_visibility(true, false);
        assert!(!is_genuine_prompt(&user_only_msg));

        // User message with text + tool response is not genuine
        let user_tool_res = Message::user()
            .with_text("Some text")
            .with_tool_response(
                "call_1",
                Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
            );
        assert!(!is_genuine_prompt(&user_tool_res));

        // Steer is not genuine
        let steer_msg = Message::user().with_text("steer me").with_steer();
        assert!(!is_genuine_prompt(&steer_msg));

        // Turn context is not genuine
        let tc_msg = Message::user()
            .with_text("turn context info")
            .with_metadata(MessageMetadata {
                turn_context: true,
                ..Default::default()
            });
        assert!(!is_genuine_prompt(&tc_msg));

        // Final output nudge is not genuine
        let nudge_msg = Message::user().with_text(FINAL_OUTPUT_CONTINUATION_MESSAGE);
        assert!(!is_genuine_prompt(&nudge_msg));

        // Assistant is not genuine
        let asst_msg = Message::assistant().with_text("assistant response");
        assert!(!is_genuine_prompt(&asst_msg));

        // Empty text is not genuine
        let empty_msg = Message::user().with_text("");
        assert!(!is_genuine_prompt(&empty_msg));

        let whitespace_msg = Message::user().with_text("   ");
        assert!(!is_genuine_prompt(&whitespace_msg));
    }

    #[test]
    fn rolling_estimate_cases() {
        // Retained usage -> ratio * sum(counts)
        let retained_usage = Usage {
            input_tokens: Some(500),
            output_tokens: None,
            total_tokens: Some(500),
            cache_read_input_tokens: None,
            cache_write_input_tokens: None,
        };
        assert!(is_retained(&retained_usage));
        let messages = vec![
            Message::user().with_text("m1"),
            Message::assistant().with_text("m2"),
        ];
        let counts = vec![100, 200];
        let est = estimate(&retained_usage, &messages, &counts, 1.5);
        assert_eq!(est, 450); // 1.5 * 300

        // Normal adds only after-anchor counts * ratio
        let normal_usage = Usage {
            input_tokens: Some(1000),
            output_tokens: Some(200),
            total_tokens: Some(1200),
            cache_read_input_tokens: None,
            cache_write_input_tokens: None,
        };
        let mut msg_with_usage = Message::assistant().with_text("anchor message");
        msg_with_usage.metadata.usage = Some(Box::new(MessageUsage {
            input_tokens: Some(1000),
            output_tokens: Some(200),
            total_tokens: Some(1200),
            ..Default::default()
        }));
        let messages_normal = vec![
            Message::user().with_text("before anchor"),
            msg_with_usage,
            Message::user().with_text("after anchor 1"),
            Message::assistant().with_text("after anchor 2"),
        ];
        let counts_normal = vec![100, 200, 50, 50];
        // Anchor at index 1. After-anchor counts sum = 50 + 50 = 100.
        // Ratio = 1.5. Estimate = 1200 + 1.5 * 100 = 1350.
        let est_normal = estimate(&normal_usage, &messages_normal, &counts_normal, 1.5);
        assert_eq!(est_normal, 1350);

        // Anchor whose usage differs from session's is ignored (fallback to ratio * sum(counts))
        let mut msg_differing = Message::assistant().with_text("stale anchor");
        msg_differing.metadata.usage = Some(Box::new(MessageUsage {
            input_tokens: Some(800),
            output_tokens: Some(100),
            total_tokens: Some(900),
            ..Default::default()
        }));
        let messages_diff = vec![Message::user().with_text("m1"), msg_differing];
        let est_diff = estimate(&normal_usage, &messages_diff, &counts, 1.0);
        assert_eq!(est_diff, 300); // fallback: 1.0 * (100 + 200)

        // No anchor falls back
        let messages_no_anchor = vec![
            Message::user().with_text("m1"),
            Message::assistant().with_text("m2"),
        ];
        let est_no_anchor = estimate(&normal_usage, &messages_no_anchor, &counts, 1.0);
        assert_eq!(est_no_anchor, 300);

        // No total falls back
        let no_total_usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            total_tokens: None,
            cache_read_input_tokens: None,
            cache_write_input_tokens: None,
        };
        let est_no_total = estimate(&no_total_usage, &messages_normal, &counts_normal, 1.5);
        assert_eq!(est_no_total, 600); // 1.5 * 400
    }

    #[test]
    fn rolling_cut_tail_starts_at_genuine_prompt() {
        // Walk lands on a tool response; cut snaps back to user prompt that started that turn.
        let messages = vec![
            Message::user().with_text("turn 1 prompt"),    // 0
            Message::assistant().with_text("turn 1 reply"), // 1
            Message::user().with_text("turn 2 prompt"),    // 2 (genuine prompt)
            Message::assistant().with_tool_request("t1", Ok(CallToolRequestParams::new("tool"))), // 3
            Message::user().with_tool_response(
                "t1",
                Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
            ), // 4
            Message::assistant().with_text("turn 2 reply"), // 5
        ];
        let counts = vec![100, 100, 50, 50, 100, 100];
        // tail_tokens = 150, ratio = 1.0.
        // Walk from end: idx 5 (100), idx 4 (100) -> sum 200 >= 150.
        // Walk stops at idx 4 (tool response).
        // Nearest genuine prompt at or before 4 is idx 2 ("turn 2 prompt").
        let plan = plan_cut(&messages, &counts, 1.0, 150)
            .unwrap()
            .expect("plan exists");
        assert_eq!(plan.cut, 2);
        assert_eq!(plan.folded, vec![0, 1]);
        assert_eq!(plan.previous_memory, None);
        assert_eq!(plan.new_material_o200k, 200); // counts[0] + counts[1]
        assert_eq!(plan.tail_o200k, 300); // counts[2..5].sum() = 50 + 50 + 100 + 100
    }

    #[test]
    fn rolling_cut_skips_turn_context_and_steer_and_nudge() {
        let mut tc_msg = Message::user().with_text("context");
        tc_msg.metadata.turn_context = true;
        let steer_msg = Message::user().with_text("steer").with_steer();
        let nudge_msg = Message::user().with_text(FINAL_OUTPUT_CONTINUATION_MESSAGE);

        let messages = vec![
            Message::user().with_text("turn 1 prompt"), // 0
            Message::assistant().with_text("turn 1 reply"), // 1
            Message::user().with_text("turn 2 prompt"), // 2: genuine
            tc_msg,                                     // 3
            steer_msg,                                  // 4
            nudge_msg,                                  // 5
            Message::assistant().with_text("turn 2 reply"), // 6
        ];
        let counts = vec![100, 100, 50, 20, 20, 20, 100];
        // tail_tokens = 120. Walk lands at 5. Nearest genuine prompt <= 5 is idx 2!
        let plan = plan_cut(&messages, &counts, 1.0, 120)
            .unwrap()
            .expect("plan exists");
        assert_eq!(plan.cut, 2);
    }

    #[test]
    fn rolling_cut_ignores_agent_invisible_history() {
        let mut messages = vec![
            Message::user().with_text("turn 1 prompt"),    // 0
            Message::assistant().with_text("turn 1 reply"), // 1
        ];
        let mut counts = vec![100, 100];

        // Add 100 agent-invisible messages
        for _ in 0..100 {
            messages.push(
                Message::user()
                    .with_text("invisible")
                    .with_visibility(true, false),
            );
            counts.push(0);
        }

        messages.push(Message::user().with_text("turn 2 prompt")); // genuine prompt (idx 102)
        counts.push(100);
        messages.push(Message::assistant().with_text("turn 2 reply")); // idx 103
        counts.push(100);

        let plan = plan_cut(&messages, &counts, 1.0, 150)
            .unwrap()
            .expect("plan exists");
        // Cut is the turn 2 prompt
        assert_eq!(plan.cut, 102);
        // folded only contains agent-visible messages < cut (0 and 1)
        assert_eq!(plan.folded, vec![0, 1]);
        assert_eq!(plan.new_material_o200k, 200);
        assert_eq!(plan.tail_o200k, 200);
    }

    #[test]
    fn rolling_cut_single_turn_bigger_than_tail() {
        // One prompt + huge tool-heavy turn after earlier turns -> cut is that prompt; everything before folds.
        let messages = vec![
            Message::user().with_text("turn 1 prompt"),    // 0
            Message::assistant().with_text("turn 1 reply"), // 1
            Message::user().with_text("turn 2 prompt"),    // 2
            Message::assistant().with_tool_request("t1", Ok(CallToolRequestParams::new("tool"))), // 3
            Message::user().with_tool_response(
                "t1",
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    "big data",
                )])),
            ), // 4
            Message::assistant().with_text("turn 2 reply"), // 5
        ];
        let counts = vec![50, 50, 50, 200, 500, 200];
        // tail_tokens = 300.
        // Walk from end: idx 5 (200), idx 4 (500) -> sum = 700 >= 300. Walk stops at idx 4.
        // Nearest genuine prompt at or before 4 is idx 2 ("turn 2 prompt").
        let plan = plan_cut(&messages, &counts, 1.0, 300)
            .unwrap()
            .expect("plan exists");
        assert_eq!(plan.cut, 2);
        assert_eq!(plan.folded, vec![0, 1]);
        assert_eq!(plan.new_material_o200k, 100);
    }

    #[test]
    fn rolling_cut_nothing_to_fold() {
        // Only one genuine prompt in visible history -> None
        let messages_one_prompt = vec![
            Message::user().with_text("only prompt"),
            Message::assistant().with_text("reply"),
        ];
        let counts_one = vec![100, 100];
        assert_eq!(
            plan_cut(&messages_one_prompt, &counts_one, 1.0, 50).unwrap(),
            None
        );

        // Visible history = previous memory pair + one turn -> None
        let (recall, memory) = make_memory_pair("What do you remember?", "I remember...");
        let messages_mem_plus_one = vec![
            recall,                                           // 0
            memory,                                           // 1
            Message::user().with_text("new turn prompt"),     // 2
            Message::assistant().with_text("new turn reply"), // 3
        ];
        let counts_mem_plus_one = vec![50, 200, 100, 100];
        assert_eq!(
            plan_cut(&messages_mem_plus_one, &counts_mem_plus_one, 1.0, 100).unwrap(),
            None
        );
    }

    #[test]
    fn rolling_cut_includes_old_compact_summary() {
        // An agent-visible old-style summary (agent_only assistant text + continuation) before the cut
        // lands in folded and in new_material_o200k.
        let summary_asst = Message::assistant()
            .with_text("Old summary")
            .with_visibility(false, true);
        let continuation = Message::user()
            .with_text("Continue from summary")
            .with_visibility(false, true);

        let messages = vec![
            summary_asst,                                   // 0
            continuation,                                   // 1
            Message::user().with_text("turn 2 prompt"),     // 2 (genuine prompt)
            Message::assistant().with_text("turn 2 reply"), // 3
        ];
        let counts = vec![250, 50, 100, 100];
        let plan = plan_cut(&messages, &counts, 1.0, 100)
            .unwrap()
            .expect("plan exists");
        assert_eq!(plan.cut, 2);
        assert_eq!(plan.folded, vec![0, 1]);
        assert_eq!(plan.previous_memory, None);
        assert_eq!(plan.new_material_o200k, 300); // 250 + 50
    }

    #[test]
    fn rolling_cut_previous_memory_detected_and_excluded_from_progress() {
        let (recall, memory) = make_memory_pair("recall", "memory");
        let messages = vec![
            recall,                                         // 0
            memory,                                         // 1: previous memory assistant row
            Message::user().with_text("turn 1 prompt"),     // 2
            Message::assistant().with_text("turn 1 reply"), // 3
            Message::user().with_text("turn 2 prompt"),     // 4 (cut)
            Message::assistant().with_text("turn 2 reply"), // 5
        ];
        let counts = vec![50, 200, 100, 100, 100, 100];
        let plan = plan_cut(&messages, &counts, 1.0, 150)
            .unwrap()
            .expect("plan exists");
        assert_eq!(plan.cut, 4);
        assert_eq!(plan.folded, vec![0, 1, 2, 3]);
        assert_eq!(plan.previous_memory, Some(1));
        // Previous memory pair (indices 0 and 1) is EXCLUDED from new_material_o200k
        assert_eq!(plan.new_material_o200k, 200); // counts[2] (100) + counts[3] (100)
    }

    #[test]
    fn rolling_cut_pairing_violation_is_error() {
        // Request before cut, response after cut -> Err(SplitToolPair)
        let messages = vec![
            Message::user().with_text("turn 1 prompt"), // 0
            Message::assistant().with_tool_request("t1", Ok(CallToolRequestParams::new("tool"))), // 1
            Message::user().with_text("turn 2 prompt"), // 2 (cut)
            Message::user().with_tool_response(
                "t1",
                Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
            ), // 3
            Message::assistant().with_text("turn 2 reply"), // 4
        ];
        let counts = vec![100, 100, 100, 100, 100];
        let err = plan_cut(&messages, &counts, 1.0, 150).unwrap_err();
        assert_eq!(err, CutError::SplitToolPair("t1".to_string()));

        // Parallel-call message (two requests in one message) where only one response crosses -> Err
        let messages_parallel = vec![
            Message::user().with_text("turn 1 prompt"), // 0
            Message::assistant()                        // 1
                .with_tool_request("t_a", Ok(CallToolRequestParams::new("tool_a")))
                .with_tool_request("t_b", Ok(CallToolRequestParams::new("tool_b"))),
            Message::user().with_tool_response(
                "t_a",
                Ok(CallToolResult::success(vec![ContentBlock::text("a ok")])),
            ), // 2
            Message::user().with_text("turn 2 prompt"), // 3 (cut)
            Message::user().with_tool_response(
                "t_b",
                Ok(CallToolResult::success(vec![ContentBlock::text("b ok")])),
            ), // 4
            Message::assistant().with_text("turn 2 reply"), // 5
        ];
        let counts_parallel = vec![100, 100, 100, 100, 100, 100];
        let err_parallel = plan_cut(&messages_parallel, &counts_parallel, 1.0, 150).unwrap_err();
        assert_eq!(err_parallel, CutError::SplitToolPair("t_b".to_string()));
    }

    #[test]
    fn rolling_cut_ratio_scales_tail() {
        // ratio 1.45 moves the cut later than ratio 1.0 for the same tail_tokens
        let messages = vec![
            Message::user().with_text("prompt 0"),    // 0
            Message::assistant().with_text("reply 0"), // 1
            Message::user().with_text("prompt 1"),    // 2
            Message::assistant().with_text("reply 1"), // 3
            Message::user().with_text("prompt 2"),    // 4
            Message::assistant().with_text("reply 2"), // 5
        ];
        let counts = vec![50, 50, 50, 50, 50, 50];
        let tail_tokens = 120;

        // Ratio 1.0: needs 120 tokens.
        // idx 5 (50) + idx 4 (50) = 100 < 120; idx 3 (50) -> sum 150 >= 120. Walk lands at 3.
        // Nearest prompt <= 3 is idx 2 ("prompt 1").
        let plan_1 = plan_cut(&messages, &counts, 1.0, tail_tokens)
            .unwrap()
            .expect("plan exists");
        assert_eq!(plan_1.cut, 2);

        // Ratio 1.45: each msg is 50 * 1.45 = 72.5 tokens.
        // idx 5 (72.5) + idx 4 (72.5) = 145 >= 120! Walk lands at 4.
        // Nearest prompt <= 4 is idx 4 ("prompt 2")!
        let plan_145 = plan_cut(&messages, &counts, 1.45, tail_tokens)
            .unwrap()
            .expect("plan exists");
        assert_eq!(plan_145.cut, 4);

        assert!(plan_145.cut > plan_1.cut);
    }

    #[test]
    fn rolling_decide_cases() {
        let settings = RollingMemorySettings {
            budget_tokens: 100_000,
            tail_tokens: 40_000,
        };
        let dummy_plan = CutPlan {
            cut: 2,
            folded: vec![0, 1],
            previous_memory: None,
            new_material_o200k: 35_000,
            tail_o200k: 30_000,
        };

        // Under budget -> Skip("under budget")
        assert_eq!(
            decide(90_000, Some(&dummy_plan), 5_000, 1.0, &settings),
            Decision::Skip("under budget")
        );

        // Plan None -> Skip("nothing to fold")
        assert_eq!(
            decide(110_000, None, 5_000, 1.0, &settings),
            Decision::Skip("nothing to fold")
        );

        // Gets-under fails: planned = ratio * (base + tail) + MEMORY_ALLOWANCE > 0.8 * budget
        // base = 50_000, tail = 30_000, ratio = 1.0 -> 80_000 + 6_000 = 86_000 > 80_000 (0.8 * 100k)
        assert_eq!(
            decide(110_000, Some(&dummy_plan), 50_000, 1.0, &settings),
            Decision::Skip("would not get under")
        );

        // Progress fails: ratio * new_material < max(30k, 0.2 * budget)
        // new_material = 10_000, ratio = 1.0 -> 10_000 < max(30k, 20k) = 30k
        let low_progress_plan = CutPlan {
            new_material_o200k: 10_000,
            ..dummy_plan.clone()
        };
        assert_eq!(
            decide(110_000, Some(&low_progress_plan), 5_000, 1.0, &settings),
            Decision::Skip("too little progress")
        );

        // Both waived over 2B (hard cap: estimate > 2 * budget = 200_000)
        // Even with high planned and low progress, hard cap triggers Fold:
        assert_eq!(
            decide(210_000, Some(&low_progress_plan), 50_000, 1.0, &settings),
            Decision::Fold
        );

        // Successful Fold:
        // estimate > budget, plan exists, not hard cap, gets under ok, progress ok
        assert_eq!(
            decide(110_000, Some(&dummy_plan), 5_000, 1.0, &settings),
            Decision::Fold
        );
    }

    #[test]
    fn rolling_backoff_cases() {
        let settings = RollingMemorySettings {
            budget_tokens: 100_000,
            tail_tokens: 40_000,
        };
        // Below 2 failures -> never backed off
        let state_1 = RollingState {
            consecutive_failures: 1,
            failed_at_estimate: Some(100_000),
            ratio: 1.45,
        };
        assert!(!backed_off(&state_1, 105_000, &settings));

        // At 2 failures:
        // tail_tokens / 2 = 20_000. failed_at_estimate = 100_000.
        // Needs estimate >= 120_000 to retry.
        let state_2 = RollingState {
            consecutive_failures: 2,
            failed_at_estimate: Some(100_000),
            ratio: 1.45,
        };
        // estimate 110_000 < 120_000 -> backed off (true)
        assert!(backed_off(&state_2, 110_000, &settings));
        // estimate 120_000 >= 120_000 -> retries (false)
        assert!(!backed_off(&state_2, 120_000, &settings));
        // estimate 125_000 >= 120_000 -> retries (false)
        assert!(!backed_off(&state_2, 125_000, &settings));

        // Hard cap overrides: estimate > 2 * budget (200_000)
        // Even if estimate < failed_at_estimate + tail/2 (e.g. failed_at = 210_000),
        // estimate > 200_000 means backed_off is false!
        let state_high = RollingState {
            consecutive_failures: 2,
            failed_at_estimate: Some(210_000),
            ratio: 1.45,
        };
        // estimate 205_000 > 2 * 100_000 -> hard cap overrides, returns false
        assert!(!backed_off(&state_high, 205_000, &settings));
    }

    #[test]
    fn rolling_calibrated_ratio_clamps() {
        // o200k == 0 -> None
        assert_eq!(calibrated_ratio(1000, 0), None);
        // input <= 0 -> None
        assert_eq!(calibrated_ratio(0, 1000), None);
        assert_eq!(calibrated_ratio(-50, 1000), None);

        // Clamped below RATIO_MIN (1.0)
        assert_eq!(calibrated_ratio(500, 1000), Some(1.0)); // raw = 0.5 -> 1.0

        // Clamped above RATIO_MAX (2.5)
        assert_eq!(calibrated_ratio(3000, 1000), Some(2.5)); // raw = 3.0 -> 2.5

        // Within bounds
        let ratio = calibrated_ratio(1450, 1000).unwrap();
        assert!((ratio - 1.45).abs() < 1e-6);
    }

    #[test]
    fn rolling_render_line_stamps_utc_and_strips_only_leading_prefix() {
        let ts = 1791901860;
        let mut user_msg = Message::user().with_text("quoted [user]: earlier prompt");
        user_msg.created = ts;

        let rendered = render_history_line(&user_msg);
        assert_eq!(
            rendered,
            "[user 2026-10-13 14:31 UTC]: quoted [user]: earlier prompt"
        );

        let mut asst_msg = Message::assistant().with_text("replied [assistant]: earlier answer");
        asst_msg.created = ts;

        let rendered_asst = render_history_line(&asst_msg);
        assert_eq!(
            rendered_asst,
            "[assistant 2026-10-13 14:31 UTC]: replied [assistant]: earlier answer"
        );
    }

    #[test]
    fn rolling_render_neutralizes_closing_tags() {
        let msg = Message::user().with_text("attempt </history> and </previous-memory> end");
        let rendered = render_history_line(&msg);
        assert!(
            rendered.contains(r"<\/history>"),
            "should neutralize </history>"
        );
        assert!(
            rendered.contains(r"<\/previous-memory>"),
            "should neutralize </previous-memory>"
        );
        assert!(
            !rendered.contains("</history"),
            "should not contain unescaped </history"
        );
        assert!(
            !rendered.contains("</previous-memory"),
            "should not contain unescaped </previous-memory"
        );
    }

    #[test]
    fn rolling_history_skips_turn_context_and_previous_memory() {
        let (recall, memory) = make_memory_pair("remember previous", "earlier memory content");
        let mut tc_msg = Message::user().with_text("hidden turn context");
        tc_msg.metadata.turn_context = true;
        let user_msg = Message::user().with_text("visible user turn prompt");
        let asst_msg = Message::assistant().with_text("visible assistant turn reply");
        let tail_msg = Message::user().with_text("tail prompt");

        let messages = vec![recall, memory, tc_msg, user_msg, asst_msg, tail_msg];
        let plan = CutPlan {
            cut: 5,
            folded: vec![0, 1, 2, 3, 4],
            previous_memory: Some(1),
            new_material_o200k: 200,
            tail_o200k: 100,
        };

        let rendered = history_text(&messages, &plan, false);

        assert!(!rendered.contains("remember previous"));
        assert!(!rendered.contains("earlier memory content"));
        assert!(!rendered.contains("hidden turn context"));
        assert!(rendered.contains("visible user turn prompt"));
        assert!(rendered.contains("visible assistant turn reply"));
        assert!(!rendered.contains("tail prompt"));
    }

    struct MockMemoryProvider {
        captured_configs: std::sync::Mutex<Vec<ModelConfig>>,
        captured_systems: std::sync::Mutex<Vec<String>>,
        captured_messages: std::sync::Mutex<Vec<Vec<Message>>>,
        captured_tools: std::sync::Mutex<Vec<Vec<rmcp::model::Tool>>>,
        responses: std::sync::Mutex<std::collections::VecDeque<Result<(Message, ProviderUsage), ProviderError>>>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl MockMemoryProvider {
        fn new(response: Result<(Message, ProviderUsage), ProviderError>) -> Self {
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(response);
            Self {
                captured_configs: std::sync::Mutex::new(Vec::new()),
                captured_systems: std::sync::Mutex::new(Vec::new()),
                captured_messages: std::sync::Mutex::new(Vec::new()),
                captured_tools: std::sync::Mutex::new(Vec::new()),
                responses: std::sync::Mutex::new(queue),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn new_queue(responses: Vec<Result<(Message, ProviderUsage), ProviderError>>) -> Self {
            Self {
                captured_configs: std::sync::Mutex::new(Vec::new()),
                captured_systems: std::sync::Mutex::new(Vec::new()),
                captured_messages: std::sync::Mutex::new(Vec::new()),
                captured_tools: std::sync::Mutex::new(Vec::new()),
                responses: std::sync::Mutex::new(responses.into()),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn last_config(&self) -> ModelConfig {
            self.captured_configs.lock().unwrap().last().cloned().unwrap()
        }

        fn last_system(&self) -> String {
            self.captured_systems.lock().unwrap().last().cloned().unwrap()
        }

        fn last_messages(&self) -> Vec<Message> {
            self.captured_messages.lock().unwrap().last().cloned().unwrap()
        }

        fn last_tools(&self) -> Vec<rmcp::model::Tool> {
            self.captured_tools.lock().unwrap().last().cloned().unwrap()
        }
    }

    #[async_trait::async_trait]
    impl Provider for MockMemoryProvider {
        fn get_name(&self) -> &str {
            "mock-memory-provider"
        }

        async fn stream(
            &self,
            _model_config: &ModelConfig,
            _system: &str,
            _messages: &[Message],
            _tools: &[rmcp::model::Tool],
        ) -> Result<goose_providers::base::MessageStream, ProviderError> {
            Err(ProviderError::NotImplemented("stream not used".to_string()))
        }

        async fn complete(
            &self,
            model_config: &ModelConfig,
            system: &str,
            messages: &[Message],
            tools: &[rmcp::model::Tool],
        ) -> Result<(Message, ProviderUsage), ProviderError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.captured_configs.lock().unwrap().push(model_config.clone());
            self.captured_systems.lock().unwrap().push(system.to_string());
            self.captured_messages.lock().unwrap().push(messages.to_vec());
            self.captured_tools.lock().unwrap().push(tools.to_vec());

            let mut queue = self.responses.lock().unwrap();
            if let Some(res) = queue.pop_front() {
                res
            } else {
                Err(ProviderError::ExecutionError("no response".to_string()))
            }
        }
    }

    #[tokio::test]
    async fn rolling_memory_request_shape() {
        let temp_dir = tempfile::tempdir().unwrap();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap()))]);

        let usage = ProviderUsage::new("test-model".to_string(), Usage::default())
            .with_finish_reasons(vec!["end_turn".to_string()]);
        let resp = Message::assistant().with_text("memory content");
        let provider = MockMemoryProvider::new(Ok((resp.clone(), usage.clone())));

        let session_cfg = ModelConfig::new("test-model").with_max_tokens(Some(4096));

        // 1. With previous memory
        let res = write_memory(
            &provider,
            &session_cfg,
            "session-1",
            Some("old memory text"),
            "history text line",
            1.45,
        )
        .await
        .expect("write_memory should succeed");

        assert_eq!(res.text, "memory content");
        assert_eq!(provider.call_count(), 1);

        let system = provider.last_system();
        assert!(system.contains("<previous-memory>"));
        assert!(system.contains("old memory text"));
        assert!(system.contains("</previous-memory>"));
        assert!(system.contains("<history>"));
        assert!(system.contains("history text line"));
        assert!(system.contains("</history>"));

        let msgs = provider.last_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].as_concat_text(), MEMORY_REQUEST_TEXT);

        assert!(provider.last_tools().is_empty());
        assert_eq!(provider.last_config().max_tokens, Some(16_000));

        // 2. Without previous memory
        let provider2 = MockMemoryProvider::new(Ok((resp, usage)));
        let res2 = write_memory(
            &provider2,
            &session_cfg,
            "session-2",
            None,
            "history text line",
            1.45,
        )
        .await
        .expect("write_memory without previous memory should succeed");

        assert_eq!(res2.text, "memory content");
        let system2 = provider2.last_system();
        assert!(!system2.contains("</previous-memory>"));
        assert!(!system2.contains("**Your previous memory.**"));
        assert!(system2.contains("<history>"));
        assert!(system2.contains("history text line"));
        assert!(system2.contains("</history>"));
    }

    #[tokio::test]
    async fn rolling_memory_failures() {
        let temp_dir = tempfile::tempdir().unwrap();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap()))]);

        let session_cfg = ModelConfig::new("test-model");

        // 1. Output limit reached (carries usage)
        let mut msg_limit = Message::assistant().with_text("partial text");
        msg_limit.metadata.output_token_limit_reached = true;
        let limit_usage = ProviderUsage::new(
            "test-model".to_string(),
            Usage::new(Some(120), Some(45), None),
        );
        let provider = MockMemoryProvider::new(Ok((msg_limit, limit_usage)));

        let err = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect_err("should fail on output limit");
        match err {
            MemoryError::OutputLimit(u) => {
                assert_eq!(u.usage.input_tokens, Some(120));
                assert_eq!(u.usage.output_tokens, Some(45));
            }
            other => panic!("expected OutputLimit, got {:?}", other),
        }

        // 2. Refusal
        let provider = MockMemoryProvider::new(Err(ProviderError::Refusal {
            details: "refused content".to_string(),
            category: None,
        }));
        let err = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect_err("should fail on refusal");
        match err {
            MemoryError::Refused(d) => assert_eq!(d, "refused content"),
            other => panic!("expected Refused, got {:?}", other),
        }

        // 3. ["max_tokens"] finish reason (carries usage)
        let msg = Message::assistant().with_text("stopped early");
        let stop_usage = ProviderUsage::new(
            "test-model".to_string(),
            Usage::new(Some(230), Some(60), None),
        )
        .with_finish_reasons(vec!["max_tokens".to_string()]);
        let provider = MockMemoryProvider::new(Ok((msg, stop_usage)));
        let err = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect_err("should fail on non-end_turn stop reason");
        match err {
            MemoryError::StopReason(reason, u) => {
                assert_eq!(reason, "max_tokens");
                assert_eq!(u.usage.input_tokens, Some(230));
                assert_eq!(u.usage.output_tokens, Some(60));
            }
            other => panic!("expected StopReason, got {:?}", other),
        }

        // 4. Empty text (carries usage)
        let msg_empty = Message::assistant().with_text("   ");
        let empty_usage = ProviderUsage::new(
            "test-model".to_string(),
            Usage::new(Some(340), Some(0), None),
        )
        .with_finish_reasons(vec!["end_turn".to_string()]);
        let provider = MockMemoryProvider::new(Ok((msg_empty, empty_usage)));
        let err = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect_err("should fail on empty text");
        match err {
            MemoryError::Empty(u) => {
                assert_eq!(u.usage.input_tokens, Some(340));
                assert_eq!(u.usage.output_tokens, Some(0));
            }
            other => panic!("expected Empty, got {:?}", other),
        }

        // 5. Absent finish_reasons + text = Ok
        let msg_ok = Message::assistant().with_text("valid memory");
        let usage_none = ProviderUsage::new("test-model".to_string(), Usage::default());
        let provider = MockMemoryProvider::new(Ok((msg_ok, usage_none)));
        let res = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect("absent finish_reasons + text should succeed");
        assert_eq!(res.text, "valid memory");

        // 6. ["end_turn"] = Ok
        let usage_end = ProviderUsage::new("test-model".to_string(), Usage::default())
            .with_finish_reasons(vec!["end_turn".to_string()]);
        let msg_end = Message::assistant().with_text("valid memory 2");
        let provider = MockMemoryProvider::new(Ok((msg_end, usage_end)));
        let res = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect("end_turn should succeed");
        assert_eq!(res.text, "valid memory 2");

        // 7. ["stop"] = Ok (normal completion for other providers)
        let usage_stop = ProviderUsage::new("test-model".to_string(), Usage::default())
            .with_finish_reasons(vec!["stop".to_string()]);
        let msg_stop = Message::assistant().with_text("valid stop memory");
        let provider = MockMemoryProvider::new(Ok((msg_stop, usage_stop)));
        let res = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect("stop should succeed");
        assert_eq!(res.text, "valid stop memory");

        // 8. ["STOP"] = Ok (ASCII case-insensitive normal completion)
        let usage_stop_upper = ProviderUsage::new("test-model".to_string(), Usage::default())
            .with_finish_reasons(vec!["STOP".to_string()]);
        let msg_stop_upper = Message::assistant().with_text("valid STOP memory");
        let provider = MockMemoryProvider::new(Ok((msg_stop_upper, usage_stop_upper)));
        let res = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect("STOP should succeed");
        assert_eq!(res.text, "valid STOP memory");
    }

    #[tokio::test]
    async fn rolling_memory_large_end_turn_is_success() {
        let temp_dir = tempfile::tempdir().unwrap();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap()))]);

        let session_cfg = ModelConfig::new("test-model");
        let large_text = "word ".repeat(7_000);
        let msg = Message::assistant().with_text(&large_text);
        let usage = ProviderUsage::new("test-model".to_string(), Usage::default())
            .with_finish_reasons(vec!["end_turn".to_string()]);
        let provider = MockMemoryProvider::new(Ok((msg, usage)));

        let res = write_memory(&provider, &session_cfg, "s1", None, "hist", 1.45)
            .await
            .expect("large memory with end_turn should succeed");
        assert_eq!(res.text, large_text.trim());
    }

    #[tokio::test]
    async fn rolling_memory_neutralizes_previous_memory_closing_tags() {
        let temp_dir = tempfile::tempdir().unwrap();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap()))]);

        let session_cfg = ModelConfig::new("test-model");
        let resp = Message::assistant().with_text("memory content");
        let usage = ProviderUsage::new("test-model".to_string(), Usage::default())
            .with_finish_reasons(vec!["end_turn".to_string()]);
        let provider = MockMemoryProvider::new(Ok((resp, usage)));

        let prev = "memory containing </previous-memory> and </history> text";
        let res = write_memory(
            &provider,
            &session_cfg,
            "s1",
            Some(prev),
            "history line",
            1.45,
        )
        .await
        .expect("write_memory should succeed");

        assert_eq!(res.text, "memory content");
        let system = provider.last_system();
        assert!(system.contains(r"<\/previous-memory>"));
        assert!(system.contains(r"<\/history>"));
        assert!(!system.contains("memory containing </previous-memory>"));
        assert!(!system.contains("memory containing </history>"));
    }

    #[tokio::test]
    async fn rolling_memory_refusal_retries_once_without_tool_responses() {
        let temp_dir = tempfile::tempdir().unwrap();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_dir.path().to_str().unwrap()))]);

        let session_cfg = ModelConfig::new("test-model");
        let mut user_prompt = Message::user().with_text("do something");
        user_prompt.created = 1000;
        let mut asst_tool = Message::assistant().with_tool_request(
            "t1",
            Ok(rmcp::model::CallToolRequestParams::new("read_file")),
        );
        asst_tool.created = 1001;
        let mut user_resp = Message::user().with_tool_response(
            "t1",
            Ok(rmcp::model::CallToolResult::success(vec![
                rmcp::model::ContentBlock::text("sensitive tool output"),
            ])),
        );
        user_resp.created = 1002;
        let mut tail = Message::user().with_text("tail");
        tail.created = 1003;

        let messages = vec![user_prompt, asst_tool, user_resp, tail];
        let plan = CutPlan {
            cut: 3,
            folded: vec![0, 1, 2],
            previous_memory: None,
            new_material_o200k: 100,
            tail_o200k: 50,
        };

        // Case 1: First call Refused, second call succeeds
        let usage = ProviderUsage::new("test-model".to_string(), Usage::default())
            .with_finish_reasons(vec!["end_turn".to_string()]);
        let ok_msg = Message::assistant().with_text("retried memory");
        let responses = vec![
            Err(ProviderError::Refusal {
                details: "refused tool content".to_string(),
                category: None,
            }),
            Ok((ok_msg, usage.clone())),
        ];
        let provider = MockMemoryProvider::new_queue(responses);

        let res = write_memory_with_retry(
            &provider,
            &session_cfg,
            "s1",
            None,
            &messages,
            &plan,
            1.45,
        )
        .await
        .expect("retry should succeed");
        assert_eq!(res.text, "retried memory");
        assert_eq!(provider.call_count(), 2);

        // First call had tool_response:
        let sys1 = provider.captured_systems.lock().unwrap()[0].clone();
        assert!(sys1.contains("tool_response:"));

        // Second call had NO tool_response:
        let sys2 = provider.captured_systems.lock().unwrap()[1].clone();
        assert!(!sys2.contains("tool_response:"));

        // Case 2: Second call also refused -> returns Refused
        let responses_both_refused = vec![
            Err(ProviderError::Refusal {
                details: "refusal 1".to_string(),
                category: None,
            }),
            Err(ProviderError::Refusal {
                details: "refusal 2".to_string(),
                category: None,
            }),
        ];
        let provider2 = MockMemoryProvider::new_queue(responses_both_refused);
        let err2 = write_memory_with_retry(
            &provider2,
            &session_cfg,
            "s2",
            None,
            &messages,
            &plan,
            1.45,
        )
        .await
        .expect_err("second refusal should be returned as Refused");
        match err2 {
            MemoryError::Refused(d) => assert_eq!(d, "refusal 2"),
            other => panic!("expected Refused, got {:?}", other),
        }
        assert_eq!(provider2.call_count(), 2);

        // Case 3: Non-refusal failure on first call is NOT retried
        let responses_server_error = vec![
            Err(ProviderError::ServerError("server down".to_string())),
        ];
        let provider3 = MockMemoryProvider::new_queue(responses_server_error);
        let err3 = write_memory_with_retry(
            &provider3,
            &session_cfg,
            "s3",
            None,
            &messages,
            &plan,
            1.45,
        )
        .await
        .expect_err("server error should fail without retry");
        assert!(matches!(err3, MemoryError::Provider(ProviderError::ServerError(_))));
        assert_eq!(provider3.call_count(), 1);
    }

    #[test]
    fn rolling_fold_rows_shape() {
        // Message 0: 2026-10-10 10:00:00 UTC (1791636000)
        let mut msg0 = Message::user().with_text("msg0").with_generated_id();
        msg0.created = 1791636000;
        // Message 1: 2026-10-11 10:00:00 UTC (1791722400)
        let mut msg1 = Message::assistant().with_text("msg1").with_generated_id();
        msg1.created = 1791722400;
        // Message 2: 2026-10-12 10:00:00 UTC (1791808800)
        let mut msg2 = Message::user().with_text("msg2").with_generated_id();
        msg2.created = 1791808800;
        // Tail: Message 3 at 2026-10-15 10:00:00 UTC (1792068000)
        let mut msg3 = Message::user().with_text("msg3").with_generated_id();
        msg3.created = 1792068000;

        let messages = vec![msg0.clone(), msg1.clone(), msg2.clone(), msg3.clone()];
        let plan = CutPlan {
            cut: 3,
            folded: vec![0, 1, 2],
            previous_memory: None,
            new_material_o200k: 100,
            tail_o200k: 100,
        };

        let conv = fold_rows(&messages, &plan, "first fold memory").unwrap();
        let rows = conv.messages();
        assert_eq!(rows.len(), 5); // 3 folded + recall + memory

        // Folded rows have agent_visible = false, IDs kept
        for (i, &folded_idx) in plan.folded.iter().enumerate() {
            assert_eq!(rows[i].id, messages[folded_idx].id);
            assert!(!rows[i].is_agent_visible());
            assert_eq!(rows[i].is_user_visible(), messages[folded_idx].is_user_visible());
        }

        // Two new rows: recall first, then memory
        let recall = &rows[3];
        let memory = &rows[4];

        assert_eq!(recall.role, Role::User);
        assert!(!recall.is_user_visible());
        assert!(recall.is_agent_visible());
        assert!(recall.id.is_some());
        assert_ne!(recall.id, msg3.id);
        assert_eq!(recall.created, msg3.created - 1);
        assert_eq!(fold_part(recall), Some("recall"));
        assert_eq!(
            recall.metadata.operation_note(OPERATION_KEY, "from").and_then(|v| v.as_str()),
            Some("2026-10-10")
        );
        assert_eq!(
            recall.metadata.operation_note(OPERATION_KEY, "to").and_then(|v| v.as_str()),
            Some("2026-10-12")
        );
        assert_eq!(
            recall.as_concat_text(),
            "What do you remember from 2026-10-10 to 2026-10-12?"
        );

        assert_eq!(memory.role, Role::Assistant);
        assert!(!memory.is_user_visible());
        assert!(memory.is_agent_visible());
        assert!(memory.id.is_some());
        assert_ne!(memory.id, recall.id);
        assert_eq!(memory.created, msg3.created - 1);
        assert_eq!(fold_part(memory), Some("memory"));
        assert_eq!(
            memory.metadata.operation_note(OPERATION_KEY, "from").and_then(|v| v.as_str()),
            Some("2026-10-10")
        );
        assert_eq!(
            memory.metadata.operation_note(OPERATION_KEY, "to").and_then(|v| v.as_str()),
            Some("2026-10-12")
        );
        assert_eq!(memory.as_concat_text(), "first fold memory");

        // Now test second fold: previous_memory note is inherited
        let mut prev_mem = memory.clone();
        prev_mem.created = 1792000000;
        let mut msg4 = Message::user().with_text("msg4").with_generated_id();
        msg4.created = 1792154400; // 2026-10-16
        let mut msg5_tail = Message::user().with_text("msg5").with_generated_id();
        msg5_tail.created = 1792240800; // 2026-10-17

        let messages2 = vec![prev_mem, msg4, msg5_tail.clone()];
        let plan2 = CutPlan {
            cut: 2,
            folded: vec![0, 1],
            previous_memory: Some(0),
            new_material_o200k: 50,
            tail_o200k: 50,
        };

        let conv2 = fold_rows(&messages2, &plan2, "second fold memory").unwrap();
        let rows2 = conv2.messages();
        assert_eq!(rows2.len(), 4);

        let recall2 = &rows2[2];
        let memory2 = &rows2[3];
        assert_eq!(recall2.created, msg5_tail.created - 1);
        assert_eq!(memory2.created, msg5_tail.created - 1);
        assert_eq!(
            recall2.metadata.operation_note(OPERATION_KEY, "from").and_then(|v| v.as_str()),
            Some("2026-10-10")
        );
        assert_eq!(
            recall2.metadata.operation_note(OPERATION_KEY, "to").and_then(|v| v.as_str()),
            Some("2026-10-16")
        );
        assert_eq!(
            recall2.as_concat_text(),
            "What do you remember from 2026-10-10 to 2026-10-16?"
        );
        assert_eq!(
            memory2.metadata.operation_note(OPERATION_KEY, "from").and_then(|v| v.as_str()),
            Some("2026-10-10")
        );
        assert_eq!(
            memory2.metadata.operation_note(OPERATION_KEY, "to").and_then(|v| v.as_str()),
            Some("2026-10-16")
        );
        assert_eq!(memory2.as_concat_text(), "second fold memory");
    }

    #[tokio::test]
    async fn rolling_fold_round_trip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let session = sm
            .create_session(
                std::path::PathBuf::from("/tmp/test"),
                "test-round-trip".to_string(),
                SessionType::User,
                GooseMode::default(),
            )
            .await
            .unwrap();

        // Add 8 messages
        // Let T = 1000.
        // Pre-cut:
        // m0: created = 900
        // m1: created = 950
        // m2: created = 999 (T-1)
        // m3: created = 1000 (T) - pre-cut sharing same second as tail first message!
        // Tail:
        // m4: created = 1000 (T) - tail first message
        // m5: created = 1005
        // m6: created = 1010
        // m7: created = 1015
        let t: i64 = 1000;
        let timestamps = [900, 950, t - 1, t, t, 1005, 1010, 1015];
        for (i, &ts) in timestamps.iter().enumerate() {
            let mut msg = if i % 2 == 0 {
                Message::user().with_text(format!("msg_{i}"))
            } else {
                Message::assistant().with_text(format!("msg_{i}"))
            };
            msg.created = ts;
            sm.add_message(&session.id, &msg).await.unwrap();
        }

        let loaded = sm.get_session(&session.id, true).await.unwrap();
        let initial_conv = loaded.conversation.unwrap();
        let initial_messages = initial_conv.messages();
        assert_eq!(initial_messages.len(), 8);

        // Plan cut by hand: cut at 4, folded = [0, 1, 2, 3]
        let plan = CutPlan {
            cut: 4,
            folded: vec![0, 1, 2, 3],
            previous_memory: None,
            new_material_o200k: 200,
            tail_o200k: 200,
        };

        let compacted = fold_rows(initial_messages, &plan, "summary of folded history").unwrap();
        sm.save_compacted_conversation(&session.id, &compacted).await.unwrap();

        let reloaded = sm.get_session(&session.id, true).await.unwrap();
        let reloaded_conv = reloaded.conversation.unwrap();

        // 1. Row count = old + 2 (nothing deleted)
        assert_eq!(reloaded_conv.messages().len(), 10);

        // 2. agent_visible_messages() == [recall, memory, tail...] (recall then memory)
        let agent_vis = reloaded_conv.agent_visible_messages();
        assert_eq!(agent_vis.len(), 6); // recall + memory + 4 tail messages
        assert_eq!(fold_part(&agent_vis[0]), Some("recall"));
        assert_eq!(fold_part(&agent_vis[1]), Some("memory"));
        assert_eq!(agent_vis[2].id, initial_messages[4].id);
        assert_eq!(agent_vis[3].id, initial_messages[5].id);
        assert_eq!(agent_vis[4].id, initial_messages[6].id);
        assert_eq!(agent_vis[5].id, initial_messages[7].id);

        // 3. In RAW row order:
        // assert only that the pair comes after every row with created <= T-1
        // and before the tail's first message; pre-cut rows with created == T sort after the pair.
        let raw = reloaded_conv.messages();
        let recall_pos = raw.iter().position(|m| fold_part(m) == Some("recall")).unwrap();
        let memory_pos = raw.iter().position(|m| fold_part(m) == Some("memory")).unwrap();
        assert_eq!(recall_pos + 1, memory_pos);

        let tail_first_pos = raw.iter().position(|m| m.id == initial_messages[4].id).unwrap();
        assert!(memory_pos < tail_first_pos);

        // Every row with created <= T-1 must come before recall_pos
        for (idx, m) in raw.iter().enumerate() {
            if m.created <= t - 1 && fold_part(m).is_none() {
                assert!(idx < recall_pos);
            }
        }

        // Pre-cut row with created == T (initial_messages[3]) sorts after the pair
        let m3_pos = raw.iter().position(|m| m.id == initial_messages[3].id).unwrap();
        assert!(m3_pos > memory_pos);

        // 4. user-visible messages unchanged in count and order (the pair is user-invisible)
        let user_vis = reloaded_conv.user_visible_messages();
        assert_eq!(user_vis.len(), 8);
        for i in 0..8 {
            assert_eq!(user_vis[i].id, initial_messages[i].id);
        }
    }

    #[test]
    fn rolling_fold_rows_rejects_missing_id() {
        let mut msg0 = Message::user().with_text("msg0");
        msg0.id = None; // Missing id!
        let msg1 = Message::assistant().with_text("msg1").with_generated_id();
        let msg2 = Message::user().with_text("msg2").with_generated_id();

        let messages = vec![msg0, msg1, msg2];
        let plan = CutPlan {
            cut: 2,
            folded: vec![0, 1],
            previous_memory: None,
            new_material_o200k: 50,
            tail_o200k: 50,
        };

        let err = fold_rows(&messages, &plan, "mem").expect_err("should reject missing id");
        assert!(err.to_string().contains("has no id"));
    }

    #[test]
    fn rolling_fold_rows_cut_out_of_bounds() {
        let msg0 = Message::user().with_text("msg0").with_generated_id();
        let messages = vec![msg0];
        let plan = CutPlan {
            cut: 5,
            folded: vec![0],
            previous_memory: None,
            new_material_o200k: 10,
            tail_o200k: 0,
        };
        let err = fold_rows(&messages, &plan, "mem").expect_err("should reject out of bounds cut");
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn rolling_fold_rows_rejects_folded_at_or_after_cut() {
        let msg0 = Message::user().with_text("msg0").with_generated_id();
        let msg1 = Message::assistant().with_text("msg1").with_generated_id();
        let msg2 = Message::user().with_text("msg2").with_generated_id();
        let messages = vec![msg0, msg1, msg2];
        let plan = CutPlan {
            cut: 2,
            folded: vec![0, 2],
            previous_memory: None,
            new_material_o200k: 50,
            tail_o200k: 50,
        };
        let err = fold_rows(&messages, &plan, "mem")
            .expect_err("should reject folded at or after cut");
        assert!(err.to_string().contains("at or after cut"));
    }

    #[test]
    fn rolling_fold_rows_rejects_empty_folded() {
        let msg0 = Message::user().with_text("msg0").with_generated_id();
        let msg1 = Message::user().with_text("msg1").with_generated_id();
        let messages = vec![msg0, msg1];
        let plan = CutPlan {
            cut: 1,
            folded: vec![],
            previous_memory: None,
            new_material_o200k: 0,
            tail_o200k: 50,
        };
        let err = fold_rows(&messages, &plan, "mem").expect_err("should reject empty folded");
        assert!(err.to_string().contains("nothing to fold"));
    }
}
