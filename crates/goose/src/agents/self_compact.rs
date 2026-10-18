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
    use serde_json::json;

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
}
