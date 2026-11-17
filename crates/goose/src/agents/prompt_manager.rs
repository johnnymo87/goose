// Modified by Jonathan Mohrbacher, 2026: subdirectory hints excluded from system prompt extras and delivered in conversation; assistant-only audience; requeue delivered hints on history replacement.
#[cfg(test)]
use chrono::DateTime;
use chrono::Utc;
use indexmap::IndexMap;
use serde::Serialize;

use crate::agents::{extension::ExtensionInfo, moim};
use crate::conversation::Conversation;
use crate::hints::load_hints::build_gitignore;
use crate::hints::{get_context_filenames, load_hint_files, SubdirectoryHintTracker};
use crate::{
    config::{Config, GooseMode},
    prompt_template,
    utils::sanitize_unicode_tags,
};
use goose_providers::conversation::message::{Message, MessageContentBlock};
use std::path::Path;

pub struct PromptManager {
    system_prompt_override: Option<String>,
    system_prompt_extras: IndexMap<String, String>,
    current_date_timestamp: String,
    subdirectory_hint_tracker: SubdirectoryHintTracker,
    pending_subdirectory_hints: Vec<(String, String)>,
    pub delivered_subdirectory_hints: IndexMap<String, String>,
}

impl Default for PromptManager {
    fn default() -> Self {
        PromptManager::new()
    }
}

#[derive(Serialize)]
struct SystemPromptContext {
    extensions: Vec<ExtensionInfo>,
    current_date_time: String,
    goose_mode: GooseMode,
    is_autonomous: bool,
    enable_subagents: bool,
    code_execution_mode: bool,
    include_extensions: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    moim_system_prompt_block: Option<String>,
}

pub struct SystemPromptBuilder<'a, M> {
    manager: &'a M,

    extensions_info: Vec<ExtensionInfo>,
    prompt_extras: IndexMap<String, String>,
    subagents_enabled: bool,
    hints: Option<String>,
    code_execution_mode: bool,
    include_extensions: bool,
    goose_mode: Option<GooseMode>,
}

impl<'a> SystemPromptBuilder<'a, PromptManager> {
    pub fn with_extension(mut self, extension: ExtensionInfo) -> Self {
        self.extensions_info.push(extension);
        self
    }

    pub fn with_extensions(mut self, extensions: impl Iterator<Item = ExtensionInfo>) -> Self {
        for extension in extensions {
            self.extensions_info.push(extension);
        }
        self
    }

    pub fn with_prompt_extras(
        mut self,
        extras: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        self.prompt_extras.extend(extras);
        self
    }

    pub fn with_code_execution_mode(mut self, enabled: bool) -> Self {
        self.code_execution_mode = enabled;
        self
    }

    pub fn without_extensions(mut self) -> Self {
        self.include_extensions = false;
        self
    }

    pub fn with_hints(mut self, working_dir: &Path) -> Self {
        let hints_filenames = get_context_filenames();
        let ignore_patterns = build_gitignore(working_dir);

        let hints = load_hint_files(working_dir, &hints_filenames, &ignore_patterns);

        if !hints.is_empty() {
            self.hints = Some(hints);
        }
        self
    }

    pub fn with_enable_subagents(mut self, subagents_enabled: bool) -> Self {
        self.subagents_enabled = subagents_enabled;
        self
    }

    pub fn with_goose_mode(mut self, mode: GooseMode) -> Self {
        self.goose_mode = Some(mode);
        self
    }

    pub fn build(self) -> String {
        let mut extensions_info = self.extensions_info;

        // Stable tool ordering is important for multi session prompt caching.
        extensions_info.sort_by(|a, b| a.name.cmp(&b.name));

        let sanitized_extensions_info: Vec<ExtensionInfo> = extensions_info
            .into_iter()
            .map(|mut ext_info| {
                ext_info.instructions = sanitize_unicode_tags(&ext_info.instructions);
                ext_info
            })
            .collect();

        let goose_mode = self
            .goose_mode
            .unwrap_or_else(|| Config::global().get_goose_mode().unwrap_or_default());

        let context = SystemPromptContext {
            extensions: sanitized_extensions_info,
            current_date_time: self.manager.current_date_timestamp.clone(),
            goose_mode,
            is_autonomous: goose_mode == GooseMode::Auto,
            enable_subagents: self.subagents_enabled,
            code_execution_mode: self.code_execution_mode,
            include_extensions: self.include_extensions,
            moim_system_prompt_block: moim::system_prompt_block(),
        };

        let base_prompt = if let Some(override_prompt) = &self.manager.system_prompt_override {
            let sanitized_override_prompt = sanitize_unicode_tags(override_prompt);
            prompt_template::render_string(&sanitized_override_prompt, &context)
        } else {
            prompt_template::render_template("system.md", &context)
        }
        .unwrap_or_else(|_| {
            "You are a general-purpose AI agent called goose, created by Block".to_string()
        });

        let mut system_prompt_extras = self.manager.system_prompt_extras.clone();
        system_prompt_extras.extend(self.prompt_extras);

        // Add hints if provided
        if let Some(hints) = self.hints {
            system_prompt_extras.insert("hints".to_string(), hints);
        }

        if goose_mode == GooseMode::Chat {
            system_prompt_extras.insert(
                "chat_mode".to_string(),
                "Right now you are in the chat only mode, no access to any tool use and system."
                    .to_string(),
            );
        }

        if system_prompt_extras.is_empty() {
            base_prompt
        } else {
            let sanitized_system_prompt_extras: Vec<String> = system_prompt_extras
                .into_values()
                .map(|extra| sanitize_unicode_tags(&extra))
                .collect();

            format!(
                "{}\n\n# Additional Instructions:\n\n{}",
                base_prompt,
                sanitized_system_prompt_extras.join("\n\n")
            )
        }
    }
}

impl PromptManager {
    pub fn new() -> Self {
        PromptManager {
            system_prompt_override: None,
            system_prompt_extras: IndexMap::new(),
            // Use the fixed current date time so that prompt cache can be used.
            // Filtering to an hour to balance user time accuracy and multi session prompt cache hits.
            current_date_timestamp: Utc::now().format("%Y-%m-%d %H:00 %:z").to_string(),
            subdirectory_hint_tracker: SubdirectoryHintTracker::new(),
            pending_subdirectory_hints: Vec::new(),
            delivered_subdirectory_hints: IndexMap::new(),
        }
    }

    #[cfg(test)]
    pub fn with_timestamp(dt: DateTime<Utc>) -> Self {
        PromptManager {
            system_prompt_override: None,
            system_prompt_extras: IndexMap::new(),
            current_date_timestamp: dt.format("%Y-%m-%d %H:%M:%S %:z").to_string(),
            subdirectory_hint_tracker: SubdirectoryHintTracker::new(),
            pending_subdirectory_hints: Vec::new(),
            delivered_subdirectory_hints: IndexMap::new(),
        }
    }

    /// Add an additional instruction to the system prompt with a key
    /// Using the same key will replace the previous instruction
    pub fn add_system_prompt_extra(&mut self, key: String, instruction: String) {
        self.system_prompt_extras.insert(key, instruction);
    }

    pub fn remove_system_prompt_extra(&mut self, key: &str) {
        self.system_prompt_extras.shift_remove(key);
    }

    pub fn record_tool_arguments(
        &mut self,
        arguments: &Option<serde_json::Map<String, serde_json::Value>>,
        working_dir: &Path,
    ) {
        self.subdirectory_hint_tracker
            .record_tool_arguments(arguments, working_dir);
    }

    pub fn load_subdirectory_hints(&mut self, working_dir: &Path) -> Vec<(String, String)> {
        let new_hints = self.subdirectory_hint_tracker.load_new_hints(working_dir);
        self.pending_subdirectory_hints.extend(new_hints);
        std::mem::take(&mut self.pending_subdirectory_hints)
    }

    pub fn set_pending_subdirectory_hints(&mut self, hints: Vec<(String, String)>) {
        self.pending_subdirectory_hints = hints;
    }

    pub fn pending_subdirectory_hints(&self) -> &[(String, String)] {
        &self.pending_subdirectory_hints
    }

    pub fn delivered_subdirectory_hints(&self) -> &IndexMap<String, String> {
        &self.delivered_subdirectory_hints
    }

    pub fn mark_hint_delivered(&mut self, key: String, content: String) {
        self.delivered_subdirectory_hints.insert(key, content);
    }

    pub fn requeue_after_history_replacement(&mut self) {
        for (key, content) in std::mem::take(&mut self.delivered_subdirectory_hints) {
            if !self
                .pending_subdirectory_hints
                .iter()
                .any(|(k, _)| k == &key)
            {
                self.pending_subdirectory_hints.push((key, content));
            }
        }
    }

    pub fn build_system_prompt(
        &mut self,
        working_dir: &Path,
        prompt_parts: Vec<(String, String)>,
        goose_mode: GooseMode,
    ) -> String {
        self.builder()
            .with_prompt_extras(prompt_parts)
            .with_hints(working_dir)
            .with_goose_mode(goose_mode)
            .without_extensions()
            .build()
    }

    /// Override the system prompt with custom text
    pub fn set_system_prompt_override(&mut self, template: String) {
        self.system_prompt_override = Some(template);
    }

    pub fn clear_system_prompt_override(&mut self) {
        self.system_prompt_override = None;
    }

    pub fn builder<'a>(&'a self) -> SystemPromptBuilder<'a, Self> {
        SystemPromptBuilder {
            manager: self,

            extensions_info: vec![],
            prompt_extras: IndexMap::new(),
            subagents_enabled: false,
            hints: None,
            code_execution_mode: false,
            include_extensions: true,
            goose_mode: None,
        }
    }
}

pub fn hint_dir_from_key(key: &str) -> &str {
    key.strip_prefix("subdir_hints:").unwrap_or(key)
}

pub fn format_marker_open(dir: &str) -> String {
    format!("<subdirectory-hints dir=\"{dir}\">")
}

pub fn format_marker_close() -> &'static str {
    "</subdirectory-hints>"
}

pub fn format_subdirectory_hint(dir: &str, content: &str) -> String {
    let hint_source = crate::hints::get_context_filenames()
        .into_iter()
        .find(|name| std::path::Path::new(dir).join(name).is_file())
        .map(|name| format!("{dir}/{name}"))
        .unwrap_or_else(|| format!("{dir}/AGENTS.md"));

    format!(
        "Repository instructions from {hint_source}, which apply to files under {dir}:\n<subdirectory-hints dir=\"{dir}\">\n{}\n</subdirectory-hints>",
        content.trim()
    )
}

pub fn conversation_contains_marker(conversation: &Conversation, marker: &str) -> bool {
    for message in conversation.messages() {
        if !message.is_agent_visible() {
            continue;
        }
        for content in &message.content {
            match content {
                MessageContentBlock::Text(t) if t.text.contains(marker) => return true,
                MessageContentBlock::ToolResponse(r) => {
                    if let Ok(res) = &r.tool_result {
                        for block in &res.content {
                            if let rmcp::model::ContentBlock::Text(t) = block {
                                if t.text.contains(marker) {
                                    return true;
                                }
                            }
                        }
                    }
                }
                MessageContentBlock::SystemNotification(n) if n.msg.contains(marker) => {
                    return true
                }
                MessageContentBlock::Error(e) if e.message.contains(marker) => return true,
                _ => {}
            }
        }
    }
    false
}

pub fn find_last_successful_tool_response(
    messages: &mut [Message],
) -> Option<&mut rmcp::model::CallToolResult> {
    for message in messages.iter_mut().rev() {
        for content in message.content.iter_mut().rev() {
            if let MessageContentBlock::ToolResponse(response) = content {
                if let Ok(result) = &mut response.tool_result {
                    if result.is_error != Some(true) {
                        return Some(result);
                    }
                }
            }
        }
    }
    None
}

pub fn deliver_subdirectory_hints(
    messages_to_add: &mut Conversation,
    hints: Vec<(String, String)>,
    existing: &Conversation,
) -> Vec<(String, String)> {
    deliver_subdirectory_hints_recording(messages_to_add, hints, existing, None)
}

pub fn deliver_subdirectory_hints_recording(
    messages_to_add: &mut Conversation,
    hints: Vec<(String, String)>,
    existing: &Conversation,
    delivered_tracker: Option<&mut IndexMap<String, String>>,
) -> Vec<(String, String)> {
    if hints.is_empty() {
        return Vec::new();
    }

    let mut hints = hints;
    hints.sort_by(|a, b| {
        let dir_a = hint_dir_from_key(&a.0);
        let dir_b = hint_dir_from_key(&b.0);
        dir_a.cmp(dir_b)
    });

    let mut candidate_hints = Vec::new();
    for (key, content) in hints {
        let dir = hint_dir_from_key(&key);
        let marker_open = format_marker_open(dir);
        if conversation_contains_marker(existing, &marker_open)
            || conversation_contains_marker(messages_to_add, &marker_open)
        {
            continue;
        }
        candidate_hints.push((key, content));
    }
    candidate_hints.dedup_by(|a, b| hint_dir_from_key(&a.0) == hint_dir_from_key(&b.0));

    if candidate_hints.is_empty() {
        return Vec::new();
    }

    if let Some(tool_result) = find_last_successful_tool_response(messages_to_add.messages_mut()) {
        let formatted_hints: Vec<String> = candidate_hints
            .iter()
            .map(|(key, content)| {
                let dir = hint_dir_from_key(key);
                format_subdirectory_hint(dir, content)
            })
            .collect();
        let combined = formatted_hints.join("\n\n");
        let hint_block = rmcp::model::ContentBlock::Text(
            rmcp::model::TextContent::new(combined).with_annotations(
                rmcp::model::Annotations::default()
                    .with_audience(vec![rmcp::model::Role::Assistant]),
            ),
        );
        tool_result.content.push(hint_block);

        if let Some(tracker) = delivered_tracker {
            for (key, content) in &candidate_hints {
                tracker.insert(key.clone(), content.clone());
            }
        }

        Vec::new()
    } else {
        candidate_hints
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;

    use super::*;

    #[test]
    fn test_build_system_prompt_sanitizes_override() {
        let mut manager = PromptManager::new();
        let malicious_override = "System prompt\u{E0041}\u{E0042}\u{E0043}with hidden text";
        manager.set_system_prompt_override(malicious_override.to_string());

        let result = manager.builder().build();

        assert!(!result.contains('\u{E0041}'));
        assert!(!result.contains('\u{E0042}'));
        assert!(!result.contains('\u{E0043}'));
        assert!(result.contains("System prompt"));
        assert!(result.contains("with hidden text"));
    }

    #[test]
    fn test_current_date_time_includes_timezone() {
        let mut manager =
            PromptManager::with_timestamp(DateTime::<Utc>::from_timestamp(0, 0).unwrap());
        manager.set_system_prompt_override("It is currently {{current_date_time}}".to_string());

        let result = manager.builder().build();

        assert_eq!(result, "It is currently 1970-01-01 00:00:00 +00:00");
    }

    #[test]
    fn test_build_system_prompt_sanitizes_extras() {
        let mut manager = PromptManager::new();
        let malicious_extra = "Extra instruction\u{E0041}\u{E0042}\u{E0043}hidden";
        manager.add_system_prompt_extra("test".to_string(), malicious_extra.to_string());

        let result = manager.builder().build();

        assert!(!result.contains('\u{E0041}'));
        assert!(!result.contains('\u{E0042}'));
        assert!(!result.contains('\u{E0043}'));
        assert!(result.contains("Extra instruction"));
        assert!(result.contains("hidden"));
    }

    #[test]
    fn prompt_contributions_are_not_retained() {
        let manager = PromptManager::new();

        let with_contribution = manager
            .builder()
            .with_prompt_extras([("operation".to_string(), "temporary instruction".to_string())])
            .build();
        let without_contribution = manager.builder().build();

        assert!(with_contribution.contains("temporary instruction"));
        assert!(!without_contribution.contains("temporary instruction"));
    }

    #[test]
    fn composed_prompt_uses_contributions_instead_of_the_extension_catalog() {
        let mut manager = PromptManager::new();
        let working_dir = tempfile::tempdir().unwrap();

        let prompt = manager.build_system_prompt(
            working_dir.path(),
            vec![(
                "extensions".to_string(),
                "# Extensions\n\n## developer".to_string(),
            )],
            GooseMode::Auto,
        );

        assert!(prompt.contains("## developer"));
        assert!(!prompt.contains("No extensions are defined"));
    }

    #[test]
    fn project_git_metadata_does_not_reach_system_prompt() {
        let project = tempfile::tempdir().unwrap();
        std::fs::create_dir(project.path().join(".git")).unwrap();
        std::fs::create_dir(project.path().join("docs")).unwrap();
        std::fs::write(
            project.path().join(".git/config"),
            "url = https://oauth2:PROMPT_SECRET@example.invalid/repo.git",
        )
        .unwrap();
        std::fs::write(
            project.path().join("docs/config.md"),
            "legitimate project configuration",
        )
        .unwrap();
        std::fs::write(
            project.path().join(crate::hints::AGENTS_MD_FILENAME),
            "project instructions\n@.git/config\n@docs/config.md",
        )
        .unwrap();
        let ignore_patterns = build_gitignore(project.path());
        let hints = load_hint_files(
            project.path(),
            &[crate::hints::AGENTS_MD_FILENAME.to_string()],
            &ignore_patterns,
        );

        let prompt = PromptManager::new()
            .builder()
            .with_prompt_extras([("hints".to_string(), hints)])
            .build();

        assert!(prompt.contains("project instructions"));
        assert!(prompt.contains("legitimate project configuration"));
        assert!(!prompt.contains("PROMPT_SECRET"));
    }

    #[test]
    fn test_build_system_prompt_sanitizes_multiple_extras() {
        let mut manager = PromptManager::new();
        manager
            .add_system_prompt_extra("test1".to_string(), "First\u{E0041}instruction".to_string());
        manager.add_system_prompt_extra(
            "test2".to_string(),
            "Second\u{E0042}instruction".to_string(),
        );
        manager
            .add_system_prompt_extra("test3".to_string(), "Third\u{E0043}instruction".to_string());

        let result = manager.builder().build();

        assert!(!result.contains('\u{E0041}'));
        assert!(!result.contains('\u{E0042}'));
        assert!(!result.contains('\u{E0043}'));
        assert!(result.contains("Firstinstruction"));
        assert!(result.contains("Secondinstruction"));
        assert!(result.contains("Thirdinstruction"));
    }

    #[test]
    fn test_remove_system_prompt_extra() {
        let mut manager = PromptManager::new();
        manager.add_system_prompt_extra("agent".to_string(), "Agent instruction".to_string());
        manager.add_system_prompt_extra("project".to_string(), "Project instruction".to_string());

        manager.remove_system_prompt_extra("agent");
        let result = manager.builder().build();

        assert!(!result.contains("Agent instruction"));
        assert!(result.contains("Project instruction"));
    }

    #[test]
    fn test_clear_system_prompt_override() {
        let mut manager = PromptManager::new();
        manager.set_system_prompt_override("Replacement prompt".to_string());
        assert!(manager.builder().build().contains("Replacement prompt"));

        manager.clear_system_prompt_override();
        assert!(!manager.builder().build().contains("Replacement prompt"));
    }

    #[test]
    fn test_build_system_prompt_preserves_legitimate_unicode_in_extras() {
        let mut manager = PromptManager::new();
        let legitimate_unicode = "Instruction with 世界 and 🌍 emojis";
        manager.add_system_prompt_extra("test".to_string(), legitimate_unicode.to_string());

        let result = manager.builder().build();

        assert!(result.contains("世界"));
        assert!(result.contains("🌍"));
        assert!(result.contains("Instruction with"));
        assert!(result.contains("emojis"));
    }

    #[test]
    fn test_build_system_prompt_sanitizes_extension_instructions() {
        let manager = PromptManager::new();
        let malicious_extension_info = ExtensionInfo::new(
            "test_extension",
            "Extension help\u{E0041}\u{E0042}\u{E0043}hidden instructions",
            false,
        );

        let result = manager
            .builder()
            .with_extension(malicious_extension_info)
            .build();

        assert!(!result.contains('\u{E0041}'));
        assert!(!result.contains('\u{E0042}'));
        assert!(!result.contains('\u{E0043}'));
        assert!(result.contains("Extension help"));
        assert!(result.contains("hidden instructions"));
    }

    #[test]
    fn test_basic() {
        let manager = PromptManager::with_timestamp(DateTime::<Utc>::from_timestamp(0, 0).unwrap());

        let system_prompt = manager.builder().build();

        assert_snapshot!(system_prompt)
    }

    #[test]
    fn test_one_extension() {
        let manager = PromptManager::with_timestamp(DateTime::<Utc>::from_timestamp(0, 0).unwrap());

        let system_prompt = manager
            .builder()
            .with_extension(ExtensionInfo::new(
                "test",
                "how to use this extension",
                true,
            ))
            .build();

        assert_snapshot!(system_prompt)
    }

    #[test]
    fn test_typical_setup() {
        let manager = PromptManager::with_timestamp(DateTime::<Utc>::from_timestamp(0, 0).unwrap());

        let system_prompt = manager
            .builder()
            .with_extension(ExtensionInfo::new(
                "extension_A",
                "<instructions on how to use extension A>",
                true,
            ))
            .with_extension(ExtensionInfo::new(
                "extension_B",
                "<instructions on how to use extension B (no resources)>",
                false,
            ))
            .build();

        assert_snapshot!(system_prompt)
    }

    #[tokio::test]
    async fn test_all_platform_extensions() {
        use crate::agents::platform_extensions::{PlatformExtensionContext, PLATFORM_EXTENSIONS};
        use crate::config::GooseMode;
        use crate::session::SessionManager;
        use std::sync::Arc;

        let tmp_dir = tempfile::tempdir().unwrap();
        let temp_root = tmp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([
            ("HOME", Some(temp_root.as_str())),
            ("GOOSE_PATH_ROOT", Some(temp_root.as_str())),
        ]);
        let session_manager = Arc::new(SessionManager::new(tmp_dir.path().to_path_buf()));
        let session = session_manager
            .create_session(
                tmp_dir.path().to_path_buf(),
                "test session".to_owned(),
                crate::session::SessionType::Hidden,
                GooseMode::default(),
            )
            .await
            .unwrap();
        let scheduler = crate::scheduler::Scheduler::new(
            tmp_dir.path().join("schedules"),
            session_manager.clone(),
        )
        .await
        .unwrap();
        let context = PlatformExtensionContext {
            extension_manager: None,
            provider: Arc::new(tokio::sync::Mutex::new(None)),
            session_manager,
            scheduler: Some(scheduler),
            use_login_shell_path: false,
        };

        let mut extensions = Vec::new();
        for def in PLATFORM_EXTENSIONS.values() {
            let Some(client) = (def.client_factory)(context.clone()) else {
                continue;
            };
            let instructions = client
                .get_instructions(&session.id, &session.working_dir)
                .await
                .unwrap_or_default();
            let has_resources = client
                .get_info()
                .and_then(|i| i.capabilities.resources.as_ref())
                .is_some();
            extensions.push(ExtensionInfo::new(def.name, &instructions, has_resources));
        }

        extensions.sort_by(|a, b| a.name.cmp(&b.name));

        let manager = PromptManager::with_timestamp(DateTime::<Utc>::from_timestamp(0, 0).unwrap());
        let system_prompt = manager
            .builder()
            .with_extensions(extensions.into_iter())
            .build();

        assert_snapshot!(system_prompt);
    }

    #[test]
    fn test_subdirectory_hints_not_in_system_prompt_and_byte_identical() {
        let temp_dir = tempfile::tempdir().unwrap();
        let working_dir = temp_dir.path();
        let sub_dir = working_dir.join("sub");
        std::fs::create_dir(&sub_dir).unwrap();
        std::fs::write(sub_dir.join("AGENTS.md"), "SUBDIR_HINT_SECRET_INSTRUCTION").unwrap();

        let mut manager =
            PromptManager::with_timestamp(DateTime::<Utc>::from_timestamp(0, 0).unwrap());

        let before = manager.build_system_prompt(working_dir, vec![], GooseMode::Auto);
        assert!(!before.contains("SUBDIR_HINT_SECRET_INSTRUCTION"));

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            serde_json::Value::String(sub_dir.join("file.txt").to_str().unwrap().to_string()),
        );
        manager.record_tool_arguments(&Some(args), working_dir);

        let after = manager.build_system_prompt(working_dir, vec![], GooseMode::Auto);
        assert!(
            !after.contains("SUBDIR_HINT_SECRET_INSTRUCTION"),
            "system prompt must not contain subdirectory hints"
        );
        assert_eq!(
            before, after,
            "system prompt must be byte-identical before and after touching subdirectory"
        );
    }

    #[test]
    fn test_deliver_subdirectory_hints_appends_to_last_successful_tool_response() {
        use goose_providers::conversation::message::{Message, MessageContent};
        use rmcp::model::{CallToolResult, ContentBlock};

        let mut messages_to_add = Conversation::default();
        messages_to_add.push(Message::assistant());
        messages_to_add.push(Message::user().with_tool_response(
            "call_1",
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "result 1",
            )])),
        ));
        messages_to_add.push(Message::assistant());
        messages_to_add.push(Message::user().with_tool_response(
            "call_2",
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "result 2",
            )])),
        ));

        let hints = vec![(
            "subdir_hints:/repo/sub".to_string(),
            "Hint content".to_string(),
        )];
        let existing = Conversation::default();

        let undelivered = deliver_subdirectory_hints(&mut messages_to_add, hints, &existing);
        assert!(undelivered.is_empty(), "all hints should be delivered");

        // Verify first tool response is untouched
        let first_resp = &messages_to_add.messages()[1];
        let MessageContent::ToolResponse(first_tr) = &first_resp.content[0] else {
            panic!()
        };
        let first_res = first_tr.tool_result.as_ref().unwrap();
        assert_eq!(first_res.content.len(), 1);

        // Verify second tool response has the hint appended as extra text item
        let second_resp = &messages_to_add.messages()[3];
        let MessageContent::ToolResponse(second_tr) = &second_resp.content[0] else {
            panic!()
        };
        let second_res = second_tr.tool_result.as_ref().unwrap();
        assert_eq!(second_res.content.len(), 2);
        let ContentBlock::Text(delivered) = &second_res.content[1] else {
            panic!()
        };
        assert!(delivered
            .text
            .contains("<subdirectory-hints dir=\"/repo/sub\">"));
        assert!(delivered.text.contains("Hint content"));
        assert!(delivered.text.contains("</subdirectory-hints>"));
        assert!(delivered
            .text
            .contains("Repository instructions from /repo/sub/AGENTS.md, which apply to files under /repo/sub:"));
    }

    #[test]
    fn test_deliver_subdirectory_hints_returns_undelivered_when_no_successful_tool_response() {
        use goose_providers::conversation::message::Message;
        use rmcp::model::ErrorData;

        let mut messages_to_add = Conversation::default();
        messages_to_add.push(Message::assistant());
        messages_to_add.push(Message::user().with_tool_response(
            "call_err",
            Err(ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                "failed",
                None,
            )),
        ));

        let hints = vec![(
            "subdir_hints:/repo/sub".to_string(),
            "Hint content".to_string(),
        )];
        let existing = Conversation::default();

        let undelivered =
            deliver_subdirectory_hints(&mut messages_to_add, hints.clone(), &existing);
        assert_eq!(
            undelivered, hints,
            "hints should remain undelivered when tool errored"
        );
    }

    #[test]
    fn test_deliver_subdirectory_hints_skips_when_marker_exists_in_existing() {
        use goose_providers::conversation::message::{Message, MessageContent};
        use rmcp::model::{CallToolResult, ContentBlock};

        let mut existing = Conversation::default();
        existing.push(Message::user().with_tool_response(
            "call_old",
            Ok(CallToolResult::success(vec![
                ContentBlock::text("old result"),
                ContentBlock::text("Directory-specific instructions for files under /repo/sub:\n<subdirectory-hints dir=\"/repo/sub\">\nHint content\n</subdirectory-hints>"),
            ])),
        ));

        let mut messages_to_add = Conversation::default();
        messages_to_add.push(Message::assistant());
        messages_to_add.push(Message::user().with_tool_response(
            "call_new",
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "new result",
            )])),
        ));

        let hints = vec![(
            "subdir_hints:/repo/sub".to_string(),
            "Hint content".to_string(),
        )];
        let undelivered = deliver_subdirectory_hints(&mut messages_to_add, hints, &existing);
        assert!(
            undelivered.is_empty(),
            "skipped hints should not be returned as undelivered"
        );

        // Verify new tool response was NOT modified (hint was skipped)
        let new_resp = &messages_to_add.messages()[1];
        let MessageContent::ToolResponse(tr) = &new_resp.content[0] else {
            panic!()
        };
        let res = tr.tool_result.as_ref().unwrap();
        assert_eq!(res.content.len(), 1);
    }

    #[test]
    fn test_deliver_subdirectory_hints_mixed_success_and_error() {
        use goose_providers::conversation::message::{Message, MessageContent};
        use rmcp::model::{CallToolResult, ContentBlock, ErrorData};

        let mut messages_to_add = Conversation::default();
        // Tool 1: success
        messages_to_add.push(Message::assistant());
        messages_to_add.push(Message::user().with_tool_response(
            "call_1",
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "result 1",
            )])),
        ));
        // Tool 2: error (last tool response in batch)
        messages_to_add.push(Message::assistant());
        messages_to_add.push(Message::user().with_tool_response(
            "call_2",
            Err(ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                "failed",
                None,
            )),
        ));

        let hints = vec![(
            "subdir_hints:/repo/sub".to_string(),
            "Hint content".to_string(),
        )];
        let existing = Conversation::default();

        let undelivered = deliver_subdirectory_hints(&mut messages_to_add, hints, &existing);
        assert!(undelivered.is_empty());

        // Verify hints were appended to Tool 1 (the last SUCCESSFUL tool response)
        let first_resp = &messages_to_add.messages()[1];
        let MessageContent::ToolResponse(first_tr) = &first_resp.content[0] else {
            panic!()
        };
        let first_res = first_tr.tool_result.as_ref().unwrap();
        assert_eq!(first_res.content.len(), 2);
    }

    #[test]
    fn test_deliver_subdirectory_hints_multiple_sorted() {
        use goose_providers::conversation::message::{Message, MessageContent};
        use rmcp::model::{CallToolResult, ContentBlock};

        let mut messages_to_add = Conversation::default();
        messages_to_add.push(Message::user().with_tool_response(
            "call_1",
            Ok(CallToolResult::success(vec![ContentBlock::text("result")])),
        ));

        let hints = vec![
            (
                "subdir_hints:/repo/z_sub".to_string(),
                "Z content".to_string(),
            ),
            (
                "subdir_hints:/repo/a_sub".to_string(),
                "A content".to_string(),
            ),
        ];
        let existing = Conversation::default();

        let undelivered = deliver_subdirectory_hints(&mut messages_to_add, hints, &existing);
        assert!(undelivered.is_empty());

        let resp = &messages_to_add.messages()[0];
        let MessageContent::ToolResponse(tr) = &resp.content[0] else {
            panic!()
        };
        let res = tr.tool_result.as_ref().unwrap();
        assert_eq!(res.content.len(), 2);
        let ContentBlock::Text(delivered) = &res.content[1] else {
            panic!()
        };
        let pos_a = delivered.text.find("/repo/a_sub").unwrap();
        let pos_z = delivered.text.find("/repo/z_sub").unwrap();
        assert!(pos_a < pos_z, "hints should be sorted deterministically");
    }

    #[test]
    fn test_deliver_subdirectory_hints_not_suppressed_by_non_agent_visible_message() {
        use goose_providers::conversation::message::{Message, MessageContent};
        use rmcp::model::{CallToolResult, ContentBlock};

        let mut existing = Conversation::default();
        // A compacted/hidden message that contains the marker but has agent_visible = false
        existing.push(
            Message::user()
                .with_tool_response(
                    "call_old",
                    Ok(CallToolResult::success(vec![
                        ContentBlock::text("old result"),
                        ContentBlock::text("<subdirectory-hints dir=\"/repo/sub\">\nold hint\n</subdirectory-hints>"),
                    ])),
                )
                .with_visibility(true, false),
        );

        let mut messages_to_add = Conversation::default();
        messages_to_add.push(Message::assistant());
        messages_to_add.push(Message::user().with_tool_response(
            "call_new",
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "new result",
            )])),
        ));

        let hints = vec![(
            "subdir_hints:/repo/sub".to_string(),
            "Sub instruction".to_string(),
        )];
        let undelivered = deliver_subdirectory_hints(&mut messages_to_add, hints, &existing);
        assert!(
            undelivered.is_empty(),
            "hint should be delivered because existing marker is not agent-visible"
        );

        let new_resp = &messages_to_add.messages()[1];
        let MessageContent::ToolResponse(tr) = &new_resp.content[0] else {
            panic!()
        };
        let res = tr.tool_result.as_ref().unwrap();
        assert_eq!(
            res.content.len(),
            2,
            "new tool response must have the hint delivered"
        );
    }

    #[test]
    fn test_requeue_after_history_replacement_moves_delivered_to_pending() {
        let mut manager = PromptManager::new();
        manager.mark_hint_delivered(
            "subdir_hints:/repo/sub".to_string(),
            "Sub instruction".to_string(),
        );
        assert!(manager.pending_subdirectory_hints().is_empty());
        assert_eq!(manager.delivered_subdirectory_hints().len(), 1);

        manager.requeue_after_history_replacement();
        assert!(manager.delivered_subdirectory_hints().is_empty());
        assert_eq!(manager.pending_subdirectory_hints().len(), 1);
        assert_eq!(
            manager.pending_subdirectory_hints()[0].0,
            "subdir_hints:/repo/sub"
        );
    }

    #[test]
    fn test_deliver_subdirectory_hints_audience_projection() {
        use goose_providers::conversation::message::{Message, MessageContent};
        use rmcp::model::{CallToolResult, ContentBlock};

        let mut messages_to_add = Conversation::default();
        messages_to_add.push(Message::user().with_tool_response(
            "call_1",
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "tool output",
            )])),
        ));

        let hints = vec![(
            "subdir_hints:/repo/sub".to_string(),
            "Sub instruction".to_string(),
        )];
        let existing = Conversation::default();

        let undelivered = deliver_subdirectory_hints(&mut messages_to_add, hints, &existing);
        assert!(undelivered.is_empty());

        let msg = &messages_to_add.messages()[0];

        // 1. Survives agent_visible_content()
        let agent_msg = msg.agent_visible_content();
        let MessageContent::ToolResponse(ref tr) = agent_msg.content[0] else {
            panic!()
        };
        let res = tr.tool_result.as_ref().unwrap();
        assert_eq!(
            res.content.len(),
            2,
            "agent_visible_content must retain the hint"
        );
        assert!(res.content[1]
            .as_text()
            .unwrap()
            .text
            .contains("<subdirectory-hints"));

        // 2. Excluded from user_visible_content()
        let user_msg = msg.user_visible_content();
        let MessageContent::ToolResponse(ref tr) = user_msg.content[0] else {
            panic!()
        };
        let res = tr.tool_result.as_ref().unwrap();
        assert_eq!(
            res.content.len(),
            1,
            "user_visible_content must exclude the hint"
        );
        assert_eq!(res.content[0].as_text().unwrap().text, "tool output");
    }
}
