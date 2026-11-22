// Modified by Jonathan Mohrbacher, 2026: wrap SkillOperation to offer no tools when a session has zero extensions.
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use goose_agent::command::Command;
use goose_agent::operation::{not_applicable, Emitter, Operation, OperationResult};
use goose_agent::skills::SkillSource;
use goose_sdk_types::custom_requests::SourceEntry;
use rmcp::model::Tool;

use super::effects::GooseEffect;
use crate::conversation::Conversation;
use crate::session::Session;

pub(crate) const NO_SKILLS_HELP: &str = "No skills installed.\n\n\
    Skills are loaded from SKILL.md files in:\n  \
    - ~/.agents/skills/ (global)\n  \
    - ~/.agents/plugins/*/skills/ (installed plugins)\n  \
    - .agents/skills/ (in current project)\n";

pub struct SkillOperation<'a> {
    inner: goose_agent::skills::SkillOperation<'a, Session>,
}

impl<'a> SkillOperation<'a> {
    pub fn new(source: Arc<dyn SkillSource<Session> + 'a>) -> Self {
        Self {
            inner: goose_agent::skills::SkillOperation::new(source),
        }
    }

    pub fn with_no_skills_help(mut self, help: impl Into<String>) -> Self {
        self.inner = self.inner.with_no_skills_help(help);
        self
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for SkillOperation<'_> {
    fn name(&self) -> &'static str {
        Operation::<Session, GooseEffect>::name(&self.inner)
    }

    async fn commands(&self, session: &Session) -> Result<Vec<Command<'_, Session, GooseEffect>>> {
        if crate::session::session_has_zero_extensions(session) {
            return Ok(Vec::new());
        }
        self.inner.commands(session).await
    }

    async fn inference_tools(&self, session: &Session) -> Result<Vec<Tool>> {
        if crate::session::session_has_zero_extensions(session) {
            return Ok(Vec::new());
        }
        Operation::<Session, GooseEffect>::inference_tools(&self.inner, session).await
    }

    async fn prompt_parts(
        &self,
        session: &Session,
        conversation: &Conversation,
    ) -> Result<Vec<(String, String)>> {
        if crate::session::session_has_zero_extensions(session) {
            return Ok(Vec::new());
        }
        Operation::<Session, GooseEffect>::prompt_parts(&self.inner, session, conversation).await
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        if crate::session::session_has_zero_extensions(session) {
            return not_applicable();
        }
        self.inner.run(session, conversation, emit).await
    }
}

pub(crate) struct GooseSkillSource;

#[async_trait]
impl SkillSource<Session> for GooseSkillSource {
    async fn skills(&self, session: &Session) -> Result<Vec<SourceEntry>> {
        crate::skills::skills_for_session(&session.working_dir)
    }

    async fn read_supporting_file(
        &self,
        skill: &SourceEntry,
        relative_path: &Path,
    ) -> Result<String> {
        let skill_dir = Path::new(&skill.path).canonicalize()?;
        let name = format!(
            "{}/{}",
            skill.name,
            relative_path.to_string_lossy().replace('\\', "/")
        );
        Ok(crate::skills::load_supporting_file(
            &skill_dir,
            relative_path,
            &name,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goose_sdk_types::custom_requests::SourceType;
    use std::collections::HashMap;

    fn skill_at(skill_dir: &Path, file: &Path) -> SourceEntry {
        SourceEntry {
            source_type: SourceType::Skill,
            name: "test-skill".to_string(),
            description: String::new(),
            content: String::new(),
            path: skill_dir.to_string_lossy().into_owned(),
            global: false,
            writable: true,
            supporting_files: vec![file.to_string_lossy().into_owned()],
            properties: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn reads_nested_supporting_file() {
        let root = tempfile::tempdir().unwrap();
        let skill_dir = std::fs::canonicalize(root.path()).unwrap();
        std::fs::create_dir(skill_dir.join("nested")).unwrap();
        let file = skill_dir.join("nested/guide.md");
        std::fs::write(&file, "Nested guidance.").unwrap();

        let content = GooseSkillSource
            .read_supporting_file(&skill_at(&skill_dir, &file), Path::new("nested/guide.md"))
            .await
            .unwrap();

        assert!(content.contains("Nested guidance."));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn reads_supporting_file_under_symlinked_directory() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tempdir.path()).unwrap();
        let references = root.join("real-skills/test-skill/references");
        std::fs::create_dir_all(&references).unwrap();
        std::fs::write(references.join("guide.md"), "Symlinked guidance.").unwrap();
        std::os::unix::fs::symlink(root.join("real-skills"), root.join("linked-skills")).unwrap();
        let linked_skill_dir = root.join("linked-skills/test-skill");

        let content = GooseSkillSource
            .read_supporting_file(
                &skill_at(
                    &linked_skill_dir,
                    &linked_skill_dir.join("references/guide.md"),
                ),
                Path::new("references/guide.md"),
            )
            .await
            .unwrap();

        assert!(content.contains("Symlinked guidance."));
    }

    #[tokio::test]
    async fn skill_operation_offers_no_tools_or_prompts_when_zero_extensions() {
        use crate::session::extension_data::{
            EnabledExtensionsState, ExtensionData, ExtensionState,
        };

        let op = SkillOperation::new(Arc::new(GooseSkillSource));

        // Session with zero extensions
        let mut extension_data = ExtensionData::default();
        EnabledExtensionsState::new(Vec::new())
            .to_extension_data(&mut extension_data)
            .unwrap();
        let session = Session {
            extension_data,
            ..Default::default()
        };

        assert!(op.inference_tools(&session).await.unwrap().is_empty());
        assert!(op
            .prompt_parts(&session, &Conversation::default())
            .await
            .unwrap()
            .is_empty());
        assert!(op.commands(&session).await.unwrap().is_empty());

        // Default session (extensions not explicitly emptied)
        let default_session = Session::default();
        let tools = op.inference_tools(&default_session).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "load_skill");
    }
}
