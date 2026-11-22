// Copyright 2026 Jonathan Mohrbacher. Licensed under the Apache License, Version 2.0.
// Fork-only module: subagent policy.
// - GOOSE_SUBAGENT_ROLES_DIR (~ expanded) is scanned before every other source, so a repo's
//   `.claude/agents/<name>.md` or `.goose/recipes/<name>.yaml` cannot shadow a role.
// - GOOSE_SUBAGENT_DEFAULT_RECIPE (~ expanded) gives a delegate whose recipe declares no
//   extensions that recipe's extensions and instructions instead of the parent's, failing closed.
// - With the default recipe set, an extension a delegate was given but could not start fails
//   the delegate, so a reviewer whose shell failed cannot run on without it and report anyway.
//   A declared platform extension this goose does not have (e.g. `skills`, removed in 1.60) is
//   dropped at creation by resolve_extensions_for_new_session, as upstream does; it is not a
//   start failure.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use goose_sdk_types::custom_requests::{SourceEntry, SourceType};

use super::scan_recipes_from_dir;
use crate::agents::StateMachineServices;
use crate::config::Config;
use crate::recipe::{Recipe, RECIPE_FILE_EXTENSIONS};

fn config_path(key: &str) -> Option<String> {
    Config::global()
        .get_param::<String>(key)
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| shellexpand::tilde(&s).into_owned())
}

fn roles_dir() -> Option<PathBuf> {
    config_path("GOOSE_SUBAGENT_ROLES_DIR").map(PathBuf::from)
}

fn default_recipe_path() -> Option<String> {
    config_path("GOOSE_SUBAGENT_DEFAULT_RECIPE")
}

pub(crate) fn is_active() -> bool {
    default_recipe_path().is_some()
}

/// Every recipe file name in the roles dir is claimed even when it fails to parse: a broken
/// role must fail closed rather than let a same-named local source take its place.
pub(super) fn scan_roles_dir(sources: &mut Vec<SourceEntry>, seen: &mut HashSet<String>) {
    let Some(dir) = roles_dir() else {
        return;
    };
    if !dir.is_dir() {
        tracing::warn!(
            "GOOSE_SUBAGENT_ROLES_DIR: path '{}' is not a directory",
            dir.display()
        );
        return;
    }
    scan_recipes_from_dir(&dir, SourceType::Recipe, false, sources, seen);
    seen.extend(recipe_file_stems(&dir));
}

fn recipe_file_stems(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| RECIPE_FILE_EXTENSIONS.contains(&ext))
        })
        .filter_map(|path| path.file_stem()?.to_str().map(str::to_string))
        .filter(|name| !name.is_empty())
        .collect()
}

pub(super) fn work_guidance() -> &'static str {
    if is_active() {
        "Delegates without a source (and agent .md sources that declare no extensions) are read-only.\n\
         Write work must go to a source whose recipe grants write tools (e.g. an implementer role)."
    } else {
        "Work (writes): partition files strictly - no two delegates touch the same file."
    }
}

/// A delegate recipe that declares no extensions takes the default recipe's extensions and
/// has its instructions prepended. A set-but-broken default is an error, never a silent
/// fallback to the parent's tools.
pub(super) fn apply_unscoped_defaults(recipe: &mut Recipe) -> Result<()> {
    if recipe.extensions.is_some() {
        return Ok(());
    }
    let Some(path) = default_recipe_path() else {
        return Ok(());
    };
    let content = std::fs::read_to_string(&path)
        .map_err(|e| anyhow!("GOOSE_SUBAGENT_DEFAULT_RECIPE: cannot read {path}: {e}"))?;
    let default = Recipe::from_content(&content)
        .map_err(|e| anyhow!("GOOSE_SUBAGENT_DEFAULT_RECIPE: {path} is not a valid recipe: {e}"))?;
    let extensions = default
        .extensions
        .ok_or_else(|| anyhow!("GOOSE_SUBAGENT_DEFAULT_RECIPE: {path} declares no extensions"))?;
    recipe.extensions = Some(extensions);
    if let Some(default_instructions) = default.instructions.filter(|s| !s.is_empty()) {
        recipe.instructions = Some(match recipe.instructions.take() {
            Some(existing) if !existing.is_empty() => {
                format!("{default_instructions}\n\n{existing}")
            }
            _ => default_instructions,
        });
    }
    Ok(())
}

/// Starts the subagent's saved extension selection and fails if any of it did not come up.
pub(crate) async fn require_started_extensions(
    services: &StateMachineServices,
    subagent_id: &str,
) -> Result<()> {
    let failures: Vec<String> = services
        .current_tools(subagent_id)
        .await?
        .start()
        .await
        .into_iter()
        .filter(|result| !result.success)
        .map(|result| {
            format!(
                "'{}': {}",
                result.name,
                result.error.as_deref().unwrap_or("failed to start")
            )
        })
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "Failed to start extensions for subagent: {}",
            failures.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{discover_filesystem_sources, DelegateParams, SummonClient};
    use super::*;
    use crate::agents::extension::PlatformExtensionContext;
    use crate::agents::extension_config::ExtensionConfig;
    use crate::session::extension_data::{EnabledExtensionsState, ExtensionState};
    use crate::session::SessionType;
    use serial_test::serial;
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn leak(s: &str) -> &'static str {
        Box::leak(s.to_string().into_boxed_str())
    }

    fn platform(name: &str, tools: &[&str]) -> ExtensionConfig {
        ExtensionConfig::Platform {
            name: name.to_string(),
            description: String::new(),
            display_name: None,
            bundled: None,
            available_tools: tools.iter().map(|t| t.to_string()).collect(),
        }
    }

    fn names(exts: &[ExtensionConfig]) -> Vec<String> {
        exts.iter().map(|e| e.name()).collect()
    }

    fn write_default(dir: &TempDir, body: &str) -> &'static str {
        let p = dir.path().join("unscoped.yaml");
        fs::write(&p, body).unwrap();
        leak(p.to_str().unwrap())
    }

    fn recipe(extensions: Option<Vec<ExtensionConfig>>, instructions: Option<&str>) -> Recipe {
        Recipe {
            version: "1.0.0".to_string(),
            title: String::new(),
            description: String::new(),
            instructions: instructions.map(str::to_string),
            prompt: None,
            extensions,
            settings: None,
            activities: None,
            author: None,
            parameters: None,
            response: None,
            sub_recipes: None,
            retry: None,
        }
    }

    /// The delegate path in create_subagent: defaults applied, then upstream resolution.
    async fn resolve(
        declared: Option<Vec<ExtensionConfig>>,
        filter: Option<Vec<&str>>,
    ) -> Result<Vec<ExtensionConfig>> {
        let client = SummonClient::new(PlatformExtensionContext {
            providers: Default::default(),
            session_manager: Arc::new(crate::session::SessionManager::instance()),
            scheduler: None,
        })?;
        let mut parent = crate::session::Session {
            provider_name: Some("openai".to_string()),
            model_config: Some(goose_providers::model::ModelConfig::new("test-model")),
            ..Default::default()
        };
        EnabledExtensionsState::new(vec![platform("summon", &[]), platform("developer", &[])])
            .to_extension_data(&mut parent.extension_data)?;
        let mut recipe = recipe(declared, None);
        apply_unscoped_defaults(&mut recipe)?;
        let params = DelegateParams {
            extensions: filter.map(|f| f.into_iter().map(String::from).collect()),
            ..Default::default()
        };
        Ok(client
            .resolve_subagent_config(&params, &recipe, &parent)
            .await?
            .extensions)
    }

    fn provider_env() -> [(&'static str, Option<&'static str>); 2] {
        [
            ("GOOSE_SUBAGENT_PROVIDER", Some("openai")),
            ("GOOSE_SUBAGENT_MODEL", Some("test-model")),
        ]
    }

    #[test]
    #[serial]
    fn subagent_policy_roles_dir_wins_over_local_sources() {
        let roles = TempDir::new().unwrap();
        let wd = TempDir::new().unwrap();
        fs::write(
            roles.path().join("code-reviewer.yaml"),
            "version: 1.0.0\ntitle: Role\ndescription: global role\ninstructions: role body\n",
        )
        .unwrap();
        fs::create_dir_all(wd.path().join(".claude/agents")).unwrap();
        fs::write(
            wd.path().join(".claude/agents/cr.md"),
            "---\nname: code-reviewer\ndescription: local impostor\n---\nimpostor",
        )
        .unwrap();
        fs::create_dir_all(wd.path().join(".goose/recipes")).unwrap();
        fs::write(
            wd.path().join(".goose/recipes/code-reviewer.yaml"),
            "version: 1.0.0\ntitle: X\ndescription: local recipe impostor\ninstructions: x\n",
        )
        .unwrap();
        let _env = env_lock::lock_env([(
            "GOOSE_SUBAGENT_ROLES_DIR",
            Some(leak(roles.path().to_str().unwrap())),
        )]);

        let sources = discover_filesystem_sources(wd.path());
        let hits: Vec<_> = sources
            .iter()
            .filter(|s| s.name == "code-reviewer")
            .collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].description, "global role");
        assert!(hits[0]
            .path
            .starts_with(roles.path().canonicalize().unwrap().to_str().unwrap()));
    }

    #[test]
    #[serial]
    fn subagent_policy_roles_dir_unset_keeps_upstream_order() {
        let wd = TempDir::new().unwrap();
        fs::create_dir_all(wd.path().join(".claude/agents")).unwrap();
        fs::write(
            wd.path().join(".claude/agents/cr.md"),
            "---\nname: code-reviewer\ndescription: local agent\n---\nbody",
        )
        .unwrap();
        let _env = env_lock::lock_env([("GOOSE_SUBAGENT_ROLES_DIR", None::<&str>)]);
        let sources = discover_filesystem_sources(wd.path());
        let hit = sources.iter().find(|s| s.name == "code-reviewer").unwrap();
        assert_eq!(hit.description, "local agent");
    }

    #[test]
    #[serial]
    fn subagent_policy_roles_dir_tilde_expanded() {
        let fake_home = TempDir::new().unwrap();
        let roles_dir = fake_home.path().join("roles");
        fs::create_dir_all(&roles_dir).unwrap();
        fs::write(
            roles_dir.join("code-reviewer.yaml"),
            "version: 1.0.0\ntitle: Role\ndescription: global role\ninstructions: role body\n",
        )
        .unwrap();
        let wd = TempDir::new().unwrap();
        let _env = env_lock::lock_env([
            ("HOME", Some(leak(fake_home.path().to_str().unwrap()))),
            ("GOOSE_SUBAGENT_ROLES_DIR", Some("~/roles")),
        ]);

        let sources = discover_filesystem_sources(wd.path());
        let hit = sources.iter().find(|s| s.name == "code-reviewer").unwrap();
        assert_eq!(hit.description, "global role");
    }

    #[test]
    #[serial]
    fn subagent_policy_roles_dir_invalid_recipe_fails_closed_without_shadowing() {
        let roles = TempDir::new().unwrap();
        let wd = TempDir::new().unwrap();
        fs::write(
            roles.path().join("code-reviewer.yaml"),
            "not: [a, valid: recipe",
        )
        .unwrap();
        fs::create_dir_all(wd.path().join(".claude/agents")).unwrap();
        fs::write(
            wd.path().join(".claude/agents/cr.md"),
            "---\nname: code-reviewer\ndescription: local impostor\n---\nimpostor",
        )
        .unwrap();
        let _env = env_lock::lock_env([(
            "GOOSE_SUBAGENT_ROLES_DIR",
            Some(leak(roles.path().to_str().unwrap())),
        )]);

        let sources = discover_filesystem_sources(wd.path());
        let hit = sources.iter().find(|s| s.name == "code-reviewer");
        assert!(hit.is_none(), "broken role was shadowed by {hit:?}");
    }

    #[tokio::test]
    #[serial]
    async fn subagent_policy_declared_extensions_replace_parent() {
        let mut env = provider_env().to_vec();
        env.push(("GOOSE_SUBAGENT_DEFAULT_RECIPE", None));
        let _env = env_lock::lock_env(env);
        let declared = vec![
            platform("developer", &["tree", "read_image"]),
            platform("todo", &[]),
        ];
        let got = resolve(Some(declared.clone()), None).await.unwrap();
        assert_eq!(got, declared);
    }

    #[tokio::test]
    #[serial]
    async fn subagent_policy_filter_only_narrows_declared() {
        let mut env = provider_env().to_vec();
        env.push(("GOOSE_SUBAGENT_DEFAULT_RECIPE", None));
        let _env = env_lock::lock_env(env);
        let declared = vec![platform("developer", &["tree"]), platform("todo", &[])];
        let got = resolve(Some(declared), Some(vec!["developer", "summon"]))
            .await
            .unwrap();
        assert_eq!(got, vec![platform("developer", &["tree"])]);
    }

    #[tokio::test]
    #[serial]
    async fn subagent_policy_undeclared_inherits_parent_when_unconfigured() {
        let mut env = provider_env().to_vec();
        env.push(("GOOSE_SUBAGENT_DEFAULT_RECIPE", None));
        let _env = env_lock::lock_env(env);
        let got = resolve(None, None).await.unwrap();
        assert_eq!(names(&got), vec!["summon", "developer"]);
    }

    #[tokio::test]
    #[serial]
    async fn subagent_policy_unscoped_gets_default_recipe_extensions() {
        let d = TempDir::new().unwrap();
        let p = write_default(
            &d,
            "version: 1.0.0\ntitle: U\ndescription: U\ninstructions: u\nextensions:\n  - type: platform\n    name: developer\n    available_tools: [tree]\n",
        );
        let mut env = provider_env().to_vec();
        env.push(("GOOSE_SUBAGENT_DEFAULT_RECIPE", Some(p)));
        let _env = env_lock::lock_env(env);
        let got = resolve(None, None).await.unwrap();
        assert_eq!(got, vec![platform("developer", &["tree"])]);
    }

    #[tokio::test]
    #[serial]
    async fn subagent_policy_declared_recipe_ignores_default() {
        let d = TempDir::new().unwrap();
        let p = write_default(
            &d,
            "version: 1.0.0\ntitle: U\ndescription: U\ninstructions: u\nextensions:\n  - type: platform\n    name: todo\n",
        );
        let mut env = provider_env().to_vec();
        env.push(("GOOSE_SUBAGENT_DEFAULT_RECIPE", Some(p)));
        let _env = env_lock::lock_env(env);
        let declared = vec![platform("developer", &[])];
        let got = resolve(Some(declared.clone()), None).await.unwrap();
        assert_eq!(got, declared);
    }

    #[test]
    #[serial]
    fn subagent_policy_missing_default_recipe_fails_closed() {
        let _env = env_lock::lock_env([(
            "GOOSE_SUBAGENT_DEFAULT_RECIPE",
            Some("/nonexistent/unscoped.yaml"),
        )]);
        let err = apply_unscoped_defaults(&mut recipe(None, None)).unwrap_err();
        assert!(
            err.to_string().contains("GOOSE_SUBAGENT_DEFAULT_RECIPE"),
            "{err}"
        );
    }

    #[test]
    #[serial]
    fn subagent_policy_default_recipe_without_extensions_fails_closed() {
        let d = TempDir::new().unwrap();
        let p = write_default(
            &d,
            "version: 1.0.0\ntitle: U\ndescription: U\ninstructions: u\n",
        );
        let _env = env_lock::lock_env([("GOOSE_SUBAGENT_DEFAULT_RECIPE", Some(p))]);
        let err = apply_unscoped_defaults(&mut recipe(None, None)).unwrap_err();
        assert!(err.to_string().contains("declares no extensions"), "{err}");
    }

    #[test]
    #[serial]
    fn subagent_policy_default_recipe_tilde_expanded() {
        let fake_home = TempDir::new().unwrap();
        fs::write(
            fake_home.path().join("unscoped.yaml"),
            "version: 1.0.0\ntitle: U\ndescription: U\nextensions:\n  - type: platform\n    name: developer\n    available_tools: [tree]\n",
        )
        .unwrap();
        let _env = env_lock::lock_env([
            ("HOME", Some(leak(fake_home.path().to_str().unwrap()))),
            ("GOOSE_SUBAGENT_DEFAULT_RECIPE", Some("~/unscoped.yaml")),
        ]);
        let mut unscoped = recipe(None, None);
        apply_unscoped_defaults(&mut unscoped).unwrap();
        assert_eq!(
            unscoped.extensions,
            Some(vec![platform("developer", &["tree"])])
        );
    }

    #[test]
    #[serial]
    fn subagent_policy_unscoped_delegate_gets_default_recipe_instructions() {
        let d = TempDir::new().unwrap();
        let p = write_default(
            &d,
            "version: 1.0.0\ntitle: U\ndescription: U\ninstructions: READ-ONLY DELEGATE\nextensions:\n  - type: platform\n    name: developer\n",
        );
        let _env = env_lock::lock_env([("GOOSE_SUBAGENT_DEFAULT_RECIPE", Some(p))]);

        let mut adhoc = recipe(None, None);
        apply_unscoped_defaults(&mut adhoc).unwrap();
        assert_eq!(adhoc.instructions.as_deref(), Some("READ-ONLY DELEGATE"));

        let mut agent = recipe(None, Some("AGENT BODY"));
        apply_unscoped_defaults(&mut agent).unwrap();
        assert_eq!(
            agent.instructions.as_deref(),
            Some("READ-ONLY DELEGATE\n\nAGENT BODY")
        );

        let mut declared = recipe(Some(vec![platform("developer", &[])]), Some("CUSTOM"));
        apply_unscoped_defaults(&mut declared).unwrap();
        assert_eq!(declared.instructions.as_deref(), Some("CUSTOM"));
    }

    #[test]
    #[serial]
    fn subagent_policy_delegate_tool_description_reflects_default_recipe() {
        let client = SummonClient::new(PlatformExtensionContext {
            providers: Default::default(),
            session_manager: Arc::new(crate::session::SessionManager::instance()),
            scheduler: None,
        })
        .unwrap();
        let description = |client: &SummonClient| {
            client
                .create_delegate_tool()
                .description
                .as_deref()
                .unwrap_or_default()
                .to_string()
        };
        {
            let _env = env_lock::lock_env([("GOOSE_SUBAGENT_DEFAULT_RECIPE", None::<&str>)]);
            let desc = description(&client);
            assert!(desc.contains("Work (writes): partition files strictly"));
            assert!(!desc.contains("are read-only"));
        }
        {
            let d = TempDir::new().unwrap();
            let p = write_default(
                &d,
                "version: 1.0.0\ntitle: U\ndescription: U\nextensions:\n  - type: platform\n    name: developer\n",
            );
            let _env = env_lock::lock_env([("GOOSE_SUBAGENT_DEFAULT_RECIPE", Some(p))]);
            let desc = description(&client);
            assert!(!desc.contains("Work (writes): partition files strictly"));
            assert!(desc.contains(
                "Delegates without a source (and agent .md sources that declare no extensions) are read-only"
            ));
            assert!(desc.contains(
                "Write work must go to a source whose recipe grants write tools (e.g. an implementer role)"
            ));
        }
    }

    async fn subagent_with(
        services: &StateMachineServices,
        dir: &Path,
        extensions: Vec<ExtensionConfig>,
    ) -> String {
        let session = services
            .session_manager
            .create_session(
                dir.to_path_buf(),
                "delegate".to_string(),
                SessionType::SubAgent,
                crate::config::GooseMode::Auto,
            )
            .await
            .unwrap();
        let mut extension_data = crate::session::extension_data::ExtensionData::default();
        EnabledExtensionsState::new(extensions)
            .to_extension_data(&mut extension_data)
            .unwrap();
        services
            .session_manager
            .update(&session.id)
            .extension_data(extension_data)
            .apply()
            .await
            .unwrap();
        session.id
    }

    fn missing_stdio(name: &str) -> ExtensionConfig {
        ExtensionConfig::External(crate::agents::extension_config::TransportConfig::Stdio {
            name: name.into(),
            description: String::new(),
            cmd: "/nonexistent/ro-shell".into(),
            args: vec![],
            envs: Default::default(),
            env_keys: vec![],
            timeout: Some(5),
            cwd: None,
            bundled: None,
            available_tools: vec![],
        })
    }

    #[tokio::test]
    async fn subagent_policy_strict_start_accepts_extensions_that_start() {
        let dir = TempDir::new().unwrap();
        let services = StateMachineServices::for_tool_tests(dir.path().to_path_buf());
        let id = subagent_with(&services, dir.path(), vec![platform("todo", &[])]).await;
        require_started_extensions(&services, &id).await.unwrap();
    }

    #[tokio::test]
    async fn subagent_policy_strict_start_fails_on_an_extension_that_does_not_start() {
        let dir = TempDir::new().unwrap();
        let services = StateMachineServices::for_tool_tests(dir.path().to_path_buf());
        let id = subagent_with(
            &services,
            dir.path(),
            vec![platform("todo", &[]), missing_stdio("ro-shell")],
        )
        .await;
        let err = require_started_extensions(&services, &id)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Failed to start extensions for subagent") && err.contains("'ro-shell'"),
            "{err}"
        );
    }

    async fn run_delegate(extensions: Vec<ExtensionConfig>) -> String {
        use futures::StreamExt;
        use goose_agent::subagent::{SubagentOutcome, SubagentUpdate};

        let dir = TempDir::new().unwrap();
        let services = StateMachineServices::for_tool_tests(dir.path().to_path_buf());
        let parent = services
            .session_manager
            .create_session(
                dir.path().to_path_buf(),
                "parent".to_string(),
                SessionType::User,
                crate::config::GooseMode::Auto,
            )
            .await
            .unwrap();
        let id = subagent_with(&services, dir.path(), extensions).await;
        let mut delegate = recipe(Some(vec![]), Some("do it"));
        delegate.settings = Some(crate::recipe::Settings {
            goose_provider: None,
            goose_model: None,
            temperature: None,
            max_turns: Some(1),
            thinking_effort: None,
        });
        services
            .session_manager
            .update(&id)
            .recipe(Some(delegate))
            .provider_name("openai")
            .model_config(goose_providers::model::ModelConfig::new("test-model"))
            .parent_session_id(Some(parent.id.clone()))
            .apply()
            .await
            .unwrap();
        let runner =
            crate::agents::subagent_handler::ForegroundSubagentRunner::new(Arc::clone(&services));
        let subagent = runner.load(&id).await.unwrap();
        let mut updates = runner.spawn(subagent, tokio_util::sync::CancellationToken::new());
        while let Some(update) = updates.next().await {
            if let SubagentUpdate::Finished(outcome) = update {
                return match outcome {
                    SubagentOutcome::Failed(reason) => reason,
                    SubagentOutcome::Completed(output) => format!("completed: {output}"),
                    SubagentOutcome::Cancelled => "cancelled".to_string(),
                };
            }
        }
        panic!("delegate stream ended without an outcome");
    }

    #[tokio::test]
    #[serial]
    async fn subagent_policy_delegate_fails_when_its_extension_does_not_start() {
        let d = TempDir::new().unwrap();
        let p = write_default(
            &d,
            "version: 1.0.0\ntitle: U\ndescription: U\nextensions:\n  - type: platform\n    name: todo\n",
        );
        let _env = env_lock::lock_env([
            ("GOOSE_SUBAGENT_DEFAULT_RECIPE", Some(p)),
            ("OPENAI_API_KEY", Some("test-key")),
            ("OPENAI_HOST", Some("http://127.0.0.1:9")),
        ]);
        let reason = run_delegate(vec![missing_stdio("ro-shell")]).await;
        assert!(
            reason.contains("Failed to start extensions for subagent")
                && reason.contains("'ro-shell'"),
            "{reason}"
        );
    }

    #[tokio::test]
    #[serial]
    async fn subagent_policy_delegate_without_policy_runs_on_after_a_failed_extension() {
        let _env = env_lock::lock_env([
            ("GOOSE_SUBAGENT_DEFAULT_RECIPE", None),
            ("OPENAI_API_KEY", Some("test-key")),
            ("OPENAI_HOST", Some("http://127.0.0.1:9")),
        ]);
        let reason = run_delegate(vec![missing_stdio("ro-shell")]).await;
        assert!(
            !reason.contains("Failed to start extensions for subagent"),
            "{reason}"
        );
    }
}
