// Modified by Jonathan Mohrbacher, 2026: test _goose/unstable/session/settings/set custom method and session/prompt maxTurns; rolling memory settings test; self-compaction over ACP test; reload excludes self_compact_and_resume for recipe response schema; session/prompt reports a failed turn in _meta.goose.turnError.
#![recursion_limit = "256"]
#[allow(dead_code)]
#[path = "acp_common_tests/mod.rs"]
mod common_tests;
use agent_client_protocol::schema::v1::{
    ContentBlock, ListSessionsRequest, ListSessionsResponse, McpServer, McpServerHttp,
    NewSessionRequest, PromptRequest, SessionConfigKind, SessionConfigOptionCategory,
    SessionConfigOptionValue, SessionInfo, SessionUpdate, SetSessionConfigOptionRequest,
    StopReason, TextContent,
};
use agent_client_protocol::ErrorCode;
use common_tests::fixtures::server::{
    assert_session_response_precedes_available_commands, AcpServerConnection,
};
use common_tests::fixtures::{
    run_test, send_custom, spawn_acp_server_in_process, Connection, OpenAiFixture,
    PermissionDecision, Session, SessionData, TestConnectionConfig,
};
#[cfg(feature = "code-mode")]
use common_tests::run_prompt_codemode;
use common_tests::{
    run_close_session, run_config_mcp, run_config_option_mode_set, run_config_option_model_set,
    run_delete_session, run_fs_read_text_file_true, run_fs_write_text_file_false,
    run_fs_write_text_file_true, run_initialize_doesnt_hit_provider, run_list_sessions,
    run_load_mode, run_load_model, run_load_session_error, run_load_session_mcp,
    run_load_session_replays_image_attachment, run_mode_set, run_model_list, run_model_set,
    run_model_set_error_session_not_found, run_new_session_returns_initial_config,
    run_new_session_uses_current_config_mode, run_permission_persistence, run_prompt_basic,
    run_prompt_error, run_prompt_image, run_prompt_image_attachment, run_prompt_mcp,
    run_prompt_model_mismatch, run_prompt_skill, run_session_name_update_notification,
    run_shell_terminal_false, run_shell_terminal_true, GENERATED_SESSION_TITLE,
    OPENAI_SESSION_NAME_RESPONSE, TURN_CONTEXT_OPEN,
};
use goose::config::GooseMode;
use goose::conversation::message::{Message, MessageMetadata};
use goose::custom_requests::{
    GetSessionInfoRequest, GetSessionInfoResponse, UpdateSessionProjectRequest,
};
use goose::recipe::{Recipe, Settings};
use goose::recipe_deeplink;
use goose::session::{SessionManager, SessionType};
use goose_test_support::{IgnoreSessionId, McpFixture, FAKE_CODE};
use serial_test::serial;
use std::path::Path;
use std::sync::Arc;

tests_config_option_set_error!(AcpServerConnection);
tests_mode_set_error!(AcpServerConnection);

async fn seed_list_sessions(data_root: &Path, working_dir: &Path, count: usize) {
    let session_manager = SessionManager::new(data_root.to_path_buf());
    for index in 0..count {
        let session = session_manager
            .create_session(
                working_dir.to_path_buf(),
                format!("Seed session {index}"),
                SessionType::Acp,
                GooseMode::default(),
            )
            .await
            .unwrap();
        session_manager
            .add_message(&session.id, &Message::user().with_text("hello"))
            .await
            .unwrap();
    }
}

async fn seed_list_session_with_message(
    data_root: &Path,
    working_dir: &Path,
    name: &str,
    session_type: SessionType,
    message: &str,
) {
    let session_manager = SessionManager::new(data_root.to_path_buf());
    let session = session_manager
        .create_session(
            working_dir.to_path_buf(),
            name.to_string(),
            session_type,
            GooseMode::default(),
        )
        .await
        .unwrap();
    session_manager
        .add_message(&session.id, &Message::user().with_text(message))
        .await
        .unwrap();
}

async fn new_connection(data_root: &Path) -> AcpServerConnection {
    let openai = OpenAiFixture::new(
        vec![],
        <AcpServerConnection as Connection>::expected_session_id(),
    )
    .await;
    <AcpServerConnection as Connection>::new(
        TestConnectionConfig {
            data_root: data_root.to_path_buf(),
            ..Default::default()
        },
        openai,
    )
    .await
}

async fn list_sessions_request(
    conn: &AcpServerConnection,
    request: ListSessionsRequest,
) -> anyhow::Result<ListSessionsResponse> {
    conn.cx()
        .send_request(request)
        .block_task()
        .await
        .map_err(Into::into)
}

async fn get_session_info_request(
    conn: &AcpServerConnection,
    request: GetSessionInfoRequest,
) -> anyhow::Result<GetSessionInfoResponse> {
    conn.cx()
        .send_request(request)
        .block_task()
        .await
        .map_err(Into::into)
}

fn assert_invalid_params(error: anyhow::Error) {
    let acp_error = error.downcast::<agent_client_protocol::Error>().unwrap();
    assert_eq!(acp_error.code, ErrorCode::InvalidParams);
}

fn session_title_meta(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    let mut meta = serde_json::Map::new();
    meta.insert("sessionTitle".to_string(), value);
    meta
}

async fn new_session_with_meta(
    conn: &AcpServerConnection,
    work_dir: &Path,
    meta: serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<String> {
    let response = conn
        .cx()
        .send_request(NewSessionRequest::new(work_dir).meta(meta))
        .block_task()
        .await?;
    Ok(response.session_id.0.to_string())
}

/// Returns the session's title and whether it is recorded as user-set.
async fn session_title(conn: &AcpServerConnection, session_id: &str) -> (String, bool) {
    let response = get_session_info_request(
        conn,
        GetSessionInfoRequest {
            session_id: session_id.to_string(),
        },
    )
    .await
    .unwrap();
    let user_set_name = response
        .session
        .meta
        .as_ref()
        .and_then(|meta| meta.get("userSetName"))
        .and_then(serde_json::Value::as_bool)
        .expect("session info should include userSetName");
    (
        response
            .session
            .title
            .expect("session info should include a title"),
        user_set_name,
    )
}

fn include_last_message_snippet_meta(
    value: serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    let mut goose = serde_json::Map::new();
    goose.insert("includeLastMessageSnippet".to_string(), value);

    let mut meta = serde_json::Map::new();
    meta.insert("goose".to_string(), serde_json::Value::Object(goose));
    meta
}

fn last_message_snippet(session: &SessionInfo) -> Option<&str> {
    session
        .meta
        .as_ref()
        .and_then(|meta| meta.get("lastMessageSnippet"))
        .and_then(serde_json::Value::as_str)
}

#[test]
fn test_config_mcp() {
    run_test(async { run_config_mcp::<AcpServerConnection>().await });
}

#[test]
fn test_config_option_mode_set() {
    run_test(async { run_config_option_mode_set::<AcpServerConnection>().await });
}

#[test]
fn test_list_sessions() {
    run_test(async { run_list_sessions::<AcpServerConnection>().await });
}

#[test]
fn test_list_sessions_emits_computed_snippet() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let cwd = Path::new("/tmp/acp-session-list-snippet");
        let session_manager = SessionManager::new(data_root.path().to_path_buf());
        let session = session_manager
            .create_session(
                cwd.to_path_buf(),
                "Live subtitle".to_string(),
                SessionType::Acp,
                GooseMode::default(),
            )
            .await
            .unwrap();
        session_manager
            .add_message(
                &session.id,
                &Message::user().with_text("**raw** _markdown_ subtitle"),
            )
            .await
            .unwrap();
        session_manager
            .add_message(
                &session.id,
                &Message::assistant()
                    .with_text("hidden newer text")
                    .with_metadata(MessageMetadata::agent_only()),
            )
            .await
            .unwrap();

        let conn = new_connection(data_root.path()).await;
        let response = list_sessions_request(
            &conn,
            ListSessionsRequest::new()
                .meta(include_last_message_snippet_meta(serde_json::Value::Null)),
        )
        .await
        .unwrap();

        assert_eq!(response.sessions.len(), 1);
        assert_eq!(last_message_snippet(&response.sessions[0]), None);

        let response = list_sessions_request(
            &conn,
            ListSessionsRequest::new().meta(include_last_message_snippet_meta(
                serde_json::Value::Bool(false),
            )),
        )
        .await
        .unwrap();

        assert_eq!(response.sessions.len(), 1);
        assert_eq!(last_message_snippet(&response.sessions[0]), None);

        let response = list_sessions_request(
            &conn,
            ListSessionsRequest::new().meta(include_last_message_snippet_meta(
                serde_json::Value::Bool(true),
            )),
        )
        .await
        .unwrap();

        assert_eq!(response.sessions.len(), 1);
        assert_eq!(
            last_message_snippet(&response.sessions[0]),
            Some("**raw** _markdown_ subtitle")
        );
    });
}

#[test]
fn test_list_sessions_pagination() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        seed_list_sessions(data_root.path(), Path::new("/tmp/acp-session-list"), 51).await;
        let conn = new_connection(data_root.path()).await;

        let first = list_sessions_request(&conn, ListSessionsRequest::new())
            .await
            .unwrap();
        assert_eq!(first.sessions.len(), 50);
        assert!(first
            .sessions
            .iter()
            .all(|session| last_message_snippet(session).is_none()));

        let second = list_sessions_request(
            &conn,
            ListSessionsRequest::new()
                .cursor(first.next_cursor.clone().unwrap())
                .meta(include_last_message_snippet_meta(serde_json::Value::Bool(
                    true,
                ))),
        )
        .await
        .unwrap();
        assert_eq!(second.sessions.len(), 1);
        assert!(second.next_cursor.is_none());
        assert_eq!(last_message_snippet(&second.sessions[0]), Some("hello"));

        let second_id = &second.sessions[0].session_id;
        assert!(first
            .sessions
            .iter()
            .all(|session| session.session_id != *second_id));
    });
}

#[test]
fn test_list_sessions_query_filters_results() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let cwd = Path::new("/tmp/acp-session-list");
        seed_list_session_with_message(
            data_root.path(),
            cwd,
            "Postgres session",
            SessionType::Acp,
            "Discuss Postgres migrations",
        )
        .await;
        seed_list_session_with_message(
            data_root.path(),
            cwd,
            "Mobile session",
            SessionType::Acp,
            "Plan the mobile release",
        )
        .await;
        let conn = new_connection(data_root.path()).await;

        let mut meta = serde_json::Map::new();
        meta.insert(
            "query".to_string(),
            serde_json::Value::String("postgres".to_string()),
        );
        let response = list_sessions_request(&conn, ListSessionsRequest::new().meta(meta))
            .await
            .unwrap();

        assert_eq!(response.sessions.len(), 1);
        assert_eq!(
            response.sessions[0].title.as_deref(),
            Some("Postgres session")
        );
        assert!(response.next_cursor.is_none());
    });
}

#[test]
fn test_list_sessions_types_override_filters_results() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let cwd = Path::new("/tmp/acp-session-list");
        seed_list_session_with_message(
            data_root.path(),
            cwd,
            "ACP session",
            SessionType::Acp,
            "ACP message",
        )
        .await;
        seed_list_session_with_message(
            data_root.path(),
            cwd,
            "User session",
            SessionType::User,
            "User message",
        )
        .await;
        let conn = new_connection(data_root.path()).await;

        let mut meta = serde_json::Map::new();
        meta.insert(
            "types".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::String("user".to_string())]),
        );
        let response = list_sessions_request(&conn, ListSessionsRequest::new().meta(meta))
            .await
            .unwrap();

        assert_eq!(response.sessions.len(), 1);
        assert_eq!(response.sessions[0].title.as_deref(), Some("User session"));
        assert!(response.next_cursor.is_none());
    });
}

#[test]
fn test_list_sessions_types_rejects_internal_session_types() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let conn = new_connection(data_root.path()).await;

        for session_type in ["hidden", "sub_agent"] {
            let mut meta = serde_json::Map::new();
            meta.insert(
                "types".to_string(),
                serde_json::Value::Array(vec![serde_json::Value::String(session_type.to_string())]),
            );

            let error = list_sessions_request(&conn, ListSessionsRequest::new().meta(meta))
                .await
                .unwrap_err();
            assert_invalid_params(error);
        }
    });
}

#[test]
fn test_list_sessions_invalid_params() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let cwd = tempfile::tempdir().unwrap();
        let other_cwd = tempfile::tempdir().unwrap();
        seed_list_sessions(data_root.path(), cwd.path(), 51).await;
        let conn = new_connection(data_root.path()).await;

        let error =
            list_sessions_request(&conn, ListSessionsRequest::new().cursor("*".to_string()))
                .await
                .unwrap_err();
        assert_invalid_params(error);

        let error = list_sessions_request(
            &conn,
            ListSessionsRequest::new().cwd(std::path::PathBuf::from("relative/path")),
        )
        .await
        .unwrap_err();
        assert_invalid_params(error);

        let first = list_sessions_request(&conn, ListSessionsRequest::new().cwd(cwd.path()))
            .await
            .unwrap();

        let error = list_sessions_request(
            &conn,
            ListSessionsRequest::new()
                .cwd(other_cwd.path())
                .cursor(first.next_cursor.unwrap()),
        )
        .await
        .unwrap_err();
        assert_invalid_params(error);

        let error = list_sessions_request(
            &conn,
            ListSessionsRequest::new().meta(include_last_message_snippet_meta(
                serde_json::Value::String("true".to_string()),
            )),
        )
        .await
        .unwrap_err();
        assert_invalid_params(error);
    });
}

#[test]
fn test_get_session_info() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let cwd = Path::new("/tmp/acp-session-info");
        let session_manager = SessionManager::new(data_root.path().to_path_buf());
        let session = session_manager
            .create_session(
                cwd.to_path_buf(),
                "Session info".to_string(),
                SessionType::Acp,
                GooseMode::default(),
            )
            .await
            .unwrap();
        session_manager
            .add_message(&session.id, &Message::user().with_text("hello"))
            .await
            .unwrap();
        let conn = new_connection(data_root.path()).await;

        let response = get_session_info_request(
            &conn,
            GetSessionInfoRequest {
                session_id: session.id.clone(),
            },
        )
        .await
        .unwrap();

        assert_eq!(
            response.session.session_id,
            agent_client_protocol::schema::v1::SessionId::new(session.id)
        );
        assert_eq!(response.session.cwd, cwd.to_path_buf());
        assert_eq!(response.session.title.as_deref(), Some("Session info"));
        assert!(response.session.updated_at.is_some());

        let meta = response
            .session
            .meta
            .expect("session info should include meta");
        assert!(meta.get("createdAt").and_then(|v| v.as_str()).is_some());
        assert_eq!(meta.get("messageCount"), Some(&serde_json::json!(1)));
        assert_eq!(meta.get("userSetName"), Some(&serde_json::json!(false)));
        assert_eq!(meta.get("sessionType"), Some(&serde_json::json!("acp")));
        assert_eq!(meta.get("hasRecipe"), Some(&serde_json::json!(false)));
    });
}

#[test]
fn test_update_session_project_rejects_hidden_session_types() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let working_dir = tempfile::tempdir().unwrap();
        let session_manager = SessionManager::new(data_root.path().to_path_buf());
        let mut hidden_sessions = Vec::new();

        for session_type in [
            SessionType::Gateway,
            SessionType::SubAgent,
            SessionType::Hidden,
            SessionType::Terminal,
        ] {
            hidden_sessions.push(
                session_manager
                    .create_session(
                        working_dir.path().to_path_buf(),
                        format!("{session_type} session"),
                        session_type,
                        GooseMode::default(),
                    )
                    .await
                    .unwrap(),
            );
        }

        let conn = new_connection(data_root.path()).await;
        for session in hidden_sessions {
            let error = conn
                .cx()
                .send_request(UpdateSessionProjectRequest {
                    session_id: session.id.clone(),
                    project_id: Some("untrusted-project".to_string()),
                })
                .block_task()
                .await
                .unwrap_err();
            assert_eq!(error.code, ErrorCode::ResourceNotFound);

            let stored = session_manager
                .get_session(&session.id, false)
                .await
                .unwrap();
            assert_eq!(stored.project_id, None);
        }
    });
}

#[test]
fn test_update_session_project_rejects_unknown_persisted_session_type() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let working_dir = tempfile::tempdir().unwrap();
        let session_manager = SessionManager::new(data_root.path().to_path_buf());
        let session = session_manager
            .create_session(
                working_dir.path().to_path_buf(),
                "Future session".to_string(),
                SessionType::User,
                GooseMode::default(),
            )
            .await
            .unwrap();
        let db_path = data_root
            .path()
            .join(goose::session::session_manager::SESSIONS_FOLDER)
            .join(goose::session::session_manager::DB_NAME);
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .connect_with(sqlx::sqlite::SqliteConnectOptions::new().filename(db_path))
            .await
            .unwrap();
        sqlx::query("UPDATE sessions SET session_type = 'future_type' WHERE id = ?")
            .bind(&session.id)
            .execute(&pool)
            .await
            .unwrap();

        assert_eq!(
            session_manager
                .get_session(&session.id, false)
                .await
                .unwrap()
                .session_type,
            SessionType::User
        );

        let conn = new_connection(data_root.path()).await;
        let error = conn
            .cx()
            .send_request(UpdateSessionProjectRequest {
                session_id: session.id.clone(),
                project_id: Some("untrusted-project".to_string()),
            })
            .block_task()
            .await
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::ResourceNotFound);

        let project_id =
            sqlx::query_scalar::<_, Option<String>>("SELECT project_id FROM sessions WHERE id = ?")
                .bind(&session.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(project_id, None);
    });
}

#[test]
fn test_update_session_project_allows_visible_session_types() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let working_dir = tempfile::tempdir().unwrap();
        let session_manager = SessionManager::new(data_root.path().to_path_buf());
        let mut visible_sessions = Vec::new();

        for session_type in [SessionType::User, SessionType::Scheduled, SessionType::Acp] {
            visible_sessions.push(
                session_manager
                    .create_session(
                        working_dir.path().to_path_buf(),
                        format!("{session_type} session"),
                        session_type,
                        GooseMode::default(),
                    )
                    .await
                    .unwrap(),
            );
        }

        let conn = new_connection(data_root.path()).await;
        for session in visible_sessions {
            conn.cx()
                .send_request(UpdateSessionProjectRequest {
                    session_id: session.id.clone(),
                    project_id: Some("project".to_string()),
                })
                .block_task()
                .await
                .unwrap();
            assert_eq!(
                session_manager
                    .get_session(&session.id, false)
                    .await
                    .unwrap()
                    .project_id,
                Some("project".to_string())
            );

            conn.cx()
                .send_request(UpdateSessionProjectRequest {
                    session_id: session.id.clone(),
                    project_id: None,
                })
                .block_task()
                .await
                .unwrap();
            assert_eq!(
                session_manager
                    .get_session(&session.id, false)
                    .await
                    .unwrap()
                    .project_id,
                None
            );
        }
    });
}

#[test]
fn test_session_name_update_notification() {
    run_test(async { run_session_name_update_notification::<AcpServerConnection>().await });
}

#[test]
fn test_close_session() {
    run_test(async { run_close_session::<AcpServerConnection>().await });
}

#[test]
fn test_config_option_model_set() {
    run_test(async { run_config_option_model_set::<AcpServerConnection>().await });
}

#[test]
fn test_config_option_thinking_effort_set() {
    run_test(async {
        let openai = OpenAiFixture::new(
            vec![],
            <AcpServerConnection as Connection>::expected_session_id(),
        )
        .await;
        let mut conn = <AcpServerConnection as Connection>::new(
            TestConnectionConfig {
                current_model: "claude-sonnet-4".to_string(),
                ..Default::default()
            },
            openai,
        )
        .await;
        let data = conn.new_session().await.unwrap();

        let response = conn
            .cx()
            .send_request(SetSessionConfigOptionRequest::new(
                data.session.session_id().clone(),
                "thinking_effort".to_string(),
                SessionConfigOptionValue::value_id("high".to_string()),
            ))
            .block_task()
            .await
            .unwrap();

        let option = response
            .config_options
            .iter()
            .find(|option| option.id.0.as_ref() == "thinking_effort")
            .expect("thinking_effort option");
        assert_eq!(
            option.category,
            Some(SessionConfigOptionCategory::ThoughtLevel)
        );
        let select = match &option.kind {
            SessionConfigKind::Select(select) => select,
            _ => panic!("thinking_effort should be a select option"),
        };

        assert_eq!(select.current_value.0.as_ref(), "high");
    });
}

#[test]
fn test_config_option_non_value_id_returns_error_and_keeps_connection() {
    run_test(async {
        let openai = OpenAiFixture::new(
            vec![],
            <AcpServerConnection as Connection>::expected_session_id(),
        )
        .await;
        let mut conn =
            <AcpServerConnection as Connection>::new(TestConnectionConfig::default(), openai).await;
        let data = conn.new_session().await.unwrap();

        let err = conn
            .cx()
            .send_request(SetSessionConfigOptionRequest::new(
                data.session.session_id().clone(),
                "mode".to_string(),
                SessionConfigOptionValue::boolean(true),
            ))
            .block_task()
            .await
            .unwrap_err();
        assert_eq!(
            err,
            agent_client_protocol::Error::invalid_params().data("Expected a value ID")
        );

        conn.cx()
            .send_request(ListSessionsRequest::new())
            .block_task()
            .await
            .expect("connection should survive an invalid set_config_option");
    });
}

#[test]
fn test_delete_session() {
    run_test(async { run_delete_session::<AcpServerConnection>().await });
}

#[test]
fn test_fs_read_text_file_true() {
    run_test(async { run_fs_read_text_file_true::<AcpServerConnection>().await });
}

#[test]
fn test_fs_write_text_file_false() {
    run_test(async { run_fs_write_text_file_false::<AcpServerConnection>().await });
}

#[test]
fn test_fs_write_text_file_true() {
    run_test(async { run_fs_write_text_file_true::<AcpServerConnection>().await });
}

#[test]
fn test_initialize_doesnt_hit_provider() {
    run_test(async { run_initialize_doesnt_hit_provider::<AcpServerConnection>().await });
}

#[test]
fn test_load_mode() {
    run_test(async { run_load_mode::<AcpServerConnection>().await });
}

#[test]
fn test_load_model() {
    run_test(async { run_load_model::<AcpServerConnection>().await });
}

#[test]
fn test_load_session_error_session_not_found() {
    run_test(async { run_load_session_error::<AcpServerConnection>().await });
}

#[test]
fn test_load_session_mcp() {
    run_test(async { run_load_session_mcp::<AcpServerConnection>().await });
}

#[test]
fn test_load_session_replays_image_attachment() {
    run_test(async { run_load_session_replays_image_attachment::<AcpServerConnection>().await });
}

#[test]
fn test_mode_set() {
    run_test(async { run_mode_set::<AcpServerConnection>().await });
}

#[test]
fn test_model_list() {
    run_test(async { run_model_list::<AcpServerConnection>().await });
}

#[test]
fn test_new_session_returns_initial_config() {
    run_test(async { run_new_session_returns_initial_config::<AcpServerConnection>().await });
}

#[test]
fn test_new_session_uses_current_config_mode() {
    run_test(async { run_new_session_uses_current_config_mode::<AcpServerConnection>().await });
}

#[test]
fn test_new_session_response_precedes_available_commands() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let work_dir = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(
            vec![],
            <AcpServerConnection as Connection>::expected_session_id(),
        )
        .await;
        let (transport, _handle, _permission_manager) = spawn_acp_server_in_process(
            openai.uri(),
            &[],
            data_root.path(),
            GooseMode::default(),
            None,
            goose_test_support::TEST_MODEL,
            true,
        )
        .await;
        assert_session_response_precedes_available_commands(
            transport,
            "session/new",
            serde_json::json!({
                "cwd": work_dir.path(),
                "mcpServers": []
            }),
        )
        .await;
    });
}

#[test]
fn test_new_session_honors_recipe_model_without_recipe_provider() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let conn = new_connection(data_root.path()).await;
        let work_dir = tempfile::tempdir().unwrap();
        let recipe_model = "gpt-4.1";
        let recipe = Recipe::builder()
            .title("Recipe model")
            .description("A recipe that only overrides the model")
            .instructions("Use the requested model")
            .settings(Settings {
                goose_provider: None,
                goose_model: Some(recipe_model.to_string()),
                temperature: None,
                max_turns: None,
            })
            .build()
            .unwrap();
        let mut meta = serde_json::Map::new();
        meta.insert(
            "recipeDeeplink".to_string(),
            serde_json::Value::String(recipe_deeplink::encode(&recipe).unwrap()),
        );

        let response = conn
            .cx()
            .send_request(NewSessionRequest::new(work_dir.path()).meta(meta))
            .block_task()
            .await
            .unwrap();
        let session_info = get_session_info_request(
            &conn,
            GetSessionInfoRequest {
                session_id: response.session_id.0.to_string(),
            },
        )
        .await
        .unwrap();
        let meta = session_info
            .session
            .meta
            .expect("session info should include meta");

        assert_eq!(
            meta.get("modelId").and_then(|v| v.as_str()),
            Some(recipe_model)
        );
        assert_eq!(
            meta.get("providerId").and_then(|v| v.as_str()),
            Some("openai")
        );
    });
}

#[test]
fn test_new_session_cleans_up_when_config_fails() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let conn = new_connection(data_root.path()).await;
        let work_dir = tempfile::tempdir().unwrap();
        let mut meta = serde_json::Map::new();
        meta.insert(
            "enabledExtensions".to_string(),
            serde_json::Value::String("invalid".to_string()),
        );

        let error: anyhow::Error = conn
            .cx()
            .send_request(NewSessionRequest::new(work_dir.path()).meta(meta))
            .block_task()
            .await
            .unwrap_err()
            .into();

        assert_invalid_params(error);

        let sessions = SessionManager::new(data_root.path().to_path_buf())
            .list_all_sessions()
            .await
            .unwrap();
        assert!(sessions.is_empty());
    });
}

#[test]
fn test_new_session_titles_session_from_meta_session_title() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let conn = new_connection(data_root.path()).await;
        let work_dir = tempfile::tempdir().unwrap();

        let session_id = new_session_with_meta(
            &conn,
            work_dir.path(),
            session_title_meta(serde_json::json!("  Duncan in #general  ")),
        )
        .await
        .unwrap();

        assert_eq!(
            session_title(&conn, &session_id).await,
            ("Duncan in #general".to_string(), true)
        );
    });
}

#[test]
fn test_new_session_prefers_recipe_title_over_meta_session_title() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let conn = new_connection(data_root.path()).await;
        let work_dir = tempfile::tempdir().unwrap();
        let recipe = Recipe::builder()
            .title("Recipe title")
            .description("A recipe with a title")
            .instructions("Follow the recipe")
            .build()
            .unwrap();
        let mut meta = session_title_meta(serde_json::json!("Client title"));
        meta.insert(
            "recipeDeeplink".to_string(),
            serde_json::Value::String(recipe_deeplink::encode(&recipe).unwrap()),
        );

        let session_id = new_session_with_meta(&conn, work_dir.path(), meta)
            .await
            .unwrap();

        // The recipe title wins, and the session is not marked user-set so
        // goose's own recipe-title naming path still applies.
        assert_eq!(
            session_title(&conn, &session_id).await,
            ("Recipe title".to_string(), false)
        );
    });
}

#[test]
fn test_new_session_without_meta_session_title_uses_default_name() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let conn = new_connection(data_root.path()).await;
        let work_dir = tempfile::tempdir().unwrap();

        for meta in [
            serde_json::Map::new(),
            session_title_meta(serde_json::Value::Null),
            session_title_meta(serde_json::json!("   ")),
        ] {
            let session_id = new_session_with_meta(&conn, work_dir.path(), meta)
                .await
                .unwrap();

            assert_eq!(
                session_title(&conn, &session_id).await,
                ("New Chat".to_string(), false)
            );
        }
    });
}

#[test]
fn test_new_session_rejects_non_string_meta_session_title() {
    run_test(async {
        let data_root = tempfile::tempdir().unwrap();
        let conn = new_connection(data_root.path()).await;
        let work_dir = tempfile::tempdir().unwrap();
        let recipe = Recipe::builder()
            .title("Recipe title")
            .description("A recipe with a title")
            .instructions("Follow the recipe")
            .build()
            .unwrap();

        // Rejected on its own, and also when a recipe title would have won —
        // validation does not depend on precedence.
        let mut with_recipe = session_title_meta(serde_json::json!(42));
        with_recipe.insert(
            "recipeDeeplink".to_string(),
            serde_json::Value::String(recipe_deeplink::encode(&recipe).unwrap()),
        );
        for meta in [session_title_meta(serde_json::json!(42)), with_recipe] {
            let error = new_session_with_meta(&conn, work_dir.path(), meta)
                .await
                .unwrap_err();
            assert_invalid_params(error);
        }

        let sessions = SessionManager::new(data_root.path().to_path_buf())
            .list_all_sessions()
            .await
            .unwrap();
        assert!(sessions.is_empty());
    });
}

/// Drives one naming-enabled turn and returns the session's title once name
/// generation has had a chance to run.
async fn title_after_naming_turn(
    data_root: &Path,
    meta: serde_json::Map<String, serde_json::Value>,
) -> (String, bool) {
    let openai = OpenAiFixture::new(
        vec![
            (
                format!("what is 1+1{TURN_CONTEXT_OPEN}"),
                include_str!("acp_test_data/openai_basic.txt"),
            ),
            (
                "Generate a short title for the above messages.".to_string(),
                OPENAI_SESSION_NAME_RESPONSE,
            ),
        ],
        <AcpServerConnection as Connection>::expected_session_id(),
    )
    .await;
    let conn = <AcpServerConnection as Connection>::new(
        TestConnectionConfig {
            data_root: data_root.to_path_buf(),
            disable_session_naming: false,
            ..Default::default()
        },
        openai,
    )
    .await;
    let work_dir = tempfile::tempdir().unwrap();
    let session_id = new_session_with_meta(&conn, work_dir.path(), meta)
        .await
        .unwrap();

    let response = conn
        .cx()
        .send_request(PromptRequest::new(
            agent_client_protocol::schema::v1::SessionId::new(session_id.clone()),
            vec![ContentBlock::Text(TextContent::new("what is 1+1"))],
        ))
        .block_task()
        .await
        .unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);

    // Naming runs in a spawned task: wait for it to land, or for the deadline
    // to prove it never will.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let title = session_title(&conn, &session_id).await;
        if title.0 == GENERATED_SESSION_TITLE || tokio::time::Instant::now() >= deadline {
            return title;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[test]
fn test_generated_name_does_not_replace_meta_session_title() {
    run_test(async {
        // Control arm: with no client title, generation names the session.
        let data_root = tempfile::tempdir().unwrap();
        assert_eq!(
            title_after_naming_turn(data_root.path(), serde_json::Map::new()).await,
            (GENERATED_SESSION_TITLE.to_string(), false),
            "name generation must work here, or the assertion below proves nothing"
        );

        // A client title is recorded as user-set, which generation must respect.
        let data_root = tempfile::tempdir().unwrap();
        assert_eq!(
            title_after_naming_turn(
                data_root.path(),
                session_title_meta(serde_json::json!("Client title"))
            )
            .await,
            ("Client title".to_string(), true)
        );
    });
}

#[test]
fn test_model_set() {
    run_test(async { run_model_set::<AcpServerConnection>().await });
}

#[test]
fn test_model_set_error_session_not_found() {
    run_test(async { run_model_set_error_session_not_found::<AcpServerConnection>().await });
}

#[test]
fn test_permission_persistence() {
    run_test(async { run_permission_persistence::<AcpServerConnection>().await });
}

#[test]
fn test_prompt_basic() {
    run_test(async { run_prompt_basic::<AcpServerConnection>().await });
}

#[test]
#[cfg(feature = "code-mode")]
fn test_prompt_codemode() {
    run_test(async { run_prompt_codemode::<AcpServerConnection>().await });
}

#[test]
fn test_prompt_error_session_not_found() {
    run_test(async { run_prompt_error::<AcpServerConnection>().await });
}

#[test]
fn test_prompt_image() {
    run_test(async { run_prompt_image::<AcpServerConnection>().await });
}

#[test]
fn test_prompt_image_attachment() {
    run_test(async { run_prompt_image_attachment::<AcpServerConnection>().await });
}

#[test]
fn test_prompt_mcp() {
    run_test(async { run_prompt_mcp::<AcpServerConnection>().await });
}

/// Selects the agent loop until dropped. Only use inside `run_test`: it holds the
/// ACP test lock, so no other test in this binary observes the override.
struct AgentLoopOverride(Option<std::ffi::OsString>);

impl AgentLoopOverride {
    fn new(state_machine: bool) -> Self {
        let previous = std::env::var_os("GOOSE_STATE_MACHINE");
        std::env::set_var("GOOSE_STATE_MACHINE", if state_machine { "1" } else { "0" });
        Self(previous)
    }
}

impl Drop for AgentLoopOverride {
    fn drop(&mut self) {
        match self.0.take() {
            Some(previous) => std::env::set_var("GOOSE_STATE_MACHINE", previous),
            None => std::env::remove_var("GOOSE_STATE_MACHINE"),
        }
    }
}

/// Two provider calls separated by a tool call. The canned responses report 2469
/// and then 2631 total tokens, so 2469 can only arrive in an update sent while
/// the prompt is still running.
async fn assert_usage_updates_during_prompt(state_machine: bool) {
    let _agent_loop = AgentLoopOverride::new(state_machine);
    let prompt = "Use the get_code tool and output only its result.";
    let mcp = McpFixture::new().await;
    let openai = OpenAiFixture::new(
        vec![
            (
                prompt.to_string(),
                include_str!("acp_test_data/openai_tool_call.txt"),
            ),
            (
                FAKE_CODE.to_string(),
                include_str!("acp_test_data/openai_tool_result.txt"),
            ),
        ],
        <AcpServerConnection as Connection>::expected_session_id(),
    )
    .await;
    let config = TestConnectionConfig {
        mcp_servers: vec![McpServer::Http(McpServerHttp::new("mcp-fixture", &mcp.url))],
        ..Default::default()
    };
    let mut conn = <AcpServerConnection as Connection>::new(config, openai).await;
    let SessionData { mut session, .. } = conn.new_session().await.unwrap();

    let output = session
        .prompt(prompt, PermissionDecision::Cancel)
        .await
        .unwrap();

    let mut used = Vec::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    while used.len() < 3 && tokio::time::Instant::now() < deadline {
        used.extend(
            session
                .session_updates()
                .into_iter()
                .filter_map(|update| match update {
                    SessionUpdate::UsageUpdate(usage) => Some(usage.used),
                    _ => None,
                }),
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(
        used,
        [2469, 2631, 2631],
        "state_machine={state_machine}, agent text: {}",
        output.text
    );
}

#[test]
fn test_prompt_usage_updates_during_turn_legacy_loop() {
    run_test(async { assert_usage_updates_during_prompt(false).await });
}

#[test]
fn test_prompt_usage_updates_during_turn_state_machine() {
    run_test(async { assert_usage_updates_during_prompt(true).await });
}

#[test]
fn test_prompt_model_mismatch() {
    run_test(async { run_prompt_model_mismatch::<AcpServerConnection>().await });
}

#[test]
fn test_prompt_skill() {
    run_test(async { run_prompt_skill::<AcpServerConnection>().await });
}

#[test]
fn test_shell_terminal_false() {
    run_test(async { run_shell_terminal_false::<AcpServerConnection>().await });
}

#[test]
fn test_shell_terminal_true() {
    run_test(async { run_shell_terminal_true::<AcpServerConnection>().await });
}

#[test]
#[serial]
fn test_session_settings_set() {
    run_test(async move {
        const FINAL_OUTPUT_RESPONSE: &str = "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_final\",\"type\":\"function\",\"function\":{\"name\":\"recipe__final_output\",\"arguments\":\"{\\\"summary\\\":\\\"done\\\"}\"}}]},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";

        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(
            vec![(
                "Special test recipe instructions".to_string(),
                FINAL_OUTPUT_RESPONSE,
            )],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let mut conn = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai,
        )
        .await;

        let SessionData { mut session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.clone();

        // (iii) invalid cacheTtl -> invalid params
        let invalid_err = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "2h",
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(invalid_err.code, ErrorCode::InvalidParams);
        assert!(
            invalid_err
                .data
                .as_ref()
                .unwrap()
                .to_string()
                .contains("invalid cacheTtl: "),
            "expected data starting with 'invalid cacheTtl: '"
        );

        // Unknown session returns resource_not_found
        let unknown_session_err = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": "unknown-nonexistent-session",
                "recipe": {
                    "title": "Some Recipe",
                    "description": "Description"
                }
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(unknown_session_err.code, ErrorCode::ResourceNotFound);

        // (i) and (ii) Set settings with cacheTtl 1h and recipe, called twice for idempotency
        let recipe = serde_json::json!({
            "title": "Settings Test Recipe",
            "description": "Test recipe description",
            "instructions": "Special test recipe instructions",
            "response": {
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string" }
                    },
                    "required": ["summary"]
                }
            }
        });

        for _ in 0..2 {
            let res = send_custom(
                conn.cx(),
                "_goose/unstable/session/settings/set",
                serde_json::json!({
                    "sessionId": session_id,
                    "cacheTtl": "1h",
                    "recipe": recipe,
                }),
            )
            .await
            .unwrap();
            assert_eq!(res, serde_json::json!({}));
        }

        let session_manager = SessionManager::new(temp_data.path().to_path_buf());

        // Verify invalid recipes are rejected and do not mutate the row or break loading
        let bad_schema_err = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "recipe": {
                    "title": "Bad Schema Recipe",
                    "description": "Empty schema",
                    "response": {
                        "json_schema": {}
                    }
                }
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(bad_schema_err.code, ErrorCode::InvalidParams);
        assert!(
            bad_schema_err
                .data
                .as_ref()
                .unwrap()
                .to_string()
                .contains("invalid recipe: "),
            "expected data starting with 'invalid recipe: '"
        );

        let bad_template_err = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "recipe": {
                    "title": "Bad Template Recipe",
                    "description": "Unclosed jinja",
                    "instructions": "Unclosed {% bad tag"
                }
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(bad_template_err.code, ErrorCode::InvalidParams);
        assert!(
            bad_template_err
                .data
                .as_ref()
                .unwrap()
                .to_string()
                .contains("invalid recipe: "),
            "expected data starting with 'invalid recipe: '"
        );

        // Verify session.recipe is persisted and unchanged by rejected recipes
        let stored_recipe = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert!(stored_recipe.recipe.is_some());
        assert_eq!(
            stored_recipe
                .recipe
                .as_ref()
                .unwrap()
                .instructions
                .as_deref(),
            Some("Special test recipe instructions")
        );

        // Verify session.model_config has cache_ttl 1h
        assert_eq!(
            stored_recipe
                .model_config
                .as_ref()
                .and_then(|mc| mc.cache_ttl()),
            Some("1h".to_string())
        );

        // Verify final_output tool exists exactly once
        let tools_res = send_custom(
            conn.cx(),
            "_goose/unstable/tools/list",
            serde_json::json!({ "sessionId": session_id }),
        )
        .await
        .unwrap();
        let tools = tools_res
            .get("tools")
            .and_then(|t| t.as_array())
            .expect("tools array");
        let final_output_count = tools
            .iter()
            .filter(|t| t.get("name").and_then(|n| n.as_str()) == Some("recipe__final_output"))
            .count();
        assert_eq!(
            final_output_count, 1,
            "final_output tool should exist exactly once"
        );

        // Verify system prompt contains the new instructions by running prompt
        let prompt_output = session
            .prompt("Hello with recipe", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(prompt_output.tool_status.is_some());

        // Survives fresh session/load on a new connection
        let openai2 = OpenAiFixture::new(vec![], Arc::new(IgnoreSessionId)).await;
        let mut conn2 = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai2,
        )
        .await;
        let _loaded = conn2.load_session(&session_id, vec![]).await.unwrap();

        // Verify final_output tool exists on fresh connection after session/load
        let tools_res2 = send_custom(
            conn2.cx(),
            "_goose/unstable/tools/list",
            serde_json::json!({ "sessionId": session_id }),
        )
        .await
        .unwrap();
        let tools2 = tools_res2
            .get("tools")
            .and_then(|t| t.as_array())
            .expect("tools array");
        let final_output_count2 = tools2
            .iter()
            .filter(|t| t.get("name").and_then(|n| n.as_str()) == Some("recipe__final_output"))
            .count();
        assert_eq!(
            final_output_count2, 1,
            "final_output tool should exist exactly once on fresh connection after load"
        );

        let stored_after_load = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored_after_load
                .model_config
                .as_ref()
                .and_then(|mc| mc.cache_ttl()),
            Some("1h".to_string())
        );

        // Third fresh connection: send settings/set directly to COLD session (no session/load first)
        let openai3 = OpenAiFixture::new(
            vec![(
                "Special test recipe instructions".to_string(),
                FINAL_OUTPUT_RESPONSE,
            )],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let conn3 = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai3,
        )
        .await;

        let set_res3 = send_custom(
            conn3.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "1h",
            }),
        )
        .await
        .unwrap();
        assert_eq!(set_res3, serde_json::json!({}));

        let stored_cold = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored_cold
                .model_config
                .as_ref()
                .and_then(|mc| mc.cache_ttl()),
            Some("1h".to_string())
        );

        // Prompt on the cold-activated session: assert recipe instructions are in force
        let prompt_res3 = send_custom(
            conn3.cx(),
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "Testing cold prompt" }],
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            prompt_res3.get("stopReason").and_then(|s| s.as_str()),
            Some("end_turn")
        );

        let updates3 = session.session_updates();
        let prompt_text = updates3
            .iter()
            .filter_map(|u| match u {
                SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                    ContentBlock::Text(t) => Some(t.text.as_str()),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert!(
            !prompt_text.contains("Ran into this error"),
            "prompt should not encounter error: {prompt_text}"
        );

        // Setting a recipe without response clears previously registered final_output tool
        let recipe_no_response = serde_json::json!({
            "title": "Recipe Without Response",
            "description": "Recipe with no output schema",
            "instructions": "Simple instructions without final output",
        });
        send_custom(
            conn3.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "recipe": recipe_no_response,
            }),
        )
        .await
        .unwrap();

        let tools_cleared = send_custom(
            conn3.cx(),
            "_goose/unstable/tools/list",
            serde_json::json!({ "sessionId": session_id }),
        )
        .await
        .unwrap();
        let tools_arr = tools_cleared
            .get("tools")
            .and_then(|t| t.as_array())
            .expect("tools array");
        let final_output_count3 = tools_arr
            .iter()
            .filter(|t| t.get("name").and_then(|n| n.as_str()) == Some("recipe__final_output"))
            .count();
        assert_eq!(
            final_output_count3, 0,
            "final_output tool should be cleared when recipe has no response"
        );
    });
}

#[test]
#[serial]
fn test_prompt_max_turns_bounds_run() {
    run_test(async move {
        // Mock provider responses: two turns of assistant text without calling final_output
        let chunk1 = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Turn 1 working\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let chunk2 = "data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"created\":1766229304,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Turn 2 working\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"created\":1766229304,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

        let openai = OpenAiFixture::new(
            vec![
                ("what is 1+1".to_string(), chunk1),
                (
                    "You MUST call the `final_output` tool NOW".to_string(),
                    chunk2,
                ),
            ],
            Arc::new(IgnoreSessionId),
        )
        .await;

        let mut conn = AcpServerConnection::new(TestConnectionConfig::default(), openai).await;

        let SessionData { session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.clone();

        // Pin a recipe requiring final_output
        let recipe = serde_json::json!({
            "title": "Recipe With Output",
            "description": "Recipe requiring final output",
            "instructions": "Always produce structured final output",
            "response": {
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "ans": { "type": "string" }
                    },
                    "required": ["ans"]
                }
            }
        });
        send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "recipe": recipe,
            }),
        )
        .await
        .unwrap();

        // 1. Invalid maxTurns -> invalid params
        let invalid_prompt_err = send_custom(
            conn.cx(),
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "what is 1+1" }],
                "_meta": {
                    "goose": {
                        "maxTurns": 0
                    }
                }
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(invalid_prompt_err.code, ErrorCode::InvalidParams);

        // 2. Valid maxTurns: 2 bounds the run and emits MAX_TURNS_MESSAGE
        let prompt_res = send_custom(
            conn.cx(),
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "what is 1+1" }],
                "_meta": {
                    "goose": {
                        "maxTurns": 2
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            prompt_res.get("stopReason").and_then(|s| s.as_str()),
            Some("end_turn")
        );

        // Check the emitted text contains max turns message
        let updates = session.session_updates();
        let agent_texts: Vec<String> = updates
            .iter()
            .filter_map(|u| match u {
                SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                    ContentBlock::Text(t) => Some(t.text.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        let combined_text = agent_texts.join("");
        assert!(
            combined_text.contains("maximum number of actions"),
            "expected max turns message in updates, got: {combined_text}"
        );
    });
}

#[test]
#[serial]
fn test_session_settings_rolling_memory() {
    run_test(async move {
        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(vec![], Arc::new(IgnoreSessionId)).await;
        let mut conn = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai,
        )
        .await;

        let SessionData { session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.clone();
        let session_manager = SessionManager::new(temp_data.path().to_path_buf());

        // 1. tailTokens >= budgetTokens -> invalid_params
        let invalid_err = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "rollingMemory": {
                    "budgetTokens": 50000,
                    "tailTokens": 60000
                }
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(invalid_err.code, ErrorCode::InvalidParams);
        assert!(
            invalid_err
                .data
                .as_ref()
                .unwrap()
                .to_string()
                .contains("invalid rollingMemory: "),
            "expected data starting with 'invalid rollingMemory: '"
        );

        // 2. set an object -> extension_data["rolling_memory.v0"] equals it
        let res = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "rollingMemory": {
                    "budgetTokens": 150000,
                    "tailTokens": 60000
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(res, serde_json::json!({}));

        let stored = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        let setting_val = stored
            .extension_data
            .get_extension_state("rolling_memory", "v0")
            .cloned();
        assert_eq!(
            setting_val,
            Some(serde_json::json!({
                "budgetTokens": 150000,
                "tailTokens": 60000
            }))
        );

        // 3. a second settings/set with only cacheTtl leaves it unchanged
        let res2 = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "1h"
            }),
        )
        .await
        .unwrap();
        assert_eq!(res2, serde_json::json!({}));

        let stored2 = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored2
                .extension_data
                .get_extension_state("rolling_memory", "v0")
                .cloned(),
            Some(serde_json::json!({
                "budgetTokens": 150000,
                "tailTokens": 60000
            }))
        );

        // 4. rollingMemory: null stores null
        let res3 = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "rollingMemory": null
            }),
        )
        .await
        .unwrap();
        assert_eq!(res3, serde_json::json!({}));

        let stored3 = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored3
                .extension_data
                .get_extension_state("rolling_memory", "v0"),
            Some(&serde_json::Value::Null)
        );

        // Re-set to an object so we can test session/load on a new connection
        let res4 = send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "rollingMemory": {
                    "budgetTokens": 120000,
                    "tailTokens": 40000
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(res4, serde_json::json!({}));

        // 5. the setting survives session/load on a new connection
        let openai2 = OpenAiFixture::new(vec![], Arc::new(IgnoreSessionId)).await;
        let mut conn2 = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai2,
        )
        .await;
        let _loaded = conn2.load_session(&session_id, vec![]).await.unwrap();

        let stored_after_load = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored_after_load
                .extension_data
                .get_extension_state("rolling_memory", "v0")
                .cloned(),
            Some(serde_json::json!({
                "budgetTokens": 120000,
                "tailTokens": 40000
            }))
        );
    });
}

#[test]
#[serial]
fn test_session_settings_idempotent_prompts_byte_identical() {
    run_test(async move {
        const FINAL_OUTPUT_RESPONSE: &str = "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_final\",\"type\":\"function\",\"function\":{\"name\":\"recipe__final_output\",\"arguments\":\"{\\\"value\\\":\\\"done\\\"}\"}}]},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";

        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(
            vec![
                ("what is 1+1".to_string(), FINAL_OUTPUT_RESPONSE),
                ("what is 2+2".to_string(), FINAL_OUTPUT_RESPONSE),
            ],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let captured = openai.captured_requests_handle();

        let mut conn = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai,
        )
        .await;

        let SessionData { mut session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.clone();

        let recipe = serde_json::json!({
            "title": "Cost Critical Recipe",
            "description": "Recipe testing byte-identical prompts",
            "instructions": "Deterministic prompt caching instructions",
            "response": {
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    },
                    "required": ["value"]
                }
            }
        });

        // First settings/set
        send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "1h",
                "recipe": recipe,
            }),
        )
        .await
        .unwrap();

        // Prompt 1
        let out1 = session
            .prompt("what is 1+1", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(out1.tool_status.is_some());

        // Identical settings/set
        send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "1h",
                "recipe": recipe,
            }),
        )
        .await
        .unwrap();

        // Prompt 2
        let out2 = session
            .prompt("what is 2+2", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(out2.tool_status.is_some());

        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs.len(), 2, "expected exactly 2 captured requests");

        let req1: serde_json::Value = serde_json::from_str(&reqs[0]).unwrap();
        let req2: serde_json::Value = serde_json::from_str(&reqs[1]).unwrap();

        assert_eq!(
            req1["messages"][0]["content"].to_string(),
            req2["messages"][0]["content"].to_string(),
            "system prompt must be byte-identical across identical settings/set"
        );

        assert_eq!(
            req1["tools"].to_string(),
            req2["tools"].to_string(),
            "tools array must be byte-identical across identical settings/set"
        );
    });
}

#[test]
#[serial]
fn test_session_settings_templated_recipe_renders_consistently() {
    run_test(async move {
        const FINAL_OUTPUT_RESPONSE: &str = "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_final\",\"type\":\"function\",\"function\":{\"name\":\"recipe__final_output\",\"arguments\":\"{\\\"summary\\\":\\\"done\\\"}\"}}]},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";

        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(
            vec![
                ("what is 1+1".to_string(), FINAL_OUTPUT_RESPONSE),
                ("what is 2+2".to_string(), FINAL_OUTPUT_RESPONSE),
            ],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let captured = openai.captured_requests_handle();

        let mut conn = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai,
        )
        .await;

        let SessionData { mut session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.clone();

        // Recipe with defaulted parameter and recipe_dir template variables
        let recipe = serde_json::json!({
            "title": "Templated Recipe",
            "description": "Recipe with template variables",
            "parameters": [
                {
                    "key": "env",
                    "description": "Environment",
                    "input_type": "string",
                    "requirement": "optional",
                    "default": "production"
                }
            ],
            "instructions": "Target env: {{ env }}. Recipe dir: {{ recipe_dir }}",
            "response": {
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string" }
                    },
                    "required": ["summary"]
                }
            }
        });

        send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "1h",
                "recipe": recipe,
            }),
        )
        .await
        .unwrap();

        let out1 = session
            .prompt("what is 1+1", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(out1.tool_status.is_some());

        // Later settings/set without recipe
        send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "1h",
            }),
        )
        .await
        .unwrap();

        let out2 = session
            .prompt("what is 2+2", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(out2.tool_status.is_some());

        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs.len(), 2);

        let req1: serde_json::Value = serde_json::from_str(&reqs[0]).unwrap();
        let req2: serde_json::Value = serde_json::from_str(&reqs[1]).unwrap();

        let sys1 = req1["messages"][0]["content"].as_str().unwrap();
        let sys2 = req2["messages"][0]["content"].as_str().unwrap();

        // Template syntax does not reach model literally
        assert!(!sys1.contains("{{ env }}"));
        assert!(!sys1.contains("{{ recipe_dir }}"));
        assert!(sys1.contains("Target env: production"));

        // System prompt is byte-identical across the settings/set without recipe
        assert_eq!(
            sys1, sys2,
            "rendered recipe must match on subsequent settings/set without recipe"
        );
    });
}

#[test]
#[serial]
fn test_prompt_self_compact() {
    run_test(async move {
        // Exchange 1: tool call for self_compact_and_resume
        let tool_call_chunk = concat!(
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1766229301,\"model\":\"gpt-5-nano\",",
            "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_compact\",",
            "\"type\":\"function\",\"function\":{\"name\":\"self_compact_and_resume\",",
            "\"arguments\":\"{\\\"prompt\\\":\\\"Resume prompt after compaction\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1766229301,\"model\":\"gpt-5-nano\",",
            "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        // Exchange 2: summarization during compaction
        let summary_chunk = concat!(
            "data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"created\":1766229302,\"model\":\"gpt-5-nano\",",
            "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":",
            "\"{\\\"user_intent\\\":[\\\"compact\\\"],\\\"technical_concepts\\\":[],\\\"files\\\":[],\\\"errors_and_fixes\\\":[],\\\"problem_solving\\\":[],\\\"user_messages\\\":[\\\"start self-compaction\\\"],\\\"pending_tasks\\\":[],\\\"current_work\\\":\\\"compacting\\\"}\"",
            "},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"created\":1766229302,\"model\":\"gpt-5-nano\",",
            "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        // Exchange 3: final assistant reply after resume
        let final_chunk = concat!(
            "data: {\"id\":\"chatcmpl-3\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",",
            "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Resumed successfully after compaction.\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-3\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",",
            "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        let openai = OpenAiFixture::new(
            vec![
                ("start self-compaction".to_string(), tool_call_chunk),
                (
                    "Please summarize the conversation history".to_string(),
                    summary_chunk,
                ),
                ("Resume prompt after compaction".to_string(), final_chunk),
            ],
            Arc::new(IgnoreSessionId),
        )
        .await;

        let mut conn = AcpServerConnection::new(TestConnectionConfig::default(), openai).await;

        let SessionData { session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.clone();

        // 1. Assert self_compact_and_resume is offered in a plain session
        let tools_res = send_custom(
            conn.cx(),
            "_goose/unstable/tools/list",
            serde_json::json!({ "sessionId": session_id }),
        )
        .await
        .unwrap();
        let tools = tools_res
            .get("tools")
            .and_then(|t| t.as_array())
            .expect("tools array");
        assert!(
            tools
                .iter()
                .any(|t| t.get("name").and_then(|n| n.as_str()) == Some("self_compact_and_resume")),
            "self_compact_and_resume tool should be offered in a plain session"
        );

        // Drain any session creation updates so we only observe updates from session/prompt
        let _ = session.session_updates();

        let prompt_res = send_custom(
            conn.cx(),
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "start self-compaction" }],
            }),
        )
        .await
        .unwrap();

        assert_eq!(
            prompt_res.get("stopReason").and_then(|s| s.as_str()),
            Some("end_turn")
        );

        let updates = session.session_updates();

        let active_run_id_from_update = |update: &SessionUpdate| -> Option<Option<String>> {
            match update {
                SessionUpdate::SessionInfoUpdate(info) => {
                    let goose = info.meta.as_ref()?.get("goose")?;
                    let run_val = goose.get("activeRunId")?;
                    if run_val.is_null() {
                        Some(None)
                    } else {
                        Some(run_val.as_str().map(|s| s.to_string()))
                    }
                }
                _ => None,
            }
        };

        let initial_run_id = updates
            .iter()
            .find_map(|u| match active_run_id_from_update(u) {
                Some(Some(id)) => Some(id),
                _ => None,
            })
            .expect("expected initial activeRunId at prompt start");

        let tool_call_idx = updates
            .iter()
            .position(|u| match u {
                SessionUpdate::ToolCall(tc) => {
                    tc.title == "self_compact_and_resume"
                        || tc
                            .meta
                            .as_ref()
                            .and_then(|m| m.get("goose"))
                            .and_then(|g| g.get("toolCall"))
                            .and_then(|t| t.get("toolName"))
                            .and_then(|n| n.as_str())
                            == Some("self_compact_and_resume")
                }
                _ => false,
            })
            .expect("expected tool_call for self_compact_and_resume");

        let resume_chunk_idx = updates
            .iter()
            .position(|u| match u {
                SessionUpdate::UserMessageChunk(chunk) => {
                    let is_steer = chunk
                        .meta
                        .as_ref()
                        .and_then(|m| m.get("goose"))
                        .and_then(|g| g.get("steer"))
                        .and_then(|s| s.as_bool())
                        .unwrap_or(false);
                    let has_text = match &chunk.content {
                        ContentBlock::Text(t) => t.text == "Resume prompt after compaction",
                        _ => false,
                    };
                    is_steer && has_text
                }
                _ => false,
            })
            .expect("expected resume prompt UserMessageChunk with _meta.goose.steer == true");

        // The session_info_update activeRunId stays the same (no activeRunId:null between tool call and resume)
        for (offset, update) in updates[tool_call_idx..=resume_chunk_idx].iter().enumerate() {
            if let Some(run_id_opt) = active_run_id_from_update(update) {
                match run_id_opt {
                    Some(run_id) => {
                        assert_eq!(
                            run_id,
                            initial_run_id,
                            "activeRunId changed at update {}",
                            tool_call_idx + offset
                        );
                    }
                    None => {
                        panic!(
                            "unexpected activeRunId: null between tool call and resume at update {}",
                            tool_call_idx + offset
                        );
                    }
                }
            }
        }

        // A "Context self-compacted." agent_message_chunk is seen
        let note_chunk_idx = updates
            .iter()
            .position(|u| match u {
                SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                    ContentBlock::Text(t) => t.text.contains("Context self-compacted."),
                    _ => false,
                },
                _ => false,
            })
            .expect("expected 'Context self-compacted.' AgentMessageChunk");

        // A usage_update follows compaction (pick the one emitted upon compaction immediately before note chunk)
        let usage_update_idx = updates[tool_call_idx..note_chunk_idx]
            .iter()
            .rposition(|u| matches!(u, SessionUpdate::UsageUpdate(_)))
            .map(|idx| tool_call_idx + idx)
            .expect("expected usage_update between tool call and note chunk");

        if let SessionUpdate::UsageUpdate(ref u) = updates[usage_update_idx] {
            assert!(
                u.used > 0,
                "UsageUpdate used should reflect retained tokens (> 0)"
            );
            assert!(
                u.used < 1000,
                "UsageUpdate used should reflect compact retained tokens, not full history: {}",
                u.used
            );
        }

        assert!(
            tool_call_idx < usage_update_idx,
            "usage_update must follow tool call"
        );
        assert!(
            usage_update_idx <= note_chunk_idx,
            "usage_update must precede or accompany compaction note chunk"
        );
        assert!(
            note_chunk_idx < resume_chunk_idx,
            "note chunk must precede resume chunk"
        );

        // The final assistant text after the resume is delivered
        let final_assistant_idx = updates
            .iter()
            .position(|u| match u {
                SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                    ContentBlock::Text(t) => {
                        t.text.contains("Resumed successfully after compaction")
                    }
                    _ => false,
                },
                _ => false,
            })
            .expect("expected final assistant text after resume");

        assert!(
            resume_chunk_idx < final_assistant_idx,
            "final assistant text must be delivered after resume chunk"
        );

        // Run completes with activeRunId: null
        let final_null_idx = updates
            .iter()
            .position(|u| active_run_id_from_update(u) == Some(None))
            .expect("expected activeRunId: null at end of prompt");
        assert!(
            final_null_idx > resume_chunk_idx,
            "activeRunId: null should arrive after the resume chunk"
        );
    });
}

/// Regression guard: a recipe session (with response schema) loaded on a fresh connection
/// must never list self_compact_and_resume.
#[tokio::test]
async fn test_recipe_session_after_reload_never_lists_self_compact_and_resume() {
    run_test(async move {
        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(vec![], Arc::new(IgnoreSessionId)).await;
        let mut conn = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai,
        )
        .await;

        let SessionData { session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.clone();

        // Recipe with a response schema (recipe final_output)
        let recipe = serde_json::json!({
            "title": "Response Schema Recipe",
            "description": "Recipe with response schema",
            "instructions": "Instructions",
            "response": {
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "result": { "type": "string" }
                    },
                    "required": ["result"]
                }
            }
        });

        send_custom(
            conn.cx(),
            "_goose/unstable/session/settings/set",
            serde_json::json!({
                "sessionId": session_id,
                "recipe": recipe,
            }),
        )
        .await
        .unwrap();

        // Verify tool list before reload
        let tools_before = send_custom(
            conn.cx(),
            "_goose/unstable/tools/list",
            serde_json::json!({ "sessionId": session_id }),
        )
        .await
        .unwrap();
        let tools_before_arr = tools_before
            .get("tools")
            .and_then(|v| v.as_array())
            .expect("tools array");
        assert!(
            !tools_before_arr
                .iter()
                .any(|t| t["name"] == "self_compact_and_resume"),
            "Before reload: session with response schema must NOT list self_compact_and_resume"
        );

        // Fresh connection to simulate restored session after reload
        let openai2 = OpenAiFixture::new(vec![], Arc::new(IgnoreSessionId)).await;
        let mut conn2 = AcpServerConnection::new(
            TestConnectionConfig {
                data_root: temp_data.path().to_path_buf(),
                ..Default::default()
            },
            openai2,
        )
        .await;

        let _loaded = conn2.load_session(&session_id, vec![]).await.unwrap();

        let tools_after = send_custom(
            conn2.cx(),
            "_goose/unstable/tools/list",
            serde_json::json!({ "sessionId": session_id }),
        )
        .await
        .unwrap();
        let tools_after_arr = tools_after
            .get("tools")
            .and_then(|v| v.as_array())
            .expect("tools array");

        assert!(
            !tools_after_arr.iter().any(|t| t["name"] == "self_compact_and_resume"),
            "After reload: restored session with response schema must NOT list self_compact_and_resume"
        );
        assert_eq!(
            tools_before_arr.len(),
            tools_after_arr.len(),
            "Tool list length must match before and after reload"
        );
    });
}

#[test]
#[serial]
fn test_prompt_turn_error_reports_a_failed_turn_in_meta() {
    run_test(async move {
        // The first prompt matches no exchange, so the mock answers 417 and the
        // provider call fails; the second prompt succeeds.
        let openai = OpenAiFixture::new(
            vec![(
                "what is 1+1".to_string(),
                include_str!("acp_test_data/openai_basic.txt"),
            )],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let mut conn = AcpServerConnection::new(TestConnectionConfig::default(), openai).await;
        let SessionData { session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.clone();

        let prompt = |text: &'static str| {
            send_custom(
                conn.cx(),
                "session/prompt",
                serde_json::json!({
                    "sessionId": session_id,
                    "prompt": [{ "type": "text", "text": text }],
                }),
            )
        };

        let failed = prompt("this prompt is not scripted").await.unwrap();
        assert_eq!(
            failed.get("stopReason").and_then(|s| s.as_str()),
            Some("end_turn")
        );
        let turn_error = failed
            .pointer("/_meta/goose/turnError")
            .unwrap_or_else(|| panic!("failed turn should carry turnError: {failed}"));
        assert_eq!(
            turn_error.get("kind").and_then(|k| k.as_str()),
            Some("provider")
        );
        assert!(
            turn_error
                .get("message")
                .and_then(|m| m.as_str())
                .is_some_and(|m| !m.is_empty()),
            "turnError should carry a message: {turn_error}"
        );

        let succeeded = prompt("what is 1+1").await.unwrap();
        assert_eq!(
            succeeded.get("stopReason").and_then(|s| s.as_str()),
            Some("end_turn")
        );
        assert!(
            succeeded.pointer("/_meta/goose/turnError").is_none(),
            "a clean turn must not carry turnError: {succeeded}"
        );
    });
}
