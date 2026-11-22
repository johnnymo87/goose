// Modified by Jonathan Mohrbacher, 2026: test _goose/unstable/session/settings/set (cache TTL, recipe, rolling memory) and session/prompt _meta.goose.maxTurns (stopReason max_turn_requests).
#![recursion_limit = "256"]
#[allow(dead_code)]
#[path = "acp_common_tests/mod.rs"]
mod common_tests;

use agent_client_protocol::schema::v1::{ContentBlock, SessionUpdate};
use agent_client_protocol::ErrorCode;
use common_tests::fixtures::server::AcpServerConnection;
use common_tests::fixtures::{
    run_test, send_custom, Connection, OpenAiFixture, PermissionDecision, Session, SessionData,
    TestConnectionConfig,
};
use goose::session::SessionManager;
use goose_test_support::IgnoreSessionId;
use serial_test::serial;
use std::sync::Arc;

const SETTINGS_SET: &str = "_goose/unstable/session/settings/set";

const FINAL_OUTPUT_SUMMARY_RESPONSE: &str = "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_final\",\"type\":\"function\",\"function\":{\"name\":\"recipe__final_output\",\"arguments\":\"{\\\"summary\\\":\\\"done\\\"}\"}}]},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";

const FINAL_OUTPUT_VALUE_RESPONSE: &str = "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_final\",\"type\":\"function\",\"function\":{\"name\":\"recipe__final_output\",\"arguments\":\"{\\\"value\\\":\\\"done\\\"}\"}}]},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";

const TEXT_TURN_1: &str = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Turn 1 working\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1766229303,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

const TEXT_TURN_2: &str = "data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"created\":1766229304,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Turn 2 working\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"created\":1766229304,\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

async fn connect(data_root: &std::path::Path, openai: OpenAiFixture) -> AcpServerConnection {
    AcpServerConnection::new(
        TestConnectionConfig {
            data_root: data_root.to_path_buf(),
            ..Default::default()
        },
        openai,
    )
    .await
}

fn offers_final_output(request_body: &str) -> bool {
    let request: serde_json::Value = serde_json::from_str(request_body).unwrap();
    request["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|tool| tool["function"]["name"] == "recipe__final_output")
}

fn data_text(err: &agent_client_protocol::Error) -> String {
    err.data.as_ref().map(|d| d.to_string()).unwrap_or_default()
}

fn agent_text(updates: &[SessionUpdate]) -> String {
    updates
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn summary_recipe() -> serde_json::Value {
    serde_json::json!({
        "title": "Settings Test Recipe",
        "description": "Test recipe description",
        "instructions": "Special test recipe instructions",
        "response": {
            "json_schema": {
                "type": "object",
                "properties": { "summary": { "type": "string" } },
                "required": ["summary"]
            }
        }
    })
}

#[test]
#[serial]
fn test_session_settings_set_rejects_invalid_input() {
    run_test(async move {
        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(vec![], Arc::new(IgnoreSessionId)).await;
        let mut conn = connect(temp_data.path(), openai).await;
        let SessionData { session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.to_string();

        let err = send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({ "sessionId": session_id, "cacheTtl": "2h" }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidParams);
        assert!(data_text(&err).contains("invalid cacheTtl: "), "{err:?}");

        let err = send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({
                "sessionId": session_id,
                "rollingMemory": { "budgetTokens": 5000, "tailTokens": 500 }
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidParams);
        assert!(
            data_text(&err).contains("invalid rollingMemory: "),
            "{err:?}"
        );

        let err = send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({
                "sessionId": "unknown-nonexistent-session",
                "recipe": { "title": "Some Recipe", "description": "Description" }
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::ResourceNotFound);

        send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({ "sessionId": session_id, "recipe": summary_recipe() }),
        )
        .await
        .unwrap();

        let rejected = [
            serde_json::json!({
                "title": "Bad Schema Recipe",
                "description": "Empty schema",
                "response": { "json_schema": {} }
            }),
            serde_json::json!({
                "title": "Bad Template Recipe",
                "description": "Unclosed jinja",
                "instructions": "Unclosed {% bad tag"
            }),
        ];
        for recipe in rejected {
            let err = send_custom(
                conn.cx(),
                SETTINGS_SET,
                serde_json::json!({ "sessionId": session_id, "recipe": recipe }),
            )
            .await
            .unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidParams);
            assert!(data_text(&err).contains("invalid recipe: "), "{err:?}");
        }

        let stored = SessionManager::new(temp_data.path().to_path_buf())
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored.recipe.and_then(|r| r.instructions).as_deref(),
            Some("Special test recipe instructions"),
            "a rejected recipe must not replace the stored one"
        );
    });
}

#[test]
#[serial]
fn test_session_settings_set_persists_and_survives_reload() {
    run_test(async move {
        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(
            vec![(
                "Special test recipe instructions".to_string(),
                FINAL_OUTPUT_SUMMARY_RESPONSE,
            )],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let captured = openai.captured_requests_handle();
        let mut conn = connect(temp_data.path(), openai).await;
        let SessionData { mut session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.to_string();
        let session_manager = SessionManager::new(temp_data.path().to_path_buf());

        let mut recipe = summary_recipe();
        recipe["retry"] = serde_json::json!({
            "max_retries": 1,
            "checks": [{ "type": "shell", "command": "false" }]
        });
        recipe["sub_recipes"] = serde_json::json!([{ "name": "child", "path": "child.yaml" }]);
        for _ in 0..2 {
            let res = send_custom(
                conn.cx(),
                SETTINGS_SET,
                serde_json::json!({
                    "sessionId": session_id,
                    "cacheTtl": "1h",
                    "recipe": recipe,
                    "rollingMemory": { "budgetTokens": 20000, "tailTokens": 2000 },
                }),
            )
            .await
            .unwrap();
            assert_eq!(res, serde_json::json!({}));
        }

        let stored = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored.model_config.as_ref().and_then(|mc| mc.cache_ttl()),
            Some("1h".to_string())
        );
        let stored_recipe = stored.recipe.expect("recipe stored");
        assert!(stored_recipe.retry.is_none(), "retry must not be stored");
        assert!(
            stored_recipe.sub_recipes.is_none(),
            "sub_recipes must not be stored"
        );
        assert_eq!(
            stored
                .extension_data
                .get_extension_state("rolling_memory", "v0"),
            Some(&serde_json::json!({ "budgetTokens": 20000, "tailTokens": 2000 }))
        );

        // Setting rollingMemory: null turns it off (explicit null)
        send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({
                "sessionId": session_id,
                "rollingMemory": null,
            }),
        )
        .await
        .unwrap();
        let stored = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored
                .extension_data
                .get_extension_state("rolling_memory", "v0"),
            Some(&serde_json::Value::Null)
        );

        // Absent rollingMemory leaves the stored state unchanged
        send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "1h",
            }),
        )
        .await
        .unwrap();
        let stored = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored
                .extension_data
                .get_extension_state("rolling_memory", "v0"),
            Some(&serde_json::Value::Null)
        );
        let out = session
            .prompt("Hello with recipe", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(out.tool_status.is_some());
        let first = captured.lock().unwrap().clone();
        assert_eq!(first.len(), 1);
        assert!(offers_final_output(&first[0]));

        let openai2 = OpenAiFixture::new(
            vec![(
                "Special test recipe instructions".to_string(),
                FINAL_OUTPUT_SUMMARY_RESPONSE,
            )],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let captured2 = openai2.captured_requests_handle();
        let mut conn2 = connect(temp_data.path(), openai2).await;
        let SessionData {
            session: mut loaded,
            ..
        } = conn2.load_session(&session_id, vec![]).await.unwrap();
        let out = loaded
            .prompt("Hello after reload", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(out.tool_status.is_some());
        let second = captured2.lock().unwrap().clone();
        assert_eq!(second.len(), 1);
        assert!(offers_final_output(&second[0]));
        let stored = session_manager
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored.model_config.as_ref().and_then(|mc| mc.cache_ttl()),
            Some("1h".to_string()),
            "session/load must keep the pinned cache TTL"
        );
    });
}

#[test]
#[serial]
fn test_session_settings_set_activates_cold_session() {
    run_test(async move {
        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(vec![], Arc::new(IgnoreSessionId)).await;
        let mut conn = connect(temp_data.path(), openai).await;
        let SessionData { session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.to_string();
        send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({ "sessionId": session_id, "recipe": summary_recipe() }),
        )
        .await
        .unwrap();
        drop(session);
        drop(conn);

        let openai = OpenAiFixture::new(
            vec![
                (
                    "Special test recipe instructions".to_string(),
                    FINAL_OUTPUT_SUMMARY_RESPONSE,
                ),
                (
                    "Simple instructions without final output".to_string(),
                    TEXT_TURN_1,
                ),
            ],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let captured = openai.captured_requests_handle();
        let cold = connect(temp_data.path(), openai).await;
        let res = send_custom(
            cold.cx(),
            SETTINGS_SET,
            serde_json::json!({ "sessionId": session_id, "cacheTtl": "1h" }),
        )
        .await
        .unwrap();
        assert_eq!(res, serde_json::json!({}));
        let stored = SessionManager::new(temp_data.path().to_path_buf())
            .get_session(&session_id, false)
            .await
            .unwrap();
        assert_eq!(
            stored.model_config.as_ref().and_then(|mc| mc.cache_ttl()),
            Some("1h".to_string())
        );

        let res = send_custom(
            cold.cx(),
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "Testing cold prompt" }],
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            res.get("stopReason").and_then(|s| s.as_str()),
            Some("end_turn"),
            "the stored recipe's instructions must reach the provider: {res}"
        );

        send_custom(
            cold.cx(),
            SETTINGS_SET,
            serde_json::json!({
                "sessionId": session_id,
                "recipe": {
                    "title": "Recipe Without Response",
                    "description": "Recipe with no output schema",
                    "instructions": "Simple instructions without final output",
                },
            }),
        )
        .await
        .unwrap();
        let res = send_custom(
            cold.cx(),
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "Testing replaced recipe" }],
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            res.get("stopReason").and_then(|s| s.as_str()),
            Some("end_turn")
        );
        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs.len(), 2);
        assert!(offers_final_output(&reqs[0]));
        assert!(
            !offers_final_output(&reqs[1]),
            "a recipe without response removes the final_output tool"
        );
        assert!(!reqs[1].contains("Special test recipe instructions"));
    });
}

#[test]
#[serial]
fn test_prompt_max_turns_bounds_run() {
    run_test(async move {
        let openai = OpenAiFixture::new(
            vec![
                ("what is 1+1".to_string(), TEXT_TURN_1),
                ("final_output".to_string(), TEXT_TURN_2),
            ],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let captured = openai.captured_requests_handle();
        let mut conn = AcpServerConnection::new(TestConnectionConfig::default(), openai).await;
        let SessionData { session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.to_string();

        send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({
                "sessionId": session_id,
                "recipe": {
                    "title": "Recipe With Output",
                    "description": "Recipe requiring final output",
                    "instructions": "Always produce structured final output",
                    "response": {
                        "json_schema": {
                            "type": "object",
                            "properties": { "ans": { "type": "string" } },
                            "required": ["ans"]
                        }
                    }
                },
            }),
        )
        .await
        .unwrap();

        for bad in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(2.5),
            serde_json::json!("two"),
        ] {
            let err = send_custom(
                conn.cx(),
                "session/prompt",
                serde_json::json!({
                    "sessionId": session_id,
                    "prompt": [{ "type": "text", "text": "what is 1+1" }],
                    "_meta": { "goose": { "maxTurns": bad } }
                }),
            )
            .await
            .unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidParams, "maxTurns {bad}");
        }
        assert!(captured.lock().unwrap().is_empty());

        let result = send_custom(
            conn.cx(),
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "what is 1+1" }],
                "_meta": { "goose": { "maxTurns": 1 } }
            }),
        )
        .await;

        assert_eq!(
            captured.lock().unwrap().len(),
            1,
            "maxTurns 1 must stop the run before the final_output continuation: {result:?}"
        );
        let response = result.expect("reaching max turns is a normal stop, not a failed turn");
        assert_eq!(response["stopReason"], "max_turn_requests");
        let text = agent_text(&session.session_updates());
        assert!(
            text.contains("maximum number of actions"),
            "expected the max turns message: {text}"
        );
    });
}

#[test]
#[serial]
fn test_session_settings_idempotent_prompts_byte_identical() {
    run_test(async move {
        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(
            vec![
                ("what is 1+1".to_string(), FINAL_OUTPUT_VALUE_RESPONSE),
                ("what is 2+2".to_string(), FINAL_OUTPUT_VALUE_RESPONSE),
            ],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let captured = openai.captured_requests_handle();
        let mut conn = connect(temp_data.path(), openai).await;
        let SessionData { mut session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.to_string();
        let settings = serde_json::json!({
            "sessionId": session_id,
            "cacheTtl": "1h",
            "recipe": {
                "title": "Cost Critical Recipe",
                "description": "Recipe testing byte-identical prompts",
                "instructions": "Deterministic prompt caching instructions",
                "response": {
                    "json_schema": {
                        "type": "object",
                        "properties": { "value": { "type": "string" } },
                        "required": ["value"]
                    }
                }
            },
        });

        send_custom(conn.cx(), SETTINGS_SET, settings.clone())
            .await
            .unwrap();
        let out1 = session
            .prompt("what is 1+1", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(out1.tool_status.is_some());
        send_custom(conn.cx(), SETTINGS_SET, settings)
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
        assert_eq!(
            req1["messages"][0]["content"].to_string(),
            req2["messages"][0]["content"].to_string(),
            "system prompt must be byte-identical across identical settings/set"
        );
        assert_eq!(
            req1["tools"].to_string(),
            req2["tools"].to_string(),
            "tools must be byte-identical across identical settings/set"
        );
    });
}

#[test]
#[serial]
fn test_session_settings_templated_recipe_renders_consistently() {
    run_test(async move {
        let temp_data = tempfile::tempdir().unwrap();
        let openai = OpenAiFixture::new(
            vec![
                ("what is 1+1".to_string(), FINAL_OUTPUT_SUMMARY_RESPONSE),
                ("what is 2+2".to_string(), FINAL_OUTPUT_SUMMARY_RESPONSE),
            ],
            Arc::new(IgnoreSessionId),
        )
        .await;
        let captured = openai.captured_requests_handle();
        let mut conn = connect(temp_data.path(), openai).await;
        let SessionData { mut session, .. } = conn.new_session().await.unwrap();
        let session_id = session.session_id().0.to_string();

        send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({
                "sessionId": session_id,
                "cacheTtl": "1h",
                "recipe": {
                    "title": "Templated Recipe",
                    "description": "Recipe with template variables",
                    "parameters": [{
                        "key": "env",
                        "description": "Environment",
                        "input_type": "string",
                        "requirement": "optional",
                        "default": "production"
                    }],
                    "instructions": "Target env: {{ env }}. Recipe dir: {{ recipe_dir }}",
                    "response": {
                        "json_schema": {
                            "type": "object",
                            "properties": { "summary": { "type": "string" } },
                            "required": ["summary"]
                        }
                    }
                },
            }),
        )
        .await
        .unwrap();
        let out1 = session
            .prompt("what is 1+1", PermissionDecision::AllowOnce)
            .await
            .unwrap();
        assert!(out1.tool_status.is_some());

        send_custom(
            conn.cx(),
            SETTINGS_SET,
            serde_json::json!({ "sessionId": session_id, "cacheTtl": "1h" }),
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
        assert!(!sys1.contains("{{ env }}"));
        assert!(!sys1.contains("{{ recipe_dir }}"));
        assert!(sys1.contains("Target env: production"));
        assert_eq!(sys1, sys2);
    });
}
