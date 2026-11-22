// Copyright 2026 Jonathan Mohrbacher. Licensed under the Apache License, Version 2.0.
// Fork-only tests: a failed stdio extension start kills the whole process group, so a
// grandchild that inherited stderr cannot keep spawn() waiting forever.

#[cfg(unix)]
#[tokio::test]
async fn test_mcp_client_startup_timeout_stdio_kills_process_group() {
    use crate::agents::mcp_startup_timeout::tests::{
        assert_dies, assert_startup_timeout_message, hung_server_ctx, read_pid,
    };
    use std::time::Duration;

    let temp_dir = tempfile::tempdir().unwrap();
    let grandchild_pid_path = temp_dir.path().join("grandchild.pid");
    let mut command = tokio::process::Command::new("sh");
    command.args([
        "-c",
        &format!(
            "sleep 600 & echo $! > \"{}\"; echo starting >&2; exec sleep 600",
            grandchild_pid_path.display()
        ),
    ]);

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        super::spawn(command, hung_server_ctx(temp_dir.path(), None)),
    )
    .await
    .expect("stdio spawn hung past the startup timeout");

    let message = match result {
        Ok(_) => panic!("spawn of a server that never initializes succeeded"),
        Err(error) => error.to_string(),
    };
    assert_startup_timeout_message(&message);
    assert!(
        message.contains("starting"),
        "error should carry the captured stderr, got: {message}"
    );
    assert_dies(read_pid(&grandchild_pid_path).await).await;
}
