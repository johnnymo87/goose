// Copyright 2026 Jonathan Mohrbacher. Licensed under the Apache License, Version 2.0.
// Fork-only module: bound MCP server startup (GOOSE_EXTENSION_STARTUP_TIMEOUT, default 25s,
// 0 treated as default) so a hung server cannot hang session/new, session/load,
// extensions/add or an extension lease.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use rmcp::service::ClientInitializeError;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::Mutex;

use crate::config::Config;

pub const DEFAULT_EXTENSION_STARTUP_TIMEOUT: u64 = 25;

pub(crate) fn resolve_startup_timeout_from_config(
    override_timeout: Option<Duration>,
    config_timeout: Option<u64>,
) -> Duration {
    override_timeout
        .filter(|d| !d.is_zero())
        .or_else(|| {
            config_timeout
                .filter(|seconds| *seconds > 0)
                .map(Duration::from_secs)
        })
        .unwrap_or_else(|| Duration::from_secs(DEFAULT_EXTENSION_STARTUP_TIMEOUT))
}

pub(crate) fn resolve_startup_timeout(override_timeout: Option<Duration>) -> Duration {
    resolve_startup_timeout_from_config(
        override_timeout,
        Config::global().get_goose_extension_startup_timeout().ok(),
    )
}

fn format_startup_timeout(d: Duration) -> String {
    if d.subsec_millis() == 0 {
        format!("{}s", d.as_secs())
    } else if d.as_secs() == 0 {
        format!("{}ms", d.as_millis())
    } else {
        format!("{:.1}s", d.as_secs_f64())
    }
}

pub(crate) async fn bound_startup<F, T>(
    override_timeout: Option<Duration>,
    startup: F,
) -> Result<T, Box<ClientInitializeError>>
where
    F: Future<Output = Result<T, ClientInitializeError>>,
{
    let limit = resolve_startup_timeout(override_timeout);
    match tokio::time::timeout(limit, startup).await {
        Ok(result) => result.map_err(Box::new),
        Err(_) => Err(Box::new(ClientInitializeError::ConnectionClosed(format!(
            "MCP server did not finish initializing within {} (GOOSE_EXTENSION_STARTUP_TIMEOUT)",
            format_startup_timeout(limit)
        )))),
    }
}

/// Reads a starting server's stderr in the background. On a failed start the
/// error carries what was read so far: a grandchild that inherited the pipe can
/// keep it open past the child's death, so waiting for EOF could hang.
pub(crate) struct StartupStderr {
    buf: Arc<Mutex<Vec<u8>>>,
    task: tokio::task::JoinHandle<()>,
}

impl StartupStderr {
    pub(crate) fn capture<R>(mut stderr: R) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
    {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&buf);
        let task = tokio::spawn(async move {
            let mut chunk = [0u8; 1024];
            while let Ok(n) = stderr.read(&mut chunk).await {
                if n == 0 {
                    break;
                }
                sink.lock().await.extend_from_slice(&chunk[..n]);
            }
        });
        Self { buf, task }
    }

    pub(crate) async fn collect(mut self) -> String {
        if tokio::time::timeout(STDERR_DRAIN_GRACE, &mut self.task)
            .await
            .is_err()
        {
            self.task.abort();
        }
        String::from_utf8_lossy(&self.buf.lock().await).into_owned()
    }
}

const STDERR_DRAIN_GRACE: Duration = Duration::from_secs(2);

/// The child is its own process-group leader (`configure_subprocess` sets
/// `process_group(0)`), so this also reaches grandchildren such as `npx`'s node.
pub(crate) fn kill_process_group(child_pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = child_pid.and_then(|pid| libc::pid_t::try_from(pid).ok()) {
        unsafe {
            libc::killpg(pid, libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    let _ = child_pid;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::action_required_manager::ActionRequiredManager;
    use crate::agents::mcp_client::{ConnectContext, GooseMcpClientCapabilities, McpClient};
    use rmcp::model::ProtocolVersion;
    use std::sync::atomic::AtomicU64;

    pub(crate) fn hung_server_ctx(
        working_dir: &std::path::Path,
        protocol_version: Option<ProtocolVersion>,
    ) -> ConnectContext {
        ConnectContext {
            timeout: Duration::from_secs(30),
            startup_timeout: Some(Duration::from_millis(500)),
            client_name: "test".to_string(),
            capabilities: GooseMcpClientCapabilities {
                protocol_version,
                ..Default::default()
            },
            working_dir: working_dir.to_path_buf(),
            docker_container: None,
            action_required: Arc::new(ActionRequiredManager::new()),
            tools_version: Arc::new(AtomicU64::new(0)),
        }
    }

    #[cfg(unix)]
    pub(crate) fn process_is_alive(pid: i32) -> bool {
        if unsafe { libc::kill(pid, 0) } != 0 {
            return false;
        }
        #[cfg(target_os = "linux")]
        if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            if let Some((_, after_name)) = stat.rsplit_once(") ") {
                if let Some(state) = after_name.chars().next() {
                    return state != 'Z' && state != 'X';
                }
            }
        }
        true
    }

    #[cfg(unix)]
    pub(crate) async fn read_pid(path: &std::path::Path) -> i32 {
        for _ in 0..100 {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(pid) = content.trim().parse::<i32>() {
                    return pid;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("no pid written to {}", path.display());
    }

    #[cfg(unix)]
    pub(crate) async fn assert_dies(pid: i32) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if !process_is_alive(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("process {pid} still alive 5s after startup timeout");
    }

    pub(crate) fn assert_startup_timeout_message(msg: &str) {
        assert!(
            msg.contains("MCP server did not finish initializing within 500ms")
                && msg.contains("GOOSE_EXTENSION_STARTUP_TIMEOUT"),
            "error should name the startup timeout, got: {msg}"
        );
    }

    #[cfg(unix)]
    async fn assert_connect_times_out_and_kills_child(protocol_version: Option<ProtocolVersion>) {
        let temp_dir = tempfile::tempdir().unwrap();
        let pid_path = temp_dir.path().join("pid");
        let mut command = tokio::process::Command::new("sh");
        command.args([
            "-c",
            &format!("echo $$ > \"{}\" && exec sleep 600", pid_path.display()),
        ]);
        let (transport, _stderr) = crate::subprocess::spawn_long_lived_mcp_subprocess(command)
            .await
            .unwrap();
        let pid = read_pid(&pid_path).await;
        assert!(process_is_alive(pid));

        let result = tokio::time::timeout(
            Duration::from_secs(3),
            McpClient::connect(
                transport,
                hung_server_ctx(temp_dir.path(), protocol_version),
            ),
        )
        .await
        .expect("McpClient::connect hung past the 500ms startup timeout");

        match result {
            Ok(_) => panic!("connect to a server that never initializes succeeded"),
            Err(error) => assert_startup_timeout_message(&error.to_string()),
        }
        assert_dies(pid).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_mcp_client_startup_timeout_auto_lifecycle() {
        assert_connect_times_out_and_kills_child(None).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_mcp_client_startup_timeout_discover_lifecycle() {
        assert_connect_times_out_and_kills_child(Some(ProtocolVersion::STANDARD_HEADERS)).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_mcp_client_startup_timeout_initialize_lifecycle() {
        assert_connect_times_out_and_kills_child(Some(ProtocolVersion::V_2025_11_25)).await;
    }

    #[test]
    fn test_mcp_client_startup_timeout_format() {
        assert_eq!(format_startup_timeout(Duration::from_secs(25)), "25s");
        assert_eq!(format_startup_timeout(Duration::from_secs(1)), "1s");
        assert_eq!(format_startup_timeout(Duration::from_millis(500)), "500ms");
        assert_eq!(format_startup_timeout(Duration::from_millis(1500)), "1.5s");
    }

    #[test]
    fn test_mcp_client_startup_timeout_resolve() {
        let default_timeout = Duration::from_secs(DEFAULT_EXTENSION_STARTUP_TIMEOUT);

        assert_eq!(
            resolve_startup_timeout_from_config(Some(Duration::from_secs(42)), Some(10)),
            Duration::from_secs(42)
        );
        assert_eq!(
            resolve_startup_timeout_from_config(Some(Duration::ZERO), Some(10)),
            Duration::from_secs(10)
        );
        assert_eq!(
            resolve_startup_timeout_from_config(Some(Duration::ZERO), Some(0)),
            default_timeout
        );
        assert_eq!(
            resolve_startup_timeout_from_config(Some(Duration::ZERO), None),
            default_timeout
        );
        assert_eq!(
            resolve_startup_timeout_from_config(None, Some(15)),
            Duration::from_secs(15)
        );
        assert_eq!(
            resolve_startup_timeout_from_config(None, Some(0)),
            default_timeout
        );
        assert_eq!(
            resolve_startup_timeout_from_config(None, None),
            default_timeout
        );
    }
}
