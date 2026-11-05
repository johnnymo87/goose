use crate::agents::extension::{Envs, ExtensionConfig};
use crate::config::{DEFAULT_EXTENSION_DESCRIPTION, DEFAULT_EXTENSION_TIMEOUT};
use crate::plugins::discovery::{discover_enabled_plugins, PluginScope};
use crate::plugins::formats::open_plugins;
use anyhow::{bail, Context, Result};
use fs_err as fs;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::warn;

const DEFAULT_MCP_CONFIG: &str = ".mcp.json";
const PLUGIN_ROOT: &str = "${PLUGIN_ROOT}";

/// Server entries stay raw JSON so that one malformed entry is skipped on its
/// own instead of failing the whole document (Agent Plugins spec §7.2.2).
#[derive(Debug, Deserialize)]
struct McpServersDocument {
    #[serde(default, rename = "mcpServers")]
    mcp_servers: HashMap<String, serde_json::Value>,
}

#[derive(Debug)]
enum McpServerConfig {
    Stdio(StdioServerConfig),
    StreamableHttp(HttpServerConfig),
}

#[derive(Debug, Deserialize)]
struct StdioServerConfig {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default)]
    cwd: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HttpServerConfig {
    url: String,
    #[serde(default)]
    headers: HashMap<String, String>,
}

fn parse_server(value: &serde_json::Value, scope: PluginScope) -> Result<McpServerConfig> {
    let Some(object) = value.as_object() else {
        bail!("server entry must be an object");
    };
    let transport = match object.get("type") {
        None => None,
        Some(serde_json::Value::String(transport)) => Some(transport.as_str()),
        Some(_) => bail!("`type` must be a string"),
    };

    let is_http = match transport {
        Some("stdio") => false,
        Some("streamable-http" | "http") => true,
        Some("sse") => bail!("the legacy HTTP+SSE transport is not supported"),
        Some(other) => bail!("unsupported transport type '{other}'"),
        None if object.contains_key("command") => false,
        None if object.contains_key("url") => true,
        None => bail!("server entry needs a `command` or a `url`"),
    };

    if is_http {
        let server: HttpServerConfig = serde_json::from_value(value.clone())?;
        validate_remote_url(&server.url, scope)?;
        validate_headers(&server.headers)?;
        Ok(McpServerConfig::StreamableHttp(server))
    } else {
        let server: StdioServerConfig = serde_json::from_value(value.clone())?;
        if server.command.trim().is_empty() {
            bail!("command must not be empty");
        }
        Ok(McpServerConfig::Stdio(server))
    }
}

fn validate_remote_url(raw: &str, scope: PluginScope) -> Result<()> {
    if raw.contains("${") {
        bail!("placeholders are not expanded in url '{raw}'");
    }
    let url = url::Url::parse(raw).with_context(|| format!("invalid url '{raw}'"))?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("url '{raw}' must use http or https");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("url must not contain user information");
    }
    if url.fragment().is_some() {
        bail!("url '{raw}' must not contain a fragment");
    }
    let is_loopback = match url.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    };
    if url.scheme() == "http" && !is_loopback {
        bail!("url '{raw}' must use https unless the host is localhost or a loopback address");
    }
    // Project plugins load from whatever repository goose is started in, without
    // consent. A remote server answering 401 would start goose's browser OAuth
    // flow from that repository, so project plugins may only reach this machine.
    if scope == PluginScope::Project && !is_loopback {
        bail!("project plugins may only connect to localhost or loopback urls, not '{raw}'");
    }
    Ok(())
}

fn validate_headers(headers: &HashMap<String, String>) -> Result<()> {
    let mut seen = HashSet::new();
    for (name, value) in headers {
        if value.contains("${") {
            bail!("placeholders are not expanded in header '{name}'");
        }
        let header_name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
            .with_context(|| format!("invalid header name '{name}'"))?;
        reqwest::header::HeaderValue::from_str(value)
            .with_context(|| format!("invalid value for header '{name}'"))?;
        if !seen.insert(header_name) {
            bail!("header '{name}' is set more than once");
        }
    }
    Ok(())
}

fn parse_servers(
    servers: HashMap<String, serde_json::Value>,
    scope: PluginScope,
) -> Vec<(String, Result<McpServerConfig>)> {
    let mut entries: Vec<_> = servers
        .into_iter()
        .map(|(name, value)| {
            let parsed = parse_server(&value, scope);
            (name, parsed)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

pub fn enabled_plugin_mcp_servers(project_root: Option<&Path>) -> Vec<ExtensionConfig> {
    let mut configs = Vec::new();
    for plugin in discover_enabled_plugins(project_root) {
        match plugin_mcp_servers(&plugin.name, &plugin.root, plugin.scope) {
            Ok(plugin_configs) => configs.extend(plugin_configs),
            Err(err) => warn!(
                plugin = %plugin.name,
                root = %plugin.root.display(),
                error = %err,
                "Failed to load plugin MCP servers; skipping",
            ),
        }
    }
    configs
}

pub fn plugin_mcp_servers(
    plugin_name: &str,
    plugin_root: &Path,
    scope: PluginScope,
) -> Result<Vec<ExtensionConfig>> {
    let manifest = open_plugins::read_manifest(plugin_root, "")?;
    let mut configs = Vec::new();
    let mut seen = HashSet::new();

    for path in mcp_config_paths(plugin_root, manifest.mcp_servers.as_ref())? {
        if !seen.insert(path.clone()) || !path.is_file() {
            continue;
        }

        let document = serde_json::from_str::<McpServersDocument>(&fs::read_to_string(&path)?)
            .with_context(|| format!("Failed to parse {}", path.display()))?;
        configs.extend(document_to_extension_configs(
            plugin_name,
            plugin_root,
            document.mcp_servers,
            scope,
        ));
    }

    if let Some(value) = manifest
        .mcp_servers
        .as_ref()
        .filter(|value| is_inline_config(value))
    {
        let servers = parse_inline_servers(value)?;
        configs.extend(document_to_extension_configs(
            plugin_name,
            plugin_root,
            servers,
            scope,
        ));
    }

    Ok(configs)
}

fn mcp_config_paths(
    plugin_root: &Path,
    config: Option<&serde_json::Value>,
) -> Result<Vec<PathBuf>> {
    let custom_paths = config
        .filter(|value| !is_inline_config(value))
        .map(open_plugins::parse_component_paths)
        .transpose()?
        .unwrap_or_default();

    let mut paths = Vec::new();
    if !custom_paths.exclusive {
        paths.push(plugin_root.join(DEFAULT_MCP_CONFIG));
    }

    for path in custom_paths.paths {
        paths.push(plugin_root.join(open_plugins::validate_relative_plugin_path(&path)?));
    }

    Ok(open_plugins::dedupe_paths(paths))
}

fn is_inline_config(value: &serde_json::Value) -> bool {
    value.as_object().is_some_and(|object| {
        !object
            .keys()
            .all(|key| matches!(key.as_str(), "paths" | "exclusive"))
    })
}

fn parse_inline_servers(value: &serde_json::Value) -> Result<HashMap<String, serde_json::Value>> {
    serde_json::from_value(value.clone())
        .with_context(|| "Failed to parse inline Open Plugins MCP servers")
}

fn document_to_extension_configs(
    plugin_name: &str,
    plugin_root: &Path,
    servers: HashMap<String, serde_json::Value>,
    scope: PluginScope,
) -> Vec<ExtensionConfig> {
    parse_servers(servers, scope)
        .into_iter()
        .filter_map(|(server_name, parsed)| match parsed {
            Ok(server) => Some(server_to_extension_config(
                plugin_name,
                plugin_root,
                server_name,
                server,
            )),
            Err(err) => {
                warn!(
                    plugin = %plugin_name,
                    server = %server_name,
                    error = %err,
                    "Skipping invalid plugin MCP server entry",
                );
                None
            }
        })
        .collect()
}

fn server_to_extension_config(
    plugin_name: &str,
    plugin_root: &Path,
    server_name: String,
    server: McpServerConfig,
) -> ExtensionConfig {
    let name = format!("{plugin_name}:{server_name}");
    let server = match server {
        McpServerConfig::Stdio(server) => server,
        McpServerConfig::StreamableHttp(server) => {
            return ExtensionConfig::StreamableHttp {
                name,
                description: DEFAULT_EXTENSION_DESCRIPTION.to_string(),
                uri: server.url,
                envs: Envs::default(),
                env_keys: Vec::new(),
                headers: server.headers,
                timeout: Some(DEFAULT_EXTENSION_TIMEOUT),
                socket: None,
                client_id: None,
                client_secret_key: None,
                scopes: Vec::new(),
                bundled: Some(false),
                available_tools: Vec::new(),
            };
        }
    };

    let root = plugin_root.to_string_lossy();
    let mut env = HashMap::from([("PLUGIN_ROOT".to_string(), root.to_string())]);
    env.extend(
        server
            .env
            .into_iter()
            .map(|(key, value)| (key, expand_plugin_root(&value, &root))),
    );

    ExtensionConfig::Stdio {
        name,
        description: DEFAULT_EXTENSION_DESCRIPTION.to_string(),
        cmd: expand_plugin_root(&server.command, &root),
        args: server
            .args
            .into_iter()
            .map(|arg| expand_plugin_root(&arg, &root))
            .collect(),
        envs: Envs::new(env),
        env_keys: Vec::new(),
        timeout: Some(DEFAULT_EXTENSION_TIMEOUT),
        cwd: server.cwd.map(|cwd| expand_plugin_root(&cwd, &root)),
        bundled: Some(false),
        available_tools: Vec::new(),
    }
}

fn expand_plugin_root(value: &str, plugin_root: &str) -> String {
    value.replace(PLUGIN_ROOT, plugin_root)
}

pub fn validate_mcp_servers_manifest_value(value: &serde_json::Value) -> Result<()> {
    if is_inline_config(value) {
        validate_servers(parse_inline_servers(value)?)?;
        return Ok(());
    }

    open_plugins::parse_component_paths(value)?;
    Ok(())
}

pub fn validate_mcp_server_document(value: &serde_json::Value) -> Result<()> {
    let document = serde_json::from_value::<McpServersDocument>(value.clone())?;
    validate_servers(document.mcp_servers)
}

fn validate_servers(servers: HashMap<String, serde_json::Value>) -> Result<()> {
    for (name, parsed) in parse_servers(servers, PluginScope::User) {
        if let Err(err) = parsed {
            bail!("Open Plugins MCP server '{}': {:#}", name, err);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::extension::ExtensionConfig;
    use crate::plugins::discovery::PluginScope;

    #[test]
    fn loads_default_mcp_json_with_plugin_root_expansion() {
        let plugin = tempfile::tempdir().unwrap();
        fs::write(
            plugin.path().join(DEFAULT_MCP_CONFIG),
            r#"{
              "mcpServers": {
                "database": {
                  "command": "${PLUGIN_ROOT}/servers/db-server",
                  "args": ["--config", "${PLUGIN_ROOT}/config.json"],
                  "env": {"DB_PATH": "${PLUGIN_ROOT}/data"},
                  "cwd": "${PLUGIN_ROOT}"
                }
              }
            }"#,
        )
        .unwrap();

        let configs = plugin_mcp_servers("test-plugin", plugin.path(), PluginScope::User).unwrap();
        assert_eq!(configs.len(), 1);
        let ExtensionConfig::Stdio {
            name,
            cmd,
            args,
            envs,
            cwd,
            ..
        } = &configs[0]
        else {
            panic!("expected stdio config");
        };

        assert_eq!(name, "test-plugin:database");
        assert_eq!(
            cmd,
            plugin
                .path()
                .join("servers/db-server")
                .to_string_lossy()
                .as_ref()
        );
        assert_eq!(
            args,
            &vec![
                "--config".to_string(),
                plugin
                    .path()
                    .join("config.json")
                    .to_string_lossy()
                    .to_string()
            ]
        );
        assert_eq!(
            envs.get_env().get("DB_PATH"),
            Some(&plugin.path().join("data").to_string_lossy().to_string())
        );
        assert_eq!(
            cwd.as_deref(),
            Some(plugin.path().to_string_lossy().as_ref())
        );
    }

    #[test]
    fn loads_inline_manifest_mcp_servers() {
        let plugin = tempfile::tempdir().unwrap();
        fs::create_dir_all(plugin.path().join(".plugin")).unwrap();
        fs::write(
            plugin.path().join(".plugin/plugin.json"),
            r#"{
              "name": "test-plugin",
              "mcpServers": {
                "api": {"command": "npx", "args": ["@company/mcp-server"]}
              }
            }"#,
        )
        .unwrap();

        let configs = plugin_mcp_servers("test-plugin", plugin.path(), PluginScope::User).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name(), "test-plugin:api");
    }

    #[test]
    fn manifest_paths_supplement_default_config() {
        let plugin = tempfile::tempdir().unwrap();
        fs::create_dir_all(plugin.path().join(".plugin")).unwrap();
        fs::write(
            plugin.path().join(".plugin/plugin.json"),
            r#"{"name":"test-plugin","mcpServers":"./custom-mcp.json"}"#,
        )
        .unwrap();
        fs::write(
            plugin.path().join(DEFAULT_MCP_CONFIG),
            r#"{"mcpServers":{"default":{"command":"default-server"}}}"#,
        )
        .unwrap();
        fs::write(
            plugin.path().join("custom-mcp.json"),
            r#"{"mcpServers":{"custom":{"command":"custom-server"}}}"#,
        )
        .unwrap();

        let names: Vec<_> = plugin_mcp_servers("test-plugin", plugin.path(), PluginScope::User)
            .unwrap()
            .into_iter()
            .map(|config| config.name())
            .collect();

        assert_eq!(names, vec!["test-plugin:default", "test-plugin:custom"]);
    }

    #[test]
    fn validates_manifest_mcp_servers_value() {
        validate_mcp_servers_manifest_value(&serde_json::json!({
            "api": {"command": "npx"}
        }))
        .unwrap();
        validate_mcp_servers_manifest_value(&serde_json::json!({
            "paths": ["./mcp.json"],
            "exclusive": true
        }))
        .unwrap();
    }

    #[test]
    fn rejects_inline_mcp_server_with_empty_command() {
        let error = validate_mcp_servers_manifest_value(&serde_json::json!({
            "api": {"command": ""}
        }))
        .unwrap_err();
        assert!(error.to_string().contains("command must not be empty"));
    }

    fn write_default_config(plugin: &Path, contents: &str) {
        fs::write(plugin.join(DEFAULT_MCP_CONFIG), contents).unwrap();
    }

    fn loaded_names(plugin: &Path) -> Vec<String> {
        plugin_mcp_servers("test-plugin", plugin, PluginScope::User)
            .unwrap()
            .into_iter()
            .map(|config| config.name())
            .collect()
    }

    #[test]
    fn loads_http_entry_as_streamable_http() {
        let plugin = tempfile::tempdir().unwrap();
        write_default_config(
            plugin.path(),
            r#"{
              "mcpServers": {
                "tec": {
                  "type": "http",
                  "url": "http://localhost:4006/mcp",
                  "headers": {"X-Tenant": "public-tenant"}
                }
              }
            }"#,
        );

        let configs = plugin_mcp_servers("test-plugin", plugin.path(), PluginScope::User).unwrap();
        assert_eq!(configs.len(), 1);
        let ExtensionConfig::StreamableHttp {
            name,
            uri,
            headers,
            timeout,
            bundled,
            ..
        } = &configs[0]
        else {
            panic!("expected streamable http config, got {:?}", configs[0]);
        };
        assert_eq!(name, "test-plugin:tec");
        assert_eq!(uri, "http://localhost:4006/mcp");
        assert_eq!(
            headers.get("X-Tenant").map(String::as_str),
            Some("public-tenant")
        );
        assert_eq!(*timeout, Some(DEFAULT_EXTENSION_TIMEOUT));
        assert_eq!(*bundled, Some(false));
    }

    #[test]
    fn accepts_agent_plugins_transport_name_and_inferred_url_entries() {
        let plugin = tempfile::tempdir().unwrap();
        write_default_config(
            plugin.path(),
            r#"{
              "mcpServers": {
                "spec": {"type": "streamable-http", "url": "https://mcp.example.com/mcp"},
                "inferred": {"url": "http://127.0.0.1:9000/mcp"},
                "local": {"type": "stdio", "command": "server"}
              }
            }"#,
        );

        let configs = plugin_mcp_servers("test-plugin", plugin.path(), PluginScope::User).unwrap();
        let kinds: Vec<_> = configs
            .iter()
            .map(|config| {
                (
                    config.name(),
                    matches!(config, ExtensionConfig::StreamableHttp { .. }),
                )
            })
            .collect();
        assert_eq!(
            kinds,
            vec![
                ("test-plugin:inferred".to_string(), true),
                ("test-plugin:local".to_string(), false),
                ("test-plugin:spec".to_string(), true),
            ]
        );
    }

    #[test]
    fn invalid_entries_are_skipped_without_dropping_valid_servers() {
        let plugin = tempfile::tempdir().unwrap();
        write_default_config(
            plugin.path(),
            r#"{
              "mcpServers": {
                "good-stdio": {"command": "server"},
                "good-http": {"type": "http", "url": "http://[::1]:4006/mcp"},
                "no-transport": {"args": ["x"]},
                "empty-command": {"command": "  "},
                "legacy-sse": {"type": "sse", "url": "https://example.com/sse"},
                "unknown-type": {"type": "websocket", "url": "wss://example.com"},
                "plain-http-remote": {"type": "http", "url": "http://example.com/mcp"},
                "userinfo": {"type": "http", "url": "https://user:pw@example.com/mcp"},
                "fragment": {"type": "http", "url": "https://example.com/mcp#frag"},
                "relative": {"type": "http", "url": "/mcp"},
                "bad-header": {"type": "http", "url": "https://example.com/mcp", "headers": {"bad header": "x"}},
                "dup-header": {"type": "http", "url": "https://example.com/mcp", "headers": {"X-A": "1", "x-a": "2"}},
                "url-placeholder": {"type": "http", "url": "https://${HOST}/mcp"},
                "header-placeholder": {"type": "http", "url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}},
                "not-an-object": "server"
              }
            }"#,
        );

        assert_eq!(
            loaded_names(plugin.path()),
            vec!["test-plugin:good-http", "test-plugin:good-stdio"]
        );
    }

    #[test]
    fn validation_accepts_http_entries_and_reports_invalid_ones() {
        validate_mcp_server_document(&serde_json::json!({
            "mcpServers": {
                "tec": {"type": "http", "url": "http://localhost:4006/mcp"},
                "local": {"command": "server"}
            }
        }))
        .unwrap();

        let error = validate_mcp_server_document(&serde_json::json!({
            "mcpServers": {
                "remote": {"type": "http", "url": "http://example.com/mcp"}
            }
        }))
        .unwrap_err();
        assert!(error.to_string().contains("remote"), "{error}");
    }

    #[test]
    fn project_plugins_may_only_reach_loopback_http_servers() {
        let plugin = tempfile::tempdir().unwrap();
        write_default_config(
            plugin.path(),
            r#"{
              "mcpServers": {
                "local": {"type": "http", "url": "http://localhost:4006/mcp"},
                "remote": {"type": "http", "url": "https://mcp.example.com/mcp"},
                "stdio": {"command": "server"}
              }
            }"#,
        );

        let names = |scope| -> Vec<String> {
            plugin_mcp_servers("test-plugin", plugin.path(), scope)
                .unwrap()
                .into_iter()
                .map(|config| config.name())
                .collect()
        };
        assert_eq!(
            names(PluginScope::Project),
            vec!["test-plugin:local", "test-plugin:stdio"]
        );
        assert_eq!(
            names(PluginScope::User),
            vec![
                "test-plugin:local",
                "test-plugin:remote",
                "test-plugin:stdio"
            ]
        );
    }
}
