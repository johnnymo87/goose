// Modified by Jonathan Mohrbacher, 2026: session headers for proxies (several names, plus a subagent's parent session id).
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::header::{HeaderName, HeaderValue};

pub const SESSION_ID_HEADER: &str = "agent-session-id";

pub const TOOL_CALL_REQUEST_ID_HEADER: &str = "agent-tool-call-request-id";
pub const WORKING_DIR_HEADER: &str = "agent-working-dir";

tokio::task_local! {
    pub static SESSION_ID: Option<String>;
}

pub async fn with_session_id<F>(session_id: Option<String>, f: F) -> F::Output
where
    F: std::future::Future,
{
    SESSION_ID.scope(session_id, f).await
}

pub fn with_session_id_stream<'a, T: Send + 'a>(
    session_id: Option<String>,
    stream: BoxStream<'a, T>,
) -> BoxStream<'a, T> {
    Box::pin(futures::stream::unfold(
        (stream, session_id),
        |(mut stream, session_id)| async move {
            with_session_id(session_id.clone(), stream.next())
                .await
                .map(|item| (item, (stream, session_id)))
        },
    ))
}

pub fn current_session_id() -> Option<String> {
    SESSION_ID.try_with(|id| id.clone()).ok().flatten()
}

pub fn session_id_request_builder() -> goose_providers::api_client::RequestBuilderDecorator {
    session_id_request_builder_with_header_name(HeaderName::from_static(SESSION_ID_HEADER))
}

pub(crate) fn session_id_request_builder_with_header_override(
    header_name_override: Option<&str>,
) -> Result<goose_providers::api_client::RequestBuilderDecorator, reqwest::header::InvalidHeaderName>
{
    let header_name = match header_name_override {
        Some(header_name) => HeaderName::from_bytes(header_name.as_bytes())?,
        None => HeaderName::from_static(SESSION_ID_HEADER),
    };

    Ok(session_id_request_builder_with_header_name(header_name))
}

fn session_id_request_builder_with_header_name(
    header_name: HeaderName,
) -> goose_providers::api_client::RequestBuilderDecorator {
    std::sync::Arc::new(move |request| {
        let (client, request) = request.build_split();
        let mut request = request?;
        let session_header = header_name.clone();
        request.headers_mut().remove(&session_header);

        if let Some(session_id) = current_session_id() {
            let value = HeaderValue::from_str(&session_id)?;
            request.headers_mut().insert(session_header, value);
        }

        Ok(reqwest::RequestBuilder::from_parts(client, request))
    })
}

/// Child session id -> parent session id, for every session linked to a parent in this
/// process (delegated subagents). Filled by `SessionUpdateBuilder::apply`, so a provider can
/// name the parent of the session it is serving without a database read on the request path.
/// A process restart empties it; a subagent does not survive one anyway.
static PARENT_SESSIONS: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<String, String>>,
> = std::sync::LazyLock::new(Default::default);

pub(crate) fn set_parent_session(session_id: &str, parent_session_id: Option<&str>) {
    let mut map = PARENT_SESSIONS
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match parent_session_id {
        Some(parent) if !parent.is_empty() => {
            map.insert(session_id.to_string(), parent.to_string());
        }
        _ => {
            map.remove(session_id);
        }
    }
}

pub fn parent_session_id(session_id: &str) -> Option<String> {
    PARENT_SESSIONS
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(session_id)
        .cloned()
}

/// Parses a comma-separated list of header names; blank entries are ignored.
pub(crate) fn parse_header_names(
    list: &str,
) -> Result<Vec<HeaderName>, reqwest::header::InvalidHeaderName> {
    list.split(',')
        .map(str::trim)
        .filter(|h| !h.is_empty())
        .map(|h| HeaderName::from_bytes(h.as_bytes()))
        .collect()
}

/// Sends the current session id under every name in `session_headers`, and, when the session
/// has a registered parent, the parent's id under `parent_header`. Every one of these headers
/// is removed first, so a caller-supplied value can never contradict the session.
pub(crate) fn session_headers_request_builder(
    session_headers: Vec<HeaderName>,
    parent_header: Option<HeaderName>,
) -> goose_providers::api_client::RequestBuilderDecorator {
    std::sync::Arc::new(move |request| {
        let (client, request) = request.build_split();
        let mut request = request?;
        for name in session_headers.iter().chain(parent_header.iter()) {
            request.headers_mut().remove(name);
        }

        if let Some(session_id) = current_session_id() {
            let value = HeaderValue::from_str(&session_id)?;
            for name in &session_headers {
                request.headers_mut().insert(name.clone(), value.clone());
            }
            if let (Some(name), Some(parent)) = (&parent_header, parent_session_id(&session_id)) {
                request
                    .headers_mut()
                    .insert(name.clone(), HeaderValue::from_str(&parent)?);
            }
        }

        Ok(reqwest::RequestBuilder::from_parts(client, request))
    })
}

/// Local OS user running goose, shared by the OTLP `user.name` resource
/// attribute and the `session.user` span attribute so the two never drift.
pub fn session_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Hostname of the machine running goose, shared by the OTLP `host.name`
/// resource attribute and the `session.host` span attribute.
pub fn session_host() -> String {
    gethostname::gethostname().to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_id_available_when_set() {
        with_session_id(Some("test-session-123".to_string()), async {
            assert_eq!(current_session_id(), Some("test-session-123".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_none_when_not_set() {
        let id = current_session_id();
        assert_eq!(id, None);
    }

    #[tokio::test]
    async fn test_session_id_none_when_explicitly_none() {
        with_session_id(None, async {
            assert_eq!(current_session_id(), None);
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_none_clears_outer_scope() {
        with_session_id(Some("outer-session".to_string()), async {
            assert_eq!(current_session_id(), Some("outer-session".to_string()));

            with_session_id(None, async {
                assert_eq!(current_session_id(), None);
            })
            .await;

            assert_eq!(current_session_id(), Some("outer-session".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_scoped_correctly() {
        assert_eq!(current_session_id(), None);

        with_session_id(Some("outer-session".to_string()), async {
            assert_eq!(current_session_id(), Some("outer-session".to_string()));

            with_session_id(Some("inner-session".to_string()), async {
                assert_eq!(current_session_id(), Some("inner-session".to_string()));
            })
            .await;

            assert_eq!(current_session_id(), Some("outer-session".to_string()));
        })
        .await;

        assert_eq!(current_session_id(), None);
    }

    #[tokio::test]
    async fn test_session_id_across_await_points() {
        with_session_id(Some("persistent-session".to_string()), async {
            assert_eq!(current_session_id(), Some("persistent-session".to_string()));

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

            assert_eq!(current_session_id(), Some("persistent-session".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_scopes_each_stream_poll() {
        let stream = futures::stream::iter([(), ()]).then(|()| async { current_session_id() });
        let mut stream =
            with_session_id_stream(Some("stream-session".to_string()), Box::pin(stream));

        assert_eq!(
            stream.next().await,
            Some(Some("stream-session".to_string()))
        );
        assert_eq!(
            stream.next().await,
            Some(Some("stream-session".to_string()))
        );
        assert_eq!(stream.next().await, None);
        assert_eq!(current_session_id(), None);
    }

    #[tokio::test]
    async fn test_session_id_request_builder_uses_custom_header() {
        with_session_id(Some("test-session-123".to_string()), async {
            let decorate =
                session_id_request_builder_with_header_override(Some("x-opencode-session"))
                    .unwrap();

            let request = decorate(reqwest::Client::new().get("http://localhost"))
                .unwrap()
                .build()
                .unwrap();

            assert_eq!(
                request.headers().get("x-opencode-session").unwrap(),
                "test-session-123"
            );
            assert!(request.headers().get(SESSION_ID_HEADER).is_none());
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_request_builder_uses_default_without_override() {
        with_session_id(Some("test-session-123".to_string()), async {
            let decorate = session_id_request_builder_with_header_override(None).unwrap();

            let request = decorate(reqwest::Client::new().get("http://localhost"))
                .unwrap()
                .build()
                .unwrap();

            assert_eq!(
                request.headers().get(SESSION_ID_HEADER).unwrap(),
                "test-session-123"
            );
        })
        .await;
    }

    fn build_with(
        decorate: &goose_providers::api_client::RequestBuilderDecorator,
    ) -> reqwest::Request {
        decorate(
            reqwest::Client::new()
                .get("http://localhost")
                .header("x-session-id", "stale")
                .header("x-parent-session-id", "stale"),
        )
        .unwrap()
        .build()
        .unwrap()
    }

    #[tokio::test]
    async fn test_session_headers_builder_sends_every_name_and_the_parent() {
        set_parent_session("child-1", Some("parent-1"));
        let decorate = session_headers_request_builder(
            parse_header_names("x-opencode-session, x-session-id").unwrap(),
            Some(HeaderName::from_static("x-parent-session-id")),
        );
        with_session_id(Some("child-1".to_string()), async {
            let request = build_with(&decorate);
            assert_eq!(
                request.headers().get("x-opencode-session").unwrap(),
                "child-1"
            );
            assert_eq!(request.headers().get("x-session-id").unwrap(), "child-1");
            assert_eq!(
                request.headers().get("x-parent-session-id").unwrap(),
                "parent-1"
            );
            assert!(request.headers().get(SESSION_ID_HEADER).is_none());
        })
        .await;
        set_parent_session("child-1", None);
        assert_eq!(parent_session_id("child-1"), None);
    }

    #[tokio::test]
    async fn test_session_headers_builder_top_level_session_has_no_parent_header() {
        let decorate = session_headers_request_builder(
            parse_header_names("x-session-id").unwrap(),
            Some(HeaderName::from_static("x-parent-session-id")),
        );
        with_session_id(Some("top-level-1".to_string()), async {
            let request = build_with(&decorate);
            assert_eq!(
                request.headers().get("x-session-id").unwrap(),
                "top-level-1"
            );
            assert!(request.headers().get("x-parent-session-id").is_none());
        })
        .await;
        // No session in scope: no session headers at all, and stale ones are stripped.
        let request = build_with(&decorate);
        assert!(request.headers().get("x-session-id").is_none());
        assert!(request.headers().get("x-parent-session-id").is_none());
    }

    #[test]
    fn test_parse_header_names() {
        assert_eq!(parse_header_names(" a , ,b ").unwrap().len(), 2);
        assert!(parse_header_names("").unwrap().is_empty());
        assert!(parse_header_names("ok,bad header").is_err());
    }

    #[test]
    fn test_session_id_request_builder_rejects_invalid_header_override() {
        assert!(session_id_request_builder_with_header_override(Some("invalid header")).is_err());
    }
}
