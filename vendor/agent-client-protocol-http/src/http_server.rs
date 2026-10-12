// Modified by Jonathan Mohrbacher, 2026, for the goose fork (patched/v1.54.0): Acp-Takeover, Acp-Stream-Features, SSE ids / Last-Event-ID, Acp-Replay. See ../VENDORED.md.
use std::{convert::Infallible, error::Error as _, sync::Arc, time::Duration};

use agent_client_protocol::{
    RawJsonRpcMessage, TransportBatchEntry, TransportFrame, schema::v1::RequestId,
    schema::v1::Response as RpcResponse,
};
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Request, StatusCode, header},
    response::{IntoResponse, Response, Sse, sse::Event},
};
use tracing::{error, info, trace};

use crate::{
    connection::{Connection, ConnectionRegistry, ResponseRoute},
    http_mailbox::Replay,
    protocol::{
        EVENT_STREAM_MIME_TYPE, HEADER_CONNECTION_ID, HEADER_LAST_EVENT_ID, HEADER_REPLAY,
        HEADER_SESSION_ID, HEADER_STREAM_FEATURES, HEADER_TAKEOVER, JSON_MIME_TYPE,
        apply_session_header_to_message, is_initialize_request, is_response_only_shape,
        method_for_message, method_requires_session_header, session_id_from_message,
    },
};

const MAX_POST_BODY_BYTES: usize = 16 * 1024 * 1024;

pub(crate) async fn handle_post(
    State(registry): State<Arc<ConnectionRegistry>>,
    request: Request<Body>,
) -> Response {
    if !request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with(JSON_MIME_TYPE))
    {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        )
            .into_response();
    }

    let connection_id = header_value(request.headers(), HEADER_CONNECTION_ID);
    let session_id = header_value(request.headers(), HEADER_SESSION_ID);
    if content_length_exceeds_limit(request.headers()) {
        return post_body_too_large_response();
    }

    let body = match axum::body::to_bytes(request.into_body(), MAX_POST_BODY_BYTES).await {
        Ok(body) => body,
        Err(e) => {
            error!("Failed to read request body: {e}");
            if is_body_limit_error(&e) {
                return post_body_too_large_response();
            }
            return StatusCode::BAD_REQUEST.into_response();
        }
    };

    let body = match std::str::from_utf8(&body) {
        Ok(body) => body,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("Invalid JSON-RPC: {error}"),
            )
                .into_response();
        }
    };
    let mut frame = TransportFrame::parse_json(body);

    if let Some((initialize_id, initialize_index)) =
        initial_initialize_request(&frame).map(|(id, index)| (id.clone(), index))
    {
        if let TransportFrame::Batch(batch) = &mut frame {
            for (index, entry) in batch.entries_mut().enumerate() {
                if Some(index) == initialize_index {
                    continue;
                }
                let TransportBatchEntry::Message(message) = entry else {
                    continue;
                };
                if let Err(error) = prepare_message_route(message, session_id.as_deref()) {
                    return (StatusCode::BAD_REQUEST, error).into_response();
                }
            }
        }
        let (connection_id, connection) = registry.create_connection().await;
        let initialize_cleanup =
            InitializeCleanup::new(registry.clone(), connection_id.clone(), connection.clone());
        if connection.send_frame_to_agent(frame).is_err() {
            initialize_cleanup.cleanup().await;
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }

        let (init_response_frame, initialize_failed) = loop {
            let Some(frame) = connection.recv_initial().await else {
                initialize_cleanup.cleanup().await;
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "agent closed before initialize response",
                )
                    .into_response();
            };
            if let Some(initialize_failed) = initialize_response_failed(&frame, &initialize_id) {
                break (frame, initialize_failed);
            }

            // A batched sibling may emit a callback or notification before all
            // response slots complete. Buffer that side traffic for the SSE
            // stream instead of mistaking it for the initialize response.
            if let Err(error) = connection.route_outbound(frame).await {
                initialize_cleanup.cleanup().await;
                return (StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
            }
        };
        let init_response = match init_response_frame.to_json() {
            Ok(response) => response,
            Err(e) => {
                initialize_cleanup.cleanup().await;
                error!("failed to serialize initialize response: {e}");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        };
        if initialize_failed {
            initialize_cleanup.cleanup().await;
            info!(connection_id = %connection_id, "Initialize rejected");
            return json_response(init_response);
        }

        connection.start_router().await;
        initialize_cleanup.disarm();
        info!(connection_id = %connection_id, "Initialize complete");
        return with_connection_header(json_response(init_response), &connection_id);
    }

    let Some(connection_id) = connection_id else {
        return (StatusCode::BAD_REQUEST, "Acp-Connection-Id header required").into_response();
    };
    let Some(connection) = registry.get(&connection_id).await else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let mut session_routes = Vec::new();
    let mut pending_routes = Vec::new();
    match &mut frame {
        TransportFrame::Single(message) => {
            let route = match prepare_message_route(message, session_id.as_deref()) {
                Ok(route) => route,
                Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
            };
            collect_route(message, route, &mut session_routes, &mut pending_routes);
            trace!(connection_id = %connection_id, ?message, "POST → agent");
        }
        TransportFrame::Batch(batch) => {
            for entry in batch.entries_mut() {
                let TransportBatchEntry::Message(message) = entry else {
                    continue;
                };
                let route = match prepare_message_route(message, session_id.as_deref()) {
                    Ok(route) => route,
                    Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
                };
                collect_route(message, route, &mut session_routes, &mut pending_routes);
            }
            trace!(connection_id = %connection_id, ?frame, "POST batch → agent");
        }
        TransportFrame::Malformed { .. } => {
            trace!(connection_id = %connection_id, ?frame, "POST malformed frame → agent");
        }
    }

    for session_id in session_routes {
        connection.ensure_session(&session_id).await;
    }
    for (request_id, route) in pending_routes {
        connection.record_pending_route(request_id, route).await;
    }

    if connection.send_frame_to_agent(frame).is_err() {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    StatusCode::ACCEPTED.into_response()
}

fn initial_initialize_request(frame: &TransportFrame) -> Option<(&RequestId, Option<usize>)> {
    fn initialize_id(message: &RawJsonRpcMessage) -> Option<&RequestId> {
        if !is_initialize_request(message) {
            return None;
        }
        let RawJsonRpcMessage::Request(request) = message else {
            unreachable!("initialize messages are requests");
        };
        Some(&request.id)
    }

    match frame {
        TransportFrame::Single(message) => initialize_id(message).map(|id| (id, None)),
        TransportFrame::Malformed { .. } => None,
        TransportFrame::Batch(batch) => {
            for (index, entry) in batch.entries().enumerate() {
                match entry {
                    TransportBatchEntry::Message(RawJsonRpcMessage::Response(_)) => {}
                    TransportBatchEntry::Message(message) => {
                        return initialize_id(message).map(|id| (id, Some(index)));
                    }
                    TransportBatchEntry::Malformed { raw, .. } if is_response_only_shape(raw) => {}
                    TransportBatchEntry::Malformed { .. } => return None,
                }
            }
            None
        }
    }
}

fn initialize_response_failed(frame: &TransportFrame, initialize_id: &RequestId) -> Option<bool> {
    fn response_failed(message: &RawJsonRpcMessage, initialize_id: &RequestId) -> Option<bool> {
        match message {
            RawJsonRpcMessage::Response(RpcResponse::Result { id, .. }) if id == initialize_id => {
                Some(false)
            }
            RawJsonRpcMessage::Response(RpcResponse::Error { id, .. }) if id == initialize_id => {
                Some(true)
            }
            RawJsonRpcMessage::Request(_)
            | RawJsonRpcMessage::Notification(_)
            | RawJsonRpcMessage::Response(_) => None,
        }
    }

    match frame {
        TransportFrame::Single(message) => response_failed(message, initialize_id),
        TransportFrame::Batch(batch) => batch.entries().find_map(|entry| match entry {
            TransportBatchEntry::Message(message) => response_failed(message, initialize_id),
            TransportBatchEntry::Malformed { .. } => None,
        }),
        TransportFrame::Malformed { .. } => None,
    }
}

fn prepare_message_route(
    message: &mut RawJsonRpcMessage,
    session_id: Option<&str>,
) -> Result<Option<ResponseRoute>, &'static str> {
    if let Some(session_id) = session_id
        && method_for_message(message).is_some()
    {
        apply_session_header_to_message(message, session_id)?;
    }

    Ok(match method_for_message(message) {
        Some(method) => match session_id_from_message(message) {
            Some(session_id) => Some(ResponseRoute::Session(session_id)),
            None if method_requires_session_header(method) => {
                return Err("Acp-Session-Id header required");
            }
            None => Some(ResponseRoute::Connection),
        },
        None => None,
    })
}

fn collect_route(
    message: &RawJsonRpcMessage,
    route: Option<ResponseRoute>,
    session_routes: &mut Vec<String>,
    pending_routes: &mut Vec<(agent_client_protocol::schema::v1::RequestId, ResponseRoute)>,
) {
    if let Some(ResponseRoute::Session(session_id)) = &route {
        session_routes.push(session_id.clone());
    }
    if let (RawJsonRpcMessage::Request(request), Some(route)) = (message, route) {
        pending_routes.push((request.id.clone(), route));
    }
}

struct InitializeCleanup {
    registry: Option<Arc<ConnectionRegistry>>,
    connection_id: String,
    connection: Arc<Connection>,
}

impl InitializeCleanup {
    fn new(
        registry: Arc<ConnectionRegistry>,
        connection_id: String,
        connection: Arc<Connection>,
    ) -> Self {
        Self {
            registry: Some(registry),
            connection_id,
            connection,
        }
    }

    async fn cleanup(mut self) {
        self.cleanup_inner().await;
    }

    fn disarm(mut self) {
        self.registry.take();
    }

    async fn cleanup_inner(&mut self) {
        let Some(registry) = self.registry.take() else {
            return;
        };
        registry.remove(&self.connection_id).await;
        self.connection.shutdown().await;
    }
}

impl Drop for InitializeCleanup {
    fn drop(&mut self) {
        let Some(registry) = self.registry.take() else {
            return;
        };
        let connection_id = self.connection_id.clone();
        let connection = self.connection.clone();
        tokio::spawn(async move {
            registry.remove(&connection_id).await;
            connection.shutdown().await;
        });
    }
}

/// Advertised on every SSE GET /acp response (200 and 409) so clients can feature-detect.
pub(crate) const STREAM_FEATURES: &str = "takeover, replay";

fn with_stream_features(mut response: Response) -> Response {
    response.headers_mut().insert(
        HEADER_STREAM_FEATURES,
        HeaderValue::from_static(STREAM_FEATURES),
    );
    response
}

fn sse_event(seq: u64, msg: String) -> Event {
    Event::default().id(seq.to_string()).data(msg)
}

pub(crate) async fn handle_get(
    registry: Arc<ConnectionRegistry>,
    request: Request<Body>,
) -> Response {
    if !request
        .headers()
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|accept| accept.contains(EVENT_STREAM_MIME_TYPE))
    {
        return (
            StatusCode::NOT_ACCEPTABLE,
            "client must accept text/event-stream",
        )
            .into_response();
    }

    let Some(connection_id) = header_value(request.headers(), HEADER_CONNECTION_ID) else {
        return (StatusCode::BAD_REQUEST, "Acp-Connection-Id header required").into_response();
    };
    let Some(connection) = registry.get(&connection_id).await else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let session_id = header_value(request.headers(), HEADER_SESSION_ID);
    let takeover =
        header_value(request.headers(), HEADER_TAKEOVER).is_some_and(|v| v.trim() == "1");
    let receiver = match (session_id.as_deref(), takeover) {
        (Some(session_id), false) => connection.subscribe_session_stream(session_id).await,
        (Some(session_id), true) => connection.take_over_session_stream(session_id).await,
        (None, false) => connection.subscribe_connection_stream(),
        (None, true) => connection.take_over_connection_stream(),
    };
    let Some(mut receiver) = receiver else {
        return with_stream_features(
            (
                StatusCode::CONFLICT,
                "outbound stream already has a subscriber",
            )
                .into_response(),
        );
    };

    let replay = request.headers().get(HEADER_LAST_EVENT_ID).map(|value| {
        if let Some(last_seen) = value
            .to_str()
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
        {
            receiver.resume_after(last_seen)
        } else {
            receiver.resume_after(0);
            Replay::Gap // unparseable id: send what is retained, but flag it
        }
    });

    let mut closed = connection.subscribe_closed();
    let stream = async_stream::stream! {
        loop {
            while let Ok((seq, msg)) = receiver.try_recv_frame() {
                trace!(payload = %msg, "SSE → client");
                yield Ok::<_, Infallible>(sse_event(seq, msg));
            }
            if *closed.borrow() {
                while let Ok((seq, msg)) = receiver.try_recv_frame() {
                    trace!(payload = %msg, "SSE → client");
                    yield Ok(sse_event(seq, msg));
                }
                break;
            }
            tokio::select! {
                biased;
                recv = receiver.recv_frame() => match recv {
                    Some((seq, msg)) => {
                        trace!(payload = %msg, "SSE → client");
                        yield Ok(sse_event(seq, msg));
                    }
                    None => break,
                },
                changed = closed.changed() => {
                    if changed.is_err() || *closed.borrow() {
                        while let Ok((seq, msg)) = receiver.try_recv_frame() {
                            trace!(payload = %msg, "SSE → client");
                            yield Ok(sse_event(seq, msg));
                        }
                        break;
                    }
                }
            }
        }
    };

    let mut response = with_stream_features(with_connection_header(
        Sse::new(stream)
            .keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(Duration::from_secs(15))
                    .text(""),
            )
            .into_response(),
        &connection_id,
    ));
    if let Some(session_id) = session_id
        && let Ok(value) = HeaderValue::from_str(&session_id)
    {
        response.headers_mut().insert(HEADER_SESSION_ID, value);
    }
    if let Some(replay) = replay {
        response.headers_mut().insert(
            HEADER_REPLAY,
            HeaderValue::from_static(match replay {
                Replay::Complete => "complete",
                Replay::Gap => "gap",
            }),
        );
    }
    response
}

pub(crate) async fn handle_delete(
    State(registry): State<Arc<ConnectionRegistry>>,
    request: Request<Body>,
) -> Response {
    let Some(connection_id) = header_value(request.headers(), HEADER_CONNECTION_ID) else {
        return (StatusCode::BAD_REQUEST, "Acp-Connection-Id header required").into_response();
    };
    let Some(connection) = registry.remove(&connection_id).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    connection.shutdown().await;
    info!(connection_id = %connection_id, "Connection terminated via DELETE");
    StatusCode::ACCEPTED.into_response()
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

fn with_connection_header(mut response: Response, connection_id: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(connection_id) {
        response.headers_mut().insert(HEADER_CONNECTION_ID, value);
    }
    response
}

fn json_response(body: String) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, JSON_MIME_TYPE)],
        body,
    )
        .into_response()
}

fn content_length_exceeds_limit(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|length| length > MAX_POST_BODY_BYTES)
}

fn is_body_limit_error(error: &axum::Error) -> bool {
    let mut source = error.source();
    while let Some(error) = source {
        if error.to_string() == "length limit exceeded" {
            return true;
        }
        source = error.source();
    }
    false
}

fn post_body_too_large_response() -> Response {
    (StatusCode::PAYLOAD_TOO_LARGE, "POST body too large").into_response()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use agent_client_protocol::{
        Channel, RawJsonRpcMessage, TransportBatch, TransportBatchEntry, TransportFrame,
        schema::v1::RequestId,
    };
    use futures::{StreamExt, future::BoxFuture};
    use serde_json::json;
    use tokio::{
        sync::mpsc,
        time::{Duration, sleep, timeout},
    };

    use super::*;
    use crate::connection::AgentFactory;

    const ISSUE_288_BURST: usize = 1_025;

    struct CapturingAgentFactory {
        forwarded: mpsc::UnboundedSender<RawJsonRpcMessage>,
    }

    impl AgentFactory for CapturingAgentFactory {
        fn spawn_agent(
            &self,
        ) -> (
            Channel,
            BoxFuture<'static, agent_client_protocol::Result<()>>,
        ) {
            let (agent, transport) = Channel::duplex();
            let forwarded = self.forwarded.clone();
            let future = Box::pin(async move {
                let Channel {
                    rx: mut incoming,
                    tx: _,
                } = agent;
                while let Some(frame) = incoming.next().await {
                    let TransportFrame::Single(message) = frame else {
                        panic!("expected a single JSON-RPC frame");
                    };
                    if forwarded.send(message).is_err() {
                        break;
                    }
                }
                Ok(())
            });

            (transport, future)
        }
    }

    fn capturing_registry() -> (
        Arc<ConnectionRegistry>,
        mpsc::UnboundedReceiver<RawJsonRpcMessage>,
    ) {
        let (forwarded_tx, forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        (registry, forwarded_rx)
    }

    fn sse_get(
        connection_id: &str,
        session_id: Option<&str>,
        extra: &[(&str, &str)],
    ) -> Request<Body> {
        let mut request = Request::builder()
            .method("GET")
            .uri("/acp")
            .header(header::ACCEPT, EVENT_STREAM_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id);
        if let Some(session_id) = session_id {
            request = request.header(HEADER_SESSION_ID, session_id);
        }
        for (name, value) in extra {
            request = request.header(*name, *value);
        }
        request.body(Body::empty()).unwrap()
    }

    async fn push_tagged(connection: &Connection, session_id: Option<&str>, tag: &str) {
        let mut params = json!({ "tag": tag });
        if let Some(session_id) = session_id {
            params["sessionId"] = json!(session_id);
        }
        connection
            .route_outbound(TransportFrame::Single(
                RawJsonRpcMessage::notification("session/update".to_string(), params).unwrap(),
            ))
            .await
            .unwrap();
    }

    fn data_tags(body: &str) -> Vec<String> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
            .filter_map(|value| value["params"]["tag"].as_str().map(str::to_owned))
            .collect()
    }

    struct RejectingInitializeAgentFactory;

    impl AgentFactory for RejectingInitializeAgentFactory {
        fn spawn_agent(
            &self,
        ) -> (
            Channel,
            BoxFuture<'static, agent_client_protocol::Result<()>>,
        ) {
            let (mut agent, transport) = Channel::duplex();
            let future = Box::pin(async move {
                match agent.rx.next().await {
                    Some(TransportFrame::Single(RawJsonRpcMessage::Request(request))) => {
                        agent
                            .tx
                            .unbounded_send(TransportFrame::Single(RawJsonRpcMessage::response(
                                request.id,
                                Err(agent_client_protocol::Error::invalid_request()
                                    .data("initialize rejected")),
                            )))
                            .unwrap();
                    }
                    Some(TransportFrame::Batch(batch)) => {
                        let responses = batch.entries().filter_map(|entry| {
                            let TransportBatchEntry::Message(RawJsonRpcMessage::Request(request)) =
                                entry
                            else {
                                return None;
                            };
                            let result = if request.method.as_ref() == "initialize" {
                                Err(agent_client_protocol::Error::invalid_request()
                                    .data("initialize rejected"))
                            } else {
                                Ok(json!({ "ok": true }))
                            };
                            Some(RawJsonRpcMessage::response(request.id.clone(), result))
                        });
                        agent
                            .tx
                            .unbounded_send(TransportFrame::Batch(
                                TransportBatch::from_messages(responses)
                                    .expect("request batch has responses"),
                            ))
                            .unwrap();
                    }
                    Some(TransportFrame::Single(_) | TransportFrame::Malformed { .. }) | None => {}
                }
                std::future::pending::<agent_client_protocol::Result<()>>().await
            });

            (transport, future)
        }
    }

    struct PendingInitializeAgentFactory;

    impl AgentFactory for PendingInitializeAgentFactory {
        fn spawn_agent(
            &self,
        ) -> (
            Channel,
            BoxFuture<'static, agent_client_protocol::Result<()>>,
        ) {
            let (agent, transport) = Channel::duplex();
            let future = Box::pin(async move {
                let Channel {
                    rx: mut incoming,
                    tx: _outgoing,
                } = agent;
                drop(incoming.next().await);
                std::future::pending::<agent_client_protocol::Result<()>>().await
            });

            (transport, future)
        }
    }

    struct BatchAgentFactory {
        forwarded: mpsc::UnboundedSender<Vec<(String, Option<String>)>>,
    }

    impl AgentFactory for BatchAgentFactory {
        fn spawn_agent(
            &self,
        ) -> (
            Channel,
            BoxFuture<'static, agent_client_protocol::Result<()>>,
        ) {
            let (mut agent, transport) = Channel::duplex();
            let forwarded = self.forwarded.clone();
            let future = Box::pin(async move {
                let Some(TransportFrame::Batch(batch)) = agent.rx.next().await else {
                    panic!("expected one batch frame");
                };
                let mut methods = Vec::new();
                let mut responses = Vec::new();
                for entry in batch.entries() {
                    match entry {
                        TransportBatchEntry::Message(RawJsonRpcMessage::Request(request)) => {
                            methods.push((
                                request.method.to_string(),
                                request
                                    .params
                                    .as_ref()
                                    .and_then(crate::protocol::session_id_from_params),
                            ));
                            responses.push(RawJsonRpcMessage::response(
                                request.id.clone(),
                                Ok(json!({ "ok": true })),
                            ));
                        }
                        TransportBatchEntry::Message(RawJsonRpcMessage::Notification(
                            notification,
                        )) => {
                            methods.push((
                                notification.method.to_string(),
                                notification
                                    .params
                                    .as_ref()
                                    .and_then(crate::protocol::session_id_from_params),
                            ));
                        }
                        TransportBatchEntry::Message(RawJsonRpcMessage::Response(_)) => {}
                        TransportBatchEntry::Malformed { raw, .. }
                            if is_response_only_shape(raw) => {}
                        TransportBatchEntry::Malformed { error, .. } => {
                            responses.push(RawJsonRpcMessage::response(
                                RequestId::Null,
                                Err(error.clone()),
                            ));
                        }
                    }
                }
                forwarded.send(methods).unwrap();
                let responses =
                    TransportBatch::from_messages(responses).expect("responses are non-empty");
                agent
                    .tx
                    .unbounded_send(TransportFrame::Batch(responses))
                    .unwrap();
                std::future::pending::<agent_client_protocol::Result<()>>().await
            });

            (transport, future)
        }
    }

    struct SideTrafficBeforeInitializeResponseAgentFactory;

    impl AgentFactory for SideTrafficBeforeInitializeResponseAgentFactory {
        fn spawn_agent(
            &self,
        ) -> (
            Channel,
            BoxFuture<'static, agent_client_protocol::Result<()>>,
        ) {
            let (mut agent, transport) = Channel::duplex();
            let future = Box::pin(async move {
                let Some(TransportFrame::Batch(batch)) = agent.rx.next().await else {
                    panic!("expected one initial batch frame");
                };
                let responses = batch.entries().filter_map(|entry| {
                    let TransportBatchEntry::Message(RawJsonRpcMessage::Request(request)) = entry
                    else {
                        return None;
                    };
                    Some(RawJsonRpcMessage::response(
                        request.id.clone(),
                        Ok(json!({ "ok": true })),
                    ))
                });

                agent
                    .tx
                    .unbounded_send(TransportFrame::Single(
                        RawJsonRpcMessage::notification(
                            "custom/during-initialize".into(),
                            json!({ "phase": "before-response" }),
                        )
                        .expect("test notification should serialize"),
                    ))
                    .unwrap();
                agent
                    .tx
                    .unbounded_send(TransportFrame::Batch(
                        TransportBatch::from_messages(responses)
                            .expect("initial batch has response-bearing requests"),
                    ))
                    .unwrap();
                std::future::pending::<agent_client_protocol::Result<()>>().await
            });

            (transport, future)
        }
    }

    #[tokio::test]
    async fn post_rejects_declared_body_larger_than_limit() {
        let (forwarded_tx, _forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .header(
                header::CONTENT_LENGTH,
                (MAX_POST_BODY_BYTES + 1).to_string(),
            )
            .body(Body::from("{}"))
            .unwrap();

        let response = handle_post(State(registry), request).await;

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn initial_batch_bootstraps_connection_and_returns_grouped_response() {
        let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(BatchAgentFactory {
            forwarded: forwarded_tx,
        })));
        let body = json!([
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "custom/after-initialize",
                "params": {}
            }
        ])
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry.clone()), request).await;

        assert_eq!(response.status(), StatusCode::OK);
        let connection_id = response
            .headers()
            .get(HEADER_CONNECTION_ID)
            .expect("successful initialize should return a connection ID")
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(
            timeout(Duration::from_secs(1), forwarded_rx.recv())
                .await
                .unwrap()
                .unwrap(),
            [
                ("initialize".to_string(), None),
                ("custom/after-initialize".to_string(), None),
            ]
        );
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let response = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
        let entries = response
            .as_array()
            .expect("initial batch response should remain grouped");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["id"], 1);
        assert_eq!(entries[1]["id"], 2);

        let connection = registry
            .remove(&connection_id)
            .await
            .expect("initialized connection should remain registered");
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn initial_batch_buffers_side_traffic_until_grouped_response() {
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(
            SideTrafficBeforeInitializeResponseAgentFactory,
        )));
        let body = json!([
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "custom/after-initialize",
                "params": {}
            }
        ])
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry.clone()), request).await;

        assert_eq!(response.status(), StatusCode::OK);
        let connection_id = response
            .headers()
            .get(HEADER_CONNECTION_ID)
            .expect("successful initialize should return a connection ID")
            .to_str()
            .unwrap()
            .to_string();
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let response = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
        assert_eq!(response.as_array().map(Vec::len), Some(2));

        let connection = registry
            .get(&connection_id)
            .await
            .expect("initialized connection should remain registered");
        let mut outbound = connection.subscribe_connection_stream().unwrap();
        let side_traffic = timeout(Duration::from_secs(1), outbound.recv())
            .await
            .unwrap()
            .unwrap();
        let side_traffic = serde_json::from_str::<serde_json::Value>(&side_traffic).unwrap();
        assert_eq!(side_traffic["method"], "custom/during-initialize");
        assert_eq!(side_traffic["params"]["phase"], "before-response");

        let connection = registry
            .remove(&connection_id)
            .await
            .expect("initialized connection should remain registered");
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn initial_batch_skips_leading_response_only_entries() {
        let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(BatchAgentFactory {
            forwarded: forwarded_tx,
        })));
        let body = json!([
            {
                "jsonrpc": "2.0",
                "id": 98,
                "result": { "ignored": true }
            },
            {
                "jsonrpc": "2.0",
                "id": 99,
                "result": { "ignored": true },
                "error": { "code": -32603, "message": "also ignored" }
            },
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "custom/after-initialize",
                "params": {}
            }
        ])
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry.clone()), request).await;

        assert_eq!(response.status(), StatusCode::OK);
        let connection_id = response
            .headers()
            .get(HEADER_CONNECTION_ID)
            .expect("successful initialize should return a connection ID")
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(
            timeout(Duration::from_secs(1), forwarded_rx.recv())
                .await
                .unwrap()
                .unwrap(),
            [
                ("initialize".to_string(), None),
                ("custom/after-initialize".to_string(), None),
            ]
        );
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let response = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
        let entries = response
            .as_array()
            .expect("initial batch response should remain grouped");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["id"], 1);
        assert_eq!(entries[1]["id"], 2);

        let connection = registry
            .remove(&connection_id)
            .await
            .expect("initialized connection should remain registered");
        connection.shutdown().await;
    }

    #[test]
    fn initial_batch_rejects_invalid_or_call_shaped_predecessors() {
        for frame in [
            TransportFrame::parse_json(
                &json!([
                    17,
                    { "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} }
                ])
                .to_string(),
            ),
            TransportFrame::parse_json(
                &json!([
                    { "jsonrpc": "2.0", "method": "custom/before-initialize" },
                    { "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} }
                ])
                .to_string(),
            ),
            TransportFrame::parse_json(
                &json!([
                    { "jsonrpc": "2.0", "id": 7, "method": "custom/before-initialize" },
                    { "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} }
                ])
                .to_string(),
            ),
        ] {
            assert!(initial_initialize_request(&frame).is_none());
        }
    }

    #[tokio::test]
    async fn post_applies_session_header_to_batch_and_routes_grouped_response() {
        let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(BatchAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        let mut connection_outbound = connection.subscribe_connection_stream().unwrap();
        let mut session_outbound = connection
            .subscribe_session_stream("session-1")
            .await
            .unwrap();
        connection.start_router().await;

        let body = json!([
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "custom/first",
                "params": {}
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "custom/second",
                "params": {}
            }
        ])
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id.as_str())
            .header(HEADER_SESSION_ID, "session-1")
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry.clone()), request).await;

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let methods = timeout(Duration::from_secs(1), forwarded_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            methods,
            [
                ("custom/first".to_string(), Some("session-1".to_string())),
                ("custom/second".to_string(), Some("session-1".to_string())),
            ]
        );
        let response = timeout(Duration::from_secs(1), session_outbound.recv())
            .await
            .unwrap()
            .expect("batch response should be emitted");
        let response = serde_json::from_str::<serde_json::Value>(&response).unwrap();
        let entries = response.as_array().expect("response should remain a batch");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["id"], 1);
        assert_eq!(entries[1]["id"], 2);
        assert!(session_outbound.try_recv().is_err());
        assert!(connection_outbound.try_recv().is_err());

        registry.remove(&connection_id).await;
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn mixed_batch_routes_fall_back_to_connection_stream() {
        let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(BatchAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        let mut connection_outbound = connection.subscribe_connection_stream().unwrap();
        let mut session_outbound = connection
            .subscribe_session_stream("session-1")
            .await
            .unwrap();
        connection.start_router().await;

        let body = json!([
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "custom/session",
                "params": {}
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "$/connection",
                "params": {}
            }
        ])
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id.as_str())
            .header(HEADER_SESSION_ID, "session-1")
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry.clone()), request).await;

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let methods = timeout(Duration::from_secs(1), forwarded_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            methods,
            [
                ("custom/session".to_string(), Some("session-1".to_string())),
                ("$/connection".to_string(), None),
            ]
        );
        let response = timeout(Duration::from_secs(1), connection_outbound.recv())
            .await
            .unwrap()
            .expect("mixed-route batch should use the connection stream");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&response)
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(session_outbound.try_recv().is_err());

        registry.remove(&connection_id).await;
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn initialize_error_response_rejects_connection() {
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(
            RejectingInitializeAgentFactory,
        )));
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry.clone()), request).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get(HEADER_CONNECTION_ID).is_none());
        assert_eq!(registry.len().await, 0);
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let message = serde_json::from_slice::<RawJsonRpcMessage>(&body).unwrap();
        assert!(matches!(
            message,
            RawJsonRpcMessage::Response(RpcResponse::Error {
                id: RequestId::Number(1),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn rejected_initialize_in_batch_returns_group_and_cleans_up_connection() {
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(
            RejectingInitializeAgentFactory,
        )));
        let body = json!([
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "custom/sibling",
                "params": {}
            }
        ])
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry.clone()), request).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get(HEADER_CONNECTION_ID).is_none());
        assert_eq!(registry.len().await, 0);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let response = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
        let entries = response
            .as_array()
            .expect("rejected initial batch response should remain grouped");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["id"], 1);
        assert_eq!(entries[0]["error"]["code"], -32600);
        assert_eq!(entries[1]["id"], 2);
        assert_eq!(entries[1]["result"], json!({ "ok": true }));
    }

    #[tokio::test]
    async fn cancelled_initialize_cleans_up_connection() {
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(
            PendingInitializeAgentFactory,
        )));
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .body(Body::from(body))
            .unwrap();

        {
            let initialize = handle_post(State(registry.clone()), request);
            tokio::pin!(initialize);
            timeout(Duration::from_secs(1), async {
                loop {
                    tokio::select! {
                        response = &mut initialize => {
                            panic!(
                                "initialize completed unexpectedly with {}",
                                response.status()
                            );
                        }
                        () = sleep(Duration::from_millis(10)) => {
                            if registry.len().await == 1 {
                                break;
                            }
                        }
                    }
                }
            })
            .await
            .unwrap();
        }

        timeout(Duration::from_secs(1), async {
            loop {
                if registry.len().await == 0 {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn sse_buffers_burst_without_polling_slow_subscriber() {
        let (forwarded_tx, _forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        let request = Request::builder()
            .method("GET")
            .uri("/acp")
            .header(header::ACCEPT, EVENT_STREAM_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id.as_str())
            .body(Body::empty())
            .unwrap();
        let response = handle_get(registry, request).await;
        assert_eq!(response.status(), StatusCode::OK);

        timeout(Duration::from_secs(1), async {
            for index in 0..ISSUE_288_BURST {
                connection
                    .push_connection_stream_for_test(format!("message-{index}"))
                    .unwrap();
            }
        })
        .await
        .expect("enqueueing must not wait for the SSE body to be polled");
        connection.shutdown().await;

        let body = timeout(
            Duration::from_secs(1),
            axum::body::to_bytes(response.into_body(), 1024 * 1024),
        )
        .await
        .expect("SSE body should close after shutdown")
        .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let messages = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect::<Vec<_>>();
        let expected = (0..ISSUE_288_BURST)
            .map(|index| format!("message-{index}"))
            .collect::<Vec<_>>();
        assert_eq!(
            messages,
            expected.iter().map(String::as_str).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn sse_allows_one_active_subscriber_per_logical_stream() {
        let (forwarded_tx, _forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        let request = |session_id: Option<&str>| {
            let mut request = Request::builder()
                .method("GET")
                .uri("/acp")
                .header(header::ACCEPT, EVENT_STREAM_MIME_TYPE)
                .header(HEADER_CONNECTION_ID, connection_id.as_str());
            if let Some(session_id) = session_id {
                request = request.header(HEADER_SESSION_ID, session_id);
            }
            request.body(Body::empty()).unwrap()
        };

        let connection_stream = handle_get(registry.clone(), request(None)).await;
        assert_eq!(connection_stream.status(), StatusCode::OK);
        let duplicate_connection_stream = handle_get(registry.clone(), request(None)).await;
        assert_eq!(duplicate_connection_stream.status(), StatusCode::CONFLICT);

        let session_one = handle_get(registry.clone(), request(Some("session-1"))).await;
        assert_eq!(session_one.status(), StatusCode::OK);
        let duplicate_session_one = handle_get(registry.clone(), request(Some("session-1"))).await;
        assert_eq!(duplicate_session_one.status(), StatusCode::CONFLICT);
        let session_two = handle_get(registry.clone(), request(Some("session-2"))).await;
        assert_eq!(session_two.status(), StatusCode::OK);

        for message in [
            "already-read",
            "queued-before-drop-1",
            "queued-before-drop-2",
        ] {
            connection
                .push_connection_stream_for_test(message.to_string())
                .unwrap();
        }
        let mut connection_body = connection_stream.into_body().into_data_stream();
        let first_event = timeout(Duration::from_secs(1), connection_body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            String::from_utf8(first_event.to_vec()).unwrap(),
            "id: 1\ndata: already-read\n\n"
        );
        drop(connection_body);

        connection
            .push_connection_stream_for_test("queued-after-drop".to_string())
            .unwrap();
        let resumed_connection_stream = handle_get(registry.clone(), request(None)).await;
        assert_eq!(resumed_connection_stream.status(), StatusCode::OK);

        connection.shutdown().await;
        let body = timeout(
            Duration::from_secs(1),
            axum::body::to_bytes(resumed_connection_stream.into_body(), 1024),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            String::from_utf8(body.to_vec()).unwrap(),
            concat!(
                "id: 2\ndata: queued-before-drop-1\n\n",
                "id: 3\ndata: queued-before-drop-2\n\n",
                "id: 4\ndata: queued-after-drop\n\n",
            )
        );
    }

    #[tokio::test]
    async fn sse_drains_queued_message_before_connection_close() {
        let (forwarded_tx, _forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        let request = Request::builder()
            .method("GET")
            .uri("/acp")
            .header(header::ACCEPT, EVENT_STREAM_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id)
            .body(Body::empty())
            .unwrap();
        let response = handle_get(registry, request).await;
        assert_eq!(response.status(), StatusCode::OK);

        connection
            .push_connection_stream_for_test("final-message".to_string())
            .unwrap();
        connection.shutdown().await;

        let body = timeout(
            Duration::from_secs(1),
            axum::body::to_bytes(response.into_body(), 1024),
        )
        .await
        .expect("SSE body should close without hanging")
        .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("data: final-message"));
    }

    #[tokio::test]
    async fn sse_subscribed_after_connection_close_drains_queued_frames() {
        let (forwarded_tx, _forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        connection
            .push_connection_stream_for_test("queued-before-subscribe".to_string())
            .unwrap();
        connection.shutdown().await;

        let request = Request::builder()
            .method("GET")
            .uri("/acp")
            .header(header::ACCEPT, EVENT_STREAM_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id)
            .body(Body::empty())
            .unwrap();
        let response = handle_get(registry, request).await;
        assert_eq!(response.status(), StatusCode::OK);

        let body = timeout(
            Duration::from_secs(1),
            axum::body::to_bytes(response.into_body(), 1024),
        )
        .await
        .expect("an already-closed SSE stream should end without hanging")
        .unwrap();
        assert_eq!(
            String::from_utf8(body.to_vec()).unwrap(),
            "id: 1\ndata: queued-before-subscribe\n\n"
        );
    }

    #[tokio::test]
    async fn post_forwards_header_session_id_to_agent_params() {
        let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": { "prompt": [] }
        })
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id.as_str())
            .header(HEADER_SESSION_ID, "session-1")
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry), request).await;

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let forwarded = timeout(Duration::from_secs(1), forwarded_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            session_id_from_message(&forwarded).as_deref(),
            Some("session-1")
        );
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn post_does_not_apply_session_header_to_cancel_request() {
        let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        let body = json!({
            "jsonrpc": "2.0",
            "method": "$/cancel_request",
            "params": { "requestId": 1 }
        })
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id.as_str())
            .header(HEADER_SESSION_ID, "session-1")
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry), request).await;

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let forwarded = timeout(Duration::from_secs(1), forwarded_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(session_id_from_message(&forwarded), None);
        let value = serde_json::to_value(forwarded).unwrap();
        assert_eq!(value["params"], json!({ "requestId": 1 }));
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn post_rejects_session_scoped_method_without_session_id() {
        let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;

        for method in ["session/delete", "session/fork"] {
            let body = json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": {}
            })
            .to_string();
            let request = Request::builder()
                .method("POST")
                .uri("/acp")
                .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
                .header(HEADER_CONNECTION_ID, connection_id.as_str())
                .body(Body::from(body))
                .unwrap();

            let response = handle_post(State(registry.clone()), request).await;

            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{method}");
            let body = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            assert_eq!(body.as_ref(), b"Acp-Session-Id header required", "{method}");
            assert!(forwarded_rx.try_recv().is_err(), "{method}");
        }
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn post_rejects_batch_session_scoped_method_without_session_id() {
        let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel();
        let registry = Arc::new(ConnectionRegistry::new(Arc::new(CapturingAgentFactory {
            forwarded: forwarded_tx,
        })));
        let (connection_id, connection) = registry.create_connection().await;
        let body = json!([
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "custom/valid",
                "params": {}
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/delete",
                "params": {}
            }
        ])
        .to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/acp")
            .header(header::CONTENT_TYPE, JSON_MIME_TYPE)
            .header(HEADER_CONNECTION_ID, connection_id.as_str())
            .body(Body::from(body))
            .unwrap();

        let response = handle_post(State(registry), request).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), b"Acp-Session-Id header required");
        assert!(forwarded_rx.try_recv().is_err());
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn sse_takeover_evicts_an_unpolled_subscriber() {
        for session_id in [None, Some("session-1")] {
            let (registry, mut forwarded_rx) = capturing_registry();
            let (connection_id, connection) = registry.create_connection().await;

            // #1 subscribes and its body is never polled: hyper's writer is stuck.
            let stuck =
                handle_get(registry.clone(), sse_get(&connection_id, session_id, &[])).await;
            assert_eq!(stuck.status(), StatusCode::OK);
            assert_eq!(stuck.headers()[HEADER_STREAM_FEATURES], STREAM_FEATURES);
            push_tagged(&connection, session_id, "before-takeover").await;

            // Without the header the 409 contract is unchanged.
            let refused =
                handle_get(registry.clone(), sse_get(&connection_id, session_id, &[])).await;
            assert_eq!(refused.status(), StatusCode::CONFLICT);
            assert_eq!(refused.headers()[HEADER_STREAM_FEATURES], STREAM_FEATURES);

            // With it, the new subscriber wins at once.
            let taken = timeout(
                Duration::from_millis(100),
                handle_get(
                    registry.clone(),
                    sse_get(&connection_id, session_id, &[(HEADER_TAKEOVER, "1")]),
                ),
            )
            .await
            .expect("takeover must not wait for the stuck stream");
            assert_eq!(taken.status(), StatusCode::OK);
            push_tagged(&connection, session_id, "after-takeover").await;

            // The new owner keeps the stream, also after the stale body is dropped.
            let again =
                handle_get(registry.clone(), sse_get(&connection_id, session_id, &[])).await;
            assert_eq!(again.status(), StatusCode::CONFLICT);
            drop(stuck);
            let again =
                handle_get(registry.clone(), sse_get(&connection_id, session_id, &[])).await;
            assert_eq!(again.status(), StatusCode::CONFLICT);

            // The agent task was never aborted.
            connection
                .send_frame_to_agent(TransportFrame::Single(
                    RawJsonRpcMessage::notification("test/ping".to_string(), json!({})).unwrap(),
                ))
                .unwrap();
            let ping = timeout(Duration::from_secs(1), forwarded_rx.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(method_for_message(&ping), Some("test/ping"));

            // Nothing the stuck stream never dequeued is lost.
            connection.shutdown().await;
            let body = timeout(
                Duration::from_secs(1),
                axum::body::to_bytes(taken.into_body(), 64 * 1024),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(
                data_tags(&String::from_utf8(body.to_vec()).unwrap()),
                ["before-takeover", "after-takeover"]
            );
        }
    }

    #[tokio::test]
    async fn sse_takeover_ends_a_still_polled_subscriber() {
        let (registry, _forwarded_rx) = capturing_registry();
        let (connection_id, _connection) = registry.create_connection().await;
        let old = handle_get(registry.clone(), sse_get(&connection_id, None, &[])).await;
        let _new = handle_get(
            registry.clone(),
            sse_get(&connection_id, None, &[(HEADER_TAKEOVER, "1")]),
        )
        .await;
        let body = timeout(
            Duration::from_secs(1),
            axum::body::to_bytes(old.into_body(), 1024),
        )
        .await
        .expect("a revoked stream must end, not idle on keepalives")
        .unwrap();
        assert!(body.is_empty());
    }

    #[tokio::test]
    async fn sse_takeover_on_idle_mailbox_delivers_queued_frames() {
        let (registry, _forwarded_rx) = capturing_registry();
        let (connection_id, connection) = registry.create_connection().await;
        push_tagged(&connection, None, "queued-1").await;
        push_tagged(&connection, None, "queued-2").await;

        let response = handle_get(
            registry.clone(),
            sse_get(&connection_id, None, &[(HEADER_TAKEOVER, "1")]),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);

        connection.shutdown().await;
        let body = timeout(
            Duration::from_secs(1),
            axum::body::to_bytes(response.into_body(), 64 * 1024),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            data_tags(&String::from_utf8(body.to_vec()).unwrap()),
            ["queued-1", "queued-2"]
        );
    }

    #[tokio::test]
    async fn sse_takeover_rejects_non_one_values() {
        let (registry, _forwarded_rx) = capturing_registry();
        let (connection_id, _connection) = registry.create_connection().await;

        let held = handle_get(registry.clone(), sse_get(&connection_id, None, &[])).await;
        assert_eq!(held.status(), StatusCode::OK);

        for non_one in ["0", "true", "yes", "2", ""] {
            let refused = handle_get(
                registry.clone(),
                sse_get(&connection_id, None, &[(HEADER_TAKEOVER, non_one)]),
            )
            .await;
            assert_eq!(refused.status(), StatusCode::CONFLICT);
            assert_eq!(refused.headers()[HEADER_STREAM_FEATURES], STREAM_FEATURES);
        }
    }

    fn event_ids(body: &str) -> Vec<u64> {
        body.lines()
            .filter_map(|line| line.strip_prefix("id: "))
            .map(|id| id.parse().unwrap())
            .collect()
    }

    #[tokio::test]
    async fn sse_frames_carry_ids_and_last_event_id_replays_the_rest() {
        let (registry, _forwarded_rx) = capturing_registry();
        let (connection_id, connection) = registry.create_connection().await;
        for tag in ["one", "two", "three"] {
            push_tagged(&connection, None, tag).await;
        }
        let first = handle_get(registry.clone(), sse_get(&connection_id, None, &[])).await;
        assert_eq!(first.headers()[HEADER_STREAM_FEATURES], "takeover, replay");
        assert!(first.headers().get(HEADER_REPLAY).is_none());
        let mut body = first.into_body().into_data_stream();
        for _ in 0..3 {
            timeout(Duration::from_secs(1), body.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        }
        drop(body); // goose wrote all three; the client only processed id 1
        push_tagged(&connection, None, "four").await;

        let resumed = handle_get(
            registry.clone(),
            sse_get(&connection_id, None, &[(HEADER_LAST_EVENT_ID, "1")]),
        )
        .await;
        assert_eq!(resumed.status(), StatusCode::OK);
        assert_eq!(resumed.headers()[HEADER_REPLAY], "complete");
        connection.shutdown().await;
        let body = String::from_utf8(
            axum::body::to_bytes(resumed.into_body(), 64 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert_eq!(event_ids(&body), [2, 3, 4]);
        assert_eq!(data_tags(&body), ["two", "three", "four"]);
    }

    #[tokio::test]
    async fn sse_last_event_id_older_than_the_ring_flags_a_gap() {
        let (registry, _forwarded_rx) = capturing_registry();
        let (connection_id, connection) = registry.create_connection().await;
        let total = crate::http_mailbox::REPLAY_RING_MAX_FRAMES + 8;
        for n in 0..total {
            connection
                .push_connection_stream_for_test(format!("m{n}"))
                .unwrap();
        }
        let first = handle_get(registry.clone(), sse_get(&connection_id, None, &[])).await;
        let mut body = first.into_body().into_data_stream();
        timeout(Duration::from_secs(5), async {
            for _ in 0..total {
                body.next().await.unwrap().unwrap();
            }
        })
        .await
        .unwrap();
        drop(body);

        let resumed = handle_get(
            registry.clone(),
            sse_get(&connection_id, None, &[(HEADER_LAST_EVENT_ID, "0")]),
        )
        .await;
        assert_eq!(resumed.headers()[HEADER_REPLAY], "gap");
        connection.shutdown().await;
        let body = String::from_utf8(
            axum::body::to_bytes(resumed.into_body(), 4 * 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        let ids = event_ids(&body);
        assert_eq!(ids.len(), crate::http_mailbox::REPLAY_RING_MAX_FRAMES);
        assert_eq!(ids.first().copied(), Some(9), "seqs 1..=8 were evicted");
    }

    #[tokio::test]
    async fn sse_takeover_with_last_event_id_recovers_frames_a_stuck_stream_dequeued() {
        for session_id in [None, Some("session-1")] {
            let (registry, _forwarded_rx) = capturing_registry();
            let (connection_id, connection) = registry.create_connection().await;
            push_tagged(&connection, session_id, "one").await;
            push_tagged(&connection, session_id, "two").await;

            // goose writes both frames into a socket the client never reads.
            let stuck =
                handle_get(registry.clone(), sse_get(&connection_id, session_id, &[])).await;
            let mut stuck_body = stuck.into_body().into_data_stream();
            for _ in 0..2 {
                timeout(Duration::from_secs(1), stuck_body.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
            }
            push_tagged(&connection, session_id, "three").await;

            let taken = handle_get(
                registry.clone(),
                sse_get(
                    &connection_id,
                    session_id,
                    &[(HEADER_TAKEOVER, "1"), (HEADER_LAST_EVENT_ID, "0")],
                ),
            )
            .await;
            assert_eq!(taken.status(), StatusCode::OK);
            assert_eq!(taken.headers()[HEADER_REPLAY], "complete");
            connection.shutdown().await;
            let body = String::from_utf8(
                axum::body::to_bytes(taken.into_body(), 64 * 1024)
                    .await
                    .unwrap()
                    .to_vec(),
            )
            .unwrap();
            assert_eq!(event_ids(&body), [1, 2, 3]);
            assert_eq!(data_tags(&body), ["one", "two", "three"]);
            drop(stuck_body); // the half-open socket lingers; harmless
        }
    }
}
