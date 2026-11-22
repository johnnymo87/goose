# Vendored: agent-client-protocol-http 2.2.0

- Source: crates.io `agent-client-protocol-http` 2.2.0,
  sha256 `18e3d7117a48e9db3c4b5e29f7539bc14e390f1f782d10a99b1d9f5c3a8cc28f`
  (the checksum upstream goose v1.54.0's Cargo.lock pins).
- Upstream VCS: github.com/agentclientprotocol/rust-sdk @ 2a78849d3eb3dcb140dade3b8fc938cf1e2b9ce5,
  path `src/agent-client-protocol-http` (from `.cargo_vcs_info.json`).
- License: Apache-2.0 (`LICENSE`, fetched from the upstream repo at that commit; the
  crate package ships none).
- Removed from the package: its nested `Cargo.lock` (ignored for workspace members).
- Wired in by the root `Cargo.toml`: workspace `members` (so `cargo test -p` works)
  and `[patch.crates-io]`.
- Formatting: never `cargo fmt --all` in this repo; it would reformat this crate. Use
  `rustfmt --edition 2024` on files you change.

## Local modifications (Apache-2.0 §4(b): each modified file carries a notice)

All on the HTTP SSE path only; the WebSocket path (`OutboundMailbox`/`OutboundLease` in
`connection.rs`, and `websocket_server.rs`) is unchanged.

- New `src/http_mailbox.rs`: owner-token HTTP mailbox (takeover) and per-stream replay ring.
- `src/connection.rs`, `src/http_server.rs`, `src/protocol.rs`, `src/lib.rs`: wiring,
  `Acp-Takeover`, `Acp-Stream-Features`, SSE `id:`, `Last-Event-ID`, `Acp-Replay`.

## When to delete this directory

When an upstream crate release contains equivalent takeover + replay and goose depends
on it. On every rebase onto a newer goose: if goose's `agent-client-protocol-http`
requirement leaves 2.x, the `[patch]` silently stops applying (cargo only warns
"patch was not used"); the release workflow's graph check fails in that case.
