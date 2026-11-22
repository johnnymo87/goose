# Upgrading the fork to a new upstream goose release

This is the brief for the **one worker per upgrade** in the stay-on-latest lane
(workstation epic `workstation-6w7i.27`, runbook in its `design` field). The
goose program's stall-watch coordinator launches you. Your job is to get one
upstream release onto the fork's patch stack, prove it downstream, and make
its workstation pin PR ready for cutover. You never deploy it.

## What you are given

The coordinator's launch brief names:

- **the version**: for example `v1.61.0`. One worker per version. If two
  versions are outstanding, the newer one supersedes (see "Superseded"
  below).
- **the PR**, which is one of two kinds:
  - **clean**: the fork watcher (L1) rebased the stack, built it, ran the
    patched test filter, tagged `patched-X.Y.Z.0`, and `update-goose.yml`
    opened the workstation pin PR, labelled `goose-upgrade`.
  - **conflicted**: L1 pushed `base/vX.Y.Z` (the bare upstream tag) and
    `upstream-sync/vX.Y.Z` (the patches that applied), then opened ONE draft
    PR `upstream-sync/vX.Y.Z -> base/vX.Y.Z` that names every patch that
    failed and includes the log tail.
- **the tag `goose`** for session-tags. Pass it to anything you launch.

**PRs that are not new upgrades.** The `pkgs/goose/**` wait in workstation
also fires for these. If the PR you were woken for is one of them, it is not an
upgrade. Say so to the coordinator and stop.

- a pin PR opened by hand, such as the port's own A4.10 PR
- a worker's PR that edits `pkgs/goose` for another reason

## Hard rules

- **Never deploy.** Do not run `goose-cutover`, `nixos-rebuild`, or
  `home-manager switch`, and do not restart `goose-serve` or `honkd`.
- **Never merge the pin PR.** PRs labelled `goose-upgrade` are excluded from
  the standing auto-merge green light. Only `goose-cutover` merges them. You
  may auto-merge your own fork PRs (see the next rule for how).
- **Use an opencode session if anything you run could restart goose-serve.**
  Nothing in this brief should. A goose tool shell runs inside goose-serve's
  cgroup and dies with it, which is why the cutover runs as its own unit.
- **Never let a goose you built touch the real `~/.local/share/goose`.**
  Before any `cargo test`, and before any run of a non-deployed goose, export
  `GOOSE_PATH_ROOT=$(mktemp -d)`. On 2026-11-21 (02:19Z on the 22nd) a 1.60
  binary run without it stamped production `sessions.db` to schema v18 while
  production was still 1.54.
- **Merge fork PRs only with `--rebase`.** Never squash and never make a merge
  commit (the fork's settings allow only rebase merges). The stack must stay one commit per patch, because L1 rebases it with
  `git rebase --onto`. You may auto-merge your own fork PRs this way, under the
  standing green light for the johnnymo87 org.
- **Worktrees only.** Never work at a shared root. Commit bare, without an
  inline identity.

## Disk and memory

- Before any build, check `df -h ~` and confirm at least 40G free. If there
  is less, stop and report.
- Use scoped tests, not `cargo test --workspace`. Release builds of the whole
  workspace exhaust the box. Use CI's flags, or a filter can match nothing or fail
  to build:
  `cargo test --release -p goose --features "$F" --lib -- <filter>`. Take `F`
  from `.github/workflows/patched-release.yml`; it includes `acp-http` and
  `code-mode`.
- When you are done, run `cargo clean` in your worktree (or delete its
  `target/`).

## Clean path

1. **Run the probe on the candidate.** Build the candidate from the pin PR's head
   commit. Do not build from its branch name: that branch is force-pushed.
   ```bash
   sha=$(gh pr view <N> -R johnnymo87/workstation --json headRefOid -q .headRefOid)
   nix build "github:johnnymo87/workstation/$sha#goose" -o /tmp/goose-cand
   goose-upgrade-probe /tmp/goose-cand/bin/goose
   ```
   Do not use the raw release tarball. It is not patchelf'd, so it will not run on
   NixOS.
   - `--old-bin` defaults to the deployed goose, which is the one rollback
     returns to.
   - Every line is PASS or FAIL, and there are no skips. A `--only` run ends in
     `RESULT PARTIAL` and exits 3. It never counts as a gate result.
   - On a new version, expect exactly one FAIL, `rendered-surface`, which reports
     "no entry approved". Step 2 clears it. Any other FAIL is your work item, not
     a flake to retry past, with one exception: `honk-suite` already reruns a
     failed test target once.
2. **The deps-lane fixture.** eng-agent-platform's maven-renovate rendered-surface
   fixture (`lanes/fixtures-rendered-surface.json`) holds one entry per goose
   version. The lane refuses a goose that has no entry.
   - Bless the new entry in an eng-agent-platform **worktree**:
     `lanes/test_rendered_surface.sh --bless`, with the candidate first on PATH.
   - Open a PR whose body says what the new version changed in the prompt. That
     PR is a review of what the lane's agent sees, so it needs a human's eyes, not
     just a green check. Tell Jonathan it is waiting.
3. **Re-run the full probe after the fixture PR merges.** Point it at a fresh
   worktree at origin/main:
   ```bash
   GOOSE_PROBE_EAP_DIR=<fresh eng-agent-platform worktree at origin/main> \
     goose-upgrade-probe /tmp/goose-cand/bin/goose
   ```
   - Never point it at your unmerged worktree. That passes the gate on a fixture
     nobody reviewed.
   - Never pull the shared `~/projects/eng-agent-platform` root either.
   - It must print `RESULT PASS` and exit 0. Paste the PASS block into the pin PR
     as a comment.
4. **The patch comment in `pkgs/goose/default.nix`** on the pin PR's branch:
   - Update the base version and the patch list.
   - For each patch you dropped, write a line saying which upstream PR absorbed
     it.
5. **Label the pin PR `goose-cutover-ready`.** The weekly cutover (L3b) picks it
   up in a quiet window.

## Conflicted path

1. Check out `upstream-sync/vX.Y.Z` in a fork worktree. Then go through the
   failing patches **one at a time**, as listed in the draft PR. The PR's top
   commit adds `fork-sync/vX.Y.Z.md`, the watcher's report (the same table). Leave it:
   L1 drops `fork-sync/` commits and `Cargo.lock`-only commits when it tags. Regenerate
   `Cargo.lock` with a build or `cargo metadata`, which change only what the patches
   need (not `cargo generate-lockfile`, which moves every dependency off the versions
   upstream released with). Never rebase it.
2. For each failing patch:
   - **Absorbed upstream?** Search the upstream changelog and PRs since the
     old base. If upstream now does the same thing, drop the patch. Record
     the absorbing upstream PR in the drop line (step 3 of the clean path) and
     in the PR body. On the branch, record the drop as an **empty commit with the
     patch's original subject** (`git commit --allow-empty -m '<original subject>'`,
     the absorbing PR in the body). L1 counts a patch whose subject is missing as
     owed and will not tag while any is owed; it drops empty commits when it tags.
   - **Otherwise, port it.** Keep the fork's code in fork-only modules and
     files. Touch upstream files only for registration lines, such as a
     `mod`, a match arm, or a builder call. That keeps the next rebase
     mechanical. Port the patch's tests with it, and keep its test names, so
     the CI filter (below) still matches them. **Keep the patch's original
     commit subject**: L1 matches patches across a carry-forward by subject.
3. After each patch, build and run that patch's tests. Run the whole patched
   filter at the end. The filter must match at least one test per entry:
   cargo exits 0 on a filter that matches nothing.
4. Push your commits onto `upstream-sync/vX.Y.Z` itself (one commit per patch, never
   a merge), mark the draft PR ready, and run `adversarial-reviewer-opus` on its diff.
5. Merge it with `gh pr merge <N> -R johnnymo87/goose --rebase`. The fork allows only
   rebase merges, and `base/*` and `patched/*` require linear history: L1 needs one
   commit per patch, and a squash would fold the stack (and the `fork-sync/` report)
   into one commit, which L1 refuses.
6. Then run L1 for that version:
   `gh workflow run upstream-sync.yml -R johnnymo87/goose -f prefix= -f target=vX.Y.Z`
   (the empty prefix makes it a real run, not a `test-l1/` rehearsal). It re-applies
   `base/vX.Y.Z` onto the bare tag, runs the patched test filter, pushes
   `patched/vX.Y.Z`, and tags `patched-X.Y.Z.0`; the release and then the pin PR
   follow. If the filter fails, the run is red and the report lands as a comment on
   your PR. Continue with the clean path.

**Superseded.** If a newer release lands while you are working, L1 carries your
`upstream-sync` branch forward onto it, so the work you pushed is kept, and closes the
old draft PR. The new report lists any original patch whose subject is not on your
branch; L1 does not re-apply those (you may have dropped one on purpose), so port or
drop each one. Fetch the new `upstream-sync/vNEW` instead of rebasing onto it, and carry
on under the new version. A port you already merged into `base/vX.Y.Z` is not lost:
L1 tags it before it moves on to a newer release. If a newer *clean* pin PR replaces
yours, stop and tell the coordinator.

**Majors.** A major-version pin PR gets the `goose-major` label. Cutover also
needs a human-applied `goose-major-ok`. Do everything above, then report to
Jonathan that it is waiting on that label.

## Patch inventory

There are two authoritative lists. Read both on the branch you are working on:

- **What each patch does and who depends on it:** the comment block at the
  top of workstation's `pkgs/goose/default.nix`. It is numbered and has one
  paragraph per patch.
- **Which tests prove it:** the `cargo test` lines in
  `.github/workflows/patched-release.yml` on the patched branch. That is the
  "patched test filter".

As of patched-1.54.0.18:

| # | Patch | Consumer | Tests in the CI filter |
|---|---|---|---|
| 1 | `GOOSE_HEADLESS_CACHE_TTL` | stall-watch timer-resumed sessions | `goose-cli --lib cache_ttl`, `goose --lib cache_ttl` |
| 2 | `GCP_VERTEX_HOST` / `_SESSION_HEADER` | goose-serve -> cfp :8789 | `goose --lib gcpvertexai session_context` |
| 3 | `GCP_AUTH_SCOPES` | goose-serve (gateway needs userinfo.email) | `goose --lib gcpauth` |
| 4 | developer shell kills the process group | every goose tool call | `goose --lib process_group` |
| 5-6 | acp-http takeover + SSE replay | honkd (refuses an upstream without them) | `agent-client-protocol-http --features server --lib`; probe `honk-live`, `stall-watch` |
| 7 | restored session keeps its cache TTL | stall-watch coordinators | `goose --lib cache_ttl` |
| 8 | `session/settings/set` (cacheTtl, recipe) | stall-watch via honk | `acp_server_test test_session_settings`; `goose-sdk-types settings_request` |
| 9 | `_meta.goose.maxTurns` | stall-watch, maven-renovate lane | `goose --lib max_turns_from_meta`; `acp_server_test test_prompt_max_turns_bounds_run` |
| 10 | fast `session/list` | honk pickers, pigeon | (no dedicated filter; probe `migration`) |
| 11 | `self_compact_and_resume` | every interactive goose session | `goose --lib self_compact compact_kind`; `--test self_compact`; `acp_server_test test_prompt_self_compact test_recipe_session_after_reload_never_lists_self_compact_and_resume` |
| 12 | rolling memory | stall-watch coordinators | `goose --lib rolling`; `--test rolling_memory`; `goose-sdk-types rolling_memory_dto` |
| 13 | `GOOSE_GLOBAL_SKILLS=false` | maven-renovate lane | `goose --lib global_skill` |
| 14 | MCP startup timeout, parallel load | honk per-session `/mcp` | `goose --lib mcp_client_startup_timeout` |
| 15 | `_meta.goose.turnError` | honkd, nvim pickers, stall-watch | `goose --lib turn_error`; `--test turn_error`; `acp_server_test test_prompt_turn_error` |
| 16 | subagent policy (roles dir, default recipe) | users/dev/goose-roles.nix | `goose --lib subagent_policy test_add_final_output_tool test_parent_link_is_visible_to_providers` |
| 17 | plugin `.mcp.json` http servers | project plugins (lane-guard) | (none in the filter; add one when porting) |
| 18 | provider stream read during tools | long tool calls | `goose --lib toolshim_provider_stream` |
| 19 | `GCP_VERTEX_GEMINI_HOST`, header list, parent header | goose-serve, aigateway ledger | `goose --lib gcpvertexai` |
| 20 | fixed system prompt, subdirectory hints as messages | prompt cache on every session | `goose --lib subdirectory_hints deliver_subdirectory_hints requeue_after_history_replacement requeue_hints_missing_from` |
| 21 | `messageId` + `truncate_from_message` | honk chat `/undo` | `goose --lib message_id_from_meta agent_capabilities_advertise_truncate_from_message truncate_conversation_from_message`; `acp_server_test test_prompt_message_id_and_truncate_from_message` |
| 22 | cache-miss breaker | Gemini delegates | `goose --lib cache_miss_breaker` |
| 23 | recipe `settings.thinking_effort` | goose-roles | `goose --lib recipe_thinking_effort recipe_without_thinking_effort thinking_effort_parsing invalid_thinking_effort` |
| 24 | Claude `thinking.display = summarized`. Expect to DROP it on any base >= v1.57.0 (upstream #13029). | honk thinking display | `goose-provider-types thinking_display summarized_thinking_round_trips test_create_request_adaptive_thinking test_create_request_enabled_thinking_with_budget` |
| 25 | streamed Claude thinking deltas | honk thinking display | `goose-provider-types test_streaming_thinking test_streamed_thinking test_streaming_omitted_thinking test_streaming_redacted_thinking native_claude_thinking` |

Every `goose` row runs with `--release --features "$F"`. The table is a map
for orientation. When it disagrees with the two sources
above, they win. Fix the table in the same PR that changed them.

## Done means

- The eng-agent-platform fixture PR for the new version has merged.
- The pin PR carries a comment with a probe `RESULT PASS`, run against origin/main
  after that merge, and the `goose-cutover-ready` label.
- The patch comment is updated, including any drops.
- Your fork worktree has been through `cargo clean`.
- You have sent ONE `swarm_send` to the goose coordinator. Say what landed,
  what was dropped and why, and anything that needs Jonathan, such as a
  fixture re-approval or `goose-major-ok`.
