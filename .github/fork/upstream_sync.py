#!/usr/bin/env python3
"""Fork watcher (L1, workstation-6w7i.27.4): move the fork's patch stack onto the newest
upstream goose release.

Runs daily from .github/workflows/upstream-sync.yml. Runnable locally for a dry run:

    L1_PREFIX=test-l1/ L1_BASE=patched-1.54.0.18 L1_TARGET=v1.60.2 L1_SKIP_TESTS=1 \
        python3 .github/fork/upstream_sync.py

Outcomes, one per run:
  clean     the stack cherry-picks onto vNEW and the patched test filter passes:
            push patched/vNEW and tag patched-NEW.0 (the tag push starts patched-release.yml).
  conflict  push base/vNEW (the bare upstream tag) and upstream-sync/vNEW (the patches that
            applied plus a fork-sync/vNEW.md report), and open ONE draft PR
            upstream-sync/vNEW -> base/vNEW naming every patch that failed.
The upgrade worker's brief is docs/fork/UPGRADING.md on this branch.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

UPSTREAM = "aaif-goose/goose"
DEBOUNCE = timedelta(hours=48)
REQUIRED_ASSETS = [
    "goose-aarch64-unknown-linux-gnu.tar.gz",
    "goose-x86_64-unknown-linux-gnu.tar.gz",
    "goose-aarch64-apple-darwin.tar.gz",
    "SHA256SUMS",
]
MARKER_DIR = "fork-sync/"
LOCK_REGEN_SUBJECT = "Cargo.lock: regenerate for {v} (fork watcher)"
MARKER_SUBJECT = "fork-sync: {v} upstream-sync report (fork watcher)"
TEST_WORKFLOW = ".github/workflows/patched-test.yml"

Version = tuple[int, int, int]


def parse_version(tag: str) -> Optional[Version]:
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag)
    return tuple(int(x) for x in m.groups()) if m else None  # type: ignore[return-value]


def parse_patched_tag(tag: str, tag_prefix: str = "") -> Optional[tuple[Version, int]]:
    m = re.fullmatch(re.escape(tag_prefix) + r"patched-(\d+)\.(\d+)\.(\d+)\.(\d+)", tag)
    if not m:
        return None
    a, b, c, n = (int(x) for x in m.groups())
    return (a, b, c), n


def vstr(v: Version) -> str:
    return "v%d.%d.%d" % v


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def pick_target(releases: list[dict], now: datetime, debounce: timedelta = DEBOUNCE) -> Optional[str]:
    """The highest stable release that stayed the latest release for at least `debounce`.

    "Latest" is the highest version published so far; a release stops being latest when a
    higher one is published (a backport to an older line never is, and never ends one).
    """
    stable = []
    for r in releases:
        v = parse_version(r["tag_name"])
        if v and not r.get("prerelease") and not r.get("draft") and r.get("published_at"):
            stable.append((_ts(r["published_at"]), v, r["tag_name"]))
    stable.sort()
    reigns: list[tuple[Version, str, datetime, datetime]] = []
    for ts, v, tag in stable:
        if reigns and v <= reigns[-1][0]:
            continue
        if reigns:
            pv, ptag, pstart, _ = reigns[-1]
            reigns[-1] = (pv, ptag, pstart, ts)
        reigns.append((v, tag, ts, now))
    best = None
    for v, tag, start, end in reigns:
        if end - start >= debounce:
            best = tag
    return best


def pick_base(fork_releases: list[dict], tag_prefix: str = "") -> Optional[str]:
    """The newest complete patched release: published, with every asset the pin needs."""
    best = None
    for r in fork_releases:
        p = parse_patched_tag(r["tag_name"], tag_prefix)
        if not p or r.get("draft"):
            continue
        names = {a if isinstance(a, str) else a["name"] for a in r.get("assets", [])}
        if not set(REQUIRED_ASSETS) <= names:
            continue
        if best is None or p > best[0]:
            best = (p, r["tag_name"])
    return best[1] if best else None


@dataclass
class Commit:
    sha: str
    subject: str
    files: list[str]


def filter_stack(commits: list[Commit]) -> list[Commit]:
    """Drop what the watcher regenerates or what only records a decision.

    Cargo.lock-only commits and fork-sync reports are regenerated. An empty commit is a
    recorded drop (a patch upstream absorbed, kept by subject so a carry-forward does not
    count it as owed); it is dropped from the next stack.
    """

    def regenerated(path: str) -> bool:
        return path == "Cargo.lock" or path.startswith(MARKER_DIR)

    kept = []
    for c in commits:
        markers = [f for f in c.files if f.startswith(MARKER_DIR)]
        if markers and len(markers) != len(c.files):
            raise ValueError(
                f"{c.sha[:12]} mixes {MARKER_DIR} with code (a squash merge?); the stack must stay one commit per patch"
            )
        if c.files and not all(regenerated(f) for f in c.files):
            kept.append(c)
    return kept


def owed(carried: list[Commit], original: list[Commit]) -> list[Commit]:
    """Original patches whose subject is not on a carried-forward branch.

    Named in the report, never re-applied: the worker may have dropped one as absorbed
    upstream or ported it under a new subject, and re-adding it would undo that.
    """
    have = {c.subject for c in carried}
    return [c for c in original if c.subject not in have]


@dataclass
class SyncPR:
    number: int
    version: str
    state: str


@dataclass
class State:
    hold: str
    target: Optional[str]
    base_tag: Optional[str]
    target_tags: list[str]
    sync_prs: list[SyncPR]
    base_branch_ahead: bool
    allow_same: bool = False
    in_flight: Optional[str] = None
    pending_base: Optional[str] = None
    target_branch_exists: bool = False


@dataclass
class Decision:
    kind: str
    reason: str = ""
    target: Optional[str] = None
    carry_from: Optional[str] = None
    close_prs: list[int] = field(default_factory=list)
    major: bool = False


def decide(s: State) -> Decision:
    if s.hold:
        return Decision("held", f"gooseHold is set ({s.hold!r})")
    if not s.target:
        return Decision("no-target", f"no stable upstream release has been latest for {DEBOUNCE}")
    if not s.base_tag:
        return Decision("no-base", "no complete patched-* release found")
    new = parse_version(s.target)
    old = parse_patched_tag(s.base_tag)
    old_v = old[0] if old else None
    if new is None or old_v is None:
        return Decision("no-base", f"cannot compare {s.target} with {s.base_tag}")
    major = new[0] > old_v[0]
    if new < old_v or (new == old_v and not s.allow_same):
        return Decision("up-to-date", f"{s.target} is not newer than {s.base_tag}")
    if s.in_flight:
        return Decision("release-in-flight", f"{s.in_flight} has no complete release yet; it is the next base")
    if s.pending_base:
        pv = parse_version(s.pending_base)
        return Decision(
            "from-base-branch",
            f"base/{s.pending_base} holds a finished port that was never tagged; it goes first",
            target=s.pending_base,
            close_prs=[p.number for p in s.sync_prs if p.version == s.pending_base and p.state == "OPEN"],
            major=bool(pv and pv[0] > old_v[0]),
        )
    if s.target_tags:
        return Decision("tagged", f"{', '.join(s.target_tags)} already exist", major=major)
    if s.target_branch_exists:
        return Decision("branch-exists", f"patched/{s.target} exists untagged; whoever made it tags it", major=major)
    for_target = [p for p in s.sync_prs if p.version == s.target]
    open_target = [p.number for p in for_target if p.state == "OPEN"]
    if s.base_branch_ahead:
        return Decision("from-base-branch", f"base/{s.target} holds the stack", target=s.target,
                        close_prs=open_target, major=major)
    if open_target:
        return Decision("worker-owns", f"upstream-sync PR #{open_target[0]} is open", major=major)
    if for_target:
        return Decision("closed-by-human", f"upstream-sync PR #{for_target[0].number} was closed unmerged", major=major)
    older_open = sorted(
        (p for p in s.sync_prs if p.state == "OPEN" and (parse_version(p.version) or (0, 0, 0)) < new),
        key=lambda p: parse_version(p.version) or (0, 0, 0),
        reverse=True,
    )
    return Decision(
        "fresh",
        "",
        target=s.target,
        carry_from=older_open[0].version if older_open else None,
        close_prs=[p.number for p in older_open],
        major=major,
    )


@dataclass
class PatchResult:
    commit: Commit
    status: str  # applied | empty | conflict | error
    conflicts: list[str]
    conflicts_with_bare_target: bool


def render_report(
    target: str,
    old: str,
    results: list[PatchResult],
    log_tail: str,
    major: bool,
    carried_from: Optional[str],
    tests_failed: bool = False,
    still_owed: Optional[list[Commit]] = None,
) -> str:
    failed = [r for r in results if r.status in ("conflict", "error")]
    out = [f"# upstream-sync {target}", ""]
    if tests_failed:
        out.append(f"Every patch applied onto {target}, but the patched test filter failed (log tail below).")
    else:
        out.append(f"{len(failed)} of {len(results)} patches failed to apply onto {target} (stack from {old}).")
    if carried_from:
        out.append(f"Carried forward from upstream-sync/{carried_from}, so its conflict work is kept.")
    if major:
        out.append(f"{target} is a **major** upgrade: its pin PR needs a human-applied `goose-major-ok`.")
    out += [
        "",
        "The upgrade worker's brief is `docs/fork/UPGRADING.md` on the fork-main branch (conflicted path).",
        f"`{MARKER_DIR}` is the watcher's report; L1 drops it when it tags.",
        "",
    ]
    if results:
        out += ["| # | commit | patch | status | conflicting files | conflicts with bare " + target + " |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(results, 1):
            status = f"**{r.status}**" if r.status in ("conflict", "error") else r.status
            files = ", ".join(f"`{f}`" for f in r.conflicts) or "-"
            out.append(
                f"| {i} | `{r.commit.sha[:12]}` | {r.commit.subject} | {status} | {files} | "
                f"{'yes' if r.conflicts_with_bare_target else 'no'} |"
            )
        out += ["", "`empty` means the patch's change is already in upstream: check which upstream PR absorbed it."]
    if still_owed:
        out += ["", f"Patches of the {old} stack whose subject is not on the carried branch (not re-applied; "
                "port them, or record the drop):", ""]
        out += [f"- `{c.sha[:12]}` {c.subject}" for c in still_owed]
    out += ["", "<details><summary>log tail</summary>", "", "```", log_tail.rstrip()[-20000:], "```", "</details>", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- side effects
#
# Two phases, so the push token never shares a machine with the candidate's code:
#   prepare  decides, cherry-picks, builds and tests (build scripts of a new upstream
#            release run here). Holds no write credential. Writes plan.json + sync.bundle.
#   publish  fetches the bundle and performs exactly the pushes and PR edits in plan.json
#            with GOOSE_SYNC_TOKEN. Runs nothing from the candidate tree.

LOG: list[str] = []
SECRET_ENV = re.compile(r"TOKEN|SECRET|PASSWORD|^ACTIONS_|^GH_|^GITHUB_(TOKEN|ENV|OUTPUT|PATH|STEP_SUMMARY|STATE)$")


def log(msg: str) -> None:
    LOG.append(msg)
    print(msg, flush=True)


def run(argv: list[str], cwd: Optional[str] = None, check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    p = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(argv[:4])} exited {p.returncode}: {p.stderr.strip()[-2000:]}")
    return p


def scrubbed_env() -> dict:
    return {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}


class Repo:
    def __init__(self, workdir: str, fork: str, push_token: str = ""):
        self.dir = workdir
        self.fork = fork
        self.push_token = push_token

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return run(["git", *args], cwd=self.dir, check=check)

    def init(self, reference: Optional[str]) -> None:
        os.makedirs(self.dir, exist_ok=True)
        self.git("init", "-q")
        if reference:
            with open(os.path.join(self.dir, ".git/objects/info/alternates"), "w") as f:
                f.write(os.path.join(reference, "objects") + "\n")
        self.git("remote", "add", "upstream", f"https://github.com/{UPSTREAM}.git")
        self.git("remote", "add", "fork", f"https://github.com/{self.fork}.git")
        if os.environ.get("GITHUB_ACTIONS"):
            self.git("config", "user.name", "github-actions[bot]")
            self.git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")

    def fetch(self, remote: str, ref: str) -> None:
        local = ref if ref.startswith("refs/tags/") else "refs/remotes/fork/" + ref.removeprefix("refs/heads/")
        self.git("fetch", "-q", "--no-tags", remote, f"+{ref}:{local}")

    def fetch_tag(self, tag: str) -> str:
        self.fetch("upstream", f"refs/tags/{tag}")
        return f"refs/tags/{tag}"

    def fetch_branch(self, branch: str) -> str:
        self.fetch("fork", f"refs/heads/{branch}")
        return f"refs/remotes/fork/{branch}"

    def remote_refs(self, pattern: str) -> list[str]:
        out = self.git("ls-remote", "fork", pattern).stdout
        refs = [line.split("\t", 1)[1] for line in out.splitlines() if "\t" in line]
        return [r for r in refs if not r.endswith("^{}")]

    def sha(self, ref: str) -> str:
        return self.git("rev-parse", f"{ref}^{{commit}}").stdout.strip()

    def commits(self, rng: str) -> list[Commit]:
        merges = self.git("rev-list", "--merges", rng).stdout.split()
        if merges:
            raise RuntimeError(f"{rng} has merge commits {merges[:3]}; the stack must be linear")
        out = []
        for sha in self.git("rev-list", "--reverse", rng).stdout.split():
            subject = self.git("log", "-1", "--format=%s", sha).stdout.strip()
            files = self.git("show", "--name-only", "--format=", sha).stdout.split()
            out.append(Commit(sha, subject, files))
        return out

    def conflicts_with(self, c: Commit, target: str) -> bool:
        p = self.git("merge-tree", "--write-tree", "--no-messages", f"--merge-base={c.sha}^", target, c.sha, check=False)
        return p.returncode == 1

    def apply(self, commits: list[Commit], target: str) -> list[PatchResult]:
        results = []
        for c in commits:
            before = self.git("rev-parse", "HEAD").stdout.strip()
            p = self.git("cherry-pick", "--allow-empty", "--keep-redundant-commits", c.sha, check=False)
            status, conflicts = "applied", []
            if p.returncode != 0:
                conflicts = self.git("diff", "--name-only", "--diff-filter=U").stdout.split()
                if conflicts == ["Cargo.lock"]:
                    self.git("checkout", "--ours", "Cargo.lock")
                    self.git("add", "Cargo.lock")
                    self.git("-c", "core.editor=true", "cherry-pick", "--continue")
                    conflicts = []
                else:
                    status = "conflict" if conflicts else "error"
                    if not conflicts:
                        conflicts = [(p.stderr.strip().splitlines() or ["?"])[-1]]
                    self.git("cherry-pick", "--abort", check=False)
                    self.git("reset", "-q", "--hard", before)
            if status == "applied" and not self.git("diff", "--quiet", before, "HEAD", check=False).returncode:
                status = "empty"  # kept as an empty commit: the recorded drop
            log(f"  {status:8} {c.sha[:12]} {c.subject}" + (f"  [{', '.join(conflicts)}]" if conflicts else ""))
            results.append(PatchResult(c, status, conflicts, self.conflicts_with(c, target)))
        return results

    def regenerate_lock(self, target: str, required: bool) -> None:
        try:
            p = run(["cargo", "metadata", "--format-version", "1"], cwd=self.dir, check=False, env=scrubbed_env())
            err = (p.stderr or f"exit {p.returncode}") if p.returncode else ""
        except FileNotFoundError:
            err = "cargo is not on PATH"
        if err:
            msg = f"cargo metadata failed: {err.strip()[-1500:]}"
            if required:
                raise RuntimeError(msg)
            log(f"Cargo.lock not regenerated ({msg})")
            return
        if self.git("diff", "--quiet", "--", "Cargo.lock", check=False).returncode:
            self.git("commit", "-q", "-m", LOCK_REGEN_SUBJECT.format(v=target), "--", "Cargo.lock")
            log("Cargo.lock regenerated")

    def push(self, refspecs: list[str], atomic: bool) -> None:
        auth = []
        if self.push_token:
            basic = base64.b64encode(f"x-access-token:{self.push_token}".encode()).decode()
            auth = ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]
        run(["git", *auth, "push", "-q", "--porcelain", *(["--atomic"] if atomic else []), "fork", *refspecs], cwd=self.dir)
        for r in refspecs:
            log(f"pushed {r.split(':', 1)[1]}")


def run_patched_tests(workdir: str) -> tuple[bool, str]:
    """Run the run-steps of the tree's own patched-test.yml, in order, as CI would."""
    import yaml  # python3-yaml: installed by upstream-sync.yml; system python refuses pip (PEP 668)

    with open(os.path.join(workdir, TEST_WORKFLOW)) as f:
        wf = yaml.safe_load(f)
    steps = wf["jobs"]["test-patches"]["steps"]
    env = dict(scrubbed_env(), GOOSE_PATH_ROOT=tempfile.mkdtemp(prefix="goose-path-root-"))
    tail: list[str] = []
    ran = 0
    for step in steps:
        if "uses" in step:
            continue
        extra = set(step) - {"name", "run"}
        if extra:
            return False, f"{TEST_WORKFLOW} step {step.get('name')!r} uses {sorted(extra)}, which the watcher cannot run"
        log(f"== {step.get('name')}")
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as script:
            script.write(step["run"])
        # GitHub's default `run` shell when a step names none (patched-test.yml names none).
        p = subprocess.Popen(["bash", "-e", script.name], cwd=workdir, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout
        for line in p.stdout:
            print(line, end="")
            tail = (tail + [line])[-150:]
        if p.wait() != 0:
            return False, "".join(tail)
        ran += 1
    if not ran:
        return False, f"{TEST_WORKFLOW} has no run steps"
    return True, "".join(tail)


def write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


def prepare(out: str) -> dict:
    fork = os.environ.get("GITHUB_REPOSITORY") or "johnnymo87/goose"
    prefix = os.environ.get("L1_PREFIX", "")
    if prefix and not (prefix.startswith("test-") and prefix.endswith("/")):
        raise SystemExit("L1_PREFIX must be empty (a real run) or look like test-<name>/")
    tp = prefix.replace("/", "-")
    skip_tests = os.environ.get("L1_SKIP_TESTS") == "1"
    allow_same = os.environ.get("L1_ALLOW_SAME") == "1"
    if (skip_tests or allow_same) and not prefix:
        raise SystemExit("L1_SKIP_TESTS and L1_ALLOW_SAME are for test-prefixed rehearsals only")
    workdir = os.environ.get("L1_WORKDIR") or tempfile.mkdtemp(prefix="upstream-sync-")
    repo = Repo(workdir, fork)
    plan: dict = {"fork": fork, "prefix": prefix, "target": None, "sources": [], "carried": None, "refspecs": [], "atomic": False, "pr": None,
                  "close": [], "comment": [], "label_major": False, "fail": None, "summary": ""}

    now = datetime.now(timezone.utc)
    up_releases = json.loads(run(["gh", "api", f"repos/{UPSTREAM}/releases?per_page=100"]).stdout)
    fork_releases = json.loads(run(["gh", "api", f"repos/{fork}/releases?per_page=100"]).stdout)
    target = os.environ.get("L1_TARGET") or pick_target(up_releases, now)
    base_tag = os.environ.get("L1_BASE") or pick_base(fork_releases)
    log(f"target={target} base={base_tag} prefix={prefix!r}")

    repo.init(os.environ.get("L1_REFERENCE"))
    target_tags, base_ahead, sync_prs, pending, branch_exists, in_flight = [], False, [], None, False, None
    tv, bv = parse_version(target or ""), parse_patched_tag(base_tag or "")
    if tv and bv:
        target_tags = [r.rsplit("/", 1)[1] for r in repo.remote_refs(f"refs/tags/{tp}patched-{target[1:]}.*")]
        branch_exists = bool(repo.remote_refs(f"refs/heads/{prefix}patched/{target}"))
        if not prefix and not os.environ.get("L1_BASE"):
            # A patched tag newer than the newest complete release is a release still being
            # built (or one that failed): it is the next base, so nothing may start from
            # the older one. Rehearsal tags never get releases, so rehearsals skip this.
            newer = [t for t in (r.rsplit("/", 1)[1] for r in repo.remote_refs("refs/tags/patched-*"))
                     if (parse_patched_tag(t) or ((0, 0, 0), 0)) > bv]
            in_flight = max(newer, key=parse_patched_tag) if newer else None
        raw = json.loads(run(["gh", "pr", "list", "-R", fork, "--state", "all", "--limit", "200",
                              "--json", "number,headRefName,state"]).stdout)
        sync_head = re.compile(re.escape(prefix) + r"upstream-sync/(v\d+\.\d+\.\d+)")
        for pr in raw:
            m = sync_head.fullmatch(pr["headRefName"])
            if m:
                sync_prs.append(SyncPR(pr["number"], m.group(1), pr["state"]))
        # Every base/v* ahead of its upstream tag and untagged is a finished port. One older
        # than the target goes first, or a newer release would strand it.
        bases = []
        for ref in repo.remote_refs(f"refs/heads/{prefix}base/v*"):
            v = ref.rsplit("/", 1)[1]
            pv = parse_version(v)
            if pv and bv[0] < pv <= tv and not repo.remote_refs(f"refs/tags/{tp}patched-{v[1:]}.*"):
                repo.fetch_tag(v)
                b = repo.fetch_branch(f"{prefix}base/{v}")
                if int(repo.git("rev-list", "--count", f"{v}..{b}").stdout) > 0:
                    bases.append(pv)
        bases.sort()
        if bases and bases[0] < tv:
            pending = vstr(bases[0])
        base_ahead = tv in bases

    d = decide(State(os.environ.get("GOOSE_HOLD", ""), target, base_tag, target_tags, sync_prs, base_ahead,
                     allow_same=allow_same and bool(prefix), in_flight=in_flight, pending_base=pending,
                     target_branch_exists=branch_exists))
    log(f"decision: {d.kind} {d.reason}".rstrip())
    plan["summary"] = f"upstream-sync: **{d.kind}** {d.reason} (target {target}, base {base_tag})"
    if d.kind not in ("fresh", "from-base-branch") or os.environ.get("L1_PLAN_ONLY") == "1":
        return plan

    assert d.target and base_tag
    target = d.target
    old_v = vstr(parse_patched_tag(base_tag)[0])  # type: ignore[index]
    repo.fetch_tag(target)
    repo.fetch_tag(old_v)
    stack_ref = repo.fetch_branch(f"patched/{old_v}")
    plan["target"] = target
    plan["sources"] = [[old_v, f"patched/{old_v}"]]
    original = filter_stack(repo.commits(f"{old_v}..{stack_ref}"))
    still_owed: list[Commit] = []
    if d.kind == "from-base-branch":
        raw_base = repo.commits(f"{target}..refs/remotes/fork/{prefix}base/{target}")
        stack = filter_stack(raw_base)
        still_owed = owed(raw_base, original)
        plan["sources"].append([target, f"{prefix}base/{target}"])
    elif d.carry_from:
        repo.fetch_tag(d.carry_from)
        carried_ref = repo.fetch_branch(f"{prefix}upstream-sync/{d.carry_from}")
        raw_carried = repo.commits(f"{d.carry_from}..{carried_ref}")
        stack = filter_stack(raw_carried)
        still_owed = owed(raw_carried, original)
        plan["sources"].append([d.carry_from, f"{prefix}upstream-sync/{d.carry_from}"])
        plan["carried"] = [f"{prefix}upstream-sync/{d.carry_from}", repo.sha(carried_ref)]
    else:
        stack = original
    for c in still_owed:
        log(f"  owed     {c.sha[:12]} {c.subject}")
    log(f"applying {len(stack)} commits onto {target}")
    repo.git("checkout", "-q", "--detach", target)
    results = repo.apply(stack, target)
    failed = [r for r in results if r.status in ("conflict", "error")]

    tests_failed, test_tail = False, ""
    if not failed and not still_owed:
        repo.regenerate_lock(target, required=not skip_tests)
        if skip_tests:
            log("tests skipped (L1_SKIP_TESTS=1, a rehearsal)")
        else:
            ok, test_tail = run_patched_tests(workdir)
            tests_failed = not ok
            log("patched test filter: " + ("PASS" if ok else "FAIL"))

    head = repo.sha("HEAD")
    if not failed and not still_owed and not tests_failed:
        branch, tag = f"{prefix}patched/{target}", f"{tp}patched-{target[1:]}.0"
        repo.git("tag", "-a", tag, "-m", f"{target} plus the fork's patches (fork watcher)", head)
        plan["refspecs"] = [f"{head}:refs/heads/{branch}", f"refs/tags/{tag}:refs/tags/{tag}"]
        plan["atomic"] = True
        plan["close"] = [[n, f"Superseded: {target} applied cleanly as {tag}."] for n in d.close_prs]
        plan["summary"] = f"upstream-sync: **clean** {target} -> {branch}, tagged {tag}"
    elif d.kind == "from-base-branch":
        body = render_report(target, old_v, results, "\n".join(LOG) + "\n" + test_tail, d.major, None, tests_failed,
                             still_owed)
        # The worker's PR is usually MERGED by now (that is what put the stack on base/).
        for_v = [p for p in sync_prs if p.version == target and p.state in ("OPEN", "MERGED")]
        plan["comment"] = [[max(p.number for p in for_v), body]] if for_v else []
        where = f"the comment on PR #{plan['comment'][0][0]}" if for_v else "this run's log"
        plan["fail"] = f"base/{target} is not ready to tag ({len(failed)} failed, {len(still_owed)} owed, " \
                       f"tests {'failed' if tests_failed else 'not run' if failed or still_owed else 'passed'}); see {where}"
        plan["summary"] = f"upstream-sync: **red** {plan['fail']}"
    else:
        base_branch, sync_branch = f"{prefix}base/{target}", f"{prefix}upstream-sync/{target}"
        if failed or still_owed:
            repo.regenerate_lock(target, required=False)
        body = render_report(target, old_v, results, "\n".join(LOG) + "\n" + test_tail, d.major, d.carry_from,
                             tests_failed, still_owed)
        marker = os.path.join(MARKER_DIR, f"{target}.md")
        os.makedirs(os.path.join(workdir, MARKER_DIR), exist_ok=True)
        with open(os.path.join(workdir, marker), "w") as f:
            f.write(body)
        repo.git("add", marker)
        repo.git("commit", "-q", "-m", MARKER_SUBJECT.format(v=target))
        head = repo.sha("HEAD")
        if not repo.remote_refs(f"refs/heads/{base_branch}"):
            plan["refspecs"].append(f"{repo.sha(target)}:refs/heads/{base_branch}")
        plan["refspecs"].append(f"+{head}:refs/heads/{sync_branch}")
        parts = [f"{len(failed)} of {len(results)} patches failed"] if failed else []
        parts += [f"{len(still_owed)} owed"] if still_owed else []
        what = "tests failed" if tests_failed else ", ".join(parts)
        title = (f"[{prefix.rstrip('/')}] " if prefix else "") + f"upstream-sync: {target} ({what})"
        plan["pr"] = {"base": base_branch, "head": sync_branch, "title": title, "body": body}
        plan["label_major"] = d.major
        plan["close"] = [[n, "Superseded by {pr_url}, which carries this branch forward."] for n in d.close_prs]
        plan["summary"] = f"upstream-sync: **conflict** {title}"

    # The bundle holds only what the fork lacks: every new commit sits on the target tag,
    # which publish fetches from upstream itself.
    os.makedirs(out, exist_ok=True)
    tips = [r.split(":", 1)[0].lstrip("+") for r in plan["refspecs"]]
    for i, t in enumerate(tips):
        repo.git("update-ref", f"refs/l1/{i}", t)
    tag_refs = [r.split(":", 1)[0] for r in plan["refspecs"] if r.startswith("refs/tags/")]
    p = repo.git("bundle", "create", os.path.join(out, "sync.bundle"), "--glob=refs/l1/*", *tag_refs,
                 "--not", f"refs/tags/{target}", check=False)
    if p.returncode != 0 and "empty bundle" not in p.stderr:
        raise RuntimeError(f"git bundle create: {p.stderr.strip()}")
    return plan


def patch_id(repo: Repo, sha: str) -> Optional[str]:
    diff = run(["git", "diff-tree", "-p", "-U0", "--no-commit-id", sha, "--", ".", ":!Cargo.lock"], cwd=repo.dir).stdout
    if not diff.strip():
        return None
    out = subprocess.run(["git", "patch-id", "--stable"], cwd=repo.dir, input=diff, text=True, capture_output=True,
                         check=True).stdout.split()
    return out[0] if out else None


def validate(plan: dict, repo: Repo, prefix: str, fork: str) -> None:
    """Refuse a plan that does anything but what prepare is allowed to ask for.

    prepare runs a new upstream release's build scripts, which could rewrite plan.json or
    the bundle. Everything publish pushes must therefore be a watcher-shaped ref, and every
    commit it pushes must be a patch the fork already holds (same patch-id), a Cargo.lock
    regeneration, the fork-sync report, or an empty recorded drop.
    """

    def refuse(why: str) -> None:
        raise SystemExit(f"::error::refusing the plan: {why}")

    if plan.get("fork") != fork or plan.get("prefix") != prefix:
        refuse(f"plan is for {plan.get('fork')!r} prefix {plan.get('prefix')!r}, expected {fork!r} {prefix!r}")
    target = plan.get("target") or ""
    if not parse_version(target):
        refuse(f"target {target!r}")
    P, T, TV = re.escape(prefix), re.escape(target), re.escape(target[1:])
    TP = re.escape(prefix.replace("/", "-"))
    forms = {
        "patched": rf"([0-9a-f]{{40}}):refs/heads/{P}patched/{T}",
        "base": rf"([0-9a-f]{{40}}):refs/heads/{P}base/{T}",
        "sync": rf"\+([0-9a-f]{{40}}):refs/heads/{P}upstream-sync/{T}",
        "tag": rf"refs/tags/({TP}patched-{TV}\.0):refs/tags/\1",
    }
    got: dict[str, str] = {}
    for r in plan["refspecs"]:
        kind = next((k for k, f in forms.items() if re.fullmatch(f, r)), None)
        if not kind or kind in got:
            refuse(f"refspec {r!r}")
        got[kind] = re.fullmatch(forms[kind], r).group(1)  # type: ignore[union-attr]
    if got and set(got) not in ({"patched", "tag"}, {"sync"}, {"base", "sync"}):
        refuse(f"refspec set {sorted(got)}")
    if "patched" in got and not plan["atomic"]:
        refuse("a branch and tag push that is not atomic")

    target_sha = repo.sha(f"refs/tags/{target}")
    if "base" in got and got["base"] != target_sha:
        refuse(f"base/{target} would not be the bare upstream tag")
    if "tag" in got and repo.sha(f"refs/tags/{got['tag']}") != got["patched"]:
        refuse(f"{got['tag']} does not point at patched/{target}")

    branch_form = re.compile(rf"patched/(v\d+\.\d+\.\d+)|{P}(?:base|upstream-sync)/(v\d+\.\d+\.\d+)")
    known_ids, known_subjects = set(), set()
    for base_v, branch in plan.get("sources", []):
        m = branch_form.fullmatch(branch)
        if not m or (m.group(1) or m.group(2)) != base_v:
            refuse(f"source {branch!r} on {base_v!r}")
        repo.fetch_tag(base_v)
        ref = repo.fetch_branch(branch)
        for c in repo.commits(f"{base_v}..{ref}"):
            known_subjects.add(c.subject)
            pid = patch_id(repo, c.sha)
            if pid:
                known_ids.add(pid)

    tip = got.get("patched") or got.get("sync")
    if tip:
        for c in repo.commits(f"{target_sha}..{tip}"):
            if not c.files:
                ok = c.subject in known_subjects
            elif c.files == ["Cargo.lock"]:
                ok = c.subject == LOCK_REGEN_SUBJECT.format(v=target)
            elif c.files == [f"{MARKER_DIR}{target}.md"]:
                ok = c.subject == MARKER_SUBJECT.format(v=target) and "sync" in got
            else:
                ok = patch_id(repo, c.sha) in known_ids
            if not ok:
                refuse(f"commit {c.sha[:12]} {c.subject!r} is not a patch the fork already holds")

    if plan.get("carried"):
        branch, sha = plan["carried"]
        if not re.fullmatch(rf"{P}upstream-sync/v\d+\.\d+\.\d+", branch):
            refuse(f"carried {branch!r}")
        now = [line.split("\t")[0] for line in repo.git("ls-remote", "fork", f"refs/heads/{branch}").stdout.splitlines()]
        if now != [sha]:
            refuse(f"{branch} moved since prepare read it (someone pushed); rerun so their work is carried")
    if plan["pr"] and (plan["pr"]["base"], plan["pr"]["head"]) != (f"{prefix}base/{target}", f"{prefix}upstream-sync/{target}"):
        refuse(f"PR {plan['pr']['head']!r} -> {plan['pr']['base']!r}")
    for n, _ in plan["close"] + plan["comment"]:
        head = run(["gh", "pr", "view", str(int(n)), "-R", fork, "--json", "headRefName", "-q", ".headRefName"]).stdout.strip()
        if not re.fullmatch(rf"{P}upstream-sync/v\d+\.\d+\.\d+", head):
            refuse(f"PR #{n} (head {head!r}) is not an upstream-sync PR")


def publish(out: str, plan: dict) -> int:
    fork = os.environ.get("GITHUB_REPOSITORY") or plan["fork"]
    prefix = os.environ.get("L1_PREFIX", "")
    token = os.environ.get("GOOSE_SYNC_TOKEN", "")
    if not token and prefix:
        # A rehearsal may push with GITHUB_TOKEN; that works only while it adds no new
        # workflow blob, which a stack re-applied onto its own base does not.
        token = os.environ.get("L1_FALLBACK_TOKEN", "")
    if not token and os.environ.get("GITHUB_ACTIONS"):
        raise SystemExit(
            "::error::GOOSE_SYNC_TOKEN is not set. GITHUB_TOKEN cannot push refs that add workflow files, "
            "which every upstream release does; see the header of .github/workflows/upstream-sync.yml."
        )
    gh_env = dict(os.environ, **({"GH_TOKEN": token} if token else {}))

    def gh(*args: str) -> str:
        return run(["gh", *args], env=gh_env).stdout

    repo = Repo(os.environ.get("L1_PUBLISH_DIR") or tempfile.mkdtemp(prefix="upstream-sync-publish-"), fork, token)
    repo.init(os.environ.get("L1_REFERENCE"))
    if plan.get("target") and parse_version(plan["target"]):
        repo.fetch_tag(plan["target"])
    bundle = os.path.join(out, "sync.bundle")
    if plan["refspecs"] and os.path.exists(bundle):
        repo.git("fetch", "-q", bundle, "+refs/l1/*:refs/l1/*")
        for r in plan["refspecs"]:
            if r.startswith("refs/tags/"):
                t = r.split(":", 1)[0].removeprefix("refs/tags/")
                repo.git("fetch", "-q", bundle, f"+refs/tags/{t}:refs/tags/{t}")
    validate(plan, repo, prefix, fork)
    if plan["refspecs"]:
        repo.push(plan["refspecs"], plan["atomic"])
    pr_url = ""
    if plan["pr"]:
        pr = plan["pr"]
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as bf:
            bf.write(pr["body"])
        pr_url = gh("pr", "create", "-R", fork, "--draft", "--base", pr["base"], "--head", pr["head"],
                    "--title", pr["title"], "--body-file", bf.name).strip()
        log(f"opened {pr_url}")
        if plan["label_major"]:
            gh("label", "create", "goose-major", "-R", fork, "--force", "--color", "5319e7",
               "--description", "upstream major-version upgrade")
            gh("pr", "edit", pr_url, "-R", fork, "--add-label", "goose-major")
    for n, body in plan["comment"]:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as bf:
            bf.write(body)
        gh("pr", "comment", str(int(n)), "-R", fork, "--body-file", bf.name)
    for n, comment in plan["close"]:
        gh("pr", "close", str(int(n)), "-R", fork, "--comment", comment.replace("{pr_url}", pr_url))
    write_summary(plan["summary"] + (f": {pr_url}" if pr_url else ""))
    if plan["fail"]:
        raise SystemExit("::error::" + plan["fail"])
    return 0


def has_work(plan: dict) -> bool:
    return bool(plan["refspecs"] or plan["pr"] or plan["comment"] or plan["close"] or plan["fail"])


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "local"
    out = argv[2] if len(argv) > 2 else tempfile.mkdtemp(prefix="upstream-sync-out-")
    os.makedirs(out, exist_ok=True)
    if mode in ("prepare", "local"):
        plan = prepare(out)
        with open(os.path.join(out, "plan.json"), "w") as f:
            json.dump(plan, f, indent=1)
        work = has_work(plan)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"work={'true' if work else 'false'}\n")
        if not work:
            write_summary(plan["summary"])
        if mode == "prepare" or not work:
            return 0
    with open(os.path.join(out, "plan.json")) as f:
        plan = json.load(f)
    return publish(out, plan)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
