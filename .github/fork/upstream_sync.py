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
    """Drop commits the watcher regenerates: Cargo.lock-only commits and fork-sync reports."""

    def regenerated(path: str) -> bool:
        return path == "Cargo.lock" or path.startswith(MARKER_DIR)

    return [c for c in commits if not c.files or not all(regenerated(f) for f in c.files)]


def carry_forward(carried: list[Commit], original: list[Commit]) -> list[Commit]:
    """A superseded upstream-sync branch, plus the original patches it does not hold yet."""
    have = {c.subject for c in carried}
    return carried + [c for c in original if c.subject not in have]


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


@dataclass
class Decision:
    kind: str
    reason: str = ""
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
    if s.target_tags:
        return Decision("tagged", f"{', '.join(s.target_tags)} already exist", major=major)
    for_target = [p for p in s.sync_prs if p.version == s.target]
    open_target = [p.number for p in for_target if p.state == "OPEN"]
    if s.base_branch_ahead:
        return Decision("from-base-branch", f"base/{s.target} holds the stack", close_prs=open_target, major=major)
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
    out += ["", "<details><summary>log tail</summary>", "", "```", log_tail.rstrip()[-20000:], "```", "</details>", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- side effects


LOG: list[str] = []


def log(msg: str) -> None:
    LOG.append(msg)
    print(msg, flush=True)


def run(argv: list[str], cwd: Optional[str] = None, check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    p = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {p.returncode}: {p.stderr.strip()[-2000:]}")
    return p


class Repo:
    def __init__(self, workdir: str, fork: str, prefix: str, push_token: str):
        self.dir = workdir
        self.fork = fork
        self.prefix = prefix
        self.tag_prefix = prefix.replace("/", "-")
        self.push_token = push_token

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return run(["git", *args], cwd=self.dir, check=check)

    def gh(self, *args: str, write: bool = False) -> str:
        env = dict(os.environ)
        if write and self.push_token:
            env["GH_TOKEN"] = self.push_token
        return run(["gh", *args], env=env).stdout

    def init(self, reference: Optional[str]) -> None:
        self.git("init", "-q")
        if reference:
            with open(os.path.join(self.dir, ".git/objects/info/alternates"), "w") as f:
                f.write(os.path.join(reference, "objects") + "\n")
        self.git("remote", "add", "upstream", f"https://github.com/{UPSTREAM}.git")
        url = f"https://github.com/{self.fork}.git"
        if self.push_token and os.environ.get("GITHUB_ACTIONS"):
            url = f"https://x-access-token:{self.push_token}@github.com/{self.fork}.git"
        self.git("remote", "add", "fork", url)
        if os.environ.get("GITHUB_ACTIONS"):
            self.git("config", "user.name", "github-actions[bot]")
            self.git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")

    def fetch_tag(self, remote: str, tag: str) -> None:
        self.git("fetch", "-q", "--no-tags", remote, f"+refs/tags/{tag}:refs/tags/{tag}")

    def fetch_branch(self, branch: str) -> str:
        self.git("fetch", "-q", "--no-tags", "fork", f"+refs/heads/{branch}:refs/remotes/fork/{branch}")
        return f"refs/remotes/fork/{branch}"

    def remote_refs(self, pattern: str) -> list[str]:
        out = self.git("ls-remote", "fork", pattern).stdout
        refs = [line.split("\t", 1)[1] for line in out.splitlines() if "\t" in line]
        return [r for r in refs if not r.endswith("^{}")]

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
            if status == "applied":
                if not self.git("diff", "--quiet", before, "HEAD", check=False).returncode:
                    self.git("reset", "-q", "--hard", before)
                    status = "empty"
            r = PatchResult(c, status, conflicts, self.conflicts_with(c, target))
            log(f"  {status:8} {c.sha[:12]} {c.subject}" + (f"  [{', '.join(conflicts)}]" if conflicts else ""))
            results.append(r)
        return results

    def regenerate_lock(self, target: str, required: bool) -> None:
        try:
            p = run(["cargo", "metadata", "--format-version", "1"], cwd=self.dir, check=False)
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

    def push(self, src: str, ref: str, force: bool = False) -> None:
        self.git("push", "-q", *(["--force"] if force else []), "fork", f"{src}:{ref}")
        log(f"pushed {ref}")


def run_patched_tests(workdir: str) -> tuple[bool, str]:
    """Run the run-steps of the tree's own patched-test.yml, in order, as CI would."""
    try:
        import yaml  # type: ignore
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])
        import yaml  # type: ignore
    with open(os.path.join(workdir, TEST_WORKFLOW)) as f:
        wf = yaml.safe_load(f)
    steps = wf["jobs"]["test-patches"]["steps"]
    env = dict(os.environ, GOOSE_PATH_ROOT=tempfile.mkdtemp(prefix="goose-path-root-"))
    tail: list[str] = []
    for step in steps:
        if "uses" in step:
            continue
        extra = set(step) - {"name", "run"}
        if extra:
            return False, f"{TEST_WORKFLOW} step {step.get('name')!r} uses {sorted(extra)}, which the watcher cannot run"
        log(f"== {step.get('name')}")
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as script:
            script.write(step["run"])
        p = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", script.name],
            cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert p.stdout
        for line in p.stdout:
            print(line, end="")
            tail = (tail + [line])[-150:]
        if p.wait() != 0:
            return False, "".join(tail)
    return True, "".join(tail)


def write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


def main() -> int:
    fork = os.environ.get("GITHUB_REPOSITORY") or "johnnymo87/goose"
    prefix = os.environ.get("L1_PREFIX", "")
    if prefix and not prefix.endswith("/"):
        raise SystemExit("L1_PREFIX must end with '/' (e.g. test-l1/)")
    push_token = os.environ.get("GOOSE_SYNC_TOKEN", "")
    if not push_token and prefix:
        # A rehearsal may push with GITHUB_TOKEN; that only works while it adds no new
        # workflow blob, which a stack re-applied onto its own base does not.
        push_token = os.environ.get("L1_FALLBACK_TOKEN", "")
    in_actions = bool(os.environ.get("GITHUB_ACTIONS"))
    plan_only = os.environ.get("L1_PLAN_ONLY") == "1"
    skip_tests = os.environ.get("L1_SKIP_TESTS") == "1"
    workdir = os.environ.get("L1_WORKDIR") or tempfile.mkdtemp(prefix="upstream-sync-")
    repo = Repo(workdir, fork, prefix, push_token)
    tp = repo.tag_prefix

    now = datetime.now(timezone.utc)
    up_releases = json.loads(run(["gh", "api", f"repos/{UPSTREAM}/releases?per_page=100"]).stdout)
    fork_releases = json.loads(run(["gh", "api", f"repos/{fork}/releases?per_page=100"]).stdout)
    target = os.environ.get("L1_TARGET") or pick_target(up_releases, now)
    base_tag = os.environ.get("L1_BASE") or pick_base(fork_releases)
    log(f"target={target} base={base_tag} prefix={prefix!r}")

    repo.init(os.environ.get("L1_REFERENCE"))
    target_tags, base_ahead, sync_prs = [], False, []
    if target and parse_version(target):
        target_tags = [r.rsplit("/", 1)[1] for r in repo.remote_refs(f"refs/tags/{tp}patched-{target[1:]}.*")]
        raw = json.loads(run(["gh", "pr", "list", "-R", fork, "--state", "all", "--limit", "200",
                              "--json", "number,headRefName,state"]).stdout)
        sync_head = re.compile(re.escape(prefix) + r"upstream-sync/(v\d+\.\d+\.\d+)")
        for pr in raw:
            m = sync_head.fullmatch(pr["headRefName"])
            if m:
                sync_prs.append(SyncPR(pr["number"], m.group(1), pr["state"]))
        if repo.remote_refs(f"refs/heads/{prefix}base/{target}"):
            repo.fetch_tag("upstream", target)
            b = repo.fetch_branch(f"{prefix}base/{target}")
            base_ahead = int(repo.git("rev-list", "--count", f"{target}..{b}").stdout) > 0

    # L1_ALLOW_SAME re-applies a stack onto its own base: the only way to rehearse the clean
    # path (tests, push, tag) when upstream has nothing newer. Test prefixes only.
    allow_same = bool(prefix) and os.environ.get("L1_ALLOW_SAME") == "1"
    d = decide(State(os.environ.get("GOOSE_HOLD", ""), target, base_tag, target_tags, sync_prs, base_ahead, allow_same))
    log(f"decision: {d.kind} {d.reason}".rstrip())
    if d.kind not in ("fresh", "from-base-branch") or plan_only:
        write_summary(f"upstream-sync: **{d.kind}** {d.reason} (target {target}, base {base_tag})")
        return 0
    if not push_token and in_actions:
        write_summary("upstream-sync: work to do but the GOOSE_SYNC_TOKEN secret is missing")
        raise SystemExit(
            "::error::GOOSE_SYNC_TOKEN is not set. GITHUB_TOKEN cannot push refs that add workflow files, "
            "which every upstream release does; see the header of .github/workflows/upstream-sync.yml."
        )

    assert target and base_tag
    old_v = vstr(parse_patched_tag(base_tag)[0])  # type: ignore[index]
    repo.fetch_tag("upstream", target)
    repo.fetch_tag("upstream", old_v)
    stack_ref = repo.fetch_branch(f"patched/{old_v}")
    original = filter_stack(repo.commits(f"{old_v}..{stack_ref}"))
    if d.kind == "from-base-branch":
        stack = filter_stack(repo.commits(f"{target}..refs/remotes/fork/{prefix}base/{target}"))
    elif d.carry_from:
        repo.fetch_tag("upstream", d.carry_from)
        carried = repo.fetch_branch(f"{prefix}upstream-sync/{d.carry_from}")
        stack = carry_forward(filter_stack(repo.commits(f"{d.carry_from}..{carried}")), original)
    else:
        stack = original
    log(f"applying {len(stack)} commits onto {target}")
    repo.git("checkout", "-q", "--detach", target)
    results = repo.apply(stack, target)
    failed = [r for r in results if r.status in ("conflict", "error")]

    tests_failed, test_tail = False, ""
    if not failed:
        repo.regenerate_lock(target, required=not skip_tests)
        if skip_tests:
            log("tests skipped (L1_SKIP_TESTS=1)")
        else:
            ok, test_tail = run_patched_tests(workdir)
            tests_failed = not ok
            log("patched test filter: " + ("PASS" if ok else "FAIL"))

    if not failed and not tests_failed:
        branch, tag = f"{prefix}patched/{target}", f"{tp}patched-{target[1:]}.0"
        repo.push("HEAD", f"refs/heads/{branch}")
        repo.git("tag", tag, "HEAD")
        repo.push(f"refs/tags/{tag}", f"refs/tags/{tag}")
        for n in d.close_prs:
            repo.gh("pr", "close", str(n), "-R", fork, "--comment", f"Superseded: {target} applied cleanly as {tag}.", write=True)
        write_summary(f"upstream-sync: **clean** {target} -> {branch}, tagged {tag}")
        return 0

    if d.kind == "from-base-branch":
        body = render_report(target, old_v, results, "\n".join(LOG) + "\n" + test_tail, d.major, None, tests_failed)
        for n in d.close_prs:
            repo.gh("pr", "comment", str(n), "-R", fork, "--body", body, write=True)
        raise SystemExit(f"::error::base/{target} does not pass: see the comment on the upstream-sync PR")

    base_branch, sync_branch = f"{prefix}base/{target}", f"{prefix}upstream-sync/{target}"
    if not repo.remote_refs(f"refs/heads/{base_branch}"):
        repo.push(f"refs/tags/{target}^{{commit}}", f"refs/heads/{base_branch}")
    if failed:
        repo.regenerate_lock(target, required=False)
    body = render_report(target, old_v, results, "\n".join(LOG) + "\n" + test_tail, d.major, d.carry_from, tests_failed)
    os.makedirs(os.path.join(workdir, MARKER_DIR), exist_ok=True)
    marker = os.path.join(MARKER_DIR, f"{target}.md")
    with open(os.path.join(workdir, marker), "w") as f:
        f.write(body)
    repo.git("add", marker)
    repo.git("commit", "-q", "-m", f"fork-sync: {target} upstream-sync report (fork watcher)")
    repo.push("HEAD", f"refs/heads/{sync_branch}", force=True)
    title = f"upstream-sync: {target} ({'tests failed' if tests_failed else f'{len(failed)} of {len(results)} patches failed'})"
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as bf:
        bf.write(body)
    url = repo.gh("pr", "create", "-R", fork, "--draft", "--base", base_branch, "--head", sync_branch,
                  "--title", (f"[{prefix.rstrip('/')}] " if prefix else "") + title, "--body-file", bf.name, write=True).strip()
    log(f"opened {url}")
    if d.major:
        repo.gh("label", "create", "goose-major", "-R", fork, "--force", "--color", "5319e7",
                "--description", "upstream major-version upgrade", write=True)
        repo.gh("pr", "edit", url, "-R", fork, "--add-label", "goose-major", write=True)
    for n in d.close_prs:
        repo.gh("pr", "close", str(n), "-R", fork, "--comment", f"Superseded by {url}, which carries this branch forward.", write=True)
    write_summary(f"upstream-sync: **conflict** {title}: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
