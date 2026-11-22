import unittest
from datetime import datetime, timedelta, timezone

import upstream_sync as us

NOW = datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc)


def rel(tag, hours_ago, prerelease=False, draft=False):
    return {
        "tag_name": tag,
        "published_at": (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prerelease": prerelease,
        "draft": draft,
    }


class VersionTest(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(us.parse_version("v1.60.2"), (1, 60, 2))
        self.assertEqual(us.parse_version("1.60.2"), (1, 60, 2))
        self.assertIsNone(us.parse_version("v1.60.2-rc1"))
        self.assertIsNone(us.parse_version("patched-1.60.2.0"))

    def test_parse_patched_tag(self):
        self.assertEqual(us.parse_patched_tag("patched-1.60.2.0"), ((1, 60, 2), 0))
        self.assertEqual(us.parse_patched_tag("patched-1.54.0.18"), ((1, 54, 0), 18))
        self.assertEqual(us.parse_patched_tag("test-l1-patched-1.54.0.3", "test-l1-"), ((1, 54, 0), 3))
        self.assertIsNone(us.parse_patched_tag("darwin-backfill-patched-1.54.0.15"))
        self.assertIsNone(us.parse_patched_tag("patched-1.54.0"))


class PickTargetTest(unittest.TestCase):
    def test_newest_release_older_than_debounce_wins(self):
        rels = [rel("v1.60.2", 58), rel("v1.60.1", 110), rel("v1.60.0", 112)]
        self.assertEqual(us.pick_target(rels, NOW), "v1.60.2")

    def test_too_young_newest_waits_and_quickly_superseded_ones_are_skipped(self):
        # 1.60.0 was latest for 2h, 1.60.1 for 45h, 1.60.2 is 10h old: nothing
        # was latest for 48h since 1.59.2, so 1.59.2 is the target.
        rels = [rel("v1.60.2", 10), rel("v1.60.1", 55), rel("v1.60.0", 57), rel("v1.59.2", 150)]
        self.assertEqual(us.pick_target(rels, NOW), "v1.59.2")

    def test_backport_on_an_older_line_is_never_latest(self):
        # v1.58.1 published after v1.59.0 but is lower: it never was latest, and
        # it does not end v1.59.0's time as latest.
        rels = [rel("v1.58.1", 100), rel("v1.59.0", 120), rel("v1.58.0", 200)]
        self.assertEqual(us.pick_target(rels, NOW), "v1.59.0")

    def test_prereleases_drafts_and_odd_tags_are_ignored(self):
        rels = [
            rel("v1.61.0", 100, prerelease=True),
            rel("v1.62.0", 100, draft=True),
            rel("v1.63.0-rc1", 100),
            rel("v1.60.2", 60),
        ]
        self.assertEqual(us.pick_target(rels, NOW), "v1.60.2")

    def test_nothing_old_enough(self):
        self.assertIsNone(us.pick_target([rel("v1.60.2", 5)], NOW))


class PickBaseTest(unittest.TestCase):
    FULL = list(us.REQUIRED_ASSETS) + ["goose-aarch64-unknown-linux-gnu.tar.bz2"]

    def test_newest_complete_release_wins(self):
        rels = [
            {"tag_name": "patched-1.60.2.0", "draft": False, "assets": us.REQUIRED_ASSETS[:1]},
            {"tag_name": "patched-1.54.0.18", "draft": False, "assets": self.FULL},
            {"tag_name": "patched-1.54.0.9", "draft": False, "assets": self.FULL},
            {"tag_name": "darwin-backfill-patched-1.54.0.15", "draft": False, "assets": self.FULL},
        ]
        self.assertEqual(us.pick_base(rels), "patched-1.54.0.18")

    def test_numeric_not_lexical_order(self):
        rels = [
            {"tag_name": "patched-1.54.0.9", "draft": False, "assets": self.FULL},
            {"tag_name": "patched-1.54.0.18", "draft": False, "assets": self.FULL},
            {"tag_name": "patched-1.9.0.30", "draft": False, "assets": self.FULL},
        ]
        self.assertEqual(us.pick_base(rels), "patched-1.54.0.18")

    def test_draft_is_not_complete(self):
        rels = [
            {"tag_name": "patched-1.60.2.0", "draft": True, "assets": self.FULL},
            {"tag_name": "patched-1.54.0.18", "draft": False, "assets": self.FULL},
        ]
        self.assertEqual(us.pick_base(rels), "patched-1.54.0.18")

    def test_none_complete(self):
        self.assertIsNone(us.pick_base([]))


class StackFilterTest(unittest.TestCase):
    def test_lock_only_and_marker_only_commits_are_dropped(self):
        commits = [
            us.Commit("a", "patch a", ["crates/goose/src/a.rs", "Cargo.lock"]),
            us.Commit("b", "Cargo.lock: regenerate for v1.60.2 (fork watcher)", ["Cargo.lock"]),
            us.Commit("c", "fork-sync: v1.60.2 report", ["fork-sync/v1.60.2.md"]),
            us.Commit("d", "patch d", ["crates/goose/src/d.rs"]),
        ]
        self.assertEqual([c.sha for c in us.filter_stack(commits)], ["a", "d"])

    def test_a_squashed_marker_and_code_commit_is_refused(self):
        with self.assertRaises(ValueError):
            us.filter_stack([us.Commit("s", "squash", ["fork-sync/v1.60.2.md", "crates/goose/src/a.rs"])])

    def test_owed_names_original_patches_missing_by_subject_without_reapplying(self):
        original = [us.Commit("1", "p1", ["x"]), us.Commit("2", "p2", ["y"]), us.Commit("3", "p3", ["z"])]
        carried = [us.Commit("1b", "p1", ["x"]), us.Commit("w", "port p2 by hand", ["y"]), us.Commit("3b", "p3", ["z"])]
        self.assertEqual([c.sha for c in us.owed(carried, original)], ["2"])


class DecideTest(unittest.TestCase):
    def state(self, **kw):
        base = dict(
            hold="",
            target="v1.61.0",
            base_tag="patched-1.60.2.0",
            target_tags=[],
            sync_prs=[],
            base_branch_ahead=False,
        )
        base.update(kw)
        return us.State(**base)

    def test_hold_wins(self):
        self.assertEqual(us.decide(self.state(hold="1.60")).kind, "held")

    def test_no_target_or_no_base(self):
        self.assertEqual(us.decide(self.state(target=None)).kind, "no-target")
        self.assertEqual(us.decide(self.state(base_tag=None)).kind, "no-base")

    def test_up_to_date(self):
        self.assertEqual(us.decide(self.state(target="v1.60.2")).kind, "up-to-date")
        self.assertEqual(us.decide(self.state(target="v1.59.0")).kind, "up-to-date")

    def test_allow_same_rehearses_onto_the_base_version(self):
        self.assertEqual(us.decide(self.state(target="v1.60.2", allow_same=True)).kind, "fresh")
        self.assertEqual(us.decide(self.state(target="v1.59.0", allow_same=True)).kind, "up-to-date")

    def test_target_already_tagged(self):
        self.assertEqual(us.decide(self.state(target_tags=["patched-1.61.0.0"])).kind, "tagged")

    def test_open_pr_for_target_is_the_workers(self):
        prs = [us.SyncPR(7, "v1.61.0", "OPEN")]
        self.assertEqual(us.decide(self.state(sync_prs=prs)).kind, "worker-owns")

    def test_worker_merged_into_base_branch(self):
        prs = [us.SyncPR(7, "v1.61.0", "OPEN")]
        d = us.decide(self.state(sync_prs=prs, base_branch_ahead=True))
        self.assertEqual((d.kind, d.close_prs), ("from-base-branch", [7]))

    def test_closed_unmerged_pr_is_a_human_decision(self):
        prs = [us.SyncPR(7, "v1.61.0", "CLOSED")]
        self.assertEqual(us.decide(self.state(sync_prs=prs)).kind, "closed-by-human")

    def test_fresh(self):
        d = us.decide(self.state())
        self.assertEqual((d.kind, d.carry_from, d.close_prs), ("fresh", None, []))

    def test_newer_release_supersedes_the_stuck_one(self):
        prs = [us.SyncPR(5, "v1.60.5", "CLOSED"), us.SyncPR(7, "v1.60.9", "OPEN"), us.SyncPR(6, "v1.60.7", "OPEN")]
        d = us.decide(self.state(sync_prs=prs))
        self.assertEqual((d.kind, d.carry_from, d.close_prs), ("fresh", "v1.60.9", [7, 6]))

    def test_a_finished_older_port_goes_before_a_newer_target(self):
        prs = [us.SyncPR(7, "v1.60.9", "MERGED"), us.SyncPR(8, "v1.60.9", "OPEN")]
        d = us.decide(self.state(pending_base="v1.60.9", sync_prs=prs))
        self.assertEqual((d.kind, d.target, d.close_prs), ("from-base-branch", "v1.60.9", [8]))

    def test_untagged_target_branch_is_left_to_whoever_made_it(self):
        self.assertEqual(us.decide(self.state(target_branch_exists=True)).kind, "branch-exists")
        # a tag wins: the branch is then simply the released one
        self.assertEqual(us.decide(self.state(target_branch_exists=True, target_tags=["patched-1.61.0.0"])).kind, "tagged")

    def test_fresh_and_base_branch_decisions_carry_the_target(self):
        self.assertEqual(us.decide(self.state()).target, "v1.61.0")
        self.assertEqual(us.decide(self.state(base_branch_ahead=True)).target, "v1.61.0")

    def test_major(self):
        self.assertTrue(us.decide(self.state(target="v2.0.0")).major)
        self.assertFalse(us.decide(self.state()).major)


class ReportTest(unittest.TestCase):
    def test_report_names_every_failing_patch(self):
        results = [
            us.PatchResult(us.Commit("aaaaaaaaaaaa", "p1", ["x"]), "applied", [], False),
            us.PatchResult(us.Commit("bbbbbbbbbbbb", "p2", ["y"]), "conflict", ["y", "z"], True),
            us.PatchResult(us.Commit("cccccccccccc", "p3", ["w"]), "empty", [], False),
        ]
        body = us.render_report("v1.61.0", "v1.60.2", results, "log line\n", major=False, carried_from=None)
        self.assertIn("1 of 3 patches failed", body)
        self.assertIn("| 2 | `bbbbbbbbbbbb` | p2 | **conflict** | `y`, `z` | yes |", body)
        self.assertIn("empty", body)
        self.assertIn("docs/fork/UPGRADING.md", body)
        self.assertIn("log line", body)

    def test_report_lists_still_owed_patches(self):
        body = us.render_report("v1.61.0", "v1.60.2", [], "", major=False, carried_from="v1.60.9",
                                still_owed=[us.Commit("dddddddddddd", "p4", ["q"])])
        self.assertIn("- `dddddddddddd` p4", body)

    def test_test_failure_report(self):
        body = us.render_report("v1.61.0", "v1.60.2", [], "boom\n", major=True, carried_from="v1.60.9", tests_failed=True)
        self.assertIn("patched test filter failed", body)
        self.assertIn("major", body)
        self.assertIn("v1.60.9", body)


class ValidateTest(unittest.TestCase):
    """publish's guard against a plan rewritten by the candidate's build scripts."""

    def setUp(self):
        import subprocess, tempfile
        self.dir = tempfile.mkdtemp()
        g = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "tag.gpgsign=false", "-c", "commit.gpgsign=false", *a], cwd=self.dir,
                                      check=True, capture_output=True, text=True).stdout.strip()
        self.g = g
        g("init", "-q")
        def commit(path, text, msg):
            with open(f"{self.dir}/{path}", "w") as f:
                f.write(text)
            g("add", path)
            g("commit", "-qm", msg)
            return g("rev-parse", "HEAD")
        commit("up.txt", "a\n", "upstream a")
        g("tag", "v0.9.0")
        commit("fork.txt", "patch\n", "patch one")
        g("update-ref", "refs/remotes/fork/patched/v0.9.0", "HEAD")
        g("checkout", "-q", "--detach", "v0.9.0")
        commit("up.txt", "a\nb\n", "upstream b")
        g("tag", "v1.0.0")
        g("-c", "user.name=t", "cherry-pick", "refs/remotes/fork/patched/v0.9.0")
        self.tip = g("rev-parse", "HEAD")
        g("tag", "-a", "patched-1.0.0.0", "-m", "t", self.tip)
        self.repo = us.Repo(self.dir, "johnnymo87/goose")
        self.repo.fetch_tag = lambda v: f"refs/tags/{v}"
        self.repo.fetch_branch = lambda b: f"refs/remotes/fork/{b}"
        self.plan = {"fork": "johnnymo87/goose", "prefix": "", "target": "v1.0.0",
                     "sources": [["v0.9.0", "patched/v0.9.0"]], "carried": None, "atomic": True,
                     "refspecs": [f"{self.tip}:refs/heads/patched/v1.0.0", "refs/tags/patched-1.0.0.0:refs/tags/patched-1.0.0.0"],
                     "pr": None, "close": [], "comment": []}

    def check(self, **kw):
        us.validate(dict(self.plan, **kw), self.repo, "", "johnnymo87/goose")

    def test_a_rebuilt_stack_passes(self):
        self.check()

    def test_any_other_ref_is_refused(self):
        with self.assertRaises(SystemExit):
            self.check(refspecs=self.plan["refspecs"] + [f"+{self.tip}:refs/heads/fork-main"])
        with self.assertRaises(SystemExit):
            self.check(refspecs=[f"{self.tip}:refs/heads/patched/v1.0.0"])
        with self.assertRaises(SystemExit):
            self.check(atomic=False)
        with self.assertRaises(SystemExit):
            self.check(prefix="test-l1/")

    def test_a_commit_the_fork_does_not_hold_is_refused(self):
        with open(f"{self.dir}/evil.yml", "w") as f:
            f.write("x\n")
        self.g("add", "evil.yml")
        self.g("commit", "-qm", "patch one")
        evil = self.g("rev-parse", "HEAD")
        self.g("tag", "-f", "-a", "patched-1.0.0.0", "-m", "t", evil)
        with self.assertRaises(SystemExit):
            self.check(refspecs=[f"{evil}:refs/heads/patched/v1.0.0", "refs/tags/patched-1.0.0.0:refs/tags/patched-1.0.0.0"])

    def test_a_tag_off_the_branch_is_refused(self):
        self.g("tag", "-f", "-a", "patched-1.0.0.0", "-m", "t", "v1.0.0")
        with self.assertRaises(SystemExit):
            self.check()


class ScrubTest(unittest.TestCase):
    def test_candidate_code_sees_no_credential(self):
        import os
        from unittest import mock
        env = {"PATH": "/bin", "HOME": "/h", "GOOSE_SYNC_TOKEN": "x", "L1_FALLBACK_TOKEN": "x", "GH_TOKEN": "x",
               "GITHUB_TOKEN": "x", "ACTIONS_RUNTIME_TOKEN": "x", "ACTIONS_ID_TOKEN_REQUEST_URL": "x", "GH_ENTERPRISE": "x"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(sorted(us.scrubbed_env()), ["HOME", "PATH"])


if __name__ == "__main__":
    unittest.main()
