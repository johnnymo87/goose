# johnnymo87/goose: fork-main

This branch holds the fork's own machinery. It contains no goose code.

- `docs/fork/UPGRADING.md` is the brief for the worker who moves the fork's patch stack onto
  a new upstream release.
- `.github/workflows/upstream-sync.yml` is the upstream-release watcher (workstation bead
  workstation-6w7i.27.4, L1); its logic is `.github/fork/upstream_sync.py`. GitHub runs
  scheduled workflows only from the default branch, which is why this branch is the repo
  default.

The patched goose itself is on `patched/v<upstream-version>`, tagged
`patched-<version>.<n>`.
