# johnnymo87/goose: fork-main

This branch holds the fork's own machinery. It contains no goose code.

- `docs/fork/UPGRADING.md` is the brief for the worker who moves the fork's patch stack onto
  a new upstream release.
- The upstream-release watcher workflow will live in `.github/workflows/` here (workstation
  bead workstation-6w7i.27.4, L1). GitHub runs scheduled workflows only from the default
  branch, so L1 will also make this branch the repo default.

The patched goose itself is on `patched/v<upstream-version>`, tagged
`patched-<version>.<n>`.
