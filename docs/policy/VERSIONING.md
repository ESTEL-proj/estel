# Versioning

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. -->

**Within one value of `STATISTICS_REFERENCE`, statistics and test results are unchanged up to the tolerances of `tests/tolerances.toml`.**

## Package version: MAJOR.MINOR.PATCH

| Bump | When | Who decides |
|---|---|---|
| MAJOR | a breaking interface change (removed or renamed function or argument, changed default); a new geometry. Whether a change of statistics values also forces a major is decided case by case with the maintainers | CODEOWNERS |
| MINOR | new features, including opt-in step-2 features and new engine configurations; a significant benchmark gain (to begin with, a factor 2 in speed or peak memory), including making an engine the default for a configuration | automatic; humans set the `bench` label for benchmark-driven minors |
| PATCH | bug fixes that leave reference values unchanged; refactors; docs | automatic |

## Two reference counters

- `estel.STATISTICS_REFERENCE`: an integer, never reset, bumped whenever any statistic value changes. Same number, same statistics, whatever the package version.
- `estel.SYNTHESIS_REFERENCE`: the same for reference synthesis trajectories, qualified by backend and pinned dependencies.

Both are recorded in every saved result (run manifest), shown in the release notes, and checked by CI. Bumping either needs CODEOWNER approval.

## Computing the required change (`tools/version_impact.py`, tier 0)

- reference tests outside tolerance, or a changed `tests/reference/` → bump a reference counter, with CODEOWNER approval;
- `api/manifest.json`: entry removed or changed → major; entry added → minor;
- geometry registry changed → major;
- `bench` label → minor.

Each PR adds `changes/<PR>.<major|minor|patch>.md` (towncrier). CI fails if the declared bump is below the required one.

## Before 1.0.0

Versions are 0.x and nothing is gated by reference values. 1.0.0 is the end of step 1: parity with STL, references signed off by the three CODEOWNERS, `STATISTICS_REFERENCE = 1`.
