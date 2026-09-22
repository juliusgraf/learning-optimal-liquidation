# Paper software release status

**Status updated 2026-09-21.** This is the public GitHub repository accompanying
the open/arXiv version of *Learning Optimal Liquidation with Closing Auctions*.
It supplies software, configurations and compact numerical evidence for the
520-run v20 study. Version **0.1.0** is preserved by the annotated Git tag
`v0.1.0`; later corrections receive a new tag. The manuscript is maintained
separately.

The private-repository observations and distribution decisions recorded in the
2026-09-08 audit are historical snapshots, superseded by this public distribution
scope. They are not current restrictions to availability upon request. Local
technical checks pass; remote tag publication and hosted CI for its exact commit
must be verified separately. Full independent reproduction is not claimed.

## Current scope

The inclusion list is [content-manifest.json](content-manifest.json). It contains
255 reviewed files plus the manifest itself, including 320 synthetic and 200
historical run records. The 52 experimental cells each contain the same ten
master seeds. Bundled input hashes, source revisions, runtime references and
0.25-minute synthetic refinement settings are consistent with those records.

`paper/`, `CLAUDE.md`, top-level `audit/`, `legacy/`, legacy-only tests and
`docs/*_v19` history are excluded. Original manuscript paths and hashes retained
in [provenance.json](provenance.json) are marked `excluded`; they are historical
identities, not available files. The frozen pre-refinement test implementation
is retained, with its docstring-only cleanup explicitly recorded.

Raw historical data, complete run trees and checkpoints are not bundled.
Neither the wheel nor the software source distribution is the compact research
bundle; obtain the repository snapshot for configs, tests, release checks and
evidence. Software uses MIT; non-code assets have the distinct scope described
in [the notices](../THIRD_PARTY_NOTICES.md).

## Local verification

The submission cleanup passed `release_validate.py --run-checks` from a clean
staging copy containing only the reviewed files, with no Git history, ignored
market data or local run directories:

- Offline CPU suite: **746 passed, 4 skipped, 28 deselected**.
- Synthetic smoke: two finite, identical episodes reach terminal settlement.
- Saved-result verification: **48 paired intervals and 30 ordinary-shortfall
  levels** agree with the supplied unrounded seed means.
- Historical-date recovery: **160,100 recorded seed/date assignments** agree.
- Wheel and software source-distribution builds pass. Their inventories exclude
  the removed directories, restricted data and model checkpoints.
- The separately installed wheel passes all eight CLI help checks and synthetic
  smoke outside the source checkout; its plotting style resource is included.
- All nine installed direct dependencies match the recorded constraints, and
  `pip check` reports no broken requirements.
- Release checksums, bundled JSON, Python compilation, Bash syntax, Markdown
  file links and section anchors pass the consistency review.

These checks use the existing local dependency environment. They do not establish
a fresh installation on every supported platform or a rerun of the full campaign.
The pinned install requires **Python 3.11+**; Python 3.10 in the package metadata
allows other compatible dependency versions, not the recorded pins.

The earlier hosted Ubuntu/Python 3.11 success applies only to commit
`02a8f00524b63b288e7e6ae813b55b9d195fb368`. Preserve that distinction: hosted CI
for the `v0.1.0` snapshot has not been verified here.
Earlier preparation checks and hosted results remain historical records in
[verification.json](verification.json).

## Result-provenance scope

| Result | Available evidence | Verification / remaining need |
|---|---|---|
| Economic performance and ordinary shortfalls | Seed means and saved aggregates | Benchmark/shortfall support checker regenerates 48 paired intervals and 30 market-policy levels; other original manuscript assembly scripts are excluded |
| Forecast accuracy | Per-seed summaries and fitted protocols | Saved outputs and hash bindings; complete forecast records are not bundled |
| Main treatments, cash-flow and preference follow-ups | Per-seed contrasts and aggregate tables | Saved outputs and hash bindings; full report regeneration needs original run trees |
| Learning and auction decomposition | Validation curves and economic seed summaries | Saved numerical inputs; figure regeneration needs full report inputs |
| AS calibration | Coefficients used in all 520 evaluations | Accepted/excluded counts and original fit samples were not saved |
| Mesh diagnostic | Original 32-path fixed-policy diagnostic | Pre-replacement policy; no finite-grid accuracy guarantee or retrained-policy comparison |

See [the exhibit map](../README.md#paper-exhibits-and-their-sources),
[replication guide](../REPRODUCING.md) and the result records in
[provenance.json](provenance.json). No full independent experiment rerun
(verification level D) is claimed. Historical test dates were visible during
development and are a reused holdout.

## Version preservation and verification

Use `git rev-parse 'v0.1.0^{commit}'` to resolve the exact version. The annotated
tag binds the reviewed source and its checksum manifest; it does not establish
that the full experiment has been independently reproduced. Keep it fixed after
publication and create a new version for later edits.

Publishing the commit and tag to GitHub, verifying hosted CI for that commit, and
checking the downloaded tagged archive are separate from local validation. The
manuscript's availability statement should cite the version and accurately state
the compact bundle's scope. Original run trees/checkpoints and restricted market
inputs are not supplied; the default launchers alone do not rebuild all 520
refined paper runs. See the [replication guide](../REPRODUCING.md).

The unresolved historical-distribution and full-artifact-availability findings
remain recorded in [provenance.json](provenance.json). Their inclusion does not
mean the public code is available only on request, nor does public code access
make unbundled inputs or checkpoints available. No history rewrite is part of
this update. The publication runbook is [RUNBOOK.md](RUNBOOK.md).
