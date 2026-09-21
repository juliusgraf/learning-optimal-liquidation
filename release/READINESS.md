# Submission readiness

The branch supplies software, configurations and compact numerical evidence for
the 520-run v20 study. Local technical validation passes. It is not yet a
published, anonymous or fully independently reproduced submission artifact.
The anonymous.4open.science mirror has not been created, as confirmed by the
owner during this review. The manuscript is maintained separately.

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
`02a8f00524b63b288e7e6ae813b55b9d195fb368`. Preserve that distinction: the current
cleanup remains uncommitted and has not run in hosted CI. Earlier preparation
checks and hosted results are recorded in [verification.json](verification.json).

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

## Remaining submission steps

1. Commit the reviewed candidate and verify hosted CI for that exact commit.
2. Create the anonymous mirror and inspect its actual reviewer-visible files and
   links. Current author/copyright/citation metadata, source URLs, validator
   literals and original diagnostic paths are identifying. This local repository
   has not been certified anonymous. Keep copyright/licensing records intact in
   the source; verify the review copy separately.
3. Check whether mirror transformations change file bytes. The release inventory,
   frozen-source regression and standard-license check bind exact bytes; an
   automatically redacted download may fail those checks. Validate the actual
   downloadable review artifact rather than assuming a passing source checkout
   proves the mirror works. Do not disable integrity checks to hide a mismatch.
4. Align the separate manuscript's availability statement with the compact scope.
   The journal expects code/data/instructions sufficient to reproduce results;
   requests for non-public-data exemptions go in the cover letter to the Area
   Editor. No exemption or manuscript compliance is established by this audit.
   See the official [code and data policy](https://pubsonline.informs.org/page/opre/code-and-data-disclosure-policy)
   and [submission guidelines](https://pubsonline.informs.org/page/opre/submission-guidelines).

The existing release blockers remain open: Git history contains materials outside
the reviewed snapshot; full runs/checkpoints lack a verified public location;
and the final submission identity/access have not been established. These limit
publication and full-replication claims, even when draft validation exits zero.
No remote upload, visibility change or history rewrite was performed.
