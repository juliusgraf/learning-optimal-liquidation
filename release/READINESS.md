# Release readiness and coverage — 2026-09-08 UTC

**Public-release preparation: BLOCKED. MIT scope: VERIFIED FOR IDENTIFIED
MATERIAL. Replication-bundle support: PARTIALLY SUPPORTED. Actual publication:
NOT PERFORMED.** These are separate assessments.

The original working tree was clean at
`64df3bc6421fe0e7d267386831b699482beb2706`. The preparation and subsequent CI
repair are committed in `02a8f00524b63b288e7e6ae813b55b9d195fb368`, the clean HEAD
at the start of this documentation review. The successful baseline CI does not
cover later documentation changes or future commits. No manuscript, simulator,
reward, selection, RNG or evaluation semantics were edited;
no result was replaced. The current source matches the synthetic experiment's
executable/configuration sources before adding release-only tooling.

The author explicitly confirmed the two named holders and 2026 MIT authority
for first-party software/documentation, and distribution permission for manuscript,
figures, derived historical-market summaries and checkpoints. Raw vendor-price
redistribution was not confirmed. No separate non-code license was selected.

## Current release gates

| Item | Status | Basis |
|---|---|---|
| `linux-portability` | RESOLVED | Hosted Ubuntu 24.04 / Python 3.11.16 validation passed at `02a8f00524b63b288e7e6ae813b55b9d195fb368`; exact current/frozen-source agreement within that runtime, not universal bitwise equality. |
| `history-distribution` | OPEN | Restricted vendor data remain in reachable history; no sanitized history or verified safe publication strategy is recorded. |
| `artifact-availability` | OPEN | Full run directories/checkpoints lack a stable public locator; any package supplied upon request also needs an explicit inventory and verification. The compact bundle does not establish exact full-experiment reproduction. |
| `as-sample-accounting` | DOCUMENTED LIMITATION | Actual A/k coefficients are preserved; accepted/excluded counts and original fit samples remain missing. Reclassified from a blocker after disclosure, not resolved by recovery. |
| `owner-candidate-review` | OWNER ACTION | The owner approved the current documentation update and private retention. Approval of any future public-release candidate remains a separate step; no final paper release tag/identifier is assigned. |

The owner explicitly approved the readiness-documentation update and confirmed
that the repository will remain private because `history-distribution` and
`artifact-availability` remain unresolved. The intended provision of source,
configurations and saved numerical summaries upon request is unchanged. This
approval does not resolve the gates for a future public release or demonstrate
availability of a full run/checkpoint package.

## What the evidence establishes

All **520** current run completion manifests matched their **15,800** bound files,
without deserializing any checkpoint. This is integrity/provenance checking, not
independent re-execution. All **200** retained historical directory hashes and
forecast fit bytes matched their original retained-history inventory. All current
run identifiers are clean original SHAs: 320 synthetic at `ffb314d…`, 200
historical at `3546566…`; full identities are in the machine-readable manifest.

The current paper's input manifests and seven figure-to-report hash bindings
matched existing bytes. No manuscript generator was run against the manuscript.
All training/validation/test/normalization seed sets were disjoint within each
run. Original input CSV/raw archive digests passed validation; recovered dates
matched **131,300** saved training and learned-policy test opening marks.
The recovered mapping is new metadata, not an invented contemporaneous record.

| Audit finding | Status | Evidence and limit |
|---|---|---|
| P-001 | PARTIALLY RESOLVED | `evidence/campaign.json`, `forecast_fits.json`, `provenance.json`, bundled numerical outputs and recorded environment identify configurations, 520 runs, selection, hashes and source revisions. Full run files, weights and safe public original-source availability remain unresolved. |
| P-002 | PARTIALLY RESOLVED | `evidence/data.json`, `historical_dates.json`, per-run complete recorded seeds and original source draw order. Data hashes and 131,300 opening-price checks passed; original preprocessing revision not recorded, vendor input remains restricted. Forecast fits preserve 512 recorded episode seeds per setting, separate protocol metadata and recovered historical fit dates. |
| P-003 | RESOLVED | `scripts/release_support.py --check` regenerates all 48 paired reference intervals and 30 ordinary-shortfall levels from bundled unrounded seed means, matching the existing supplement. Equal seed weighting and original 10,000-resample pointwise procedure preserved. Level B. |
| P-004 | PARTIALLY RESOLVED | `evidence/as_calibration.json`: saved A/k from every actual evaluation, source hashes, configured 5,000 attempts, component seed derivation and fitting scope. K_hat explicitly derived from saved k/gamma_m. Accepted/excluded counts and original fit samples not saved; no new fit substituted. |
| P-005 | RESOLVED | `evidence/mesh_diagnostic.json` preserves the existing 32-path common-driver diagnostic with grids, seeds, per-episode quantities and measured differences. Level A. Supports residual sensitivity only, not finite-grid accuracy or retrained-policy robustness. |

| Central paper result | Supplied numerical/output evidence | Verification |
|---|---|---|
| Table `tab:economic_performance`; figure `fig:economic_shortfall` | `evidence/revision_v20/audit/economic_by_seed.csv`, `shortfall_summary.csv`, existing economic figure | A |
| Table `tab:benchmark_supplement` | `evidence/benchmark_supplement.csv` and paired seed means; regenerated by release_support | B |
| Table `tab:forecast_accuracy` | `forecast_summary.csv`, `forecast_by_algorithm_seed.csv`, `forecast_fits.json`; original forecast input hashes retained | A |
| Table `tab:treatments`; figure `fig:forecast_information` | `revision_v20/tables/treatments.csv`, main treatment seed records, both follow-up comparisons | A |
| Figure `fig:auction_value_inventory` | `revision_v20/tables/auction_mechanism.csv` and economic seed decomposition | A |
| Figures `fig:learning_guidance`, `fig:headline_learning` | validation/credit seed curves, follow-up seed means, seven original figure hashes | A |
| AS reference and mesh sentence | `as_calibration.json`, `mesh_diagnostic.json` | A, with limits above |

Full identifiers, SHA-256 values, inputs, script entry points, availability and
missing dependencies appear in `provenance.json`. Original exact shell argv were
not universally retained; documented entry points must not be read as fabricated
historical command logs. Recorded configurations are the stronger executable
specification. No level D verification was performed.

## Historical preparation checks and their limits

The records in this section predate the successful hosted baseline run below.
They retain their original tested scope and outcomes, including superseded failures.

Commands were inspected before execution. Original weights were hashed only.
Private inspection/extraction scripts and command logs are outside this release;
public validation is repeatable with the new release tools.

- `.venv/bin/python -m pip check`: exit 0, no broken requirements.
- `.venv/bin/python -m pytest -q tests/test_revision_acceptance.py`: exit 0,
  26 passed on the original clean source.
- `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/lmm-release-audit/mpl
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LMM_TORCH_INTRAOP_THREADS=1
  LMM_TORCH_INTEROP_THREADS=1 .venv/bin/python -m pytest -q -m
  'not network and not slow and not market_data'`: exit 0, 736 passed,
  34 deselected, 14 expected fixture/legacy warnings, 199.35 seconds.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_release_tools.py`:
  exit 0, 15 passed (checksum tampering, path/symlink safety, finalization gates, seed-matrix refusal,
  and actual P-003 regeneration).
- `PYTHONPATH=src .venv/bin/python scripts/smoke_release.py`: exit 0; two
  identical finite synthetic CPU episodes reach terminal settlement, no training.
- `PYTHONPATH=src .venv/bin/python scripts/release_support.py --check`: exit 0;
  48 paired intervals and 30 ordinary-shortfall levels match supplied unrounded data.
- CFF validated against the official versioned CFF 1.2.0 JSON Schema using a
  locally available `jsonschema 4.17.3` Draft7Validator under Python 3.10;
  schema fetched read-only, no private material transmitted. Focused CFF/license
  checks also run in the offline release validator. No release date/version/DOI
  was invented. Standard MIT text was preserved exactly.

Run-specific dependency records uniformly identify Python 3.14.4, NumPy 2.4.6,
pandas 3.0.3, Torch 2.12.0, SB3 2.7.1, Gymnasium 1.2.3, SciPy 1.17.1,
Matplotlib 3.11.0, PyYAML 6.0.3 and certifi 2026.5.20. These direct versions
are saved in `paper-environment.txt`; a complete transitive historical lock and
hardware specification are not available.

The first complete archive-copy test run exposed three tests that assumed an
enclosing `.git` directory (743 passed, three failed, four skipped). Those tests
now use explicit test source identities; production provenance gates are unchanged.
The final staged `python scripts/release_validate.py --run-checks` exited 0:
**746 passed, four skipped, 34 deselected**, 14 fixture/legacy warnings, 216.65
seconds, followed by successful synthetic smoke, P-003 regeneration and 160,100
recorded seed/date checks. A subsequently added checksum-tamper test passed in the
15-test focused release suite; no production script changed after the staged run.
The four skips reflect separately obtained historical data, not replaced inputs.
Wheel and sdist builds exited 0. Their inspected inventories contain the MIT
license and third-party notices, with no vendor data, manuscript, checkpoints or
run directories. The wheel was installed into a separate target and passed import,
CLI-parser and synthetic smoke checks. Software packages are not the compact
replication archive.

Clean staging/package verification is recorded in `verification.json`. It reuses
the local dependency installation with staged source/wheel imports. A fresh
network installation was attempted only for a small citation validator dependency;
package-index access was unavailable. No global dependency was installed or upgraded.
The eventual final paper-release candidate must be tested at its own clean commit.
The later hosted baseline run below verifies installation on its stated target;
it does not replace the original experimental environment records.

### Subsequent CI regression repair — historical, hosted follow-up now complete

The earlier author-supplied Linux/Python 3.11.16 log reported 745 passed, five skipped,
34 deselected and one failure in the old fixed-hash legacy episode test.
The test now compares the current simulator exactly with the frozen synthetic
simulator dependency closure from `881f6aa9ffcd7ce3928a5c68ad5b71e9129ec231`,
under the same interpreter and installed dependencies. The fixture's source and
configuration hashes are checked before execution; no Git/network access is
required. All original trajectory fields and the final RNG state are compared
without rounding or tolerance. A one-ULP mutation is rejected.

The original source reproduces the old capture hash on Python 3.14.4 / NumPy
2.4.6 / macOS arm64. Current and frozen source also agree exactly on Python
3.10.6 / NumPy 1.26.4 / macOS arm64, with a different hash. That second environment
is a diagnostic outside the project's declared NumPy dependency range. Python's
floating-point sum change explains some variation, but the entire old Linux-hash
cause was not isolated. At that repair stage, hosted verification was pending;
the successful baseline run below supersedes that pending status and closes the
portability blocker. The regression criterion compares current and frozen source
within one runtime; differing hashes across runtimes do not require further
investigation as a release gate. No scientific implementation changed.
See [fixture provenance](../tests/fixtures/pre_refinement/README.md).

For this repair, `python scripts/release_validate.py --run-checks` passed locally:
**753 passed, 34 deselected**, 14 existing fixture/legacy warnings, 213.41 seconds,
followed by the synthetic smoke, saved-result and date-assignment checks. The
focused refinement/acceptance selection passed all 51 tests. The three legacy
reference checks also passed in a clean staging copy without Git history or
ignored local artifacts. These historical local results remain in
`verification.json`, separately from the subsequent hosted verification.

## Successful hosted baseline validation

[Tests run #14, attempt 1](https://github.com/juliusgraf/learning-market-making/actions/runs/34193586813)
completed successfully for `02a8f00524b63b288e7e6ae813b55b9d195fb368` on
2026-09-08. Authenticated read-only GitHub run/job metadata and the job log were
checked during this review. Job `101956457673` completed at 06:19:26 UTC on
hosted Ubuntu **24.04.4**, Python **3.11.16**. Constrained CPU dependency installation
passed, followed by:

- `python scripts/release_validate.py --run-checks`: **748 passed, 5 skipped,
  34 deselected**, 14 fixture/legacy warnings; synthetic smoke PASS; **48 paired
  intervals and 30 ordinary-shortfall levels** PASS; **160,100 recorded seed/date
  assignments** PASS.
- `python -m pip wheel . --no-deps --wheel-dir dist`: PASS.

This supersedes the earlier failed Linux checks (the initial inspected result was
730 passed, 16 skipped, 23 deselected, one failure; the subsequent supplied log
was 745 passed, five skipped, 34 deselected, one failure). Hosted Linux validation
is complete and passing for the identified commit. The regression establishes
exact current-versus-frozen-source agreement in the same runtime, including final
RNG state. It does **not** establish bitwise serialized-trajectory equality across
Python/library versions, platforms or hardware, or level D experiment reproduction.
Later documentation/manuscript commits require separate local and hosted checks.

## Retained manuscript versus author-reported corrections

`paper/main.tex` remains unchanged at its existing hash. Its AS calibration
paragraph (line 1028 in the baseline) describes 5,000 attempted samples and the
discard rules, but omits the disclosure that the actual coefficients are preserved
while accepted/excluded counts and original fit samples are missing. It also lacks
the author-reported Code and Data Availability statement covering source code,
experiment configurations and saved numerical summaries upon request, with
historical Alpaca inputs obtained separately. It still refers to the replication
bundle in the experimental protocol (baseline line 766).

The latest author-approved manuscript text is not present in the inspected
manuscript sources. The owner will import it later using the procedure in
[RUNBOOK.md](RUNBOOK.md#later-author-approved-manuscript-update). This known
mismatch is not a prerequisite for completing this documentation-only update;
the retained manuscript is not represented as synchronized.

## Coverage ledger and unresolved exposure

Unless explicitly updated by the hosted baseline verification above, the counts
and scan results below describe the original preparation audit, not a new
exhaustive history/security scan of the later HEAD.

| Surface | Coverage | Limit / action |
|---|---|---|
| Current source and artifacts | 303 initial tracked files; code/config/tests/legacy, manuscript sources, figures, local saved runs and metadata inventoried | Full raw runs remain local; no broad refactor or deletion |
| Local Git | Non-shallow; 87 reachable commits, six local branches, remote-tracking refs, stash; no local tags/submodules/LFS pointers found | Remote-only/unreachable objects not covered; no fetch or history rewrite |
| Security scan | 2,117 reachable blob objects plus 15 expanded archive members across history/current files; tracked files and extracted bundle; values kept out of reports | Twenty large vendor archives exceeded expansion bound; ignored full trees/caches and binary visual content not comprehensively scanned |
| Credential findings | Credential-like matches reviewed as synthetic tests; no live secret identified in bounded inspected scope | Heuristics are not an unconditional absence claim; no credentials tested or rotated |
| Local paths / logs | Machine paths in old reports, manuscript build files and metadata identified; new bundle copies normalize root prefixes | Original evidence retained; tracked build debris and old metadata require reviewed archival/removal before final exposure |
| PDF / binaries | Current manuscript PDF metadata inspected; original model files integrity-hashed | Historical compressed PDF/image content and model internals not semantically audited; no untrusted deserialization |
| Hosted repository | Original audit: private, releases list empty, nine Actions runs enumerated with empty first artifact pages. Baseline run metadata reconfirms private status | No public availability established; original artifact/log coverage is not an exhaustive current hosted-surface audit |
| Hosted CI | Tests run #14 at `02a8f00524b63b288e7e6ae813b55b9d195fb368`: 748 passed, 5 skipped, 34 deselected; smoke, saved-result/date checks and wheel PASS | Earlier failures superseded; exact same-runtime reference criterion, no cross-runtime bitwise guarantee; later commits need new checks |
| Publication | None performed; preparation committed in the baseline | Final candidate approval, tag, release assets, visibility and signed-out access require owner action |

**Do not change repository visibility yet.** Vendor raw/processed historical data
exist in reachable Git history. The author's permission for derived summaries
and checkpoints is not permission to redistribute the vendor input archives.
A source archive allowlist, `.gitignore` or MIT license cannot solve this.
No private scan values or specific sensitive historical locations are reproduced
here; owner-only details are retained outside the intended public package.

For any future public release, resolve distribution/history scope and promised
artifact availability, retain the disclosed AS evidence limitation, review the exact
inclusion list, commit the reviewed final candidate, validate that clean SHA/tag, publish only
with separate authorization, and verify signed-out access. The unfinished
manuscript template and draft release notes are in [RUNBOOK.md](RUNBOOK.md).
