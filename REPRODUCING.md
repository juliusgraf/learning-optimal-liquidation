# Replication scope and commands

The current paper uses the completed replacement v20 bundle: 440 main runs plus
two 40-run follow-ups. Ten master seeds per cell are **42, 7, 99, 123, 2024, 314,
577, 811, 1618, 2718**. There are 240 main synthetic runs (six schemes × four
learners × ten seeds), 200 historical runs (five tickers × four learners × ten
seeds), and 80 synthetic follow-up runs. No seed is excluded.

## Original source, current source and release identity

| Material | Original identity | Interpretation |
|---|---|---|
| All 320 replacement synthetic runs | `ffb314dd736a750e026861284e437aafa1e935ee` | 0.25-minute internal mesh; new forecast fit, normalization, training and selection |
| All 200 retained historical runs | `354656645a7e035be718e03cbf100664d1a87262` | Original historical bytes retained, not relabeled as refined-source runs |
| Preparation starting checkout | `64df3bc6421fe0e7d267386831b699482beb2706` | Manuscript changes since the synthetic run revision; executable/configuration sources matched that revision before preparation |
| Historical hosted preparation baseline | `02a8f00524b63b288e7e6ae813b55b9d195fb368` | Includes release preparation and the revised regression test; hosted Linux validation passed for this exact commit |
| Paper software version | `v0.1.0` | Annotated tag for version 0.1.0; resolve its full commit with `git rev-parse 'v0.1.0^{commit}'`. Later changes receive a new tag; the old hosted baseline does not verify this version |

All 520 current run records have clean original SHA identifiers. The earlier
pre-replacement main campaign included 138 dirty identifiers and an author
attestation; that older disclosure remains in the retained-history evidence.
It is not proof of a captured historical dirty snapshot. Original manifests are
untouched. The current code differs from the historical revision principally by
rough-Heston refinement and campaign/reporting support; historical reset/pool
routines were compared directly when recovering date assignments.

`release/evidence/campaign.json` extracts existing records, normalizes repository
path prefixes, and retains original file hashes. This preparation record is not
a contemporaneous campaign log. Full source history is available locally;
which reviewed history will be made public remains an owner decision.

## Three execution levels

1. **Synthetic smoke**: `python scripts/smoke_release.py`. Two fixed synthetic
   CPU episodes with identical seed, finite-state/accounting and terminal checks.
   No training, data access, deserialization or uploads. Test fixtures under
   `tests/fixtures/` are deliberately synthetic/development evidence, not paper inputs.
2. **Saved-result analysis**: `python scripts/release_support.py --check` verifies
   48 paired learner/reference intervals and 30 ordinary-shortfall levels using
   the bundled unrounded seed means. `--output /tmp/lmm-support-table` regenerates
   a compact CSV in a new external directory. This is verification level B.
3. **Full training/evaluation**: explicit opt-in, using original configuration,
   seed, input and source records. Requires original historical input archives
   and much more computation. It was not run during release preparation.

Verification levels: **A** traced to existing output; **B** figure/table
regenerated from saved numerical results; **C** analysis rerun from supplied
inputs; **D** complete experiment independently rerun. A smoke test is never D.
See the result-provenance table in [release/READINESS.md](release/READINESS.md).

## Environment and tests

The pinned installation in README requires Python 3.11+ (the recorded NumPy,
pandas, SciPy and Matplotlib versions require it). `pyproject.toml` remains the
canonical package specification; no research dependency was upgraded during
preparation. `release/paper-environment.txt` records the nine direct runtime
versions from every v20 run, **not** today's environment inferred as historical.
It is not a complete transitive lock. Paper runtime: Python 3.14.4, macOS 26.6.2,
arm64, one native numerical/Torch thread per worker; production launcher used
10 workers. The owner identified the campaign machine during submission
preparation as this MacBook Pro: Apple M5 Pro, 15 CPU cores, 24 GiB unified memory
(model Mac17,9). Those hardware details were read from the machine, not the
original run metadata. Allow 5–7 hours for 520 runs at ten workers; see
[the measured timings and estimate](release/evidence/runtime_estimate.json).
A complete transitive environment was not captured in the original records.

```sh
pytest -q tests/test_revision_acceptance.py
pytest -q -m 'not network and not slow and not market_data'
python scripts/smoke_release.py
python scripts/release_support.py --check
python scripts/recover_historical_dates.py
python scripts/release_validate.py --run-checks
```

Tests are CPU bounded and offline. Historical integration tests are opt-in:
`pytest --require-market-data -q -m 'market_data and not network and not slow'`.
Never replace missing or invalid data with a fixture to make a paper run pass.

The original local preparation checks used modified/staged source and the existing
dependency installation. That clean staging copy contained no ignored data/results
or model weights; package-index access was unavailable for a fresh local install.
Hosted CI subsequently installed the constrained CPU dependencies successfully on
Ubuntu 24.04 / Python 3.11.16 and passed validation and wheel construction at
`02a8f00524b63b288e7e6ae813b55b9d195fb368`; see the commit-specific
[verification record](release/verification.json).
The former `test_legacy_golden_episode`
compared serialized floating-point values with one runtime's fixed hash, which
failed on Linux/Python 3.11. The test now compares the current simulator exactly
against the pre-refinement source frozen in
[tests/fixtures/pre_refinement](tests/fixtures/pre_refinement/README.md), using
the same interpreter/dependencies in an isolated subprocess. All 71 observations,
rewards, times and H values must match without rounding or numerical tolerance;
the final RNG state must also match. Source hashes are checked before execution.
Local checks reproduced the original capture and showed exact current/reference
agreement in two runtimes whose output hashes differ. Python's floating-point
`sum` change explains some variation. The revised check has now passed on hosted
Linux at the baseline above, establishing exact current-versus-frozen-source
agreement within that runtime. The earlier serialized-hash failure is superseded
as a release gate: cross-runtime hash identity and a complete explanation of the
old hash difference are not required by this regression criterion. No simulator
or RNG code was changed to repair the test. This CI result does not verify later
commits or establish full-experiment reproduction.
Torch enables deterministic algorithms with `warn_only=True`; bitwise equality
across hardware, Python/BLAS versions or accelerator kernels is not guaranteed.
Wall-clock timings are not reproducibility targets.

## Data, dates and random streams (P-002)

[data/README.md](data/README.md) documents Alpaca SIP access, August 2026 coverage,
13:30–16:00 America/New_York sessions, latest-valid quote-at-or-before sampling,
60-second maximum quote age, crossed/nonpositive/zero-size rejection, no future
fill, and fail-closed incomplete sessions. The CSV has UTC timestamps and five
price columns, 151 rows per session; raw prices are rebased within the environment
to S0=100 and rounded half-up to a 0.01 model tick. The row-120 midquote is frozen
through the auction. The schema-3 quote-size label erratum is preserved; current
schema-4 preprocessing is not claimed to be the original producer revision.

`release/evidence/data.json` identifies the original CSV, sidecar, twenty raw
archive hashes, date pools, filtering metadata and access limit. No price rows
or vendor quote-size/spread summaries are included there. The original CSV and
raw archive checks passed locally. New downloads need not be byte-identical.
The original preprocessing commit was not saved in the sidecar.

`campaign.json` preserves each run's complete recorded seeds, including the 128
validation paths, 100 test paths, 16 normalization paths and actually executed
training episodes. The normalizer uses training data and a dedicated policy
stream; it also fits the inventory reference from 12 of its 16 paths. Existing
selection records bind the chosen checkpoint and eligibility rules.

`seed_everything` sorts the full component set: `as_calibration`, `env_eval`,
`env_final_eval`, `env_train`, `exploration`, `normalizer_env`, `normalizer_policy`,
`replay_auction`, `replay_clob`. NumPy SeedSequence(master_seed) spawns one child
per sorted component, then its first child initializes that component's generator.
Adding a component would change some streams. Final tests use
SeedSequence([master_seed, 19001, 791923]); episode integers are drawn using
`draw_seed`. AS uses its own component, not the final-evaluation stream.
All recorded training/validation/test/normalization sets were disjoint within
each of the 520 runs. This is not a proof of statistical independence of finite
paths; reference conditioning intentionally shares the normalization paths.

`historical_dates.json` **recovers** mappings during preparation from recorded
seeds, original CLOB tape draw order, then the next pool-index draw. Training and
normalization use configured ticker order × sorted training dates; validation
and test use the run ticker and their separate date pools. All recorded training
and learned-policy test opening prices were checked against the recovered dates.
Normalization/validation date mappings are recovered, not contemporaneous logs.
Forecast-fit records include all 512 recorded train/validation episode seeds per
setting and recovered historical fit dates; these fit seeds do not overlap the
recorded learning/evaluation/normalization seeds. Development-visible test
dates remain a reused holdout, not new dates or independent exchange backtests.

## Saved analysis and calibration (P-001, P-003, P-004, P-005)

Point estimates average episodes within seed, then weight all ten seed means
equally. Historical macro summaries average five tickers within seed first.
Paired comparisons retain episode/env-seed matching before seed aggregation;
10,000 percentile resamples of the ten paired seed differences use PCG64 seed 0
and sorted values, as the original report implementation does. Ordinary shortfall
negates saved PnL and reverses interval endpoints. Intervals are pointwise,
without multiplicity correction; 48 positive intervals do not imply simultaneous
95% coverage. The P-003 checker uses no rounded manuscript values.

**AS calibration evidence limit (P-004).** The actual AS coefficients used in the
paper are preserved in
[release/evidence/as_calibration.json](release/evidence/as_calibration.json)
for all 520 evaluations. A and k are contemporaneous outputs; K_hat = k/gamma_m
is explicitly a derived quantity. Accepted/excluded calibration counts and
original fit samples are missing from the supplied records; their absence
prevents a sample-level audit of the original fit. Missing counts are represented
as `null`, not zero, in the machine-readable records.

The origin-constrained least-squares fit was configured to attempt 5,000 simulated
books/orders per seed, excluding emptied books and unchanged best quotes. This
configured attempt count does not establish the number of accepted samples.
No fresh fit was substituted for the missing evidence. The coefficient is fitted
to simulated order flow, not historical price data; k and K_hat have inverse
model-price units. P-004 remains partially resolved.

`mesh_diagnostic.json` preserves the existing 32-path, fixed-DQN comparison of
legacy, 1, 0.5 and 0.25-minute grids with common Brownian drivers and coupled
auction proposals. It concerns the **pre-replacement** policy hash recorded in
the diagnostic, not the current retrained checkpoint at a reused path. It reports
residual/nonmonotone mesh sensitivity and negative variance frequencies; it does
not establish finite-grid accuracy or broad policy robustness. No new paper-supporting diagnostic
was executed during this preparation (tests still exercise bounded diagnostics).

Full figure/report regeneration needs local original runs and their provenance:
`python -m lmm.experiments.make_report --root results/revision_v20 --seeds 42 7 99 123 2024 314 577 811 1618 2718 --publication`.
Inspect source-attestation/retained-history gates before use; they intentionally
reject incompatible source or artifacts. Main plots and tables must read saved
numbers without stepping an environment. Manuscript sources and their assembly
scripts are maintained separately and are not bundled in this branch. See the
[exhibit-to-input mapping](README.md#paper-exhibits-and-their-sources) for the
retained numerical evidence; `scripts/release_support.py --check` verifies the
paired benchmark table directly from bundled seed means.

## Full experiment opt-in and preserved originals

Use a separate, authorized checkout of the original experiment revision and a
new output root; inspect each launcher before running. Do not run the
`--replace-synthetic` operation against the preserved completed study: it is an
archiving/retraining workflow, not a harmless regeneration command.

The original launch and settings are identified in
`docs/rough_heston_refinement/rerun_v20.md`, the completion-bound configurations,
and `release/evidence/campaign.json`. That document is a historical launch recipe;
its preparation-time checks predate the completed campaign records. The full replacement was invoked as:
`scripts/run_multiseed.sh --replace-synthetic --jobs 10 --threads-per-job 1`.
It depended on the already completed v20 baseline, retained-history inventory
and authorized local input archives. This is an original recipe, not a portable
one-command reconstruction from the compact public bundle.

Single-run primitives are `python -m lmm.experiments.train --config CONFIG
--run-name NAME --seed SEED`, followed by `python -m lmm.experiments.evaluate
--run-dir RUN --n-episodes 100`, and paired-analysis/reporting entry points.
Use exact extracted configurations and original source identities, rather than
default overlays that still preserve legacy generator/forecast settings.
Committing the extracted metadata does not recover unavailable original artifacts
or justify labeling a new run an exact replication. Full independent experiment
replication (level D) remains unperformed.
