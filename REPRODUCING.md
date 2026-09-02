# Reproducing the revised experiments

This repository implements the revised manuscript environment for the
synthetic rough-Heston setting and the historical-midprice setting, with DQN
and the DDPG/TD3/SAC continuous-action relaxations. Current artifacts use
schema 11, are written below `results/revision_v11/`, and carry environment
contract `unified-minute-auction-mdp-2026-09-02-v8`; checkpoints and results
from earlier contracts are rejected.

## Setup

Python 3.10 or newer is required. The reference configs use CPU Torch.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyproject.toml` is the canonical dependency specification. For tooling that
expects a requirements file, `python -m pip install -r requirements.txt` is
equivalent: that file delegates to the same editable project and development
extras instead of maintaining a second dependency list.

## Mandatory pre-run gate

Run the acceptance checklist before starting a long experiment:

```bash
pytest -q tests/test_revision_acceptance.py
pytest -q -m 'not network and not slow'
```

The checklist covers the realized grid and interval partition, leak-free
chronology, residual-book carry-over, transactional auction validity,
indicative-price lags, action counts and canonical no-ops, cross-phase Bellman
targets, one update per transition, common unclipped reward scaling, pro-rata
fills, inventory conservation, self-trade exclusion, shaped-training/economic-evaluation accounting,
annualized minute-clock rough-Heston time, the capped benchmark schedule, and rejection of
stale artifacts.

Do not infer publication readiness from the legacy characterization tests in
`tests/audit/`; they document the superseded implementation.

## Common physical clock

Both active settings use minutes. One integer grid interval is one minute,
`tau_op=120` opens the auction after a two-hour CLOB phase, and `tau_cl=150`
ends a 30-minute auction. CLOB Poisson intensities are therefore per minute and
auction Bernoulli probabilities are per one-minute decision. The synthetic
rough-Heston model remains parameterized in trading years, using
`s_star=252*6.5*60=98,280` trading minutes per year. The numerical grid has not
grown, so this change does not add environment steps; it changes the physical
meaning and the synthetic mid-price calendar-time increment.

## One shared simulator

The active setting overlays select only run identity and the exogenous
mid-price source. Both inherit the same grid, CLOB and auction flows, action
space, cancellation/order semantics, reward, validation design, and
algorithm-specific hyperparameters from `configs/base.yaml` and
`configs/algo/*.yaml`. In particular, both use `lambda0=1`, `V_inf=2`,
`rho_lob=0.96`, exogenous `L_max=200`, agent CLOB offset bound 12, and the
same 1,346-action cancellation-enabled DQN auction head. The acceptance suite
compares every shared resolved section for all four learners.

The manuscript notation is used directly in the active action configuration.
`actions.B_inf=150` is the absolute bound on the executed frozen-mid offset
`b`, so `b in [-B_inf,B_inf]` and
`S_t^a=S_{tau_op}^{mid}+alpha*b_t^a`. The same absolute support sets the
exogenous quote bounds `M1=-B_inf` and `M2=B_inf`; configuration validation
requires `actions.B_inf == auction_flow.B_inf`. Every learned policy uses the
same 21 local manuscript actions `ell in [-B_max,B_max]`, where
`actions.B_max=10`. The explicit `actions.auction_anchor` resolves them around
the lagged observable indicative price in H-on arms and around the frozen
auction-open midprice in H-off arms. The latter prevents the ablated signal
from leaking through action execution or admissibility. Thus `B_inf` is the
absolute market-coordinate bound and `B_max` is the local proposal bound; they
are neither aliases nor interchangeable settings.

## Historical data requirement

The repository includes `data/historical_sp500_midquotes_1m.csv` and its
schema-3 provenance sidecar. A clean checkout can therefore run the historical
experiment without a data account, network access, or a preprocessing step.
The environment verifies the processed-file digest, every tracked raw-archive
digest, source/feed declarations, and split contract before constructing a
historical path.

The artifact was built from timestamped bid/ask quotes (SIP for publication),
stores raw USD midpoints without normalization, and maps each decision time to
the latest valid quote at or before that time. The environment freezes the
auction-open midquote during the call and rebases a selected session to
`S0=100` only as an explicit model-coordinate transformation. Order flow,
books, auction proposals, clearing, and allocation remain simulated.

The optional credential-safe regeneration command and raw-event archive
contract are documented in `data/README.md`:

```bash
python3 -m lmm.data.load_midquote_data --help
```

Generation must be followed by the policy-free carry-over gate:

```bash
python3 -m lmm.experiments.diagnose_simulator \
  --config configs/base.yaml \
  --config configs/historical_sp500_midquotes.yaml \
  --episodes 20 --assert-ready
```

Run the same gate for the rough-Heston source:

```bash
python3 -m lmm.experiments.diagnose_simulator \
  --config configs/base.yaml \
  --config configs/synthetic_rough_heston.yaml \
  --episodes 100 --assert-ready
```

## Canonical runs

The matched confirmation budget is 800 episodes in both settings.
Each run fits its feature normalizer on training-only paths. A checkpoint can
enter the validation race only after both phase learners pass their configured
optimizer-update maturity thresholds. The headline DQN exposes the complete
auction grid from episode zero and uses the same masked epsilon-greedy schedule
in both phases, so all eligible auction updates count normally. Early-stopping
patience starts with the first eligible validation. The initial economic score is a
non-reportable safety floor; if no mature candidate beats it, the run records
selection failure instead of creating `best.pt`. Otherwise `best.pt` is
selected on validation risk-adjusted PnL and evaluated on 100 held-out test
episodes.

Normal training uses the manuscript-shaped reward with `q=1`, `lambda_inv=2`,
and exact cancellation clawback: canceling an order reverses the fictive
interim shaping credited when that order was submitted, in addition to the
unchanged fee with `d=0.1`. Validation, checkpoint selection, and final
learned/AS/TWAP comparisons all use the common economic-only reward. Centering
subtracts the same policy-invariant initial-inventory value in both contracts.

One synthetic run:

```bash
scripts/run_synthetic_dqn.sh --seed 42
scripts/run_synthetic_ddpg.sh --seed 42
scripts/run_synthetic_td3.sh --seed 42
scripts/run_synthetic_sac.sh --seed 42
```

One historical ticker:

```bash
scripts/run_historical_dqn.sh --seed 42 --symbol MSFT
```

All standard runs and output generation:

```bash
scripts/reproduce_all.sh --seed 42
```

This command is intentionally long. It runs four synthetic experiments and
four algorithms across the configured historical tickers.

For isolated development or CI runs, set `LMM_RESULTS_ROOT` to an absolute
scratch directory. Every launcher and report generator honors the override,
and training records the same path in `config_resolved.yaml`, so discovery and
provenance cannot disagree. Leave it unset for the canonical
`results/revision_v11/` layout. For example:

```bash
LMM_RESULTS_ROOT=/absolute/scratch/lmm-results \
  scripts/reproduce_all.sh --smoke --symbol MSFT --seed 9001
```

A short pipeline smoke is available and is not a training result:

```bash
scripts/reproduce_all.sh --smoke --symbol MSFT --seed 9001
```

The non-headline synthetic treatment runs—three additional H/anchor-bundle by shaping
arms, no auction, and no cancellation for all four learners—are launched with:

```bash
scripts/run_synthetic_treatments.sh --seed 42
# or exercise only the pipeline:
scripts/run_synthetic_treatments.sh --smoke --seed 9001
```

The fourth H/anchor-bundle by shaping cell (`H` on, shaping on) is exactly the canonical
synthetic headline configuration. The treatment launcher therefore does not
train it again: paired reporting reuses `synthetic_rough_heston/*_seed<N>` as
that arm. Run the canonical synthetic launchers (or `reproduce_all.sh`) for the
same seed before aggregating treatments.

The no-cancellation treatment gives DQN 673 auction actions; the enabled
regime has 1,346. Continuous agents use two or three normalized auction
proposal coordinates respectively while keeping the same 18-feature network
input.

The no-auction treatment removes both terminal auction participation and every
auction-derived training input: it disables auction shaping and the projected
clearing feature in addition to terminating at auction open. It is therefore a
comparison against a policy that does not anticipate the modeled auction.

The canonical five-seed publication command covers the headline, historical,
and complete synthetic treatment matrix, then generates all aggregates:

```bash
scripts/run_multiseed.sh --jobs 5 --threads-per-job 2
```

On a 15-core host, five workers with two numerical-library/Torch intra-op
threads and one Torch inter-op thread each leave some headroom. The launcher
also caps Apple Accelerate/vecLib and NumExpr, preventing hidden oversubscription;
the effective settings are saved in `runtime_versions.json`. Lower `--jobs` if memory is tighter. Workers write distinct run
directories. Shared per-run, combined, and multiseed output generation remains
serial and begins only after every worker finishes. A small development check
can use `--smoke --jobs 2`, which defaults to the deliberately nonpublication
seeds `9001 9002`; smoke artifacts are explicitly rejected by publication
mode. The launcher rejects canonical publication seed names in smoke mode so a
development artifact cannot block the later full run at the same path.

Full publication mode also requires a clean Git worktree: `git_sha.txt` records
the exact commit, not an uncommitted patch. The long launcher is safely
restartable across already finished runs. A run is skipped only when its saved
resolved config exactly matches the current command, its complete evaluation
uses `best.pt`, and `pipeline_complete.json` cryptographically binds the
resolved config, seed, Git SHA, runtime and split-seed provenance, training
metrics and feature normalizer, initial/best/final checkpoints, best-selection
sidecar, and the complete evaluation artifact tree (including metadata,
records, traces, diagnostics, and both paired-difference files) by SHA-256.
Any subsequent change makes the manifest invalid. An interrupted or
drifted run fails closed with its path. Move that partial directory aside for
diagnosis and rerun; automatic checkpoint resume is intentionally not used
because post-checkpoint metric/grid rows would also need transactional
truncation. Publication aggregation additionally exact-compares every resolved
run against the current checked-in base, setting, algorithm, and treatment
config stack. The sole accepted operational difference is the launcher's
disk-safe `checkpoint_interval_episodes=10000000`, which suppresses periodic
replay-heavy resume checkpoints without changing learning or selection.

To regenerate reports without training:

```bash
scripts/make_multiseed_outputs.sh --publication
```

That command also builds the cross-treatment table. To rebuild only that
table, use `scripts/make_treatment_outputs.sh --publication`.

Multiseed policy comparisons use confidence intervals over paired per-seed
policy-minus-benchmark differences. Historical cross-asset tables are emitted
both in currency units and in basis points of initial notional. The treatment
table reports six explicit within-algorithm contrasts: the H/anchor bundle at
each shaping level, shaping at each H/anchor level, the full auction-aware
headline treatment versus the bundled no-auction comparator, and cancellation
on minus off. Each evaluation episode is matched by environment
seed before its difference is reduced to a master-seed mean; positive values
favor the first-named condition. The summary `.tex`/`.csv` and auditable
per-seed provenance CSV are written under
`results/revision_v11/synthetic_rough_heston/_cross_treatment/tables/`.

Generated result artifacts are never copied into `paper/` automatically.
Promotion into the manuscript artifact directory remains an explicit manual
review step.

## Equivalent module commands

For a synthetic DQN run named `dqn_seed42`:

```bash
RUN=results/revision_v11/synthetic_rough_heston/dqn_seed42

python3 -m lmm.experiments.train \
  --config configs/base.yaml \
  --config configs/synthetic_rough_heston.yaml \
  --config configs/algo/dqn.yaml \
  --run-name dqn_seed42 --seed 42

python3 -m lmm.experiments.evaluate --run-dir "$RUN" --trace-episodes 1
python3 -m lmm.experiments.policy_differences --run-dir "$RUN" --benchmark as
python3 -m lmm.experiments.policy_differences --run-dir "$RUN" --benchmark twap
python3 -m lmm.experiments.make_figures --run-dir "$RUN"
python3 -m lmm.experiments.make_tables --run-dir "$RUN"
```

The installed command names are `lmm-train`, `lmm-evaluate`,
`lmm-policy-differences`, `lmm-make-figures`, and `lmm-make-tables`.
Fixed-policy cumulative differences are not called regret.

## Artifact map

Every completed run contains the following raw and provenance artifacts:

- `config_resolved.yaml`, `seed.txt`, `git_sha.txt`, and
  `runtime_versions.json`;
- `historical_data_manifest.json` for historical runs;
- `feature_normalizer.yaml`, `split_seeds.yaml`, and realized-grid records;
- `metrics.csv` and training forecast diagnostics;
- `checkpoints/{initial,best,final}.pt` plus optional resume checkpoints;
- `eval/records.csv`, `eval/metadata.yaml`, realized grids, forecast summaries,
  proposal/action/clearing diagnostics, and episode traces;
- `eval/policy_difference_{as,twap}.csv`.

The canonical multiseed launcher generates per-run and single-seed combined
figures/tables for its first requested seed only; producing duplicate anatomy
plots for all five seeds would add substantial runtime and disk use without
entering the publication summaries. It generates the cross-seed tables and
figures from all five seeds under each headline setting's `_multiseed/`
directory, and the paired treatment outputs under `_cross_treatment/`. Run
`scripts/make_all_outputs.sh --seed N` explicitly if per-run plots and tables
are wanted for another seed.

Primary evaluation fields are `pnl` and `risk_adjusted_pnl`. The latter is

```text
risk_adjusted_pnl = pnl - terminal_inventory_penalty
```

where `pnl` already includes cancellation fees. Shaped training return and all
shaping components are retained as diagnostics but never enter reported PnL or
checkpoint selection. Historical tables additionally use
`risk_adjusted_pnl_bps`.

The full field-level schema is in `docs/metrics_schema.md`.

## Compile the manuscript

The two generated inputs currently referenced by `paper/main.tex` are tracked
under `paper/results/`, so compilation does not depend on a local experiment
output tree. With a TeX distribution providing `latexmk` and the packages
listed in `paper/packages.tex`, build from a clean checkout with:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

Publication figures and tables should still be regenerated after the final
experiments. Keeping the currently referenced inputs tracked guarantees build
completeness; it does not promote an old smoke or pre-revision artifact to a
new empirical result.

## Determinism

One master `SeedSequence` creates private streams for normalizer calibration,
training, validation, final evaluation, exploration, replay, and the AS
order-book impact regression. The environment never uses global NumPy or
Python RNG state. The AS benchmark is risk-neutral (`gamma=0`), so no
volatility parameter is calibrated or stored.
For a fixed config, seed, machine, and dependency build, numeric trajectories
are reproducible across fresh processes; `wall_clock_s` is intentionally not.
Common-random-number comparisons reuse the same exogenous realization for each
policy on a matched evaluation episode.
