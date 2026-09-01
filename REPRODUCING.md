# Reproducing the revised experiments

This repository implements the revised manuscript environment for the
synthetic rough-Heston setting and the historical-midprice setting, with DQN
and the DDPG/TD3/SAC continuous-action relaxations. Current minute-clock
artifacts are written below `results/revision_v9/`; checkpoints and results
from earlier environment contracts are rejected.

## Setup

Python 3.10 or newer is required. The reference configs use CPU Torch.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

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
same 254-template cancellation-enabled DQN auction head. The acceptance suite
compares every shared resolved section for all four learners.

The exogenous auction quote support is `B_inf=150` ticks, so the exogenous
bounds are `M1=-150` and `M2=150`. The strategic manuscript coordinate remains
the absolute frozen-mid offset `b in [-B_max,B_max]`, with `B_max=150` and
`S_t^a=S_{tau_op}^{mid}+alpha*b_t^a`. For tractability, every learned policy
uses 21 local templates `ell in [-10,10]` around the lagged observable
indicative price and resolves them at decision time to an absolute `b` inside
the ambient band. `B_inf` and `B_max` happen to be numerically equal but have
different exogenous and strategic roles.

## Historical data requirement

New historical experiments require
`data/historical_sp500_midquotes_1m.csv` and its schema-3 sidecar. The artifact
is built from timestamped bid/ask quotes (SIP for publication), stores raw USD
midpoints without normalization, and maps each decision time to the latest
valid quote at or before that time. The environment freezes the auction-open
midquote during the call and rebases a selected session to `S0=100` only as an
explicit model-coordinate transformation. Order flow, books, auction
proposals, clearing, and allocation remain simulated.

The exact credential-safe build command and raw-event archive contract are
documented in `data/README.md`:

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
optimizer-update maturity thresholds; DQN auction updates made while its
behavior policy is locked to no-order do not count. Early-stopping patience
starts with the first eligible validation. The initial economic score is a
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

A short pipeline smoke is available and is not a training result:

```bash
scripts/reproduce_all.sh --smoke --symbol MSFT --seed 42
```

The matched synthetic treatment matrix—four `H_cl`/shaping arms, no auction,
and no cancellation for all four learners—is launched explicitly with:

```bash
scripts/run_synthetic_treatments.sh --seed 42
# or exercise only the pipeline:
scripts/run_synthetic_treatments.sh --smoke --seed 42
```

The no-cancellation treatment gives DQN 127 auction actions; the enabled
regime has 254. Continuous agents use two or three normalized auction
proposal coordinates respectively while keeping the same 18-feature network
input.

For cross-seed reporting:

```bash
scripts/run_multiseed.sh --seeds "42 7 99"
scripts/make_multiseed_outputs.sh --seeds "42 7 99"
```

Multiseed policy comparisons use confidence intervals over paired per-seed
policy-minus-benchmark differences. Historical cross-asset tables are emitted
both in currency units and in basis points of initial notional.

## Equivalent module commands

For a synthetic DQN run named `dqn_seed42`:

```bash
RUN=results/revision_v9/synthetic_rough_heston/dqn_seed42

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

Each run contains:

- `config_resolved.yaml`, `seed.txt`, `git_sha.txt`, and
  `runtime_versions.json`;
- `historical_data_manifest.json` for historical runs;
- `feature_normalizer.yaml`, `split_seeds.yaml`, and realized-grid records;
- `metrics.csv` and training forecast diagnostics;
- `checkpoints/{initial,best,final}.pt` plus optional resume checkpoints;
- `eval/records.csv`, `eval/metadata.yaml`, realized grids, forecast summaries,
  proposal/action/clearing diagnostics, and episode traces;
- `eval/policy_difference_{as,twap}.csv`;
- generated figures and booktabs/CSV tables.

Primary evaluation fields are `pnl` and `risk_adjusted_pnl`. The latter is

```text
risk_adjusted_pnl = pnl - terminal_inventory_penalty
```

where `pnl` already includes cancellation fees. Shaped training return and all
shaping components are retained as diagnostics but never enter reported PnL or
checkpoint selection. Historical tables additionally use
`risk_adjusted_pnl_bps`.

The full field-level schema is in `docs/metrics_schema.md`.

## Determinism

One master `SeedSequence` creates private streams for normalizer calibration,
training, validation, final evaluation, exploration, replay, and benchmark
calibration. The environment never uses global NumPy or Python RNG state.
For a fixed config, seed, machine, and dependency build, numeric trajectories
are reproducible across fresh processes; `wall_clock_s` is intentionally not.
Common-random-number comparisons reuse the same exogenous realization for each
policy on a matched evaluation episode.
