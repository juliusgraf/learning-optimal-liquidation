# Reproducing the revised experiments

This repository implements the revised manuscript environment for the
synthetic rough-Heston setting and the historical-midprice setting, with DQN
and the DDPG/TD3/SAC continuous-action relaxations. Revised artifacts are
written only below `results/revision_v2/`; checkpoints and results from earlier
environment contracts are rejected.

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
fills, inventory conservation, self-trade exclusion, unshaped accounting,
annualized rough-Heston time, the capped benchmark schedule, and rejection of
stale artifacts.

Do not infer publication readiness from the legacy characterization tests in
`tests/audit/`; they document the superseded implementation.

## Historical data requirement

The configured input, `data/historical_sp500_1m.csv`, contains 20 frozen
one-minute sessions for CAT, PG, GOOGL, JPM, and MSFT. The chronological pools
are 10 training sessions (2026-08-03–14), 5 validation sessions
(2026-08-17–21), and 5 test sessions (2026-08-24–28). The accompanying sidecar
records source provenance, per-session fills, split membership, and the CSV
digest. Dataset ID, digest, tickers, timezone, missing-data treatment, split
ranges, and nonempty pools are verified before use and copied into every run.

The loader maps each realized decision time to the most recent observation at
or before that time and never interpolates from a future bar. It uses the
historical observation at auction open and freezes it throughout the call. The
historical series supplies midprices only; order flow, books, auction proposals,
clearing, and allocation remain simulated.

The range-capable yfinance builder and exact regeneration command are documented
in `data/README.md`:

```bash
python3 -m lmm.data.load_yfinance_data --help
```

## Canonical runs

The baseline budgets are 1,000 synthetic episodes and 500 historical episodes.
Each run fits its feature normalizer on training-only paths, selects `best.pt`
on validation risk-adjusted PnL, and evaluates that fixed checkpoint on 100
held-out test episodes per seed/configuration.

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

The no-cancellation treatment gives DQN 511 auction actions; the enabled
regime has 1,022. Continuous agents use two or three normalized auction
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
RUN=results/revision_v2/synthetic_rough_heston/dqn_seed42

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
