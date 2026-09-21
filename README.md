# Learning Optimal Liquidation with Closing Auctions

<<<<<<< HEAD
Code and saved results for the paper *Learning Optimal Liquidation with Closing
Auctions*. The paper studies end-of-day
liquidation through a continuous limit order book followed by a closing
auction. DQN, DDPG, TD3 and SAC share one simulator and one executable action
grid and are compared with Avellaneda–Stoikov and TWAP references. Historical
inputs supply midprices only; order flow, auction clearing and allocation are
simulated.

The reported results come from the v20 campaign: 320 synthetic runs
(rough-Heston internal step 0.25 min; 240 main runs plus two 40-run follow-ups)
and 200 historical runs (five tickers), ten master seeds per cell. This
repository contains the source, the resolved configuration and seeds of every
run, and the saved numerical summaries behind every table and figure. Full run
directories and model checkpoints are not included.

## Contents

| Path | What it holds |
|---|---|
| `src/lmm/` | Simulator, clearing mechanism, learners, benchmarks, experiment entry points (`lmm-train`, `lmm-evaluate`, `lmm-make-report`, …) |
| `configs/` | Base configuration, synthetic/historical overlays, learner overlays (`algo/`), treatment overlays (`treatment/`), campaign definition (`campaign/`) |
| `scripts/` | Launchers (`run_*.sh`, `run_multiseed.sh`, `reproduce_all.sh`), release checks (`smoke_release.py`, `release_support.py`, `release_validate.py`) |
| `release/evidence/` | Run identities, resolved configurations, seeds, selection records, fitted normalization state, artifact hashes, and the saved seed-level and aggregate summaries (see [Paper exhibits](#paper-exhibits-and-their-sources)) |
| `release/paper-environment.txt` | Direct dependency versions recorded in all 520 paper runs |
| `data/` | Instructions for obtaining and preprocessing the historical midprice data (the data themselves are not included) |
| `tests/` | Offline CPU test suite, including an exact regression against the frozen pre-refinement simulator |
| `docs/` | Model, learner, metrics and v20 reproduction documentation |

## Requirements

**Software.** Python 3.11 or later for the pinned installation below. The package
metadata permits Python 3.10 with other compatible dependency versions, but that
is not the recorded environment. The paper runs used Python 3.14.4 on
macOS 26.6.2 (arm64) with the direct dependency versions pinned in
`release/paper-environment.txt` (NumPy, pandas, SciPy, PyTorch, Gymnasium,
Stable-Baselines3, Matplotlib, PyYAML, certifi). No proprietary software is
needed. Manuscript sources and PDF are maintained separately and are not included.

**Hardware.** CPU only; no GPU is used or required. The campaign ran with 10
parallel workers, one numerical/Torch thread each, on
an Apple M5 Pro MacBook Pro (model identifier Mac17,9), with 15 CPU cores and
24 GiB unified memory. Allow approximately **3–15 minutes per training and evaluation run** and **5–7 hours for 520 runs with ten workers**. These are
planning estimates: the saved training logs span 1.4–10.3 minutes per run, and
the 320-run synthetic pipeline took 3.09 hours. See the
[timing evidence and estimation method](release/evidence/runtime_estimate.json).
Levels 1 and 2 below run on a laptop in minutes.

## Installation

These commands assume a repository checkout or complete repository snapshot.
The software wheel/source distribution alone does not include `release/` evidence.
=======
Code for end-of-day liquidation through a continuous limit order book followed
by a closing auction. DQN, DDPG, TD3 and SAC share one simulator and one
executable action grid. Historical inputs supply midprices only; order flow,
auction clearing and allocation are simulated.

The paper's results come from the v20 campaign: 320 synthetic runs (rough-Heston
internal step 0.25 min) and 200 historical runs. Run identities, resolved
configurations, seeds, selection records, fitted normalization state and
artifact hashes are in `release/evidence/`. Saved numerical summaries behind
every table and figure, and the scripts that regenerate them, are included;
full run directories and checkpoints are not. Older v19 summaries in `docs/`
are development history.

## Setup and smoke test

Python 3.10+. The paper environment was Python 3.14.4 on macOS arm64; pinned
versions are in `release/paper-environment.txt`.
>>>>>>> f35440f51678c676f70a02c77dda1d7d27f8e077

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -c release/paper-environment.txt -e '.[dev]'
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib"
<<<<<<< HEAD
```

A fresh install with these constraints was verified on Ubuntu 24.04 /
Python 3.11.16 at commit `02a8f005` (hosted CI; see
`release/verification.json`).

## Reproducing the results

Reproduction is organized in three levels so that the parts a reader can check
in minutes are separated from the longer retraining. Details, caveats and
the original launch recipe are in [REPRODUCING.md](REPRODUCING.md).

| Level | Command | Establishes | Needs |
|---|---|---|---|
| 1. Smoke | `python scripts/smoke_release.py` | Simulator runs two synthetic episodes end to end with finite states and consistent accounting | Nothing beyond the install; ~1 min |
| 2. Saved-result verification | `python scripts/release_support.py --check` | Regenerates all 48 paired learner/reference intervals and 30 ordinary-shortfall levels from the bundled unrounded seed means and checks them against the reported values | Nothing beyond the install; seconds |
| 2. Tests | `pytest -q -m 'not network and not slow and not market_data'` | Unit and contract tests, plus exact agreement of the current simulator with the frozen pre-refinement source | CPU only; a few minutes |
| 3. Full retraining | Follow [the original configuration/source recipe](REPRODUCING.md#full-experiment-opt-in-and-preserved-originals) for all 520 runs | Rerun training, evaluation and paired contrasts with the recorded inputs | Synthetic arm: no market data. Historical arm: authorized data, see below. Estimate: 5–7 hours with ten workers on the recorded machine |

Notes on level 3:

- `scripts/reproduce_all.sh --seed 42` runs the headline learners in both
  settings for one seed. `scripts/run_multiseed.sh --jobs 10 --threads-per-job 1`
  runs the 440-run main matrix for ten seeds. Neither command alone recreates
  the refined 520-run paper campaign, including both follow-ups.
- The exact resolved configuration, seeds, source revision and selection record
  of every paper run are in `release/evidence/campaign.json`. Use those when
  matching a specific reported run; the default overlays are a convenience, not
  a certificate of equivalence.
- The ten master seeds are 42, 7, 99, 123, 2024, 314, 577, 811, 1618, 2718.
  Derivation of every component stream from a master seed is documented in
  REPRODUCING.md.
- `run_multiseed.sh --replace-synthetic` is an archiving/retraining workflow
  against an existing result tree, not a regeneration command; do not run it
  without reading `docs/rough_heston_refinement/rerun_v20.md`.
- Torch runs with deterministic algorithms enabled (`warn_only=True`). Exact
  agreement with the frozen source was verified within one runtime; bitwise
  equality across platforms, BLAS builds or library versions is not claimed.

## Paper exhibits and their sources

The saved numerical inputs for the paper exhibits are under `release/evidence/`;
paths in the table below are relative to that directory. The retained
`lmm.experiments.make_report` and `lmm.experiments.cashflow_comparison` generators
read saved results without stepping an environment and expect the full run
trees, which are not included. Manuscript-specific assembly scripts were removed
with `paper/`; `scripts/release_support.py` remains available to regenerate and
verify the paired benchmark table from the bundled seed means.

| Exhibit | Saved inputs |
|---|---|
| Headline ordinary and inventory-penalized shortfalls (synthetic and equal-ticker historical) | `shortfall_summary.csv`; per seed in `revision_v20/audit/economic_by_seed.csv` |
| Economic performance by market | `revision_v20/tables/economic_performance.csv` |
| Auction contribution and inventory management | `revision_v20/tables/auction_mechanism.csv`; per seed in `revision_v20/audit/economic_by_seed.csv` |
| Market-specific shortfalls and paired learner/reference effects | `benchmark_supplement.csv` (regenerated by `release_support.py --check`) |
| Forecast accuracy by market and phase | `forecast_summary.csv`, `forecast_by_algorithm_seed.csv`; fit protocol in `forecast_fits.json` |
| Matched synthetic effects (auction access, dense credit, explicit forecast feature, anchoring) | `revision_v20/tables/treatments.csv`; per seed in `revision_v20/audit/treatments_by_seed.csv` |
| Dense auction credit versus raw cash-flow training | `revision_v20_cashflow/comparison.csv`, `revision_v20_cashflow/by_seed.csv` |
| Added preferences with conditioning fixed | `revision_v20_economic_dense_h/comparison.csv`, `revision_v20_economic_dense_h/by_seed.csv` |
| Validation curves and checkpoint selection | `revision_v20/audit/validation_by_seed.csv`, `revision_v20/audit/validation_common_support.csv`, `revision_v20/audit/credit_learning_*.csv` |
| Avellaneda–Stoikov calibration coefficients used in every evaluation | `as_calibration.json` |
| Rough-Heston mesh refinement diagnostic | `mesh_diagnostic.json` |

Point estimates average episodes within seed, then weight the ten seed means
equally; historical summaries average the five tickers within seed first.
Paired intervals use 10,000 percentile resamples of the ten paired seed
differences (PCG64, seed 0). Intervals are pointwise and not corrected for
multiple comparisons.

## Data

The historical midprice paths were obtained through the Alpaca Market Data API
(SIP consolidated quotes). Redistribution rights are not established, so neither
the vendor quote archives nor the processed CSV is included. What is included:
the tickers (CAT, GOOGL, JPM, MSFT, PG), the session dates (training 2026-08-03
to 08-14, validation 08-17 to 08-21, test 08-24 to 08-28), the preprocessing
rules and command, the SHA-256 hashes of the original CSV and the twenty raw
archives, and the recovered date assignments for historical runs
(`release/evidence/data.json`, `release/evidence/historical_dates.json`). Readers
with the required Alpaca data entitlements can rebuild the input using
[data/README.md](data/README.md); a fresh
download need not be byte-identical to the original.

The synthetic campaign, all four learners' matched comparisons and both
follow-ups require no market data.

## Known limits

- Full run directories and checkpoints are not distributed; level 3 is the only
  route to them.
- The AS coefficients used in the paper are preserved, but the calibration
  sample counts and fit inputs were not recorded, so the original fit cannot be
  audited at the sample level. Missing counts are stored as `null`, not zero.
- The full campaign was not rerun during release preparation (level D in the
  terminology of REPRODUCING.md).
- Development-visible test dates are a reused holdout, not an independent
  out-of-sample backtest.

## Documentation

[Model](docs/model.md) · [Learners](docs/rl_design.md) ·
[Metrics schema](docs/metrics_schema.md) · [Reproduction guide](REPRODUCING.md) ·
[Data setup](data/README.md) · [Release readiness ledger](release/READINESS.md)

## License and citation

First-party software and software documentation use the MIT license; see [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for non-code assets and vendor
data terms. Cite the paper and the software via [CITATION.cff](CITATION.cff).
=======
python scripts/smoke_release.py
pytest -q -m 'not network and not slow and not market_data'
```

The smoke test runs two synthetic episodes without training or data. Full
experiments, table regeneration (`scripts/release_support.py --check`) and the
validation script are described in [REPRODUCING.md](REPRODUCING.md).
Exact agreement with the frozen source was verified on Ubuntu 24.04 /
Python 3.11.16 at commit `02a8f005`; bitwise equality across platforms and
library versions is not claimed.

## Data

Vendor quote archives and processed price paths are not included. Historical
reproduction needs separately licensed inputs; see [data/README.md](data/README.md)
for the tickers, dates and preprocessing.

## Known limits

The AS coefficients used in the paper are preserved, but the calibration
sample counts and fit inputs are not; see the disclosure in REPRODUCING.md.

## Documentation and license

[Model](docs/model.md) · [Learners](docs/rl_design.md) · [Metrics](docs/metrics_schema.md)

MIT license (Julius Graf and Thibaut Mastrolia, 2026); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for non-code assets and vendor
data. Cite via [CITATION.cff](CITATION.cff).
>>>>>>> f35440f51678c676f70a02c77dda1d7d27f8e077
