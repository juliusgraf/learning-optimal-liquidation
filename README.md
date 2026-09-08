# Learning Liquidation with Closing Auctions

Research code for end-of-day liquidation through a continuous limit order book
followed by a closing call auction. DQN uses discrete controls; DDPG, TD3 and SAC
use continuous proposals projected onto the same executable grid. Synthetic
rough-Heston and historical-midquote settings share the market mechanics.
Historical inputs determine prices only; order flow and auction clearing remain
simulated.

The revised volume-maximizing auction projection is implemented as
`max_volume_v2` (schema 16). Existing configurations, fitted forecasts and saved
paper results retain legacy nearest-tick clearing. The multiseed launcher now defaults to v2, refits forecasts automatically and
writes separate v20 outputs. See the [rerun workflow](docs/revised_clearing_reruns.md).

## Quick start

Python 3.10 or newer is required. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q -m 'not network and not slow'
```

The public test suite runs without a market-data account. Tests requiring the
private historical dataset skip explicitly when it is absent. Synthetic
experiments require no external data:

```bash
# Legacy standalone smoke: four training and three evaluation episodes only.
LMM_RESULTS_ROOT=results/smoke scripts/run_synthetic_dqn.sh --smoke

# Inspect the revised 440-run matrix and two forecast prerequisites.
scripts/run_multiseed.sh --jobs 10 --threads-per-job 1 --dry-run

# After committing a clean worktree, fit, train/reselect, evaluate and report:
# scripts/run_multiseed.sh --jobs 10 --threads-per-job 1
```

The smoke run is not a scientific result. Full training budgets, data setup and
report generation are documented in [REPRODUCING.md](REPRODUCING.md).

## Model and results

- [Model specification](docs/model.md): timing, controls, clearing and objectives.
- [Learning design](docs/rl_design.md): observations, conditioning and selection.
- [Results and protocol index](docs/README.md): completed-study summaries and evidence.
- [Metric definitions](docs/metrics_schema.md): economic and training accounting.
- [Historical-data setup](data/README.md): obtain and validate your own authorized inputs.

The completed study comprises 440 main runs and two 40-run synthetic comparisons,
with ten training seeds per cell. Headline training uses weighted preference
rewards; selection and comparisons use economic PnL less terminal inventory risk.
The negative of economic PnL is implementation shortfall relative to initial
midprice under the stated residual-marking convention. Machine-readable metric
names and saved results retain their original sign conventions.

The findings support dense auction credit and method-dependent incremental
forecast value. Additional preference terms have no established marginal benefit
when conditioning is matched. See the [main results](docs/revision_v19_verdict.md)
and [matched comparison](docs/economic_dense_h_results_verdict_v19.md).

The manuscript is maintained separately. The public repository includes source,
tests, configurations and compact analysis summaries. Market data, trained
checkpoints and full report bundles are excluded. Historical test dates were
inspected during development; these experiments are not exchange-auction backtests.

## Reproducibility and distribution

The executable specification is `configs/base.yaml` composed with a setting,
algorithm and optional treatment overlay. Saved runs retain their resolved
configurations and source identity. Changes must not overwrite completed runs
or weaken provenance validation.

[CONTRIBUTING.md](CONTRIBUTING.md) describes validation and contribution practices.
The software is available under the [MIT license](LICENSE); this does not grant
rights to third-party market data. Software citation metadata is in
[CITATION.cff](CITATION.cff).

For the paper submission, cite this GitHub repository and record the commit
used for the experiments. The current source tree excludes manuscript files
and market data; earlier commits retain the project's development history.

Optional rough-Heston internal refinement and coupled diagnostics are documented in
[docs/rough_heston_refinement/README.md](docs/rough_heston_refinement/README.md).
The default and retained manuscript results use the legacy decision-grid generator.

To replace only the 320 synthetic runs of the saved v20 campaign at the selected
0.25-minute mesh while retaining its 200 historical runs, see
[the prepared v20 rerun command](docs/rough_heston_refinement/rerun_v20.md).
