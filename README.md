# Learning Optimal Liquidation with Closing Auctions

A research simulator for end-of-day liquidation through a continuous limit order
book followed by a closing call auction. DQN is the discrete baseline; DDPG, TD3
and SAC use projected continuous proposals on the same executable controls.
Synthetic rough-Heston and historical-midquote experiments share the market
mechanics. Historical inputs supply midprice paths; order flow, books, clearing
and allocation remain simulated.

The study is complete: **440 main runs plus two 40-run economic-training
comparisons**, with ten training seeds per cell. Headline policies train on
weighted shaped J with dense auction credit; checkpoint selection and evaluation
use economic risk-adjusted PnL. The current task is the research write-up.

## Read the completed study

- [Research findings and manuscript plan](docs/research_writeup_plan_v19.md)
- [Main results](results/revision_v19/_publication/index.html)
- [Raw economic cash-flow comparison](results/revision_v19_cashflow/_comparison/index.html)
- [Matched dense economic comparison](results/revision_v19_economic_dense_h/_comparison/index.html)
- [Documentation index](docs/README.md), including the result verdicts and analysis records

Results are local, ignored artifacts. They are available on the research machine;
a source checkout alone does not contain trained checkpoints or reports.

## Sources of truth

- [Manuscript](paper/main.tex): mathematical specification. Pending alignment
  recommendations are in [the review patch](docs/manuscript_recommendations_v19.patch)
  and the manuscript plan; neither is applied automatically.
- `configs/base.yaml` plus setting, algorithm and treatment overlays: executable
  configuration. Saved runs retain their own complete resolved configurations.
- [Learning design](docs/rl_design.md): controls, forecast, rewards and selection.
- [Reproduction guide](REPRODUCING.md): setup, verification and report generation.

## Setup and checks

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q -m 'not network and not slow'
```

The frozen historical CSV, provenance sidecar and raw archives are tracked;
ordinary historical runs need no credentials or download. See [the data
contract](data/README.md).

Superseded development output and one-off tools have been removed from the
working tree, with exact recovery instructions in [the cleanup record](docs/cleanup.md).
`legacy/` and `audit/` retain the small implementation and characterization
records used by regression tests; they are not active specifications.
