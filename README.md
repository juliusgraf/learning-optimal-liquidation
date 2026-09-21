# Learning Optimal Liquidation with Closing Auctions

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

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -c release/paper-environment.txt -e '.[dev]'
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib"
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
