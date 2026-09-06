# Learning Optimal Liquidation with Closing Auctions

This repository implements the simulator and reinforcement-learning methods
for an end-of-day liquidation problem with a continuous limit-order-book phase
followed by a two-sided closing call. The controlled CLOB policy is a
one-sided liquidator. At auction open, the admissible controls expand to
signed schedules, so terminal inventory can be positive or negative.

The implemented learners are DQN and the projected continuous-proposal methods
DDPG, TD3, and SAC. Synthetic rough-Heston and historical-midquote experiments
use the same market mechanics, action semantics, rewards, and learning
configuration; only the exogenous mid-price source changes. Historical runs
replay frozen SIP midquotes while order flow, books, auction updates, clearing,
and allocation remain simulated.

The [current repair report](docs/pathology_repair_v18.md) documents the
author-approved shaping correction, calibration, learning checks and auction
diagnostics. The market remains stylized; the development results do not
establish universal benchmark superiority or empirical auction calibration.
The [DQN auction follow-up](docs/dqn_auction_repair_v18.md) records the later
projection, exploration and replay-target fixes, with all rejected trials.

## Sources of truth

- `paper/main.tex` defines the mathematical model.
- `configs/base.yaml`, setting overlays, algorithm overlays, and treatment
  overlays define the active executable configuration.
- `REPRODUCING.md` is the operational guide for installation, validation,
  training, evaluation, and paper compilation.
- `docs/rl_design.md` documents how the mathematical controls are represented
  by the learners.

Legacy implementations and the reports under `audit/` are retained for
provenance and characterization only. They are not active specifications.

## Quick start

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q tests/test_revision_acceptance.py
pytest -q -m 'not network and not slow'
```

The repository includes the frozen processed historical artifact and its
verified provenance sidecar, so the historical simulator does not require API
credentials after checkout. See `REPRODUCING.md` for the complete workflow.

After committing the reviewed changes and leaving the tree clean, launch the
440-run publication matrix with `scripts/run_multiseed.sh --jobs 10 --threads-per-job 1`
from the activated environment. The resulting report is
`results/revision_v19/_publication/index.html`: five focused figures, three
tables, and seed-level audit records. The [output guide](docs/research_outputs.md)
explains the statistical protocol, file locations, regeneration and exact
unapplied manuscript inclusion instructions. Comprehensive diagnostics are
available separately with `--diagnostics`.

The protected TeX sources are unchanged. The [exact review patch](docs/manuscript_recommendations_v19.patch)
aligns them with weighted shaped J for headline training and economic
risk-adjusted PnL for validation and evaluation. Earlier reports are retained
as development history.
