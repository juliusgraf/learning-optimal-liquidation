# Learning Liquidation with Closing Auctions

Research software for end-of-day liquidation through a continuous limit order
book and a closing call auction. DQN, DDPG, TD3 and SAC share a simulator and
executable action grid. Historical inputs supply midquotes; order flow, auction
clearing and allocation remain simulated.

This checkout is a **draft public-release candidate**, with a compact replication
evidence bundle. It has not been published. Repository preparation, support for
the paper's replication claim, and verified public availability are separate
outcomes; see [the readiness report](release/READINESS.md).

The current manuscript uses the completed **v20** campaign: 320 synthetic runs
at a maximum internal rough-Heston step of 0.25 minutes, and 200 retained
historical runs. The original synthetic and historical source revisions differ.
The [provenance manifest](release/provenance.json) and
[replication guide](REPRODUCING.md) identify them explicitly. The older v19
summaries in `docs/` are development history, not the current paper's results.

## Install and run a small check

Python 3.10+ is declared by the package. The recorded paper environment is Python
3.14.4 on macOS arm64; [recorded direct dependency versions](release/paper-environment.txt)
are preserved separately from the broad installation requirements.

From a source checkout, create a fresh environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -c release/paper-environment.txt -e '.[dev]'
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python scripts/smoke_release.py
pytest -q -m 'not network and not slow and not market_data'
```

The smoke check runs two identical synthetic CPU episodes, without training,
market data, credentials, model weights, tracking or uploads. It is not a paper
result. The tests use explicit synthetic fixtures in `tests/fixtures/`; they
never replace missing paper inputs. The legacy episode regression compares the
current simulator exactly with frozen original source under the same runtime;
cross-platform bitwise reproducibility is not assumed. The revised Linux check
still needs a hosted run; see the readiness report.

Regenerate and verify the paired benchmark supporting table from bundled,
unrounded seed means:

```sh
python scripts/release_support.py --check
# Optional: writes to a NEW directory outside the checkout.
python scripts/release_support.py --output /tmp/lmm-support-table
python scripts/release_validate.py
```

Draft validation checks contents and provenance. Its success does **not** clear
reported blockers or establish publication. Full experiments are explicit opt-in
operations described in [REPRODUCING.md](REPRODUCING.md).

## What is supplied

- Simulator, learners, configuration overlays, tests and execution/reporting entry points.
- All 520 run identities, resolved configurations, complete recorded training,
  validation, test and normalization seeds, selection records, fitted normalization
  state, dependency records, AS coefficients and original artifact hashes in
  `release/evidence/campaign.json` and its companion records.
- Compact saved numerical outputs for current headline, treatment and follow-up
  comparisons, plus the existing paired-reference support table.
- The existing mesh diagnostic and recovered historical episode-to-date mappings,
  with their limits and preparation-time transformations identified.
- Manuscript and figure sources under `paper/`, retained without modification in
  this preparation. The full run directories and original checkpoints remain local;
  their hashes do not make those objects publicly accessible.

**AS calibration evidence limit:** the actual AS coefficients used in the paper
are preserved, but accepted/excluded calibration counts and original fit samples
are missing from the supplied records. See the
[calibration disclosure](REPRODUCING.md#saved-analysis-and-calibration-p-001-p-003-p-004-p-005)
and [saved coefficient records](release/evidence/as_calibration.json).

Vendor quote archives and processed price paths are **not** included in the
proposed package. Historical reproduction requires separately authorized,
matching inputs; see [data access and preprocessing](data/README.md). Historical
test dates were inspected during development. No untouched-date claim is made.

The Git repository's reachable history contains historical data and other
superseded materials. `.gitignore` and the curated staging package do not make
that history safe to expose. Owner review is required before changing visibility.

## Model, license and citation

See [model mechanics](docs/model.md), [learner design](docs/rl_design.md),
[metric definitions](docs/metrics_schema.md), and [contribution guidance](CONTRIBUTING.md).

First-party software and software documentation use the standard [MIT license](LICENSE),
with confirmed holders Julius Graf and Thibaut Mastrolia, copyright 2026.
[Distribution scope and third-party notices](THIRD_PARTY_NOTICES.md) distinguish
software, installed dependencies, research assets and restricted vendor data.
Permission to distribute non-code research assets does not assign them an MIT license.
Academic citation is requested separately in [CITATION.cff](CITATION.cff).

The [owner publication runbook](release/RUNBOOK.md) supplies final commit/tag,
checksumming and unauthenticated-access checks. Do not fill a manuscript release
identifier or release commit from this uncommitted preparation.
