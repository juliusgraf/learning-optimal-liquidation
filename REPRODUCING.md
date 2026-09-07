# Reproducing the completed study

Training is finished: 440 main v19 runs and two subsequent 40-run synthetic
comparisons. Use the saved results for writing. This guide documents how to check
and regenerate them, and how to reproduce experiments if needed later.
The [documentation index](docs/README.md) links the scientific protocol and
findings; [rl_design.md](docs/rl_design.md) defines the learning contract.

## Environment and validation

Python 3.10 or newer is required. `pyproject.toml` is the dependency source of
truth, including Stable-Baselines3 2.7.1; `requirements.txt` delegates to it.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
pytest -q tests/test_revision_acceptance.py
pytest -q -m 'not network and not slow'
```

When market data are absent, data-dependent integration tests skip explicitly.
Use `pytest --require-market-data -q -m 'not network and not slow'` on the
research machine to require all private-data checks.

The offline suite covers environment chronology, action projection, rewards,
checkpoint selection, data provenance, launchers and reporting. Tests under
`tests/audit/` also characterize the retained legacy implementation; they do not
establish the validity of the current economics. No full training is required to
run the offline suite.

The manuscript is maintained separately and is not part of the software release.

## Data and model contract

The locally retained historical CSV, schema-3 sidecar and raw archives are required
for reproducing the original historical runs. They are excluded from the current source tree.
The environment checks their digests and split contract. Synthetic experiments need no data account. Historical setup and authorized
regeneration are described in [data/README.md](data/README.md).
Training uses pooled stock-session paths from training dates. Validation and test
remain ticker-specific. The previously inspected historical test dates are a
reused holdout, not fresh-date or unseen-stock evidence.

The resolved configuration is `configs/base.yaml`, one setting overlay, one
`configs/algo/` overlay and any `configs/treatment/` overlay. Forecast reliability
coefficients in the setting overlays are frozen training-only fits; their
protocols remain under `docs/verification_v19/forecast_{synthetic,historical}/`.
Feature normalizers are fitted within each run on training paths and saved.
See the learning design for the distinction between forecast information and
quote anchoring, and for the exact reward-conditioning equations.

Headline training uses weighted shaped J. Checkpoint selection and all learned,
AS and TWAP evaluation use economic risk-adjusted PnL. Economic controls change
only their declared interventions. Raw cash-flow training omits reward
conditioning; the matched dense economic control retains it. PnL never includes
fictive rewards or replay-only potentials.

For a policy-free simulator check:

```bash
python -m lmm.experiments.diagnose_simulator \
  --config configs/base.yaml --config configs/synthetic_rough_heston.yaml \
  --episodes 100 --assert-ready
python -m lmm.experiments.diagnose_simulator \
  --config configs/base.yaml --config configs/historical_sp500_midquotes.yaml \
  --episodes 20 --assert-ready
```

## Completed outputs

| Campaign | Runs | Report |
|---|---:|---|
| Main synthetic, historical and mechanism study | 440 | `results/revision_v19/_publication/index.html` |
| H-anchored raw economic cash flow | 40 | `results/revision_v19_cashflow/_comparison/index.html` |
| H-anchored dense economic training | 40 | `results/revision_v19_economic_dense_h/_comparison/index.html` |

The main bundle contains five PDF/PNG figure groups, three CSV/LaTeX tables,
seed-level audit records and an input/output hash manifest. Each follow-up
contains a comparison figure, table, seed-level differences and manifest.
[research_outputs.md](docs/research_outputs.md) explains each contrast and the
statistical interpretation. The public [analysis summaries](docs/README.md) provide compact completed-study evidence.

Every completed run retains its resolved configuration, Git SHA, dependency and
split-seed provenance, feature normalizer, training metrics, selected/initial/final
checkpoints, evaluation records, traces, diagnostics and paired benchmark
comparisons. `pipeline_complete.json` binds the complete required inventory by
SHA-256. Do not delete intermediate-looking files inside completed runs: many
are required by that integrity contract. Generated reports are separate from the
bound run inputs. Neither report generation nor analysis needs to step an environment.

## Regenerating reports without training

This section requires the locally retained saved runs and their original source
commits. Full run outputs are not included in a source checkout. Users can run
new campaigns and report them using their own committed source identity.

The follow-ups validate against each run's saved source identity:

```bash
python -m lmm.experiments.cashflow_comparison --report-only
python -m lmm.experiments.cashflow_comparison --conditioned --report-only
```

The main strict publication generator additionally requires a clean checkout at
the **saved training commit**. A later documentation or cleanup commit is still a
different Git identity. Preserve that gate: use a separate checkout instead of
changing saved provenance or weakening validation. From this repository root:

```bash
LMM_REPO="$PWD"
LMM_TRAIN_SHA="$(cat results/revision_v19/synthetic_rough_heston/dqn_seed42/git_sha.txt)"
LMM_REPORT_TREE="$(mktemp -d /tmp/lmm-v19-report.XXXXXX)"
git worktree add --detach "$LMM_REPORT_TREE" "$LMM_TRAIN_SHA"
(
  cd "$LMM_REPORT_TREE"
  PYTHONPATH="$LMM_REPORT_TREE/src" MPLCONFIGDIR="$LMM_REPO/.cache/matplotlib" \
    "$LMM_REPO/.venv/bin/python" -m lmm.experiments.make_report \
    --root "$LMM_REPO/results/revision_v19" \
    --seeds 42 7 99 123 2024 314 577 811 1618 2718 --publication
)
git worktree remove "$LMM_REPORT_TREE"
```

This uses the installed dependency environment with the recorded source checkout
and regenerates the existing publication directory from saved run inputs. For a
new campaign at its own clean training commit, the shell equivalent is
`LMM_RESULTS_ROOT=/absolute/path/to/campaign scripts/make_multiseed_outputs.sh --publication`.
Specify the root explicitly when reporting a different campaign. Extra per-run plots remain opt-in with `--diagnostics`.

## Reproducing training later

These commands document the completed protocols; they are not needed for the
write-up. Use a fresh results root and a clean committed tree. Never overwrite
completed campaigns or mix runs from different source commits.

```bash
# Inspect the main 440 jobs without training:
LMM_RESULTS_ROOT=/absolute/scratch/new-study \
  scripts/run_multiseed.sh --jobs 10 --threads-per-job 1 --dry-run
# Remove --dry-run only when intentionally reproducing the campaign.
```

For the two extensions, `python -m lmm.experiments.cashflow_comparison --help`
documents `--root`, `--headline-root`, `--jobs`, `--threads-per-job`, `--dry-run`
and `--conditioned`. Their controls reuse the selected synthetic headline runs.
The original matrix and both follow-ups use ten canonical master seeds:
42, 7, 99, 123, 2024, 314, 577, 811, 1618 and 2718.

The production cap is 800 episodes, with economic validation on 128 paths and
final evaluation on 100 paths. The maturity gate, patience and stopping rules
are recorded in the resolved configuration. The selected reportable checkpoint
is the best mature economic-validation policy; non-improving seeds remain in
the study. Worker count changes scheduling, not the learning budget or seed set.
Queue state and console logs live under `_orchestration/`.

The launchers skip only completed runs whose configuration, source identity and
full completion inventory validate. Interrupted or changed runs fail closed.
They suppress replay-heavy periodic resume checkpoints; selected, initial and
final checkpoints are retained. Exact numerical replay also depends on the saved
dependency versions and machine. Independent RNG streams separate normalization,
training, validation, test, exploration and replay; wall-clock timings are not
expected to reproduce.

For a bounded development investigation, the retained `scripts/diagnose_learning.py`
accepts `--algo`, `--seed`, `--episodes`, `--validation-size` and `--output`.
It caps training at 200 episodes and rejects an existing output directory.
Use `--resolved-config` to investigate a saved diagnostic configuration.
The remaining audit scripts cover forecast credit, physical market units,
shaping calibration and policy economics. Superseded one-off scripts and trial
overlays are archived as described in [cleanup.md](docs/cleanup.md).
