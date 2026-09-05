# Current v17 workflow

The active contracts and algorithm details are in `docs/rl_design.md` and
`docs/model_refinement_v17.md`. The headline is H-on/shaping-on, trained on the
author-approved weighted J and
selected/evaluated on economic risk-adjusted PnL. Use a fresh results/revision_v17
namespace; older outputs remain diagnosis inputs only. Stable-Baselines3 2.7.1
is installed by the project dependency metadata.

Bounded development verification (180 episodes, no final-test paths):

```bash
MPLCONFIGDIR=/tmp/lmm-mpl LMM_TORCH_INTRAOP_THREADS=1 LMM_TORCH_INTEROP_THREADS=1 \
  .venv/bin/python scripts/diagnose_learning.py --algo dqn --seed 628 \
  --episodes 180 --validation-size 64 --output results/my_dqn_diagnostic
```

Use `--algo ddpg`, `td3`, or `sac` with distinct output directories. Historical
checks additionally pass `--setting historical_sp500_midquotes --symbol MSFT`.
The command refuses more than 200 episodes or an existing output directory.

DDPG/TD3 use phase-specific normalization and signed-asinh auction inventory
coordinates; DQN/SAC retain their v16 representation. To run a matched
representation control, append either
`--config configs/treatment/representation_pooled.yaml` or
`--config configs/treatment/representation_phase_asinh.yaml` to the bounded
diagnostic command. These supplementary controls have distinct labels and
are not added to the production treatment matrix. To reproduce a completed
run exactly, pass its saved complete `config.yaml` via `--resolved-config`.
The production launchers below retain the 800-episode cap and full treatment
matrix. No production run was launched as part of the repair.

# Reproducing the revised experiments

This repository implements the revised manuscript environment for the
synthetic rough-Heston setting and the historical-midprice setting, with DQN
and the DDPG/TD3/SAC continuous-action relaxations. Current artifacts use
schema 15, are written below `results/revision_v17/`, and carry environment
contract `shaped-j-economic-eval-sb3-2026-09-05-v15`; checkpoints and results
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
same 1,346-action cancellation-enabled DQN auction grid. The acceptance suite
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

The matched confirmation budget is 800 episodes in both settings. Each run
fits its feature normalizer on training-only paths. All algorithms use the
common 32-episode structured warmup; DQN retains its complete action grids.
A checkpoint can enter the validation race only after both phase learners pass
the configured optimizer-update maturity thresholds (only CLOB for no-auction).
Early-stopping patience starts with the first eligible validation. The active
`checkpoint_require_initial_improvement=true` also requires improvement over
the initial policy's economic validation score. The initial policy is never
reportable. `best.pt` is the highest-scoring mature, reportable joint policy,
selected on economic validation and evaluated on 100 held-out test episodes.
If no checkpoint qualifies, the run fails closed and retains its diagnostics;
the publication report will not silently omit that seed.

Headline training uses the manuscript's shaped J. Initial-inventory centering
subtracts the constant S0*I0. A training-only telescoping potential redistributes
rewards without changing the full centered J return. Validation, checkpoint
selection and learned/AS/TWAP comparisons use economic risk-adjusted PnL.
The common calibration has q=0, while CLOB and signed interim auction shaping
remain active. The separate shaping-off treatments remove both shaping terms.
See `docs/rl_design.md` for the exact conditioning and replay equations.

DQN still selects from the exact masked action grids, but its Q function does
not give each of the 1,346 auction actions an unrelated output vector. A shared
two-layer state trunk scores normalized action coordinates through a rank-32
embedding. The replay update evaluates only its sampled action, while greedy
and Double-DQN target selection enumerate the complete admissible grid. This
coordinate-conditioned head shares evidence across nearby `(K,ell,c)` actions
and is recorded in checkpoints as `coordinate_conditioned_bilinear_v1`.

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
`results/revision_v17/` layout. For example:

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

The remaining H/anchor-bundle by shaping cell (`H` on, shaping on) is exactly the canonical
synthetic headline configuration. The treatment launcher therefore does not
train it again: paired reporting reuses `synthetic_rough_heston/*_seed<N>` as
that arm. Run the canonical synthetic launchers (or `reproduce_all.sh`) for the
same seed before aggregating treatments.

The shaping-on arms share exactly the same parameters as the headline.
Signed fictive submission credit need not equal eventual execution cash;
cancellation reverses only the original credits of schedules actually canceled.
Therefore shaped J and the economic criterion are different objectives even
with q=0. The matched shaping contrasts measure this distinction. The
`ablation_h_on_shaping_on.yaml` overlay remains an alias for the headline and
is not separately launched.

The no-cancellation treatment gives DQN 673 auction actions; the enabled
regime has 1,346. Continuous agents use two or three normalized auction
proposal coordinates respectively while keeping the same 18-feature network
input.

The no-auction treatment removes both terminal auction participation and every
auction-derived training input: it disables auction shaping and the projected
clearing feature in addition to terminating at auction open. It is therefore a
comparison against a policy that does not anticipate the modeled auction.

The canonical five-seed publication command covers the headline, historical,
and complete synthetic treatment matrix, then generates the focused report:

```bash
scripts/run_multiseed.sh --jobs 5 --threads-per-job 2
```

On a 15-core host, five workers with two numerical-library/Torch intra-op
threads and one Torch inter-op thread each leave some headroom. The launcher
also caps Apple Accelerate/vecLib and NumExpr, preventing hidden oversubscription;
the effective settings are saved in `runtime_versions.json`. Lower `--jobs` if memory is tighter. Workers write distinct run
directories. Report generation runs once, serially, after every worker finishes. A small development check
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

Open `results/revision_v17/_publication/index.html` after completion. It contains
four figures and three tables, with captions and methods. Figures are vector
PDF plus PNG; tables are LaTeX (`longtable`/`booktabs`) plus numeric CSV. The
report retains **mean** economic performance, including poor seeds, because
that is the expected-value estimand. It does not substitute an IQM or a maximum
validation score for held-out economic performance. Small plot markers expose
every seed; 95% intervals resample training-seed means, not individual test
episodes as independent training replications. Five seeds still limit precision.

The paired treatment figure/table covers H/anchor at both shaping levels,
shaping at both H/anchor levels, the full auction-aware versus bundled
no-auction comparator, and cancellation on minus off. The separate auction
mechanism figure derives the exact same-CLOB no-order counterfactual from
saved accounting, separating execution price edge, fees and inventory-risk
relief. It does not reinterpret the bundled no-auction treatment as a pure
auction-access experiment. Negative contributions are shown unchanged.

All headline comparisons use basis points of initial notional. Historical
markets are shown individually; the auction summary additionally uses the
fixed-ticker equal-weight mean within each seed. This is not a dollar forecast
or uncertainty over unseen historical dates/stocks. See
[`docs/research_outputs.md`](docs/research_outputs.md) for the full artifact map,
statistical protocol, and exact unapplied manuscript inclusion instructions.

Comprehensive legacy diagnostics remain opt-in:
`scripts/make_multiseed_outputs.sh --publication --diagnostics` and
`scripts/make_all_outputs.sh --seed 42 --diagnostics`. They are outside the
focused publication bundle. The old standalone paired treatment table remains
available via `scripts/make_treatment_outputs.sh --publication`.

Generated result artifacts are never copied into `paper/` automatically.
Promotion into the manuscript artifact directory remains an explicit manual
review step.

## Equivalent module commands

For a synthetic DQN run named `dqn_seed42`:

```bash
RUN=results/revision_v17/synthetic_rough_heston/dqn_seed42

python3 -m lmm.experiments.train \
  --config configs/base.yaml \
  --config configs/synthetic_rough_heston.yaml \
  --config configs/algo/dqn.yaml \
  --run-name dqn_seed42 --seed 42

python3 -m lmm.experiments.evaluate --run-dir "$RUN" --trace-episodes 1
python3 -m lmm.experiments.policy_differences --run-dir "$RUN" --benchmark as
python3 -m lmm.experiments.policy_differences --run-dir "$RUN" --benchmark twap
python3 -m lmm.experiments.publication --write-completion-manifest "$RUN"
# Optional per-run debugging, not the publication report:
# python3 -m lmm.experiments.make_figures --run-dir "$RUN"
# python3 -m lmm.experiments.make_tables --run-dir "$RUN"
```

The installed command names are `lmm-train`, `lmm-evaluate`,
`lmm-policy-differences`, `lmm-make-report`, `lmm-make-figures`, and `lmm-make-tables`.
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

The canonical launcher writes one report under
`results/revision_v17/_publication/`: `index.html`, `README.md`,
`figures/{economic_performance,learning,auction_mechanism,treatments}.{pdf,png}`,
`tables/{economic_performance,auction_mechanism,treatments}.{tex,csv}`,
`audit/` with seed-level estimates and validation series, and `manifest.json`
with input and output hashes. Per-run figures, critic-loss plots, cumulative
fixed-policy differences, and selected episode anatomy are not generated by
default. Their underlying raw data remain in each run directory.

`scripts/make_all_outputs.sh --seed N` now writes a focused, explicitly
nonpublication single-seed report under `_single_seedN/`; it requires all four
algorithms in each included market and omits confidence intervals.
Nonpublication multiseed reports use `_development/`. Only `--publication`
produces `_publication/` and applies the strict complete-matrix/configuration/
commit/checkpoint checks. Reports are read-only with respect to run artifacts
and completion manifests. Regeneration never executes an environment.

Primary evaluation fields are `pnl` and `risk_adjusted_pnl`. The latter is

```text
risk_adjusted_pnl = pnl - terminal_inventory_penalty
```

where `pnl` already includes cancellation fees. Headline training return is
centered shaped J. Its components and the telescoping replay adjustment are
retained as training diagnostics and never enter reported PnL or checkpoint
selection. Historical tables additionally use
`risk_adjusted_pnl_bps`.

The full field-level schema is in `docs/metrics_schema.md`.

## Compile the manuscript

The generated parameter-table input currently referenced by `paper/main.tex`
is tracked under `paper/results/`, so compilation does not depend on a local
experiment output tree. With a TeX distribution providing `latexmk` and the
packages listed in `paper/packages.tex`, build from a clean checkout with:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

LaTeX build intermediates are ignored by Git. To remove them while keeping
the compiled PDF, run:

```bash
latexmk -cd -c paper/main.tex
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
