# Repository guidance

## Authoritative sources

The active executable contract is the resolved composition of:

1. `configs/base.yaml`;
2. one setting overlay (`configs/synthetic_rough_heston.yaml` or
   `configs/historical_sp500_midquotes.yaml`);
3. one learner overlay in `configs/algo/`;
4. any explicitly requested treatment overlay in `configs/treatment/`.

Do not copy version numbers, action counts, episode budgets, or reward switches
into this file. Read them from the resolved configuration so this guidance
cannot become a competing specification. The current result namespace and
artifact schema are likewise defined by `experiment.results_root` and
`experiment.artifact_schema_version` in `configs/base.yaml`.

`paper/main.tex` is the mathematical specification. Do not edit it unless the
user explicitly requests a manuscript change. `docs/rl_design.md` explains
the learner representation, and `REPRODUCING.md` is the operational guide.
Files under `legacy/` and `audit/` document superseded implementations and are
not implementation specifications.

## Non-negotiable contracts

- Synthetic and historical settings share the simulator, action, reward,
  learner, and evaluation configuration. Setting overlays select the
  exogenous mid-price source, run identity and training-only forecast
  reliability coefficients.
- A policy observes only current state. Future random grid points and future
  arrivals remain simulator-private.
- The strategic CLOB order is one-sided, lives for one realized interval, and
  never enters residual carry-over.
- The auction processes current exogenous proposals before revealing the
  state. The indicative price is lagged, the final agent action participates
  in clearing, and no exogenous event follows it.
- Multiple strategic auction schedules may remain live. Cancel-all removes
  all schedules submitted strictly before the current action and never removes
  the replacement submitted by that same action.
- The manuscript auction coordinates use `B_inf` for the absolute admissible
  offset of executed `b` and `B_max` for the local policy coordinate `ell`.
  Config and documentation must preserve those distinct roles.
- H-off treatments use the `frozen_mid` auction anchor so masks, projection,
  and execution cannot leak the ablated indicative signal. H-visible controls
  may use either anchor to distinguish forecast information from quote anchoring.
- Training uses the reward selected by the resolved configuration: the
  headline uses the author-approved weighted, centered shaped J. The exact
  unapplied manuscript edits are in docs/manuscript_recommendations_v19.patch,
  with current integration guidance in docs/research_writeup_plan_v19.md.
  Shaping-off specifications are explicit treatments. The telescoping
  potential and frozen exogenous-price reference preserve policy differences;
  neither enters reported training J or economic PnL. Validation, checkpoint
  selection, final evaluation, and AS/TWAP comparisons always use economic
  risk-adjusted PnL. Initial-inventory centering is policy invariant and must
  never enter reported PnL twice.
- PnL fields never include shaping. Clearing and accounting use actual signed
  pro-rata fills, with no hidden auction inventory clipping.
- Both phases use the common 18-coordinate feature map documented in
  `docs/rl_design.md`; phase-inactive coordinates are zero.
- Feature-normalization statistics are fit on training paths, stored with the
  checkpoint, and reused unchanged for validation and test.
- Result readers must reject incompatible artifact schemas and environment
  contracts. Never silently reuse old checkpoints, tables, figures, or
  metrics.

## Engineering workflow

- Python 3.10+, `src/` package layout, YAML overlays.
- Install from `pyproject.toml` with `python -m pip install -e ".[dev]"`.
  `requirements.txt` delegates to that canonical metadata.
- Run `pytest -q tests/test_revision_acceptance.py` before long experiments,
  then `pytest -q -m 'not network and not slow'`.
- Treat the frozen historical CSV and its sidecar as version-controlled inputs.
  Provider-side regeneration is optional provenance work, not a prerequisite
  for running a clean checkout; tracked raw archives support digest validation.
- Preserve unrelated changes. Generate figures and tables from saved outputs;
  output generation must not step an environment.
- The completed study and its follow-ups are the basis for writing. Keep
  diagnostic output under `results/`; retain only compact evidence in `docs/`.
  Superseded development files are indexed with exact recovery hashes in
  `docs/cleanup_manifest.json`; do not restore them into the active workflow
  merely to follow an archived instruction.
