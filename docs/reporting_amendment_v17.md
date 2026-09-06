# Reporting non-improving runs in revision v17

The full campaign trained all 220 policies. The no-auction DQN run at seed 123
completed 800 episodes, but the original initial-improvement gate rejected its
best mature checkpoint: economic validation 12.127631 versus initial 13.017725.
It therefore has a saved `best_mature.pt` and `selection_failure.yaml`, but no
`best.pt` or original completion manifest. The missing evaluation was previously
misdiagnosed by reporting as an old environment-contract artifact.

The reporting amendment includes **every run's best mature economic-validation
policy**, irrespective of improvement over initialization. This is a post-run
protocol change, prompted by the observed failure, and must be disclosed. It
does not establish that every learner improves or beats the reference policies.
Maturity thresholds, validation/test streams, training configurations and actual
early-stopping histories are preserved. Changing stopping histories would require
new training and is outside this amendment.

## Recovery and report generation

From the repository root with `.venv` activated:

```bash
python -m lmm.experiments.mature_reporting --root results/revision_v17
scripts/make_multiseed_outputs.sh --publication
```

The first command audits the complete canonical matrix under the queue lock.
For 219 completed runs it verifies the original SHA-256 completion inventories,
matching selection episodes/scores and exact model-state equivalence between
`best.pt` and `best_mature.pt`. ZIP timestamps and the outer archive root name
are serialization details; tensor or nested archive payload differences fail.
It reuses their existing evaluations without modifying any files in those runs.

For a completed non-improving run, it evaluates only the saved `best_mature.pt`
using the existing 100 held-out test paths, writes the ordinary paired reference
differences and traces, and binds its full original training tree plus new
evaluation tree in the root-level amendment. It does not create `best.pt`, edit
the original gate decision, change a resolved configuration, or retrain a seed.
Reinvocation verifies an existing amendment; an interrupted evaluation can be
continued only if its source/training receipt still matches. Unrelated existing
evaluation artifacts are never overwritten.

The amendment lives at
`results/revision_v17/_reporting_protocol/best_mature_v1.json`. Evaluation logs
and the recovery receipt are beside it. The original 219 completion manifests
and all 220 saved training Git SHAs remain unchanged.

The amended report verifies the saved training revision against executable
source/configuration files, allowing only the six reporting/provenance modules
listed in `mature_reporting.REPORTING_FILES` to differ. It then binds the exact
source snapshot used for this amendment. Subsequent executable-source changes
fail validation; committing these same audited files does not invalidate it.
This explicit amended provenance path is separate from the ordinary clean-HEAD
publication checks. Training and completed-run skipping still use their original
strict checks. Do not rerun the training launcher merely to produce the report.

If reporting code subsequently changes, an explicit re-audit is available with
`python -m lmm.experiments.mature_reporting --root results/revision_v17 --refresh-reporting-source`.
It rechecks all artifacts, preserves the previous source snapshot in the audit
history, and permits changes only to reporting modules. Changes to the evaluation
module or any training/simulator/configuration source remain prohibited; no
evaluation or training is rerun by this refresh.

## Outputs and manuscript disclosure

The standard four figures and three tables remain in
`results/revision_v17/_publication`. Its HTML and README prominently disclose the
post-run amendment and identify the non-improving policy. Additional audit files:

- `audit/checkpoint_selection.csv`: all 220 validation scores, initial scores,
  improvement flags, selected zero-based episodes and evaluated checkpoint names.
- `audit/reporting_protocol.json`: the full source and artifact binding.
- `audit/reporting_amendment.tex`: ready-to-include disclosure, with human-readable
  episode numbering (500 episodes for the failed DQN run).

Neither protected manuscript file is edited. In `paper/main.tex`, insert the
following immediately after the introductory paragraph of `Numerical Simulations`
(currently line 670), before `\subsection{Generative Stochastic Market Model}`
(currently line 674). Compile from `paper/`, consistent with its existing inputs:

```tex
We report the checkpoint maximizing economic validation performance among
checkpoints meeting the prescribed phase-specific update thresholds, including
policies that do not improve over initialization. We retain every training seed.
\input{../results/revision_v17/_publication/audit/reporting_amendment.tex}
```

Keep the distinction between training on shaped rewards and selecting/evaluating
on the economic objective. No parameter-table changes are needed for this
reporting amendment.
