# Replace the synthetic portion of v20 at 0.25 minutes

This is the historical launch recipe for the completed 320-run synthetic
replacement, using a maximum internal step of **0.25 minutes**. It requires the
pre-existing full v20 result tree and is not a command for rebuilding the study
from the compact evidence bundle. Preparation-time checks below predate the
actual launch; completed records are in `release/evidence/campaign.json`.
The selected step is not an established accuracy threshold.

Commit the prepared changes so the worktree is clean, then run from the repository:

```sh
scripts/run_multiseed.sh --replace-synthetic --jobs 10 --threads-per-job 1
```

The launcher selects `.venv/bin/python` automatically. All files required for
this launch, including the pre-replacement inventory and configuration, belong
in that commit. Do not discard the prepared changes when cleaning the worktree.
No manual removal of existing results is needed.

Optional read-only preview (works before committing):

```sh
scripts/run_multiseed.sh --replace-synthetic --jobs 10 --threads-per-job 1 --dry-run
```

## Exact scope

The existing 520-run evidence consists of a 440-run main matrix and two 40-run
follow-ups. The normal `run_multiseed.sh` matrix by itself is 440 runs.

| Existing root | Replaced synthetic runs | Retained historical runs |
|---|---:|---:|
| `results/revision_v20` | 240: headline plus five treatment arms | 200: five tickers × four algorithms × ten seeds |
| `results/revision_v20_cashflow` | 40 | 0 |
| `results/revision_v20_economic_dense_h` | 40 | 0 |
| Total | **320** | **200** |

All original run names, seed identities, treatment definitions, maximum training
budgets, validation/selection rules, and evaluation budgets are retained. The
synthetic price mesh and its dependent fitted quantities change. No v21 root is
created. The main report continues to contain 440 runs, while the two follow-up
comparisons account for the remaining 80.

## What the command does

1. Require a clean worktree and lock all three existing v20 roots. Check the
   saved 520-run inventory against the checked-in baseline and verify every
   retained historical file, including logs, against its recorded directory hash.
2. Move the eight synthetic setting directories, old synthetic forecast fit,
   synthetic orchestration logs, and obsolete combined reports into
   `_superseded/synthetic_refinement_v1/` **inside their existing roots**.
   Historical run directories, their fit, and historical worker logs stay in
   place. Archive moves are journaled and restartable. They retain the original
   results for auditing and do not copy their large checkpoint files.
3. Install the versioned campaign overlay from
   `configs/campaign/v20_synthetic_refinement.yaml` in each affected root as
   `_provenance/synthetic_refinement.yaml`. It selects 0.25 minutes and clears
   the legacy synthetic forecast coefficients before fitting.
4. Fit one new synthetic forecast on 256 training paths; the separate 256
   validation paths remain diagnostic-only. The historical forecast is retained
   without refitting. Each synthetic worker applies the mesh overlay **before**
   the new forecast overlay.
5. Retrain all 320 synthetic policies with ten concurrent workers and one native
   numerical/Torch thread per worker. Each fresh training job refits its own
   training-only feature normalizer and inventory reference. Policies are newly
   selected and evaluated, and paired differences and completion manifests are
   regenerated. Replay-heavy periodic resume checkpoints remain disabled, as in
   the original multiseed command.
6. Only after every job succeeds, rebuild:
   - `results/revision_v20/_publication/index.html` and all its tables, figures,
     audit CSVs and manifest;
   - `results/revision_v20_cashflow/_comparison/`;
   - `results/revision_v20_economic_dense_h/_comparison/`.
7. Verify retained historical bytes again and mark the replacement complete.

This is a **retrained-policy campaign**, distinct from the previous fixed-policy
numerical sensitivity diagnostic. Synthetic forecasts, input transformations,
reference conditioning and policy selection are all refreshed. Historical
results retain the original source revisions and files. The combined report
explicitly discloses these different source histories rather than claiming that
all runs used the new commit.

The byte-bound pre-replacement inventory is
`docs/rough_heston_refinement/v20_replacement_baseline.json`. The original source
attestation is retained with the archived report. The active report uses a new
`audit/retained_history.json` record and lists the source identity of every run.
Historical config/completion/selection checks remain active; retention is not a
blanket bypass for older checkpoints or altered files.

## Logs, interruption, and storage

The queue status and worker logs are under:

```text
results/revision_v20/_orchestration/synthetic_replacement/
```

The restart journal is:

```text
results/revision_v20/_provenance/synthetic_replacement.json
```

If a worker fails, dispatch stops and active workers drain; reports are not
regenerated from an incomplete matrix. Relaunch the same command from the same
clean source revision. Completed refined pipelines must pass the existing
exact-config/source/completion checks before being skipped. Incomplete new
attempts and partial forecast fits are moved into timestamped archive folders
before a fresh attempt. A corrupt completed pipeline fails validation instead
of being silently reused or erased.

The archives retain the original synthetic checkpoints, so the replacement
requires storage for both old and new synthetic artifacts until the user decides
to remove the archives. Historical artifacts are never duplicated or regenerated.
The journal records original-to-archive paths for interpreting old absolute
paths in the archived manifests.

The command refreshes generated v20 research reports and comparisons. The
manuscript is maintained separately and is not modified by this workflow.
Its claims must be checked against the matching completed campaign records.

## Preparation checks

- Dry-run inventory: 320 synthetic runs, 200 retained historical runs, 0.25-minute
  maximum step, ten workers and one thread per worker.
- All 200 actual historical runs passed publication configuration, selection,
  completion, original-source and full-directory retention validation (15.78 s).
- Replacement tests exercise archive recovery, source/config mismatch rejection,
  historical file/forecast protection, incomplete-attempt handling, config order
  for all eight synthetic arms, actual forecast smoke fitting, and report dispatch
  only after successful training. Mocked orchestration exercises the real
  follow-up report CLI parser without launching training.
- The forecast smoke tests use temporary directories and two paths per split;
  they do not replace the real campaign fit.

- Final regression suite: **736 passed, 34 deselected**, 14 expected fixture/legacy
  warnings (208.73 s). Bash syntax checks, Ruff for the new modules/tests, and
  `git diff --check` passed.
- All 320 planned config stacks were compared with their saved counterparts:
  controls, hyperparameters, training/evaluation budgets and seeds are unchanged;
  the mesh and dependent forecast identity change, followed by refitting weights.
- Final filesystem checks confirmed that no replacement journal or archive has
  been created in any real v20 root.
