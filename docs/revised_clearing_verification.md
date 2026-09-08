# Auction projection verification — 2026-09-07

The revised mechanism is implemented as `max_volume_v2`, schema 16. The base
and retained setting configs still select `nearest_tick_v1`, schema 15. Old
fitted weights/checkpoints cannot silently acquire revised semantics. No
retained result, fitted coefficient or paper number was changed or relabeled.

Implementation was developed from commit
`c79c905783bd489daa585036cdddaa3817b0afd4` (changes are uncommitted).
The supplied specification SHA-256 is
`912afde0dca509a64d7c8c8492f5ad93e40d6a7df7abdae202c308cebdee1c65`;
the supplied review SHA-256 is
`2965726f35fa64d5a9831cc87435354e68d49b37c1a4db28f81e8514cf16882b`.
Only the auction-projection request and finding 1 informed implementation;
unrelated document instructions/findings were not treated as additional tasks.

## Verification performed

- Final `.venv/bin/python -m pytest -m 'not network' -q`: **708 passed**
  in 322.07 seconds, including available historical-data and slow integration
  checks. Its 14 warnings concern legacy string escapes and deliberate
  missing-artifact/failing-report tests; there are no test failures.
- Integrated launcher/queue/reporting focused batch: **94 passed** in 58.57
  seconds. Independent publication-fit tests are also included in the full
  suite. Counts overlap and must not be added together.
- Independent exhaustive oracle: 150 linear, 150 capped-external and 150 smooth
  signed-external books; bounded tick-volume maxima and balanced pro-rata
  executions verified. Deterministic cases cover the review's
  `100.01 / 0.93 / 0.06` example, ties, zero/on-grid roots, market sides,
  aggregate strategic exposure, cancellation and indication/settlement timing.
- Native DQN/DDPG/TD3/SAC and SB3 checkpoint compatibility checked in both
  directions, including missing legacy mechanism/provenance fields. Retagged
  legacy outputs and mixed reports are rejected.
- Two actual forecast prerequisite jobs completed through the bounded queue
  under `/tmp/lmm-v2-launch-check`: synthetic and historical, two paths per
  split, standard model horizon, one computation thread per worker. Their
  manifests validate; production validation rejects these smoke-sized fits.
- One actual revised shell worker (`9007 dqn synthetic 1`) completed there:
  four training episodes with reduced smoke maturity requirements, initial and
  selected checkpoints, three matched evaluation paths for learned/initial/AS/
  TWAP policies, both paired benchmark reports and the pipeline completion
  manifest. Its metadata, trace/report loading and completion hashes validate.
  No smoke coefficients or policy outcomes were promoted into production.
- New tests cover forecast success/reuse, tampering, missing completion,
  failure before learning, legacy-root isolation, two-stage queue ordering,
  all four algorithms and all setting/control types, and canonical publication
  configuration checks against independently validated forecast coefficients.
- The exact user command with `--dry-run` resolves to 440 jobs, ten workers,
  one computation thread each, two forecast prerequisites and the separate
  `results/revision_v20` root. It performs no fitting/training.
- `git diff --check`, shell syntax checks and Python compilation passed.
  No network-marked tests are currently collected.

Earlier tiny-horizon forecast/reference checks and the one-episode direct
train/evaluate test also passed. The larger checks above exercise the new
launcher; all are implementation validation, not revised scientific results.
An existing report test was updated because synthetic reports now explicitly
name their clearing mechanism rather than having an empty scope statement.

## Ready but unrun

Run after reviewing/committing a clean worktree:

```bash
scripts/run_multiseed.sh --jobs 10 --threads-per-job 1
```

This now runs two training-only forecast fits, then the original **440-run main
matrix**, with per-run fresh reference calibration, training/reselection,
matched evaluation and one final research report under `results/revision_v20`.
The previous unexecuted 520-run manual plan and planner were removed. Its extra
80 follow-up runs are outside this command. The production root contains no
forecasts, trained policies or evaluation results at handoff.

Full 256-path forecast fits, production calibration, all 440 training/reselection
runs, matched 100-path final evaluations and any paper refresh remain unrun.
The user will start the campaign; the agent has not cleaned or committed the
worktree. Retained v19 results and manuscript numbers remain untouched.
See [the full workflow and compatibility decisions](revised_clearing_reruns.md)
and [the official-source market-practice assessment](auction_market_practice.md).

## Changed paths

- [README.md](../README.md)
- [REPRODUCING.md](../REPRODUCING.md)
- [configs/base.yaml](../configs/base.yaml)
- [configs/clearing/max_volume_v2.yaml](../configs/clearing/max_volume_v2.yaml)
- [docs/README.md](../docs/README.md)
- [docs/auction_market_practice.md](../docs/auction_market_practice.md)
- [docs/metrics_schema.md](../docs/metrics_schema.md)
- [docs/model.md](../docs/model.md)
- [docs/revised_clearing_reruns.md](../docs/revised_clearing_reruns.md)
- [docs/revised_clearing_verification.md](../docs/revised_clearing_verification.md)
- [scripts/_common.sh](../scripts/_common.sh)
- [scripts/diagnose_forecast_credit.py](../scripts/diagnose_forecast_credit.py)
- [scripts/refit_clearing_reference.py](../scripts/refit_clearing_reference.py)
- [scripts/run_multiseed.sh](../scripts/run_multiseed.sh)
- [src/lmm/agents/base.py](../src/lmm/agents/base.py)
- [src/lmm/agents/continuous_base.py](../src/lmm/agents/continuous_base.py)
- [src/lmm/agents/dqn.py](../src/lmm/agents/dqn.py)
- [src/lmm/agents/sb3.py](../src/lmm/agents/sb3.py)
- [src/lmm/config.py](../src/lmm/config.py)
- [src/lmm/env/mdp.py](../src/lmm/env/mdp.py)
- [src/lmm/experiments/clearing_campaign.py](../src/lmm/experiments/clearing_campaign.py)
- [src/lmm/experiments/diagnose_simulator.py](../src/lmm/experiments/diagnose_simulator.py)
- [src/lmm/experiments/evaluate.py](../src/lmm/experiments/evaluate.py)
- [src/lmm/experiments/make_report.py](../src/lmm/experiments/make_report.py)
- [src/lmm/experiments/plotting.py](../src/lmm/experiments/plotting.py)
- [src/lmm/experiments/policy_differences.py](../src/lmm/experiments/policy_differences.py)
- [src/lmm/experiments/protocol.py](../src/lmm/experiments/protocol.py)
- [src/lmm/experiments/publication.py](../src/lmm/experiments/publication.py)
- [src/lmm/experiments/run_matrix.py](../src/lmm/experiments/run_matrix.py)
- [src/lmm/experiments/tracing.py](../src/lmm/experiments/tracing.py)
- [src/lmm/experiments/train.py](../src/lmm/experiments/train.py)
- [src/lmm/market/clearing.py](../src/lmm/market/clearing.py)
- [src/lmm/utils/logging.py](../src/lmm/utils/logging.py)
- [tests/test_clearing_campaign.py](../tests/test_clearing_campaign.py)
- [tests/test_clearing_provenance.py](../tests/test_clearing_provenance.py)
- [tests/test_pipeline_launchers.py](../tests/test_pipeline_launchers.py)
- [tests/test_research_report.py](../tests/test_research_report.py)
- [tests/test_run_matrix.py](../tests/test_run_matrix.py)
- [tests/test_skeleton.py](../tests/test_skeleton.py)
- [tests/test_tick_projection.py](../tests/test_tick_projection.py)
