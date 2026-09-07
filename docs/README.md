# Research documentation

## Model and operation

- [Model specification](model.md)
- [Learning design](rl_design.md) and [continuous-control projection](continuous_action_extension.md)
- [Metrics schema](metrics_schema.md) and [research outputs](research_outputs.md)
- [Reproduction guide](../REPRODUCING.md) and [data setup](../data/README.md)

## Completed study

- [Main 440-run results](revision_v19_verdict.md)
- [Raw cash-flow comparison results](cashflow_results_verdict_v19.md)
- [Matched dense economic comparison results](economic_dense_h_results_verdict_v19.md)
- [Initial 440-run protocol](final_run_scope_v19.md)
- [Raw cash-flow follow-up protocol](economic_cashflow_comparison_v19.md)
- [Matched dense economic follow-up protocol](economic_dense_h_comparison_v19.md)
- [Structural repairs and limitations](pathology_repair_v19.md)

The study contains 520 completed runs. Protocol records describe decisions at
launch; the result summaries record the completed outcomes.

## Evidence and provenance

`analysis_v19/`, `analysis_cashflow_v19/` and `analysis_dense_h_v19/` contain
numeric summaries, analysis scripts and verification records. Scripts consuming
full run records require the separately retained research artifacts.

`verification_v19/` retains development ledgers, critic-calibration summaries
and forecast-calibration records. These are development evidence, not additional
final training seeds. `research_v19/` contains the supporting source ledger.

The [earlier cleanup record](cleanup.md) indexes archived development work.
The current source tree excludes market-data inputs, manuscript files and
private writing materials. Earlier commits remain in Git history.
