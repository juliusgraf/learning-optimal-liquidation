# Research documentation

## Findings and writing

- [Research narrative and manuscript plan](research_writeup_plan_v19.md)
- [Proposed manuscript prose](manuscript_drafts/research_framing_v19.tex)
- [Main study verdict](revision_v19_verdict.md)
- [Cash-flow comparison verdict](cashflow_results_verdict_v19.md)
- [Matched dense economic comparison verdict](economic_dense_h_results_verdict_v19.md)
- [Referee requirements and ablation audit](referee_ablation_audit_v19.md)
- [Pending mathematical and parameter alignment](manuscript_recommendations_v19.patch)

## Model and protocol

- [Learning design](rl_design.md) and [continuous-control projection](continuous_action_extension.md)
- [Metrics schema](metrics_schema.md) and [research output guide](research_outputs.md)
- [Initial 440-run scope](final_run_scope_v19.md)
- [Raw cash-flow follow-up](economic_cashflow_comparison_v19.md)
- [Matched dense economic follow-up](economic_dense_h_comparison_v19.md)
- [V19 structural repairs and limitations](pathology_repair_v19.md)
- [Installation, reproduction and verification](../REPRODUCING.md)

The initial scope and follow-up protocols record the decisions at their respective
launch dates. The study now comprises all 520 completed runs; the writing plan
supersedes prospective language in those protocol records.

## Evidence and provenance

`analysis_v19/`, `analysis_cashflow_v19/` and `analysis_dense_h_v19/` contain
analysis scripts, numeric summaries and verification records for the completed
study. `research_v19/` retains the supporting research source ledger.

`verification_v19/` keeps the compact development ledgers, critic-calibration
summary and both forecast-calibration records. These are development evidence,
not additional final seeds. Per-trial files and earlier revisions are recoverable
from the exact Git revision recorded in [cleanup.md](cleanup.md). This archival
rule applies to successful and rejected trials alike.
