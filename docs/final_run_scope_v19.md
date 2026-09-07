# Final v19 scope decision — September 6, 2026

The final matrix is **440 runs**, reduced from the initial 520-run plan before
launching production. The CLOB-only and auction-only reward-preference arms
are optional diagnostics. Each would have required four algorithms × ten
seeds = 40 runs. Removing both saves 80 runs, or 15.4% of scheduled experiments.
Actual elapsed-time savings depend on each arm's early stopping and throughput.
No training, selection, simulator or forecast parameters change in this scope
revision. Headline training remains weighted shaped J; selection/test use
Jbar_lambda.

| Retained block | Runs | Purpose |
|---|---:|---|
| Synthetic headline | 40 | Compare all four learned methods and the two references |
| Five historical headline markets | 200 | Same comparison across the fixed price-path settings |
| Fixed-anchor full J | 40 | Separate quote anchoring from forecast information; provide the matched full-shaping control |
| Economic dense | 40 | Shared comparator for H information, credit and combined preferences |
| Economic sparse | 40 | Test the thirty-decision auction credit-assignment motivation |
| Economic dense, H feature off | 40 | Isolate H observation at a fixed anchor; provide the auction-access comparator |
| No auction | 40 | Test auction access under matched H-off/economic training |
| Total | 440 | Ten seeds and all four algorithms in every retained cell |

The five reported synthetic effects are dense minus sparse credit, H feature
on minus off, indicative minus fixed-mid anchor, combined manuscript
preferences on minus off, and auction access on minus off. The existing
same-CLOB auction decomposition needs no additional training. All five figure
groups and three tables remain; the treatment figure now has five contrasts.

The omitted controls answered a secondary decomposition question: which
manuscript reward term drives a combined effect? Their bounded development
results remain in the complete evidence ledger, including negative findings.
The production campaign cannot attribute a combined effect to one term alone.
The two YAML configurations remain available for a separately disclosed
follow-up, not as selectively reported members of the final balanced matrix.

Keep ten seeds: the development evidence already shows meaningful variation
across seeds. Cutting independent replications to preserve secondary ablations
would weaken stability and performance comparisons. This priority is a
project-specific judgment consistent with the attention to variability and
experimental scope in [Patterson et al., JMLR 2024](https://www.jmlr.org/papers/v25/23-0183.html).
The number 440 is not a research standard. A smaller matrix would be valid
with narrower claims, for example limiting mechanism studies to two declared
algorithms, but would no longer characterize those effects for all four.

## Readiness decision

The implementation is ready for a frozen final evaluation campaign after the
reviewed changes are committed and the working tree is clean. The prior
595-test offline suite, 43 bounded learning runs and complete smoke pipeline
provide the operational and numerical evidence. The reduced matrix has its
own launcher/report tests, dry-run manifest and regenerated report from the
retained 56 jobs of the earlier 72-job development smoke. No new training was
needed for this reduction.

This is not a claim that every performance pathology has disappeared. DDPG's
identified late critic-exploitation regression is substantially mitigated in
paired checks, not ruled out for every future seed. Small negative synthetic
auction contributions and mixed economic rankings remain. The choice to
retain shaped J permits an economic cost from its preferences. These outcomes
must be shown rather than treated as bugs whenever they occur. The previously
inspected historical week remains a reused holdout; the new campaign is not
independent evidence on new historical dates.

Do not continue adjusting hyperparameters simply to make every treatment
positive. Freeze this protocol, retain every seed, and use the final results
to determine which claims the paper supports.

```bash
# Run from the repository root.
source .venv/bin/activate
unset LMM_RESULTS_ROOT
scripts/run_multiseed.sh --jobs 10 --threads-per-job 1
```

Outputs remain `results/revision_v19/_publication/index.html`. Verification records are
`docs/verification_v19/reduced_matrix_tests_pass.log`,
`reduced_production_dry_run.json`, `reduced_report.log` and
`reduced_matrix_checks.json`.
