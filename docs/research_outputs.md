# Launch and research outputs

The full launcher trains the current v17 specification and then produces a
focused research report. Training uses the approved weighted shaped J;
checkpoint selection and final comparisons use the economic objective.
The production experiment has not been run in this output-refinement task.
Short launcher tests check operation, not the eventual scientific conclusions.

After reviewing/committing the changes and leaving the tree clean, run:

```bash
cd /Users/juliusgraf/Learning-Market-Making
source .venv/bin/activate
unset LMM_RESULTS_ROOT
scripts/run_multiseed.sh --jobs 10 --threads-per-job 1
```

Use the normal `results/revision_v17` namespace; unset `LMM_RESULTS_ROOT` first
if your shell has a development override. Ten workers each have one numerical
thread, a starting point for the current 15-core Apple M5 Pro with 24 GB RAM.
The queue assigns individual experiments across all seeds, algorithms, tickers
and treatments; a worker immediately takes another experiment when it finishes.
`--jobs` is no longer capped by the five master seeds. Lower it to eight if other
applications need resources. More workers are supported but do not guarantee
more throughput once CPU or memory bandwidth is saturated. Keep the commit/configuration
unchanged during the experiment. No API credentials or new quote download are
needed: historical runs consume the frozen repository inputs.

The matrix contains 220 learned-policy runs:

| Block | Runs |
|---|---:|
| Synthetic headline: four algorithms × five seeds | 20 |
| Historical headline: four algorithms × five tickers × five seeds | 100 |
| Five additional synthetic treatments × four algorithms × five seeds | 100 |

Seeds are 42, 7, 99, 123, 2024; tickers are MSFT, JPM, PG, GOOGL and CAT.
Each run has an 800-episode maximum, 100-path validation every 100 episodes,
and 100 test episodes. Early stopping can end training sooner. Initial
validation, feature fitting and benchmark calibration are additional work.
The short queue smoke checks verify scheduling and complete output generation;
they do not establish steady-state speedup for 800-episode training.

The headline supplies the H-on/shaping-on factorial cell, so it is not trained
twice. Representation controls remain optional development experiments, outside
this publication matrix. DDPG/TD3 retain their declared phase/asinh preprocessing;
DQN/SAC retain their validated preprocessing. The comparison includes those
implementation choices.

The launcher requires a clean commit. It skips an existing run only after
exact configuration and completion-manifest validation. Partial, drifted or
nonreportable runs stop the pipeline with their path. Retain/move a partial
run for diagnosis and restart the command; there is no automatic partial
checkpoint resume. A failed training seed is not dropped or replaced by a
hand-picked seed. Failures stop new dispatch; already running jobs finish.
Ctrl-C terminates active worker process groups. A kernel lock prevents two
launchers writing the same results root, including during final reporting. If any run has no mature checkpoint improving on its initial
economic validation, the full publication report will not be generated.

## Local concurrency check

The queue was checked on the current 15-core/24-GB Mac using the same 56-job
smoke matrix with four training episodes per job and three evaluation episodes:

| Resource configuration | Training/evaluation queue | Through final report |
|---|---:|---:|
| 5 workers × 2 threads | 117 seconds | 122 seconds |
| 10 workers × 1 thread | 68 seconds | 73 seconds |

This single short comparison was about 40% shorter with ten workers. It includes
startup, small replay buffers and output overhead, so it is not a measured
speedup for 800-episode training. Both matrices completed, and all 112 completion
manifests remained valid after reporting. Queue tests cover more than five
simultaneous jobs, immediate refill, failed-job dispatch stoppage, interruption
cleanup, exclusive results-root locking, and one final report. Evidence is in
`docs/verification_outputs/queue_verification.json`.

## What to open

Start with **`results/revision_v17/_publication/index.html`**. It is a local,
static report with all figures, interpretation, and download links. The same
material is in `README.md`. The output bundle is deliberately small:

| Artifact basename | Question it answers | Formats |
|---|---|---|
| `economic_performance` | How do DQN, DDPG, TD3 and SAC compare with AS/TWAP on the economic objective in synthetic data and each historical ticker? The table also shows net PnL and paired objective differences against DQN and both references. | Figure PDF/PNG; table TeX/CSV |
| `learning` | Does economic validation improve during shaped training? Are there late regressions or seed instability? Which checkpoint was actually selected? | Figure PDF/PNG |
| `auction_mechanism` | How much does auction participation add with the CLOB policy fixed? How much is execution price edge net of fees versus inventory-risk relief? How much inventory reaches the auction and remains afterward? | Figure PDF/PNG; table TeX/CSV |
| `treatments` | What changes when shaping, the H/anchor bundle or cancellation is switched? How does the full model compare with the bundled no-auction specification? | Figure PDF/PNG; table TeX/CSV |

Figures live in `figures/`, tables in `tables/`. Colorblind-safe colors identify
algorithms consistently; figures use dots/intervals or learning curves, readable
small multiples, zero reference lines and vector PDF text. No arbitrarily
ordered cumulative episode plots, reward-composition pies, per-run histograms,
or critic-loss grids enter the default bundle. Raw data are retained.

All comparisons use basis points of initial notional. This permits a common
interpretation across the normalized markets without claiming empirical dollar
profits. The primary figure shows every ticker separately, retaining mixed
rankings. Net PnL includes cancellation fees. The economic objective subtracts
terminal inventory penalty; neither includes training shaping.

## Uncertainty and interpretation

The primary estimand is the **mean** test outcome across episodes, then across
training seeds. An IQM would change that expected-value estimand and discard
some extreme seed outcomes. Each training seed receives equal weight. Tiny
points show every seed; larger points and lines show means with deterministic
95% percentile bootstrap intervals (10,000 resamples, bootstrap seed zero).
One-seed development reports display point estimates without an interval.

References are checked for identical economic outcomes and CRN keys across
algorithm runs and are counted once per seed. Comparisons pair both episode
number and environment seed before forming within-seed differences. Treatment
contrasts use the same pairing. Cross-ticker auction means average the fixed
reported ticker set within each seed and resample whole seed blocks, preserving
cross-ticker dependence. There is no claim of population uncertainty over new
stocks, new dates, or all market regimes. Five training seeds provide limited
precision; intervals are pointwise and exploratory, with no multiple-testing
significance stars. Do not infer pairwise significance from overlapping or
non-overlapping separate-policy intervals; use the paired-difference columns.

Learning plots use **validation**, never test results or the maximum observed
validation as a final outcome. Individual seed traces and selected-checkpoint
stars are retained. The aggregate mean/band only uses checkpoints observed for
every requested seed; it stops at the end of common support. There is no
forward-fill, smoothing or best-so-far envelope to conceal late regression.
The horizontal coordinate counts completed training episodes, with the initial
untrained policy at zero.

For the auction, let `I_open = I0 - clob_exec_qty`, let `Z` be the actual signed
auction execution, and let `P_open` be the frozen residual-inventory mark. The
report computes the exact objective difference against no auction orders on
the same CLOB trajectory:

```text
auction value = Z * (S_cl - P_open) - cancellation fees
                + lambda * (I_open**2 - I_final**2)
```

Actual signed settlement is preserved, including negative final inventory and
negative contributions. The no-order counterfactual retains inventory at the
same policy-independent mark and incurs no auction cancellation fees. It does
not require another simulation. Inventory plots show **mean absolute** exposure,
so short and long positions cannot cancel in the average. The bundled
no-auction treatment separately changes training and observables; it must not
be described as isolating auction participation alone.

These reporting choices use the emphasis on uncertainty and raw run variability
in [Agarwal et al., NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
and the experimental-design guidance of
[Patterson et al., JMLR 2024](https://www.jmlr.org/papers/v25/23-0183.html).
A broad benchmark-suite performance-profile plot is not needed for this
focused economic comparison.

## Audit and regeneration

Within `_publication/audit/`:

- `economic_by_seed.csv`: each policy's seed mean, paired DQN/benchmark/initial
  differences, auction components, residual inventory and short frequency.
- `validation_by_seed.csv`: each observed validation point and selected flag.
- `validation_common_support.csv`: the exact points and intervals on the heavy
  learning curves, with seed counts.
- `treatments_by_seed.csv`: each paired treatment estimate, run paths and digest
  of the common evaluation seed vector.

`manifest.json` records all contributing runs, key input hashes, the report
implementation hash and generated output hashes. Publication generation also
validates the existing run completion manifests, including checkpoint contents,
exact configs and clean Git commit. Older result namespaces are never mixed in.
No result or checkpoint is re-evaluated by output generation.

The raw run directories are unchanged:

```text
results/revision_v17/synthetic_rough_heston/<algo>_seed<seed>/
results/revision_v17/historical_sp500_midquotes/<algo>_<ticker>_seed<seed>/
results/revision_v17/synthetic_rough_heston__<arm>/<algo>__<arm>_seed<seed>/
```

Each contains resolved configuration/seed/commit/runtime provenance,
`metrics.csv`, normalizer and split-seed state, `checkpoints/best.pt` with
selection sidecars, initial/final checkpoints, `eval/records.csv`, economic
metadata, traces/action/forecast diagnostics, paired reference differences,
and `pipeline_complete.json`. The launcher console shows starts/completions. Detailed training progress is
in each run's `logs/run.log`. `_orchestration/status.json` records queued,
active, completed and failed job identities, elapsed time and the concurrency
limit; `_orchestration/<block>__<algo>_seed<seed>.log` captures each full pipeline.
The results-root lock file may remain after completion; kernel locking, not
its mere existence, determines whether another launcher is active. Periodic replay-heavy resume snapshots are suppressed by the full
launcher; selected and final policies remain available.

Inspect the full 220-job command plan without training (works before committing):

```bash
scripts/run_multiseed.sh --jobs 10 --threads-per-job 1 --dry-run
```

Regenerate the focused report, without training:

```bash
scripts/make_multiseed_outputs.sh --publication
```

For optional detailed debugging only:

```bash
scripts/make_all_outputs.sh --seed 42 --diagnostics
scripts/make_multiseed_outputs.sh --publication --diagnostics
```

Those comprehensive legacy outputs remain outside `_publication`. Without
`--diagnostics`, `make_all_outputs.sh --seed N` now produces a focused
single-seed development report under `_single_seedN/`. A nonpublication
multiseed report is under `_development/` and clearly labelled as such.

## Exact manuscript recommendations, not applied

`paper/main.tex` and `paper/results/tables_params/params_generative.tex` remain
untouched. The existing calibration/reward recommendations in
`docs/manuscript_recommendations_v17.patch` remain separate from output inclusion.
Do not treat the old parameter table as the active run specification.

The current manuscript has no result figure inclusions to replace. After
reviewing the full report, copy its `figures/` and `tables/` directories to
`paper/results/research_report/`. In `paper/main.tex`, insert
`\usepackage{longtable}` immediately after `\input{packages}`. Insert the
following immediately before `\bibliographystyle{plain}` (after the existing
`\newpage`). The captions below deliberately contain no numerical conclusions
until the production results have been read:

```tex
\section{Learning and Economic Evaluation}
We train the headline policies on the weighted shaped objective $J$ and
select mature checkpoints using economic validation performance. Held-out
comparisons report economic risk-adjusted PnL in basis points of initial
notional. We average episodes within each training seed and then average
seeds equally. Intervals are pointwise 95\% percentile bootstrap intervals
of seed means; five seeds limit inferential precision. AS and TWAP are
stylized references. The algorithm implementations include the declared
preprocessing choices. No test outcomes enter checkpoint selection.

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{results/research_report/figures/economic_performance.pdf}
\caption{Held-out economic performance. Large points and intervals summarize
training-seed means; small points show every seed. Historical tickers are
reported separately.}
\label{fig:economic_performance}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{results/research_report/figures/learning.pdf}
\caption{Economic validation during shaped training. Thin lines show seeds;
stars identify selected checkpoints. Aggregate means and pointwise intervals
stop when common observed checkpoint support ends.}
\label{fig:economic_learning}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{results/research_report/figures/auction_mechanism.pdf}
\caption{Auction contribution relative to no auction orders with the CLOB
trajectory fixed. Total value combines execution price edge, cancellation
fees and inventory-risk relief. Inventory panels show mean absolute exposure.
Historical summaries average the fixed ticker set within each seed.}
\label{fig:auction_mechanism}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{results/research_report/figures/treatments.pdf}
\caption{Paired synthetic treatment effects. Positive values favor the first
condition. H changes the observed signal and action anchoring. The full versus
no-auction contrast bundles auction access, H/anchor and shaping. Intervals
are pointwise, with no familywise significance claim.}
\label{fig:treatments}
\end{figure}

\clearpage
\input{results/research_report/tables/economic_performance}
\input{results/research_report/tables/auction_mechanism}
\input{results/research_report/tables/treatments}
```

The three tables are multipage `longtable` environments; do not wrap them in
`table` floats. They can instead be placed in an appendix if journal page limits
favor the four figures in the main text. For the protected generative-parameter
table, use the exact numerical recommendations in the existing v17 patch;
output generation does not require modifying that file.
