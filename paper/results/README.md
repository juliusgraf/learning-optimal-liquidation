# Manuscript exhibits from the synthetic v20 replacement

`paper/main.tex`, its four empirical tables and seven figure assets use the
completed replacement v20 bundle: 320 synthetic runs at maximum internal
rough-Heston step 0.25 minutes and 200 unchanged historical runs. The synthetic
forecast fit, normalization, reference conditioning and policies were refitted
or retrained in that campaign. No fitting, training or evaluation was performed
while refreshing the paper. The selected mesh has no established accuracy threshold.

[The before/after comparison](refinement_comparison.md) compares the archived
pre-replacement report with the latest results; its CSV preserves 260 means and
interval endpoints. It is a comparison of recalibrated, retrained campaigns,
not a fixed-policy numerical sensitivity experiment. Identical seed labels do
not establish coarse/fine Brownian coupling. Historical report table values
agree exactly, and all 200 historical run directories plus their forecast fit
are checked against the original retained-history inventory.

Main changes from the previous paper:

- SAC replaces DDPG as the lowest synthetic mean penalized shortfall; paired
  learner comparisons still do not establish a ranking.
- Auction-access intervals are positive for all four learners, including DQN.
- Explicit H access worsens DQN; the former DDPG benefit is no longer established.
- Indicative anchoring worsens DDPG, TD3 and SAC; DQN is uncertain.
- Indicative-anchor preferences worsen SAC; all frozen-anchor effects remain uncertain.
- Dense credit still helps all learners (0.73–2.08 bps). The broad cash-flow
  comparison still wins 39/40 seed pairs (3.53–5.16 bps on average), but the
  exception is now DQN seed 314, losing 0.15 bps.
- Synthetic TD3 replaces DQN as the only uncertain realized auction contribution.
- Forecast errors and synthetic reliability weights were refreshed. The first
  historical weight was also corrected from 0.193 to 0.192: the unchanged saved
  coefficient is 0.1924916, so this is a rounding correction, not a new fit.

The existing empirical table layouts are preserved:

| Manuscript exhibit | Source |
|---|---|
| Table 4, headline shortfalls | `results/revision_v20/_publication/audit/economic_by_seed.csv` |
| Table 6, forecast accuracy | 240 headline runs' `eval/h_forecasts.csv`, checked against completion manifests |
| Table 7, five main contrasts | `results/revision_v20/_publication/tables/treatments.csv` |
| Table 7, two follow-ups | Each v20 follow-up's `_comparison/comparison.csv` |
| Table 5, paired benchmark effects and ordinary shortfalls | Unrounded `_publication/audit/economic_by_seed.csv` |
| Figures 1, 2, 3, 5 and Figure 4's upper panels | Main publication report's five vector PDFs |
| Figure 4's lower panels | Cash-flow and conditioned-economic comparison PDFs |

`refresh_manifest.json` binds figures and table inputs to exact source hashes.
The `.pdf.txt` names retain the supplied manuscript convention; contents are
actual vector PDFs. `analysis_inputs.json` binds forecast logs and fit protocols.
`comparison_inputs.json` binds the archived and latest comparison evidence.

The forecast estimand averages errors over pre-action decisions within each
phase/episode, then equally over episodes, algorithms and seeds; it excludes
terminal observations. RMSE is the square root of the final mean squared error.
Each of the 240 headline runs contributes 100 episodes. Historical shortfall
summaries average tickers within seed before constructing intervals. Shortfall
negates the saved economic score and reverses interval endpoints. Every seed
is retained. Pointwise intervals are not corrected for multiple comparisons.

Reproduce from the repository root:

```bash
.venv/bin/python paper/results/analyze.py
.venv/bin/python paper/results/refresh.py
.venv/bin/python paper/results/compare.py
.venv/bin/python paper/results/supplement.py
latexmk -cd -pdf -outdir=build paper/main.tex
cp paper/build/main.pdf paper/main.pdf
.venv/bin/python paper/results/verify.py
```

The refresh script updates numerical table bodies and figures. Prose is reviewed
separately; `verify.py` deliberately checks the current qualitative findings so
a subsequent change in their signs forces another prose review. Verification
checks 450 numerical values across four main-body tables, seven figure hashes, all 320 synthetic completion
manifests and mesh/source identities, all retained historical directory hashes,
publication and follow-up input manifests, selected empirical claims, and LaTeX
warnings. It records results in `verification.json`. The verified PDF is copied
to the tracked `paper/main.pdf`; build files and visual QA renders stay in
ignored `paper/build/`.

Table 5 in the results section displays all 48 learner/reference/market penalized-shortfall comparisons
and 30 market-specific ordinary-shortfall levels (four learners plus AS in six
markets). Paired differences are formed within master seed before bootstrapping;
policy-level intervals are never substituted for paired-difference intervals.
The generator uses unrounded saved seed means, checks pairing against the saved
episode-paired gaps, and rounds only the LaTeX display. `benchmark_supplement.csv`
retains the unrounded estimates and `supplement_manifest.json` binds their input,
generator and output hashes. Ordinary-shortfall intervals reverse the original
PnL interval endpoints, following the existing publication convention. The
verification independently recomputes paired intervals from the saved gap columns
and checks all four MSFT ordinary-shortfall intervals. Intervals are pointwise;
48 positive intervals do not imply simultaneous 95% coverage across comparisons.
