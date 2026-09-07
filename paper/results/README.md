# Manuscript exhibits from the completed revised-clearing campaign

`paper/main.tex` and its seven figure assets now use the completed v20 campaign:
440 main runs plus the two separately saved 40-run follow-ups. Legacy v19
results are not mixed into these exhibits. Source experiment directories and
checkpoints were not changed, and no fitting, training or evaluation was run
while preparing this update.

The existing empirical table layouts are preserved:

| Manuscript exhibit | Source |
|---|---|
| Table 4, headline shortfalls | `results/revision_v20/_publication/audit/economic_by_seed.csv` |
| Table 5, forecast accuracy | All 240 headline runs' `eval/h_forecasts.csv` files, checked against their completion manifests |
| Table 6, five main contrasts | `results/revision_v20/_publication/tables/treatments.csv` |
| Table 6, two follow-ups | Each v20 follow-up's `_comparison/comparison.csv` |
| Figures 1, 2, 3, 5 and Figure 4's upper panels | The v20 publication report's five vector PDFs |
| Figure 4's lower panels | The v20 cash-flow and conditioned-economic comparison PDFs |

`refresh_manifest.json` maps every figure to its exact source and hash. The
`.pdf.txt` names are retained for compatibility with the supplied manuscript;
their contents are actual vector PDFs.

`forecast_summary.csv` and `forecast_by_algorithm_seed.csv` preserve the paper's
forecast estimand: average errors over pre-action decisions within each phase
and episode, then equally over episodes, algorithms and seeds. Exclude terminal
observations. Compute RMSE as the square root of the final average squared error.
The 240 runs contribute 100 episodes each. The input manifest records their
forecast-log hashes and the fit protocols used for the reported reliability
weights.

`shortfall_summary.csv` averages historical tickers within seed before computing
intervals. Shortfall is the negative of the saved economic score; interval
endpoints are negated and reversed from the same bootstrap calculation used
by the figures. Every seed is retained.

The prose now reflects the revised anchor effects, the 39/40 cash-flow wins,
the updated auction-credit range, DDPG's lowest synthetic mean without a clear
algorithm ranking, and the distinction between economic and ordinary-shortfall
performance. With the author's explicit confirmation, the mechanism description
now states that both lagged auction indications and settlement use the
volume-maximizing projection. Timing and the inherited opening signal are
unchanged.

Reproduce from the repository root with the existing environment:

```bash
.venv/bin/python paper/results/analyze.py
.venv/bin/python paper/results/refresh.py
latexmk -cd -pdf -outdir=build paper/main.tex
.venv/bin/python paper/results/verify.py
```

The refresh script updates the three numerical table bodies and seven figures;
prose changes were reviewed separately. `verification.json` records checks on
216 rounded numerical table values, figure hashes, selected qualitative claims,
publication source attestation, and LaTeX warnings. The final PDF is
`paper/build/main.pdf`. Generated build files and visual QA renders remain in
ignored `paper/build/`.
