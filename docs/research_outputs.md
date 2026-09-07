# Launch and research outputs (v19)

Headline training retains the author-approved weighted shaped objective
`J_omega`, with `omega=.0001`, `q=0` and both manuscript shaping terms enabled.
Selection and all economic comparisons use `Jbar_lambda`. V19 adds a calibrated
CLOB forecast, critic LayerNorm for native SB3 DDPG and mechanism-specific
controls. The completed evidence is retained under `results/revision_v19/`.

The main campaign and both economic-training follow-ups are complete. The table
below records the original 440-run scope; each follow-up adds 40 synthetic runs.
Use [the completed-study writing plan](research_writeup_plan_v19.md) for the
current interpretation and [the reproduction guide](../REPRODUCING.md) for
report-only commands and source-identity requirements. No further training is
needed to use these outputs.

| Block | Runs |
|---|---:|
| Synthetic headline: four algorithms × ten seeds | 40 |
| Historical headline: four algorithms × five tickers × ten seeds | 200 |
| Five additional synthetic cells × four algorithms × ten seeds | 200 |
| Total | 440 |

The seeds are 42, 7, 99, 123, 2024, 314, 577, 811, 1618 and 2718. Each run
has an 800-episode cap, economic validation on 128 paths every 50 episodes,
and the existing maturity gate and patience rule. Each selected policy is
then evaluated on 100 separate simulation paths. The new simulation namespace
19001 does **not** make the previously inspected historical dates an untouched
holdout. Do not claim out-of-sample evidence on new securities or dates from
this rerun. See [the evidence and limitations](pathology_repair_v19.md).

## The report to read

Open **`results/revision_v19/_publication/index.html`**.
It contains five figure groups and three tables. Figures are vector PDF plus
PNG, and tables are numeric CSV plus LaTeX. The report preserves negative
results and poor seeds. Its pointwise 95% bootstrap intervals resample
training-seed means; 100 episodes are not 100 independent trained policies.
Historical tickers appear separately. The auction summary's historical mean
weights the fixed tickers equally within each seed.

| Output basename | Question answered | Formats |
|---|---|---|
| `economic_performance` | How do economic PnL and risk-adjusted PnL compare with DQN, AS and TWAP? | Figure and table |
| `learning` | Does economic validation improve, and where do individual seeds regress? | Figure |
| `auction_mechanism` | How much auction inventory arrives, and what is the direct execution/risk contribution? | Figure and table |
| `treatments` | Which of the five isolated mechanisms changes selected economic performance? | Figure and table |
| `credit_assignment` | Does dense auction guidance improve economic learning at the same training budget? | Figure |

The two learning figures show actual observed validation values. The aggregate
is drawn only where all seeds have observations; stopped runs are never
carried forward and raw curves are never replaced by their running maximum.
The credit figure pairs dense and sparse validation paths and budgets,
rejecting a mismatched random-seed set. This is the figure most directly
relevant to the original thirty-step credit-assignment motivation.

The direct auction contribution uses the same learned CLOB trajectory and
then compares actual auction execution with submitting no auction orders:

```
auction_value = (S_cl - frozen_mid) * auction_qty - cancellation_cost
                + lambda * (opening_inventory^2 - final_inventory^2)
```

This separates price edge and fees from inventory-risk relief. It is not the
value of an optimally retrained no-auction policy. Small or negative values
remain visible. A controller that already liquidates in the CLOB has little
remaining liquidation risk for the auction to remove.

## What the five treatment contrasts mean

All treatments are retrained, and each contrast pairs the same master seeds
and evaluation market paths. The canonical headline retains H, both shaped
preferences, the indicative anchor and dense replay conditioning.

| Contrast | What is held fixed |
|---|---|
| Economic dense − economic sparse | Economic training objective, H feature and frozen-mid anchor; only action-dependent auction potential guidance changes |
| H feature on − off | Economic training, dense conditioning and frozen-mid anchor |
| Indicative − frozen-mid anchor | H feature, complete shaped J and dense conditioning |
| Combined preferences on − off | H, frozen-mid anchor and dense conditioning |
| Auction access on − off | H off and economic training; auction-on arm uses the frozen-mid grid |

Sparse here means sparse **auction control credit**. It retains CLOB potential
conditioning and market-return centering. Both potentials are terminal-corrected,
so dense versus sparse does not change the episodic economic objective.
Removing a manuscript preference does change the training objective. These
are different scientific questions; neither contrast guarantees a positive sign.
The CLOB-only and auction-only preference arms are optional follow-ups outside
the final matrix. Omitting them saves 80 runs (15.4% of the initial plan), but
means the production results identify only the combined manuscript-preference
effect, not its individual components. All four algorithms, ten seeds, five
historical tickers and the distinct information/anchor controls are retained.
Legacy bundled H/anchoring and no-cancellation controls are also optional.

## Supporting records and forecast diagnostics

The report's `audit/` directory contains:

- `economic_by_seed.csv`: economic outcomes, benchmark gaps and auction decomposition.
- `validation_by_seed.csv` and `validation_common_support.csv`: unaltered learning observations and supported aggregates.
- `treatments_by_seed.csv`: paired treatment effects with provenance.
- `credit_learning_by_seed.csv` and `credit_learning_common_support.csv`: matched dense-minus-sparse validation results.

The report `manifest.json` records inputs and scope. Each run has its resolved
configuration, training metrics, checkpoint/selection metadata and `eval/`
records. Run directories are:

```text
results/revision_v19/synthetic_rough_heston/<algo>_seed<seed>/
results/revision_v19/historical_sp500_midquotes/<algo>_<ticker>_seed<seed>/
results/revision_v19/synthetic_rough_heston__<arm>/<algo>__<arm>_seed<seed>/
```

In each `eval/`, `h_forecast_summary.csv` reports bias, MAE, RMSE and improvement
against contemporaneous midprice by **policy, phase and time-to-close bin**.
Inspect CLOB and auction separately: a good pooled H score can conceal a poor
CLOB forecast. Raw forecast records support more detailed diagnostics; use
market paths, rather than individual correlated forecast times, as resampling
units. These auxiliary data do not create extra default publication figures.

## Regeneration and manuscript integration

Inspect the entire command plan without training:

```bash
scripts/run_multiseed.sh --jobs 10 --threads-per-job 1 --dry-run
```

Regenerate from completed matching artifacts at the saved training commit.
For later checkouts, follow the isolated-checkout procedure in
[REPRODUCING.md](../REPRODUCING.md#regenerating-reports-without-training):

```bash
LMM_RESULTS_ROOT=results/revision_v19 scripts/make_multiseed_outputs.sh --publication
```

`--diagnostics` adds the legacy comprehensive outputs outside the publication
bundle. Smoke and development reports are labelled separately and are not
scientific training results. The bounded repair ledger remains under
`verification_v19/`; its superseded one-off generator is recoverable using
[the development archive instructions](cleanup.md).

Both protected TeX files remain untouched. The exact consolidated proposed
changes are in [manuscript_recommendations_v19.patch](manuscript_recommendations_v19.patch).
They document the forecast, learning implementation, calibration and experimental
protocol; they do not insert conclusions from an unrun campaign. After reading
the final report, copy only chosen figures/tables into `paper/results/` and
cite their supplied methods/captions. Prefer economic performance, auction
mechanism and the mechanism/credit figures for the main argument; use the raw
learning curves to substantiate stability. Tables can provide appendix numbers
without duplicating every figure in the main text.
