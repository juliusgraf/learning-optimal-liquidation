# Research narrative for the completed study

Training is finished. This plan uses the completed 440-run v19 campaign and its
two 40-run synthetic extensions. No further training, retuning or new ablation
is proposed. The initial campaign and follow-ups should be identified separately
in the experimental protocol, rather than described as one preregistered matrix.

## Central question and contribution

**Can a learning agent use information about a future closing auction to balance
continuous execution with end-of-day inventory management?**

The contribution is a two-phase execution framework with an explicit closing
call, a forecast connecting the phases, and learning guidance for decisions whose
cash consequences are delayed until clearing. DQN is the baseline; DDPG, TD3 and
SAC are projected continuous-proposal extensions on the same executable grid.
The empirical study asks whether this framework produces economically useful
policies and what its information, training guidance and auction access contribute.

Do not make the paper a contest to show that every added component is individually
beneficial. Also do not claim all components are necessary. The completed controls
identify which mechanisms help and where the simpler variant remains competitive.

## Three manuscript questions

### 1. Does information about future clearing help?

The forecast has an economic purpose: help distinguish an attractive CLOB sale
from inventory worth carrying into the closing call, and inform auction decisions.
Algorithm 1's conversion of exogenous residual liquidity into auction schedules
is part of the environment specification; the H-feature ablation does not remove
that mechanism. It removes explicit access to the forecast while preserving the
same simulated market and other observed liquidity information.

Two pieces of evidence answer different parts of the question:

- H predicts the eventual simulated clearing price more accurately than
  contemporaneous midprice in both phases across all six markets. Synthetic
  MAE is 0.1800 versus 0.1925 during CLOB and 0.0488 versus 0.1403 during auction.
- At a common fixed anchor and with economic training, adding the H observation
  changes economic performance by +0.17 bps (DQN), +0.72 (DDPG), +0.15 (TD3)
  and -0.17 (SAC). Only DDPG's interval excludes zero: [0.12, 1.36].

The supported finding is **predictively informative auction information with
method-dependent incremental decision value**. H-off retains other clearing-
relevant information, so the comparison does not test an uninformed trader
against a trader who alone knows anything about clearing. Limited incremental
value could reflect redundancy or incomplete use by the learner; this study
does not identify which explanation applies.

Keep quote anchoring as a secondary design check. H-centred and frozen-mid
grids have different executable price coverage, so an accurate forecast need
not be the best anchor. Report the negative anchor results in a compact
supplementary panel and acknowledge them in the main forecast discussion.
The declared H-anchored headline is retained. No claim of optimal anchoring is
needed to motivate supplying the forecast as information.

### 2. Does intermediate guidance make useful auction behavior learnable?

Yes. This is the strongest methodological result and directly addresses the
original thirty-decision credit-assignment motivation. Orders can be submitted,
retained and canceled before any auction cash settles. Dense feedback makes
their projected execution and inventory consequences visible during learning.

Distinguish the design purpose from the empirical attribution:

- The headline combines manuscript J (including weighted fictive rewards) with
  dense potential credit and reward conditioning. It beats raw economic cash-flow
  training by 3.57, 3.06, 4.68 and 5.37 bps for DQN, DDPG, TD3 and SAC, with
  positive paired intervals and all 40 paired seed means favoring the headline.
- The matched dense-versus-sparse auction-credit comparison improves economic
  performance by 1.18, 1.69, 2.30 and 0.85 bps, all with positive intervals.
  At episode 250, paired economic validation gains are also positive for all four
  algorithms. This isolates an auction-credit effect while retaining common
  CLOB conditioning, and supports learning benefits beyond final selection.
- The final H-anchored comparator shares all conditioning and removes only the
  manuscript preferences. J-minus-economic effects are -0.06, -0.06, +0.19 and
  -0.20 bps; all intervals include zero. Conditioned economic training improves
  selected validation over initialization in 39/40 runs and uses the auction.

The useful paper finding is **dense guidance improves economically evaluated
learning; the extra forecast-relative preferences are a tested design choice
with no established marginal benefit at this calibration**. Give this last
result one clear paragraph and the complete contrast, rather than allowing it
to displace the broader question or disappear from the evidence.

Fictive rewards are teaching signals, not cash flows. That is precisely why
evaluation excludes them. Their exclusion does not prevent them from improving
economic results through a better learned policy. The comparison tests this
indirect benefit. We cannot justify a term merely because it changes its own
training score, and lack of a clear final-score gain does not establish that
its learning trajectories are identical or that its effect is exactly zero.

Use J for the manuscript's specified weighted preference objective; use
"potential-based credit" for the telescoping addition; use "training scheme"
for their combination. State which configuration is being discussed whenever
using the broader phrase "reward shaping."

### 3. Does auction access improve liquidation and inventory management?

Yes for the continuous methods in the matched access comparison, with a smaller
and uncertain DQN effect. Economically trained H-off policies with auction access
outperform retrained no-auction controls by 0.22 bps (DQN), 0.56 (DDPG), 0.48
(TD3) and 0.61 (SAC); the continuous-method intervals are positive. The no-auction
control ends at the CLOB boundary, marks residual inventory and applies the
inventory penalty; it is not forced market-order liquidation.

The direct headline auction decomposition provides a second, distinct view:

| Policy | Synthetic auction value | Historical fixed-ticker mean auction value | Historical inventory: auction open to terminal |
|---|---:|---:|---|
| DQN | 0.18 bps | 0.39 bps | 7.93 to 7.35 units |
| DDPG | 0.70 bps | 1.48 bps | 8.95 to 6.58 units |
| TD3 | 0.60 bps | 1.35 bps | 8.27 to 5.72 units |
| SAC | 0.48 bps | 1.54 bps | 8.64 to 6.50 units |

Inventories are expected absolute holdings, from an initial 100. Historical
summaries weight the five fixed tickers equally within seed. Approximately
82-89% of the continuous methods' historical auction value is inventory-risk
relief; DQN's share is about 98%. Contributions are modest because most inventory
has already been sold. This is an economically meaningful use of a final
liquidity opportunity, not evidence that the auction generates most total PnL.

At episode level, relative to no auction orders after the same CLOB trajectory,

    auction value = Z * (S_close - S_frozen_mid) - cancellation fees
                    + lambda * (I_open^2 - I_terminal^2).

This separates execution price edge from inventory-risk relief. All headline
market/method mean contributions are positive; 23/24 intervals exclude zero,
with synthetic DQN the exception. Continuous-minus-DQN direct auction-value
gaps have positive intervals in both synthetic and historical-macro summaries.
Those gaps include different endogenous CLOB trajectories and opening inventory;
they do not identify superior auction skill conditional on identical inventory.

Describe the outcome as **liquidation and terminal inventory-risk management**.
The auction permits signed positions: mean absolute inventory need not fall in
every setting, even when expected squared inventory falls. In particular, synthetic
TD3/SAC increase mean absolute holdings slightly through the auction while reducing
the quadratic risk penalty on average. Do not promise complete liquidation or
claim the auction mechanically reduces every measure of inventory.

## Benchmark comparisons provide context

The headline beats both stylized AS and TWAP on mean economic objective in all
six settings, with positive pointwise paired intervals. This establishes that
the framework produces useful policies relative to the stated references.
AS is the implemented risk-neutral variant with a static auction extension;
its higher raw PnL in many settings coexists with greater residual-inventory risk.
Do not make AS's lack of active auction control the sole evidence for auction
value: the matched learner-access control supplies the stronger comparison.

Continuous methods are competitive with DQN on total economic performance;
universal superiority is not established. Their stronger realized auction
contribution is the more informative algorithm finding. Keep adverse historical
results, including MSFT's negative economic means. Historical input is midprice
data inside a simulated market, not an observed exchange-auction backtest.

## Results order and focused figure plan

Begin with a short protocol subsection identifying economic evaluation, ten
independent training seeds, 100 test episodes per selected policy, paired seeds,
separate validation, and the chronology of the two follow-ups. Use equal-seed
means and pointwise intervals; keep all seeds and distinguish synthetic mechanism
evidence from the historical headline settings.

The results should then proceed in this order:

1. **Economic performance and auction use.** Establish useful learned policies,
   then show the cash-edge/risk-relief decomposition and retrained access contrast.
2. **Forecast information.** Show prediction accuracy and incremental H-feature
   value together, avoiding the implication that one guarantees the other.
3. **Learning from delayed auction outcomes.** Show the matched learning-credit
   curves, the complete scheme versus raw cash flow, and the final matched
   preference comparison. Explain the latter as a boundary on attribution.

Target four main figures, assembled from existing artifacts and saved records:

| Figure | Research purpose | Existing source |
|---|---|---|
| Economic policy performance | Risk-adjusted outcomes and reference gaps across markets | `results/revision_v19/_publication/figures/economic_performance.pdf` |
| Auction value and inventory | Direct edge/risk decomposition, inventory and matched access effect | Existing `auction_mechanism` figure/table and `treatments.csv` |
| Forecast information | Phase-specific forecast accuracy plus H-on/off economic effect | `docs/analysis_v19/forecast_summary.csv` and H-feature treatment records |
| Learning guidance | Credit-learning curves plus three H-anchored training schemes | Existing `credit_assignment` figure and both follow-up comparison records |

Use the existing economic-performance table for precise levels. Put the complete
eight-cell synthetic treatment specification, all contrast intervals, anchoring
and fixed-anchor preference controls, and full learning traces in the appendix.
This is an editorial organization of the completed evidence, not selective
removal of negative results. Numerical settings retain a concise parameter table.
Assembling these figures later requires reporting work only, not new training.

## Exact manuscript integration recommendations

The protected files are unchanged. `docs/manuscript_drafts/research_framing_v19.tex`
contains proposed prose blocks, not an applied patch.

1. Replace only the abstract body between `\\begin{abstract}` and the keywords
   with block A. The current title can remain; an optional more cautious title is
   `Learning to Liquidate with Closing Auctions`.
2. In `Methodology, Contributions and Financial Insights`, replace the first
   paragraph and the final paragraph beginning `The numerical experiments compare`
   with block B's question/contribution paragraphs. Move the intervening detailed
   DQN training explanation to the learning section or shorten it to avoid
   repeating the algorithm descriptions. Retain the benchmark introduction.
3. After the paragraph beginning `Without reward shaping or initial-inventory
   centering` and before `\\section{Learning Algorithms}`, add block C. Separately,
   describe the explicit potential and reference control variate in the learning
   section using the implementation-alignment recommendations already supplied.
4. Align the existing shaped-reward equations with the runs: in the definition
   of `u_t`, multiply the complete bracket `K_t^a H_t^cl(H_t^cl-S_t^a) + f^a(...)`
   by `omega_A = 10^{-4}`. The cancellation clawback then uses the weighted `u_s`.
   Multiply the terminal `f^a(S_close Z)` term by the same `omega_A`. Keep
   `q=0` as the headline numerical value. Describe the weighted objective as J
   consistently. These are proposed changes to the current unweighted equations,
   not alterations to the completed simulator or objective choice.
5. The current main file ends after the rough-Heston/AS calibration subsection
   and has no empirical results or conclusion. Insert the evaluation protocol
   and results blocks D-F after that subsection and before the bibliography;
   add block G as the conclusion. Add figure/table references when the four
   selected exhibits are assembled, rather than importing the complete dashboards.
6. Keep parameter and forecast-calibration reconciliation explicit. In particular,
   the protected parameter table's `lambda_0` entry must be recommended as
   `0.5/min/side`, not its current `1/min/side`. Reuse the exact pending numerical
   alignment recommendations in `docs/manuscript_recommendations_v19.patch`, but
   do not apply that older patch wholesale: its experiment matrix and narrative
   predate the two completed follow-ups. The new findings above supersede its
   prospective result language.

The empirical story is ready for writing. The purpose of the remaining work is
clear explanation, accurate mathematical/implementation alignment and focused
presentation of the completed evidence.
