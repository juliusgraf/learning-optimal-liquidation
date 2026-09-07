# Technical review for the completed v19 write-up

Reviewed 6 September 2026, repository commit `272ca910f60126b92358df71e9c0fc5159bd8eb6`.
This is a manuscript–implementation consistency review, not a new experiment or
a certification of market realism. Manuscript, original patch, code, and saved
campaigns were left unchanged. Paths and line numbers below refer to that commit.

## Assessment

The simulator, saved numerical records, and current writing plan form a coherent
basis for the paper. The manuscript is **not yet a reproducible description of
the completed study**: it specifies an older objective, parameter calibration,
and learner implementation and contains no empirical results or conclusion.
The pending patch fixes much of this, but is neither a complete writing plan nor
a fully consistent replacement specification.

The appropriate next step is to align the write-up with the frozen experiments.
Changing the simulator to match obsolete manuscript parameters would instead
define a different experiment. If a modeling change is desired, treat it as a
separate study and do not attach the existing numbers to the changed model.

## Working model of the project

The agent starts with 100 inventory units at rebased price 100. It submits sell
orders during 120 minutes of CLOB trading; the random decision clock waits for
arrivals on both sides subject to the manuscript's time caps. Its order has
same-price priority and expires after one decision interval. Exogenous depth
is refreshed, and the strategic remainder never carries into the auction.

Algorithm 1 tracks first and second volume moments of exogenous price levels.
The residual final CLOB snapshot initializes two-sided linear auction schedules.
A persistent background schedule enforces nondegeneracy. Thirty auction actions
permit new signed schedules, persistence, and cancellation of all previously
live strategic schedules. Current exogenous proposals arrive before each action;
the indicative price is lagged. The last strategic action enters clearing and
no exogenous proposal follows it.

Terminal clearing uses the continuous aggregate root, half-up tick rounding,
and pro-rata allocation, after aggregating the agent's schedules to avoid
self-trading. Actual signed fills can oversell inventory. Residual inventory is
marked at the frozen exogenous midprice and penalized quadratically.

DQN enumerates a finite grid (391 CLOB / 1,346 auction actions). DDPG, TD3 and
SAC optimize continuous proposals projected onto the same executable controls.
Two phase learners communicate through the auction continuation value. The
18 raw features are reduced observations, not a claim of Markov sufficiency.
Synthetic rough-Heston prices and pooled historical-midquote sessions feed the
same market mechanism; historical order books and exchange auctions are not replayed.

The experimental pipeline fits training-only forecast/feature transforms, trains
with a declared reward scheme, selects a joint mature checkpoint using economic
validation, and compares policies on common test paths. Reporting consumes saved
records. The completed study contains 440 main runs and two subsequent 40-run
synthetic comparisons, with ten master seeds per cell.

## Required manuscript changes, in priority order

### R1 — Specify the objective that generated the headline policies

**High priority.** `paper/main.tex:402` defines unweighted fictive auction credit;
`:407` also omits its terminal weight. In `src/lmm/env/rewards.py:61` and
`src/lmm/env/mdp.py:451`, the headline multiplies the entire fictive credit by
`auction_shaping_weight=0.0001`; cancellation reverses that exact weighted
credit. Actual cash, cancellation fees, and inventory risk are not multiplied.
The configured purchase attenuation is `q=0`, which does not disable the other
preferences.

Use one symbol, e.g. `omega_A`, for the weight; define **J as the weighted
manuscript preference objective**, consistent with the latest writing plan.
Keep economic PnL, risk-adjusted PnL, J, and the reward stored in replay distinct.
The old patch switches to `J_omega` only in some places: either complete that
notation consistently or retain J throughout. Weighting fictive rewards changes
the objective; centering and telescoping credit do not have the same status.

### R2 — Replace obsolete simulator and learner parameter tables

**High priority.** `paper/results/tables_params/params_generative.tex` disagrees
with the saved runs and `configs/base.yaml`:

| Quantity | Manuscript | Completed runs |
|---|---:|---:|
| Arrival intensity per minute per side | 1 | 0.5 |
| Exogenous auction slope support | [0.1, 2] | [1, 20] |
| Inventory penalty lambda | 2 | 0.01 |
| Purchase attenuation q | 1 | 0 |
| CLOB clipping scale k-star | 1,000 | 10,000 |
| Cancellation increment d | 0.1 | 0.001 |
| Slope increment beta | 1 | 0.25 |
| Fictive auction credit weight | absent | 0.0001 |

The table at `paper/main.tex:584` also needs 512 transitions per phase **and 32
completed structured warm-up episodes**, two hidden layers of width 256, reward
scale 1, DQN AdamW/MSE/gradient bound 10, and DQN Polyak coefficient 0.005.
The DQN epsilon schedule is 5/95 episodes and ends at 0.05. Rates decay by the
fixed episode schedule; table rates should be labeled initial rates. SAC's
entropy coefficient starts at 0.001 and adapts. DDPG's critic has LayerNorm.
The original patch covers most table corrections; compare with the resolved
algorithm configs rather than inferring settings from inactive compatibility fields.

### R3 — Distinguish the raw estimator from the forecast supplied to the learner

**High priority.** `paper/main.tex:234`–`:265` presents Algorithm 1's raw output
as H. `src/lmm/env/mdp.py:360` additionally applies
`H = mid + w[time_bin] * (H_raw - mid)` during CLOB trading. Four weights are fit
on 256 independent no-order training paths, separately for the synthetic setting
and the historical training pool. Exact coefficients are in the setting overlays.
The raw recursion and residual-book conversion remain unchanged.

Define raw H in Algorithm 1, then define the calibrated observed H immediately
after it. State that opening H inherits the final CLOB forecast, while subsequent
auction H follows the existing lagged indicative rule. Update the state and
reward definitions to refer to the appropriate H. Adding calibration prose only
much later in the learning section leaves the earlier mathematical definition
ambiguous. Calibration against simulated future clearing is not exchange forecast validation.

### R4 — Describe the actual numerical representation and training reward

**High priority.** The raw feature vector in `paper/main.tex:346`–`:354` is a valid
description of observations, but not the numerical inputs to the networks.
`src/lmm/env/features.py:156` transforms relative prices and replaces the two
weighted-quote coordinates by continuous-root residual exposure and price
displacement. DQN/DDPG/TD3 then use phase-specific standardization and signed
asinh inventory transforms; SAC uses pooled linear standardization. The patch
has the fuller description. `docs/rl_design.md` by itself does not spell out
the weighted-quote-to-exposure replacement as clearly.

Replay also adds `Phi(next)-Phi(now)`, with exactly zero terminal potential, and
subtracts a frozen reference inventory curve's realized exogenous price return
(`src/lmm/rl/conditioning.py`, `src/lmm/rl/loops.py:211`). Write the finite-horizon
telescoping identity explicitly; it holds pathwise and does not require claiming
that the reduced features are Markov. For a fixed fitted reference, its adjustment
is policy-independent on the same exogenous path. It need not have zero historical
expectation. These transformations do not enter reported PnL or J.

The comparison of the four headline learners includes differences in preprocessing
and critic architecture. Describe implementations, not an isolated causal comparison
of algorithm update rules.

### R5 — Repair the DQN pseudocode beyond the existing patch

**High priority for reproducibility.** The current auction Q architecture and
initialization at `paper/main.tex:464`–`:476` are obsolete. The implemented auction
head uses `e(a)-e(noop)` and a fixed zero non-noop coefficient; there is no special
no-order initialization. The patch corrects those points and the exploration mixture.

However, even **after applying that patch**, Algorithm 3 still:

1. Gates updates only on buffer size (`paper/main.tex:521`). It must also require
   completion of the 32 warm-up episodes. With 30 auction transitions per episode,
   a 512-row buffer can fill before warm-up finishes. The actual gate is in
   `src/lmm/agents/dqn.py:424`.
2. Calls its update Adam (`:541`), despite the patch's table and code using AdamW.
3. Returns the final networks (`:559`), while reported policies use the best
   eligible joint economic-validation checkpoint. Add the selection/stopping
   procedure or explicitly separate the inner learning loop from that wrapper.
4. Uses raw `varphi(X)` without explicitly showing the frozen network-input
   transformation. Keep raw inputs for masks/potentials and transformed inputs
   for the learner distinct.

Prefer one stored reward that includes terminal settlement once. If retaining
the pseudocode's split `r`/`g` representation, explain its algebraic equivalence
to the code's single total transition reward. Do not re-add G to a terminal
environment reward that already includes it.

### R6 — Add the completed protocol, results, and qualified conclusions

**High priority.** The manuscript ends after rough-Heston/AS calibration
(`paper/main.tex:817`) and before the bibliography. The patch describes only
the original 440-run campaign; the current plan incorporates both follow-ups.
Use `docs/manuscript_drafts/research_framing_v19.tex` as proposed prose blocks,
not as an entire file to input verbatim: its abstract and introduction paragraphs
belong in different places.

The supported narrative is useful auction-aware liquidation, benefits of dense
auction credit, and method-dependent incremental forecast value. The additional
forecast-relative preferences have no established marginal benefit once guidance
is matched. Preserve negative anchor results and adverse historical outcomes.
Do not infer separate CLOB versus auction preference effects, cancellation value,
historical mechanism effects, or performance with the same static auction rule
as AS from the existing matrix. Those controls were not completed in this study.

The referee audit proposes a further shared-AS-auction comparison, but the later
writing plan explicitly chooses to finish with existing evidence. Treat that as
a claim restriction, not an instruction to launch more runs. Likewise, the old
patch's representation-control and liquidity-sensitivity prose must not suggest
completed production evidence merely because configurations once existed.

### R7 — Correct an overgeneralized return convention in the documentation

**Medium priority; new documentation finding.** `docs/rl_design.md` says
evaluation returns equal risk-adjusted PnL, and `docs/metrics_schema.md:47` says
all validation/final evaluation rewards are centered. This is false for the
literal cash-flow extension: its overlay disables initial-value centering, and
`src/lmm/config.py:472` preserves that switch during economic evaluation.

Across its 16,000 saved evaluation rows (learned, initial, AS, TWAP),
`return_undisc = risk_adjusted_pnl + 10,000` to floating-point precision.
`risk_adjusted_pnl` itself is correctly centered and the comparisons use it.
This is a documentation/metric-interpretation problem, not evidence of a
10,000-unit gain or corrupted headline results. State that return aliases follow
the resolved centering setting; always use `risk_adjusted_pnl_bps` for comparisons
across training schemes. Do not rewrite saved records to force the aliases equal.

Also, CSV `auction_interim_shaping` is already net of cancellation clawbacks
(`src/lmm/rl/loops.py:299`). The separate clawback field is diagnostic; subtracting
it again when reconstructing J double counts it.

## Evidence limits that belong in the paper

- The independent training units are ten seeds, not the 100 test episodes or
  all 520 runs pooled together. Use equal-seed means and paired seed differences.
- The report uses pointwise 95% percentile bootstrap intervals with 10,000
  resamples. These are not simultaneous or multiplicity-adjusted guarantees.
- Historical training pools five stocks over August 3–14; validation uses
  August 17–21 and test August 24–28, 2026. The test dates had been inspected
  during development. Fresh simulation seeds do not make them a fresh holdout.
- Historical macro summaries concern these five fixed tickers; they do not
  estimate a population distribution over stocks.
- Same-CLOB auction contribution is a realized accounting counterfactual.
  Retrained auction access is a different intervention. Neither establishes
  superior auction skill at identical opening inventory across learners.
- Auction access, same-price priority, linear supply schedules, refreshed depth,
  and terminal residual marks are maintained stylizations. A clearing existence
  theorem does not prove policy optimality, convergence, or real-market profitability.

## Verification performed in this review

See `verification.json` for machine-readable counts and residuals.

- Fresh offline suite: **607 passed, 23 deselected**, with 14 warnings.
- Confirmed 440 + 40 + 40 saved run configurations and corresponding completion
  files. Rehashed 4,424 main publication input/output entries without mismatches,
  plus config, evaluation, training-metric and split-seed files against completion
  manifests across all 520 runs.
- Checked accounting over **208,000 evaluation rows** and reward identities
  over **282,450 training rows**. After respecting centering and net-clawback
  conventions, maximum accounting/reward residuals are below 1e-10.
- Confirmed the current environment, market, agents, RL modules, config model,
  base config and algorithm overlays match the main training commit exactly.
  Later changes add follow-up/reporting support and remove archived diagnostics.
- The untouched manuscript compiles successfully from a temporary copy. Three
  overfull boxes and one underfull box remain, including a wide parameter table;
  scientific alignment should precede final layout work.
- `git apply --check docs/manuscript_recommendations_v19.patch` succeeds. That
  tests patch applicability, not scientific completeness.

No fresh training, policy rollout, external literature audit, or complete rehash
of large checkpoint/forecast binaries was performed. Existing full provenance
checks remain available through the reproduction workflow. Successful tests
support implementation contracts; they do not establish empirical external validity.
