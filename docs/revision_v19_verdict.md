# Verdict on the completed v19 campaign

**Yes: these results are sufficient to start the research write-up.** The
completed campaign supports a coherent liquidation/auction-control study.
It provides convincing evidence for dense auction credit assignment and for
the economic value of auction participation by the continuous policies.
It does not support uniform superiority of continuous algorithms, universal
positive cash PnL, or an economic benefit from every manuscript preference.
The next step should be an accurate write-up of the frozen results, rather
than another round of tuning to make every contrast positive.

This assessment reads the complete 440-run publication bundle, its seed-level
records and raw training curves. It does not retrain, reselect checkpoints,
modify result artifacts, or edit either protected TeX file.

## Integrity and numerical readiness

All 440 runs use commit `95fc54841616d2c2fd27865ec77e095d51e2b82b`, ten master
seeds and 100 test episodes per evaluated policy. Training actually consumed
243,750 episodes across the matrix, with the specified cap/early stopping.
The publication manifest's recorded input hashes match the files on disk.
The full market/algorithm/seed matrix and matched treatment configurations
validate, and all 200 seed-level treatment estimates were reproduced from raw
records. All selected checkpoints satisfy the economic selection metric and
maturity gate; recorded validation and test simulation seed sets are disjoint.

Recorded training returns, economic outcomes, inventories and observed critic
losses are finite. Economic evaluation contains zero manuscript shaping,
returns equal risk-adjusted PnL, and cash/risk and inventory-conservation
identities hold. Across all training rows, the shaped-return accounting
identity has maximum residual below 1e-11 and replay conditioning below 2e-13.
The logged interim auction shaping already includes cancellation clawbacks;
they must not be subtracted a second time when analyzing training logs.

These checks support implementation correctness and finite learning in this
campaign. They are not a convergence theorem or evidence that all individual
seeds are profitable. Reproducible checks and summaries are in
[analysis_v19](analysis_v19/verification.json).

## Economic performance

Mean test economic objective, in basis points of initial notional:

| Market | DQN | DDPG | TD3 | SAC | AS | TWAP |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic | 3.27 | 3.69 | 4.07 | 3.92 | .12 | −6.45 |
| CAT | 15.00 | 15.97 | 15.02 | 15.33 | 12.56 | 6.60 |
| GOOGL | 2.79 | 3.05 | 3.72 | 3.35 | −.54 | −7.38 |
| JPM | 4.58 | 5.79 | 5.62 | 5.82 | 3.22 | −2.19 |
| MSFT | −2.30 | −2.68 | −2.02 | −3.03 | −5.66 | −11.48 |
| PG | 9.77 | 10.16 | 10.07 | 10.07 | 7.38 | 1.79 |

Every learned method beats both stylized references on the economic objective
in every market at the mean level. All 24 paired gaps against AS and all 24
against TWAP have positive pointwise 95% seed-bootstrap intervals. This is
substantially better evidence for useful learned liquidation than mere finite
losses or a rising training reward. It is still a comparison with the stated
stylized benchmarks, which differ in policy class; it is not superiority over
an optimized real-world execution desk or every possible auction strategy.

All four learned synthetic objectives and cash PnLs have positive intervals.
MSFT is the exception to a universal profitability narrative: learned mean
objectives are negative, with intervals below zero; mean cash PnLs range from
−1.47 to −.61 bps, with intervals including zero. The methods nevertheless
substantially reduce the benchmark losses. This is compatible with a
liquidation problem on adverse or difficult paths and does not itself imply
numerical failure. Individual weak outcomes also remain, notably MSFT/SAC
seed 2718, which is worse than its initial policy and AS on test paths.

The benchmark advantage needs its correct economic interpretation. AS often
has higher raw cash PnL, but substantially greater terminal inventory risk.
For example, in the synthetic setting AS has cash PnL 5.17 and objective .12;
TD3 has cash PnL 4.65 and objective 4.07. The −.52 bps cash difference is more
than offset by 4.46 bps lower expected inventory penalty. The contribution is
better risk-adjusted liquidation, not a universal cash-profit improvement.
The fixed penalty calibration must be disclosed and economically motivated.
[Cash/risk comparison by market](analysis_v19/cash_vs_as.csv).

Continuous methods have higher mean synthetic objectives than DQN by .42–.80
bps, but all three paired intervals include zero. Their historical fixed-ticker
mean advantages are .49 for DDPG, .51 for TD3 and .34 for SAC, and those
intervals also include zero. Some individual-market contrasts are clearer:
DDPG beats DQN on CAT and JPM, TD3 on GOOGL and JPM, and SAC on JPM, by the
reported pointwise intervals. The data support competitive continuous
extensions and more effective realized auction use; they do not establish a
universal ranking. Do not treat fixed tickers as independent training seeds.

## Learning stability

| Method | Selected validation improves on initial | Final validation improves on initial | Mean selected-to-final drop | Largest drop |
|---|---:|---:|---:|---:|
| DQN | 60/60 | 60/60 | .42 bps | 1.40 bps |
| DDPG | 58/60 | 56/60 | .33 bps | 2.17 bps |
| TD3 | 57/60 | 55/60 | .53 bps | 2.99 bps |
| SAC | 59/60 | 55/60 | .45 bps | 1.76 bps |

Overall, 234/240 headline selected policies improve on their initial economic
validation. On separate test paths, 234/240 selected policies also improve on
initialization, although the individual exceptions differ. DQN's large mean
gain over initialization reflects its often very poor random initial policy;
it should not be used as evidence that it learns more effectively than the
continuous algorithms. The relevant comparisons are economic outcomes and
matched learning budgets.

The earlier broad DDPG regression is not present at comparable severity.
On the previously problematic CAT seed 811, validation goes from initial
7.89 to selected 11.41 at episode 400, then ends at 10.98 at episode 600.
Some late regression remains, especially TD3/CAT seed 577 (5.37 to 2.37).
Selection/early stopping therefore remain meaningful parts of the method.
The wording should be “numerically stable learning with residual seed-level
regressions,” not “monotone convergence” or “every seed learns successfully.”
[Learning audit](analysis_v19/learning_audit.csv).

The existing learning overview is useful, but the very negative untrained
DQN values compress the post-warm-up region. During figure preparation, add
an inset or supplementary view of that region using the same raw curves;
keep the complete initial-policy comparison visible elsewhere. No new
training is needed.

## Closing-auction value

Direct auction contribution to the economic objective, relative to submitting
no auction orders after the same policy's CLOB trajectory:

| Method | Synthetic mean [95% CI] | Historical fixed-ticker mean [95% CI] |
|---|---:|---:|
| DQN | .18 [−.02, .43] | .39 [.23, .58] |
| DDPG | .70 [.26, 1.22] | 1.48 [1.07, 1.88] |
| TD3 | .60 [.29, .93] | 1.35 [.93, 1.73] |
| SAC | .48 [.27, .69] | 1.54 [.98, 2.10] |

All four have positive mean direct auction value in every market. DQN's
synthetic interval includes zero; the other 23 market/method intervals are
positive. DQN has four negative synthetic seed means, DDPG two, SAC one and
TD3 none. These exceptions remain in the reported averages.

In both the synthetic setting and the historical macro summary, the paired
continuous-minus-DQN auction-value intervals are positive. Synthetic gaps are
.52 bps for DDPG, .41 for TD3 and .30 for SAC; historical gaps are 1.09, .96
and 1.15 bps. This is stronger evidence for the continuous extensions in
auction use than their total-objective ranking. It compares each policy's
realized contribution, including its endogenous opening inventory, rather
than isolating auction skill conditional on identical CLOB behavior across
algorithms. [Paired auction gaps](analysis_v19/auction_gaps_vs_dqn.csv).

The independent, retrained auction-access contrast confirms economic value
for all three continuous methods: DDPG +.56 [.27,.90], TD3 +.48 [.21,.77],
SAC +.61 [.15,.97] bps. DQN is +.22 [−.25,.67]. This contrast holds information
and the economic training objective matched. It is distinct from the direct
same-CLOB decomposition and supports the same broad conclusion.

Auction value is mostly inventory-risk relief in the historical setting;
synthetic SAC/TD3 also have positive cash-edge components. The effect is
moderate in absolute size because roughly 88–96% of inventory is already
liquidated at auction opening, depending on policy/market. Moderate value is
still economically interpretable, and both an independently retrained access
contrast and a direct decomposition support it. The manuscript should not
inflate these results into a claim that most value comes from auction trading.

## The mechanism results answer the original motivation

Mean synthetic economic treatment effects, bps:

| Retained contrast | DQN | DDPG | TD3 | SAC |
|---|---:|---:|---:|---:|
| Dense minus sparse auction credit | 1.18 | 1.69 | 2.30 | .85 |
| H observation on minus off | .17 | .72 | .15 | −.17 |
| Indicative minus frozen-mid anchor | −.51 | −.55 | −.35 | −.20 |
| Combined manuscript preferences on minus off | −.13 | −.00 | .34 | .02 |
| Auction access on minus off | .22 | .56 | .48 | .61 |

**Dense credit is the strongest mechanism result.** All four selected-policy
intervals exclude zero: DQN [ .65,1.93 ], DDPG [ .92,2.45 ], TD3 [ .70,4.54 ],
SAC [ .29,1.47 ]. Positive seed effects occur in 10/10, 9/10, 8/10 and 8/10
respectively. TD3's mean is influenced by one large sparse-training failure
(the largest difference is 10.87 bps), but its median remains +1.81 and most
seed pairs favor dense credit. Retain the full mean and all seeds.

The raw learning comparison gives matching evidence without checkpoint
selection: at episode 250, dense-minus-sparse validation is +3.34 for DQN,
+2.76 for DDPG, +1.75 for TD3 and +1.82 for SAC, with positive pointwise
intervals for all four. Effects are not uniformly positive at episode 50, so
avoid claiming improvement from the first update. Common observed support
ends as runs stop; later surviving traces do not represent all ten seeds.

This directly supports the original thirty-decision credit-assignment
motivation. Terminal-corrected auction potential guidance exposes projected
execution and inventory risk before settlement while preserving the episode
objective. The sparse arm retains the common CLOB conditioning, so the result
is specifically about additional auction credit, not removal of every
numerical transformation.

**The extra manuscript preferences have no demonstrated economic benefit at
this calibration.** All four combined-preference intervals include zero.
TD3's positive mean is driven by one relatively large seed effect; its median
is slightly negative and only five of ten effects are positive. These are
not equivalence tests: absence of a clear gain does not prove exact equality.
The full results justify separating useful temporal credit from the additional
CLOB opportunity-cost and fictive auction preferences. The omitted single-term
arms cannot be used to identify their individual production effects.

**Forecast prediction and decision value differ.** The calibrated H has lower
MAE and RMSE than contemporaneous midprice in both phases across all six
markets. For example, synthetic CLOB MAE is .1800 versus .1925, and auction
MAE .0488 versus .1403. Phase summaries average within episodes, then within
algorithm/seed, with the fixed algorithm set equally weighted. They describe
prediction errors against realized simulator clearing; they do not validate
forecasting actual exchange auction prices or establish the causal gain from
calibration versus raw Algorithm 1. There is no full-production raw-versus-
calibrated H intervention. [Forecast audit](analysis_v19/forecast_summary.csv).

Only DDPG shows a clear economic gain from the added H observation: +.72
[.12,1.36] bps. The other intervals include zero. H-off retains midprice and
other book/auction information, including the common auction-exposure
representation. This is an incremental-feature comparison, not removal of
all information useful for anticipating clearing. Redundancy is a plausible
interpretation of limited incremental value, not a demonstrated causal
explanation from these results alone.

Indicative anchoring lowers the mean objective for all four algorithms;
pointwise intervals exclude zero for DQN, DDPG and TD3, but not SAC. A bounded
grid centered on H changes available executable orders, not just coordinates.
The result therefore argues against assuming that an accurate price forecast
is necessarily a good center for the control grid. Keep the frozen
headline and report this ablation; selecting the better anchor after seeing
test results would create a different, post hoc headline.

## A defensible paper narrative

A suitable empirical narrative is:

> A phase-aware RL framework learns economically useful liquidation policies
> in a stylized continuous market followed by a closing auction. Dense,
> terminal-corrected auction credit improves learning and economic evaluation
> across all four algorithms. Continuous-control policies realize more auction
> value than the DQN baseline, although their total-performance rankings remain
> uncertain. Forecast accuracy alone does not ensure economic improvement:
> its value as an observation differs from its value as an action-grid anchor,
> and additional non-invariant reward preferences show no clear economic gain
> at the specified calibration.

The scope should be fixed historical-midprice scenarios embedded in a
stylized simulator, with reused historical test dates disclosed. The final
experiment is not an untouched test on new historical periods, an empirical
fit of exchange auction liquidity, or a deployed-profitability demonstration.
Market/risk/shaping scales must be described as calibrated or stylized according
to their actual provenance. The paper must also include the forecast extension
beyond Algorithm 1, the weighted shaped J and terminal-corrected replay
conditioning documented in the existing unapplied review patch.

Confidence intervals throughout are pointwise seed-bootstrap summaries, not
familywise multiple-comparison guarantees. Ten seeds limit precision. Fixed
historical markets and repeated dates are not independent market samples.
These limitations constrain the strength and breadth of claims, but do not
prevent a useful research write-up of the modeled control problem.

The immediate remaining work is editorial and methodological: articulate the
correct objective, specify the complete implementation and calibration,
explain benchmark policy-class differences, present the five controlled
mechanisms, and show poor seeds alongside averages. Additional training is
not necessary to begin that work. Submission suitability will also depend on
the paper's novelty, theoretical contribution and positioning, which cannot
be decided from numerical results alone.
