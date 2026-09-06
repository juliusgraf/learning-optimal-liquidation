# V18 repair and bounded verification

**Development history:** the subsequent [DQN auction repair](dqn_auction_repair_v18.md)
supersedes this report's DQN three-step specification and auction results.
The earlier results and frozen artifacts below are retained as the record of
that stage, including its negative findings.

V18 addresses a reproducible numerical failure and several weaknesses in the
experimental protocol. It does **not** establish that every learned policy beats
AS, that every auction contribution is positive, or that these simulated markets
are empirically calibrated exchanges. In particular, DQN's synthetic auction
controller remains weaker than the continuous controllers in the bounded checks.
The negative results are retained. A publication must allow that scientific
outcome rather than use a success-only seed or checkpoint rule.

The executable specification is `configs/base.yaml` plus the setting, algorithm
and treatment overlays. Headline training still uses weighted shaped **J**;
validation, selection and evaluation use economic **bar-J-lambda**, after
subtracting initial inventory value. Price formation, admissible controls,
benchmark formulas, signed auction settlement, the terminal quadratic inventory
cost, the 120/30-minute horizon and discount factor one are preserved.

## Diagnosis and changes

1. **DQN value amplification.** The v17 bilinear Q approximator could amplify
   both learned factors under undiscounted bootstrapping. This was visible as
   very large critic losses and catastrophic late policy deterioration. The
   state trunk now uses non-affine LayerNorm; the action embedding uses
   non-affine LayerNorm and tanh. The linear query and value heads remain
   unrestricted. AdamW decay is .0001. Three-step returns reduce repeated
   bootstrapping; shorter tails flush at phase boundaries. MSE still includes
   rare inventory losses, and targets/rewards are not clipped. Gradient clipping
   is 10 rather than 1, which was too restrictive in the bounded DQN comparisons.
   Every exact admissible action remains available. The new architecture has a
   distinct checkpoint identifier; legacy loading keeps legacy defaults.

2. **Uneven phase learning.** Transition-count warm-ups let the auction learner
   update well before a lower-flow CLOB learner. All four methods now collect
   the same 32 complete structured exploration episodes before either phase
   updates, with a 512-transition minimum per buffer. DDPG, TD3 and SAC retain
   native Stable-Baselines3 2.7.1 losses/optimizers/target updates and their
   existing networks, rates and noise settings. Their optimizer search was not
   expanded. DQN/DDPG/TD3 use phase-specific normalization and signed-asinh
   auction inventory; SAC retains pooled standardization.

3. **Selection and late regression.** Eligibility requires 2,000 updates in
   each active phase. The first eligible checkpoint is evaluated immediately:
   DQN can cross maturity just after periodic episode 100, and waiting until
   episode 150 discarded part of the learned trajectory. Validation uses 128
   paths every 50 episodes, with four eligible non-improvements stopping
   training and the same 800-episode cap. Selection remains of a joint policy;
   phase networks are not spliced. Every mature seed is reportable even when it
   fails to improve on initialization. The untrained policy is a diagnostic.
   These changes limit exposure to late deterioration; they do not prove that
   DDPG or another off-policy learner has a monotone learning curve.

4. **Generalization protocol.** Historical training and normalizer fitting now
   pool all 50 rebased stock--sessions on the training dates. Validation/test
   stay stock-specific and on disjoint date partitions. Ten fixed master seeds
   replace five: 42, 7, 99, 123, 2024, 314, 577, 811, 1618, 2718. The full matrix
   is 440 runs. The final simulated stream has namespace 18001 and preserves
   cross-policy/treatment pairing. The August 24--28 historical test week was
   already inspected in v17; a new simulation stream is **not** a fresh
   historical holdout. This limitation is now explicit in generated reports.

5. **Auction mechanism and capacity.** Continuous intensity is .5 arrivals per
   minute per side, giving expected contra-side volume 198.62 for the 100-unit
   parent. This declared substantial-participation scenario leaves economically
   relevant residual exposure without forcing an auction reserve or constraining
   signed fills. The report adds a pure auction-access contrast using the
   existing H-off/shaping-off and no-auction arms. The original bundled contrast
   and exact same-CLOB no-order decomposition remain distinct. No extra
   treatment runs or plots are needed.

Layer normalization/regularization are motivated by the TD-stability analysis in
[Gallici et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/c23f3852601f6dd7f0b39223d031806f-Paper-Conference.pdf).
That result is not a convergence theorem for this environment or implementation.

## Bounded evidence and its limits

No 800-episode production training was launched. The development ledger contains
36 attempts, each capped at 200 episodes, totaling 7,080 training episodes
(including immature and rejected candidates). These are 2.01% of the maximum
352,000 episodes in the new campaign, plus separate offline replay stress and
short pipeline checks. Repeated adaptive development is not an independent
confirmatory experiment. All development policy evaluation uses validation dates;
none uses the final-test stream. New order-flow paths on validation dates are
called confirmation paths below, not new historical test dates.

For a matched synthetic seed (853), 128 fresh development confirmation paths give:

| Policy | Net PnL (bps) | Economic objective (bps) | Same-CLOB auction value (bps) |
|---|---:|---:|---:|
| DQN, final representation and clip 10 | 2.686 | 1.920 | -0.289 |
| DDPG | 6.032 | 4.570 | 1.527 |
| TD3 | 5.006 | 3.970 | 1.256 |
| SAC | 6.288 | 3.988 | 1.329 |
| AS | 7.182 | 2.525 | — |
| TWAP | 2.252 | -3.652 | — |

This is one training seed, not a statistically established algorithm ranking.
SAC's selected validation score is below its initialization score; it is retained.
A positive objective does not demonstrate learning improvement by itself.
DQN still underperforms AS on this seed. Its auction value has a conditional
path standard error .077 bps, so the negative mean must not be dismissed as
roundoff. This standard error excludes training-seed variability.

A subsequently tested DQN synthetic seed 881 gives PnL 8.400 and objective
7.798 bps versus AS 3.984, with auction value -.079 (conditional path SE .036).
The historical MSFT seed 887 gives PnL 7.077, objective 4.936 versus AS 1.681,
and auction value +1.198 (SE .261). Those are separate seeds and markets, not
paired evidence that one algorithm dominates another. The final DQN selected
checkpoints all have positive economic objectives in these three checks.

The earlier shared specification was frozen before seeds 853/857 in
`verification_v18/frozen/`. The first-maturity fix and final DQN clipping choice
were made afterward and are disclosed here; they are not retrospective changes
to those frozen files. `verification_v18/final_specification/` records the final
prospective specification. No failed candidate is erased from the ledger.

### Late-update stress

A saved DQN replay buffer was revisited for 60,000 additional CLOB and 20,000
auction updates, without generating new training trajectories. This intentionally
reuses a fixed off-policy dataset much more heavily than ordinary training.
The learning-rate counter was held fixed at the value saved after diagnostic
validation (127 for the final two checks), so the learning rate stayed above the
late-production floor. The exact counter for every trial is in the protocol.
This is a numerical stress test, not an economic performance estimate.

| Final DQN replay source | Max observed CLOB / auction loss | Max absolute Q, CLOB / auction | Validation objective before / after (bps) |
|---|---:|---:|---:|
| Synthetic 881 | 16.809 / .147 | 72.022 / 27.932 | 10.719 / 9.564 |
| MSFT 887 | 12.992 / .113 | 59.087 / 28.275 | 4.785 / 3.997 |

Losses and Q values remained finite; the v17-style value explosion was not
reproduced. Economic deterioration is still possible and the 48-path endpoint
standard errors are 3.888 and 2.588 bps. An old-architecture control deteriorated
from 12.377 to -22.193, while normalization alone bounded Q values but still
lost economic performance. Those trials have different replay data and, in
some cases, different learning-rate counters/liquidity settings. Do not turn
this table into an isolated causal effect of LayerNorm or claim convergence.

### Frozen-policy historical transfer

Without any learner updates or reselecting checkpoints, the MSFT-selected
historical policies were evaluated on 128 newly drawn validation flow paths
per stock. All 20 policy/stock combinations had positive mean PnL and positive
economic objectives. DQN used training seed 887; the continuous policies used
857. They share training dates/stocks and are not an unseen-security test.
The common reference calibration and evaluation paths are fixed in the probe.

| Stock | DQN | DDPG | TD3 | SAC | AS |
|---|---:|---:|---:|---:|---:|
| MSFT | 1.263 | 1.112 | 1.062 | .705 | -1.135 |
| JPM | 4.313 | 7.196 | 7.448 | 6.743 | 5.030 |
| PG | 5.717 | 8.001 | 8.032 | 6.974 | 3.888 |
| GOOGL | 4.175 | 4.782 | 4.795 | 4.095 | 1.204 |
| CAT | 12.286 | 10.933 | 10.835 | 10.617 | 7.940 |

Values are economic objective in bps. DQN still trails AS on JPM. Conditional
path variation, common historical dates and one trained policy per algorithm
preclude significance or universal-ranking claims. This provides a useful
check that positive performance does not depend solely on the stock used for
checkpoint selection. The full scope is in
`verification_v18/transfer_validation_protocol.json`.

### Auction weakness that remains

The continuous controllers have positive auction effects on both the shared
synthetic and MSFT confirmation checks. In the synthetic seed above, the gains
are 1.26–1.53 bps, a substantial fraction of their 3.97–4.57 bps objectives.
Mean absolute inventory need not fall even when squared-inventory risk falls:
DDPG's open/final absolute inventories are 7.61/7.93 on that seed.

DQN sometimes adds small exposure and repeatedly cancels after liquidating
most of the parent in the CLOB. Its synthetic seed 881 opens with mean absolute
inventory 3.99 and closes at 4.16. Feasible reserve probes do not justify a
blanket positive-auction claim either. On 64 synthetic paths with about 20.52
units arriving at the auction, the selected DQN finishes at 20.03 and loses
.068 bps versus no orders; TD3's earlier frozen probe reduces inventory to
7.03 and gains 2.52 bps. These are conditional probes using a different CLOB
policy, not headline performance or added reference benchmarks.

Shaped J and economic bar-J-lambda are different objectives. With the current
auction weight, the local fictive sale credit is .01 per nominal unit; even a
zero-inventory toy case can prefer a small sale under J while bar-J-lambda
prefers no order. There is no mathematical guarantee that training on shaped J
improves the economic objective or that every learned auction adds value.
An exact-clearing, policy-free example confirms the conflict without any
learning error: with zero residual inventory and 30 exogenous schedules of
slope 10 (15 quoted at 99.9 and 15 at 100.1), enumerate all 673 no-cancellation
orders at an indicative/frozen price of 100. The economic maximizer is no order,
with value zero. The shaped maximizer has slope 8 and offset -7 ticks; actual
tick clearing and pro-rata allocation sell .53985 units, giving economic
value -.002914 and shaped value +.002686. The full specification/results are
in `verification_v18/objective_conflict_example.json`. This small illustrative
loss does not explain away DQN's larger cancellation-driven losses.

A guaranteed nonnegative auction effect would require an additional control
restriction, an appropriate fallback, or changing the training objective and
would change the algorithm comparison. The report instead
exposes the economic effect and the matched shaping/auction treatments.

### Rejected alternatives

The ledger retains the old architecture, normalization-only and pooled-training
controls. A .25 continuous intensity made liquidation too capacity-constrained:
all four selected economic means were negative in the synchronized-warm-up
checks, despite exceeding the still worse references. It is a liquidity
sensitivity, not the headline. An existing conservative DQN auction
initialization improved neither market consistently and produced a negative
MSFT economic result; it remains disabled. These attempts cannot be represented
as successful independent replications of the final configuration.

## Calibration verdict

The values are dimensionally coherent for a stylized liquidation study. They
are not all statistically estimated parameters. The full audit is
`verification_v18/calibration.json`; physical-unit illustrations use training
quotes only under `verification_v18/market_units/`.

- **k-star = 10,000 ticks.** Yes: with alpha=.01 its clipping distance is 100
  price units, the entire rebased initial price. It is implausible as a typical
  market gap. Define eta-C = S0/(k-star*alpha) = 1 instead. Before clipping,
  the CLOB shaping deduction is eta-C*(p/S0)*E*(H-p)-positive. Thus one tick of
  shortfall deducts approximately one tick per executed unit, and k-star is
  derived from a unit opportunity-cost preference. Reducing k-star to 100
  while retaining the reward formula would multiply that local penalty by
  100. The exact paper patch makes this interpretation explicit.
- **q = 0.** This turns off the purchase-side attenuation/subsidy, including
  its terminal shaping term. It does not turn off signed interim auction
  shaping or CLOB shaping. q is a preference parameter, not an exchange datum.
- **beta = .25.** Units are inventory units per price unit. The maximum single
  slope is 8. A ten-tick local displacement implies .8 nominal units per
  maximum schedule; 30 such schedules imply 24 nominal units. These are scale
  illustrations, not fill bounds: actual signed clearing and inventory are
  unaltered. In an illustrative physical mapping, beta becomes beta*M/A
  shares per dollar, where P=A*S and shares=M*I; it is not ".25 shares".
- **lambda = .01.** A ten-unit residual costs one model objective unit, which
  is one basis point of the 10,000 initial notional. Its marginal risk cost at
  ten units is 20 ticks per additional unit. This states risk appetite rather
  than estimating a market friction.
- **Cancellation fees.** d=.001 is the increment in d-j=j*d. The largest single
  fee is .029 bps, but cancel/replacing at every subsequent slot totals .435
  bps. The audit previously mislabeled the single-fee bound as the total;
  that reporting defect is fixed. Repeated unnecessary cancellation is
  economically material at these execution-profit scales.
- **Liquidity and price units.** The parent is about 50.3% of expected eligible
  contra-side model flow, a large participation rate. The model tick is 1 bp,
  not necessarily the exchange's one-cent tick after rebasing. Using the
  training depth illustration gives about 151 shares per inventory unit,
  a roughly 15,126-share parent and a .050-dollar MSFT model tick. Those
  choices are disclosed as an eligible-liquidity/coarse-price model, not as a
  fitted reconstruction of all SIP trades.

Exchange evidence establishes the importance and capacity of closing auctions,
not these particular simulator coefficients. Nasdaq reports that the Closing
Cross accounts for approximately 17% of daily volume
([Nasdaq Closing Cross](https://www.nasdaq.com/products/north-american-markets/nasdaq-stock-market/closing-cross)).
NYSE studies relate closing-auction impact to participation and trading costs
([NYSE impact study](https://beta.nyse.com/data-insights/closing-auction-immediate-market-impact-price-drift-and-transaction-cost-of-trading)).
Neither identifies this model's beta or says an optimal parent must reserve
17% for the close. Empirical beta identification would require actual auction
supply/demand or imbalance/price-response data, beyond the midpoint archive.

## Publication scope and files

The existing 580-test non-slow/non-network suite passed after the core repairs;
88 targeted tests passed after the final DQN clipping and reporting changes;
seven calibration tests passed, including an actual cancel/replace episode
that attains the corrected total-fee bound.
A 56-job, two-development-seed smoke campaign completed training, evaluation
and the focused four-figure/three-table report before the final DQN clipping
change; subsequent targeted launcher tests exercised the final setting. All
15 final report tests passed, including coverage of the seventh contrast. The
smoke report was re-rendered and visually inspected after that addition. A dry run enumerates exactly
440 production jobs. Verification logs, counts, source/artifact hashes and
all attempted diagnostics are retained in `verification_v18/`. No production
result was substituted with a development estimate.

The new production outputs will be in
`results/revision_v18/_publication/index.html`. The four figure groups remain
economic performance, learning, auction mechanism and treatments. Seed means,
paired differences, selected-checkpoint markers and negative effects remain
visible; there is no test-based selection or trimming of poor seeds.
See [the output guide](research_outputs.md) and [REPRODUCING](../REPRODUCING.md).

`paper/main.tex` and `paper/results/tables_params/params_generative.tex` are
untouched. [manuscript_recommendations_v18.patch](manuscript_recommendations_v18.patch)
is an exact, unapplied patch against both current files. It updates the reward
specification, parameter interpretation, algorithm/optimizer table, seed count,
selection protocol, training pool and historical-scope disclosure. It contains
no invented full-campaign results. Review it rather than treating the current
protected parameter table as the executable configuration.
