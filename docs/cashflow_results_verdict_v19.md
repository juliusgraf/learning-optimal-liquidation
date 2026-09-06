# Verdict on the completed economic cash-flow comparison

The complete headline training scheme beats the requested literal economic
cash-flow training comparator in every one of the 40 synthetic algorithm/seed
pairs. All four pointwise 95% seed-bootstrap intervals for the paired mean gap
are positive. This supports the practical training scheme relative to this
specific baseline. It does not establish that manuscript J's additional
preferences are the source of the improvement.

## Economic test results

Units are basis points of initial notional, evaluated on the economic objective.

| Algorithm | Headline | Cash-flow training | Paired gap [95% interval] |
|---|---:|---:|---|
| DQN | 3.270 | -0.298 | 3.568 [1.295, 6.911] |
| DDPG | 3.692 | 0.629 | 3.064 [2.045, 4.185] |
| TD3 | 4.068 | -0.610 | 4.677 [3.882, 5.491] |
| SAC | 3.920 | -1.445 | 5.365 [4.187, 6.632] |

Every method wins 10/10 paired seed means. DQN's mean is amplified by seed 2024:
cash-flow training scores -16.47 bps versus headline 0.19. The median DQN gap
is still 1.96 bps; excluding the largest gap as a diagnostic leaves a 2.11 bps
mean advantage. The primary result retains every seed. The other median gaps
are 2.56, 4.88 and 5.47 bps for DDPG, TD3 and SAC.

Unpaired uncertainty intervals for the cash-flow policy means all include zero.
Strong paired evidence is compatible with this because the comparisons share
evaluation paths and preserve training-seed pairing.

## Learning explains the limits of this comparison

| Algorithm | Cash-flow selected validation improves over initialization | Headline selected validation improves over initialization |
|---|---:|---:|
| DQN | 10/10 | 10/10 |
| DDPG | 1/10 | 10/10 |
| TD3 | 0/10 | 10/10 |
| SAC | 0/10 | 10/10 |

Cash-flow DQN improves from very weak initialization but remains economically
less effective. Cash-flow continuous methods usually deteriorate from their
initial policies, even after economic validation selection. The protocol
permits reporting a mature learned policy that is worse than initialization;
the initial policy is a separately evaluated diagnostic, not a hidden fallback.

All recorded nonmissing critic losses and required economic values are finite.
There is no observed NaN/Inf failure or economic-accounting discrepancy. However,
finite arithmetic does not imply successful learning.

Literal cash-flow replay returns retain roughly 10,000 units of gross economic
wealth, while economic performance differences are only a few units. Headline
replay returns are of order a few units. Late cash-flow critic losses are typically
of order 10^4-10^5, versus much smaller headline losses. Absolute loss levels are
not directly comparable across these reward representations and do not by
themselves prove the cause of failure. Together with the validation regressions,
they motivate an interpretation involving target scale and delayed credit. The
experiment does not isolate these mechanisms from the extra preferences.

The same allowed episode cap and early-stopping rule are used, but actual training
durations differ: cash-flow mean episodes are DQN 430, DDPG 355, TD3 255 and SAC
410; headline means are 430, 575, 635 and 720. This is a comparison under the
declared selection/stopping protocol, not a claim about asymptotic optimality or
equal realized gradient-update budgets. The new arm consumed 14,500 episodes.

## Inventory and auction behavior

| Algorithm | Cash-flow inventory at auction open | Headline inventory at auction open | Cash-flow direct auction value | Headline direct auction value |
|---|---:|---:|---:|---:|
| DQN | 8.67 | 6.40 | -1.30 | 0.18 |
| DDPG | 1.10 | 6.15 | -0.41 | 0.70 |
| TD3 | 0.52 | 4.25 | -0.03 | 0.60 |
| SAC | 0.70 | 4.37 | -1.22 | 0.48 |

Inventories are mean absolute units from an initial 100. Auction value is in bps,
relative to no auction orders after that policy's own CLOB trajectory; it includes
price edge, fees and inventory-risk relief. It is not a cross-policy intervention
holding all CLOB decisions fixed.

The cash-flow continuous policies deplete almost all inventory before the auction.
DDPG and SAC subsequently increase mean absolute inventory, to 2.36 and 4.31 units
respectively, with negative economic auction contributions. TD3 finishes close to
flat and obtains virtually no auction value. Cash-flow DQN finishes with 9.98
absolute units and a larger inventory penalty than its headline counterpart.

The headline gain is not exclusively lower terminal risk. Relative to cash-flow
training, headline DDPG and TD3 earn 3.57 and 5.26 bps more raw PnL while incurring
0.50 and 0.58 bps more terminal risk penalty. They retain inventory and execute
more profitably rather than maximizing liquidation speed. SAC's gain is mostly
raw PnL; DQN's gain is mostly lower inventory penalty. These statements are
descriptive decompositions, not identification of a particular reward term.

## Appropriate paper claim

> With forecast observations and H-centred action grids held fixed, the complete
> shaped-training scheme improves held-out economic performance over literal
> cash-flow training for all four algorithms and all ten paired seeds. Cash-flow
> training generally fails to improve the continuous policies from initialization
> and produces little or negative auction value. Separate controls show that
> dense credit assignment is useful, while the additional manuscript reward
> preferences have no demonstrated average economic benefit at the tested
> fixed-anchor calibration.

The earlier economic-dense results remain relevant: economically trained policies
can perform well with appropriate reward conditioning. Do not interpret the new
comparison as proof that optimizing the economic objective is intrinsically
inferior, or that each fictive reward is beneficial. This is synthetic evidence;
there is no historical cash-flow training extension in the completed matrix.

## Verification

All 40 control/headline pairs passed completion-hash, config-matching, reward and
replay checks. All 40 published seed effects were reproduced. Training, validation
and test simulation streams are disjoint. Evaluation includes no fictive rewards;
the cash-flow evaluation's gross-wealth-minus-initial-value identity has maximum
error 5.4e-12. The commits differ only by the follow-up configuration, orchestration,
reporting, tests and documentation; learner and simulator source are unchanged.

Evidence tables are in `docs/analysis_cashflow_v19/`. No training, checkpoint
selection, production-output edits, or protected manuscript edits were performed
for this analysis.
