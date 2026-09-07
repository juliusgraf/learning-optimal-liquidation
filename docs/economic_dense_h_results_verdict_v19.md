# Verdict: H-anchored economic training with matched conditioning

All 40 follow-up runs completed during this analysis. The automatically generated
report is `results/revision_v19_economic_dense_h/_comparison/index.html`.

The extra manuscript reward preferences have no demonstrated average economic
benefit when H anchoring and the complete reward conditioning are held fixed.
All four paired 95% seed-bootstrap intervals include zero. This is not an
equivalence test and does not prove that the preferences have exactly zero effect.

| Method | Headline J | Conditioned economic | J minus economic [95% CI], bps |
|---|---:|---:|---|
| DQN | 3.270 | 3.329 | -0.059 [-0.396, 0.276] |
| DDPG | 3.692 | 3.748 | -0.056 [-0.440, 0.485] |
| TD3 | 4.068 | 3.879 | 0.188 [-0.116, 0.493] |
| SAC | 3.920 | 4.120 | -0.200 [-0.564, 0.131] |

J wins 5/10 DQN seeds, 2/10 DDPG seeds, 7/10 TD3 seeds and 5/10 SAC seeds.
DDPG's small mean difference masks one relatively large favorable J seed;
its median J-minus-economic difference is -0.146 bps. There is no consistent
preference advantage across algorithms or seeds.

## What resolves the raw-cash-flow learning difficulty

Relative to the completed raw-cash-flow comparator, conditioned economic training
improves the economic objective by:

| Method | Conditioning-package gain [95% CI], bps |
|---|---|
| DQN | 3.627 [1.250, 7.152] |
| DDPG | 3.120 [2.264, 4.086] |
| TD3 | 4.489 [3.704, 5.357] |
| SAC | 5.565 [4.500, 6.684] |

These are paired contrasts with the same H observation and H anchor. They
measure the full conditioning package: dense auction and CLOB potentials,
initial-value centering and the market control variate. They do not identify
which component alone supplies the gain. The earlier matched dense-versus-sparse
auction experiment separately supports the auction-credit contribution.

Conditioned economic training improves selected economic validation over the
initial policy in 39/40 cases: 10/10 DQN, 10/10 DDPG, 9/10 TD3 and 10/10 SAC.
The remaining TD3 case is seed 314. Final validation improves over initial in
38/40 cases. Required metrics and observed critic losses are finite. Late loss
levels are similar in order of magnitude to headline learning, in contrast to
the gross-cash-flow comparator's much larger losses and widespread continuous
policy regressions. Residual seed-level regression is still possible.

The new run consumed 24,200 training episodes. Mean episode counts were 455 for
DQN, 575 for DDPG, 620 for TD3 and 770 for SAC. Early stopping and checkpoint
selection remain part of the comparison; this is not a statement about exact
optimal policies or equal realized update counts.

## Auction use does not require the extra preferences

Mean direct economic auction contributions are +0.141 bps for DQN, +0.601 for
DDPG, +0.492 for TD3 and +0.463 for SAC. Headline means are +0.182, +0.703,
+0.597 and +0.483 respectively. These are positive averages, not a claim that
every interval or seed is positive. Direct contributions hold each policy's own
CLOB trajectory fixed against submitting no auction orders; they do not hold
CLOB trajectories fixed across different learned policies.

Economically trained policies therefore obtain useful auction behavior without
the extra fictive preferences, given the common information and conditioning.
The original motivation of making delayed auction consequences learnable is
supported. The stronger claim that the additional manuscript preferences improve
economic performance is not supported at this calibration.

## Write-up consequence

Keep the declared headline results and report this comparator prominently.
The defensible finding is that dense credit and reward conditioning make the
economic problem learnable, while the extra forecast-relative reward preferences
yield small, mixed, statistically inconclusive economic effects. Do not use the
large J-versus-raw-cash-flow gap as evidence for those preferences specifically.
Do not equate no demonstrated advantage with a proved absence of economic cost.
All these follow-up mechanism findings are synthetic, not new historical tests.

## Verification and scope

All 40 pair configurations, completion hashes, economic/replay identities and
training/validation/test seed separation checks passed. All reported paired
effects were reproduced from episode records. Maximum economic training-return
identity error is below 9.3e-12. The reusable audit and evidence tables are in
`docs/analysis_dense_h_v19/`.

No training was launched or interrupted for this analysis; the already-running
campaign finished normally. No production artifacts or protected TeX files were
modified.
