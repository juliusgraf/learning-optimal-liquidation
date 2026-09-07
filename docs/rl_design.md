# Learning and evaluation contract (v19)

`configs/base.yaml` and the algorithm/treatment overlays are the executable
specification. `paper/main.tex` defines the control problem. The protected
manuscript has not been edited: `docs/manuscript_recommendations_v19.patch` gives
precise proposed changes, and `docs/pathology_repair_v19.md` records the diagnosis
and limits of the bounded verification.

## Economics and observations

Both price sources share the 120-minute CLOB and 30-minute closing auction,
one-sided CLOB liquidation, signed persistent auction schedules, cancellation
chronology, tick clearing and actual pro-rata allocation. The historical
setting replays midprices within the same stylized simulated market; it is
not a replay or calibration of historical order flow or exchange liquidity.
The calibration uses alpha=.01, beta=.25, lambda=.01, k_star=10000,
q=0, auction shaping weight=.0001, d=.001, and exogenous slope support
[1,20]. Price units are rebased; the liquidity parameters are stylized.
Continuous arrival intensity is .5 per minute per side, about twice the parent
in expected contra-side volume. This is a substantial-participation scenario,
not an estimate of exchange tape liquidity. The earlier calibration audit is
archived as `docs/pathology_repair_v18.md`; see [recovery instructions](cleanup.md).

All 391 CLOB and 1,346 auction actions remain. The auction slope lattice is
`0.25 * {0,...,32}`; its maximum is 8 per schedule. Multiple schedules
may remain live, and aggregate fills and terminal inventory are never clipped.
AS/TWAP retain their manuscript formulas and the same per-schedule slope cap.
They are deterministic reference policies, not deterministic market outcomes
or members of the same two-sided auction policy class.

The raw observation remains the common 18-coordinate manuscript vector:

```
(time, inventory, h_cl, s_mid, decision_index,
 depth_ask, depth_bid, top_ask, top_bid,
 n_mm, n_buy, n_sell, cancel_admissible,
 own_slope, own_weighted_quote, exogenous_slope,
 auction_imbalance, exogenous_weighted_quote)
```

Before fitting standardization, `h_cl` becomes `h_cl-s_mid`; each weighted
quote becomes `weighted_quote-s_mid*slope`. The original midprice and slopes
are retained, so this is a deterministic invertible coordinate change in
H-on runs. H-off runs set the transformed H coordinate to zero. H observation and auction
anchor now have independent switches; the controlled H comparison holds the
anchor at frozen midprice in both arms. No future grid, arrivals or prices
enter any feature or action.

Before use as a future-close signal during the CLOB, Algorithm 1's raw output
is calibrated as `H=mid+w[b]*(H_raw-mid)` in four equal time bins. Restricted,
no-intercept least squares fits the four weights on 256 independent training
no-order paths, with actual closing price as the target. Synthetic weights
are `[0, .190153, .234565, .414256]`; historical training-pool weights are
`[.192784, .314044, .386180, .525631]`. All weights are frozen for every algorithm
and treatment. The internal raw estimator and exogenous carryover are unchanged.
During the auction, the original indicative calculation is used. This is a
change to the implemented forecast and hence the forecast-dependent shaped J,
not a change to settlement or order flow. Its validation improves CLOB MAE;
it does not guarantee economic improvement. The original indicative auction
anchor remains the headline default. See the v19 report for errors by phase.

The normalizer is fitted on 16 independent training paths and frozen. Its
four calibration modes cover random actions, persistent auction schedules,
CLOB abstention followed by persistent schedules, and auction abstention.
Uniform cancellation-heavy paths alone underrepresent accumulated exposure.
Time and decision index divide by tau_cl; the cancellation bit passes through.
Checkpoints store and validate the fitted transform.

DQN, DDPG and TD3 fit separate CLOB/auction population statistics. Before
fitting, their auction inventory and continuous-root residual exposure use
the signed transform `asinh(x / I_s)`, where `I_s=alpha/lambda` (I0 if lambda
is zero). This inventory scale equates one-tick execution value with the
quadratic inventory cost. The transform is invertible and unbounded. Time
and decision index retain their existing scaling. SAC retains pooled linear standardization. Thus the headline comparison includes the specified
preprocessing, not an isolated causal comparison of algorithm update rules.

The supplementary `representation_pooled.yaml` and
`representation_phase_asinh.yaml` overlays in `configs/treatment/` apply
one representation uniformly to all four methods. They are bounded
representation controls outside the existing publication treatment matrix;
their distinct labels prevent confusing them with headline artifacts.

## Rewards

Headline training uses the author-approved weighted shaped J: only fictive
auction credits and their exact cancellation clawbacks are multiplied by
.0001. Economic cash, fees, risk and clearing are unchanged. This is a change
to the manuscript objective, not an invariant normalization. q=0 removes the purchase
subsidy; it does not remove the CLOB multiplier or the interim auction term.
Initial-inventory centering subtracts the policy-invariant S0*I0 once across
the episode. Thus `training_return` reports a sample of J-S0*I0.

A training-only control variate uses `r_t + Phi(next)-Phi(now)` before replay
scaling. During CLOB trading,
`Phi=(s_mid-S0)*I-lambda*max(I-I0*(1-t/tau_op),0)^2`. During the auction,
using the observed slope and weighted-quote sums,

```
delta = (W_own-mid*G_own + W_exo-mid*G_exo + taker_imbalance)
        / (G_own+G_exo)
z_hat = G_own*delta - (W_own-mid*G_own)
Phi = (mid-S0)*inventory + delta*z_hat - lambda*(inventory-z_hat)^2
```

The terminal potential is exactly zero. The economic-sparse mechanism removes
only `delta*z_hat-lambda*(inventory-z_hat)^2` in the auction through
`rl.learning_auction_inventory_potential=false`; mark-to-mid centering and
CLOB conditioning remain. Since midprice is frozen during the auction,
a persistent unchanged schedule then receives no action-dependent intermediate
potential credit. Both dense and sparse versions have the same episode
objective after the terminal correction. This approximate continuous-root
liquidation value is only a control variate; actual clearing, fills and cash
continue to use the original simulator. The potential telescopes. In the
rebased settings Phi(initial)=0. In addition, replay subtracts the realized
price return of a frozen deterministic inventory reference curve, evaluated
at the midpoint of each CLOB decision interval. The reference averages the
normalizer's training inventory trajectories, excluding deliberate CLOB
abstention, and is interpolated onto minute times. On a common exogenous
price/clock path the adjustment is identical for every policy, including
historical nonmartingales; its historical expectation need not be zero.
Thus policy differences are preserved, while the diagnostic replay sum is
`training_return + potential_adjustment + market_baseline_adjustment`.
Reported J and PnL omit both adjustments. No reward clipping is used.

Validation, checkpoint selection, AS/TWAP comparisons and final evaluation
instantiate `economic_evaluation_config`, which disables all manuscript
shaping and the training control variate. Their metric is

```
pnl = CLOB cash + actual auction cash + frozen-mid residual mark
      - initial inventory value - cancellation fees
risk_adjusted_pnl = pnl - lambda * final_inventory^2
```

Reported PnL never includes shaping or centering a second time. Evaluation
returns equal risk-adjusted PnL. The training and economic metrics must not
be conflated in plots or interpretation.

## Algorithms and replay

DQN uses the existing coordinate-conditioned rank-32 Q architecture with two
256-unit hidden layers per phase, masked Double-DQN targets, MSE loss,
learning rate 0.00015, Polyak coefficient 0.005 and norm clipping at 10.
Each state hidden layer has non-affine LayerNorm before its activation. The
learned action embedding uses non-affine LayerNorm followed by tanh. Query
and value heads remain unrestricted; rewards and Q targets are not clipped.
AdamW applies weight decay .0001. The CLOB architecture identifier is
`coordinate_conditioned_normalized_v2`. The auction network uses
`coordinate_conditioned_reference_v3`, with
`Q(s,a)=V(s)+u(s)·(e(a)-e(noop))/sqrt(32)` and its former global non-noop
coefficient fixed at zero. The learned reference value is exactly `Q(s,noop)=V(s)`.
No-op transitions cannot create spurious action-embedding gradients, including
through batch-dependent floating-point roundoff. The CLOB network retains its
uncentered embedding and trainable global non-noop coefficient. Checkpoints
record architecture identifiers separately for both phases; legacy loading
preserves legacy defaults.

Both heads use ordinary initialization without a no-order margin. Epsilon=1
for five episodes, declining to 0.05 over the following 90 episodes. After the
common warm-up, CLOB epsilon exploration remains uniform over admissible actions.
Auction epsilon exploration uses an equal mixture of uniform sampling among
admissible zero-slope controls (wait/cancel-only) and uniform sampling over all
admissible actions. This addresses control undercoverage on the 1,346-action
grid; every admissible action retains positive sampling probability. Greedy
evaluation is still the exact masked Q argmax with no benchmark fallback.

All four methods use one-step replay; terminal reward is included once.
The auction follow-up reverted DQN's intermediate three-step setting because
its uncorrected off-policy returns charged earlier actions for subsequent
exploratory trades and cancellations. DQN again uses the standard one-step
Double-DQN target. DDPG, TD3 and SAC retain their native library updates.
Every method uses undiscounted continuation across the phase boundary.
Earlier multi-step and known-fee Q-decomposition trials remain in the
development ledger; the known-fee decomposition is inactive.

DDPG, TD3 and SAC use Stable-Baselines3 2.7.1 through `lmm.agents.sb3.SB3Agent`.
The production factory selects `algo.backend=sb3`. The older handwritten
classes remain solely for regression characterization; they are not selected
by the active continuous algorithm overlays. Their tests use explicit frozen
algorithm fixtures, while active integration tests exercise SB3 itself.

DDPG now uses critic-only hidden LayerNorm (two 256-unit layers, learned affine
parameters, before ReLU). The final scalar output is unrestricted. The SB3
policy factory builds both online and target critics this way. Actor layers,
single critic, actor update every step, absence of target smoothing, rates,
Polyak coefficient and native DDPG losses are unchanged. This addresses the
observed feedback between critic overestimation and actor over-selling; it
is not a conversion to TD3. A config switch allows the paired old-critic
control. TD3 and SAC do not enable this architecture change.

Each phase has a separate library model and replay buffer. `model.train(1)`
performs the library losses, backward passes, optimizer and target updates.
At a CLOB/auction junction, the replay adapter computes the current auction
continuation at sampling time, adds it to that row's reward and suppresses the
library's CLOB continuation for that row. It uses the auction target actor
and critic for DDPG/TD3 (including TD3 smoothing), and the current auction SAC
actor plus target critics and that phase's entropy coefficient for SAC.
Terminal cash/mark/penalty is included once with no network continuation.
Actor and critic optimizer pre-step hooks apply the common gradient norm
bound of 1; native library loss construction and updates remain intact.
These hooks are reinstalled when loading checkpoints.

The continuous methods retain the manuscript's normalized proposal mapping,
raw proposal replay, rounded executable orders and cancellation threshold.
A shared finite exploration design runs first; subsequent DDPG/TD3 exploration is Gaussian
with standard deviation 0.1. SAC uses automatic entropy adjustment initialized
at 0.001 in replay units, avoiding the old temperature of 1 overwhelming
small execution gains. There is no auction-specific saturated actor
initialization or benchmark imitation. The finite exploration phase below
collects experience; it does not supply optimized action labels.

Every method uses reward scale 1, replay capacity 50,000 per phase,
batch size 128 and a replay minimum of 512 transitions in each phase. Both
learners wait until all 32 structured warm-up episodes are complete. At most
one optimizer update is permitted per realized environment step. The common
physical horizon and Bellman factor of one are unchanged.

## Selection and treatments

Joint checkpoints are selected by mean economic validation score after at
least 2,000 CLOB and 2,000 auction updates (the auction gate is inactive in
no-auction runs). Phase networks are never spliced. The initial policy is a
diagnostic comparator, not a selectable checkpoint. Every training seed is
retained even if it does not improve on initialization. Validation uses 128
paths every 50 episodes; the first eligible checkpoint is also evaluated
immediately. Four eligible evaluations without improvement stop training,
with the same 800-episode cap. Final evaluation uses
separate seeds and common random numbers for all policies. A larger training
return is not evidence of greater economic value or benchmark superiority.

The canonical headline is H-on/shaping-on/auction-on with the indicative anchor.
Five additional synthetic cells isolate mechanisms, with identical training
seeds, market parameters and economic evaluation:

| Cell | H feature | Anchor | CLOB preference | Auction preference | Auction potential |
|---|---|---|---|---|---|
| Headline | on | indicative | on | on | on |
| Fixed-anchor full | on | frozen mid | on | on | on |
| Economic dense | on | frozen mid | off | off | on |
| Economic sparse | on | frozen mid | off | off | off |
| H feature off | off | frozen mid | off | off | on |
| No auction | off | frozen mid | off | off | inactive |

The report compares dense versus sparse economic training at common observed
budgets as well as at selected checkpoints; this measures credit assignment
without changing the economic objective. The H comparison holds the anchor
and rewards fixed. The anchor comparison retains H and the complete shaped J.
The fixed-anchor full arm tests the combined manuscript preferences. The
CLOB-only and auction-only arms remain optional; the final matrix does not
identify their separate effects. Auction access compares H-off economic
training with and without the auction. A same-CLOB auction-noop counterfactual
also measures direct auction execution value, without claiming to represent a
retrained no-auction optimum. Legacy bundled H/shaping and no-cancellation
overlays remain available outside the canonical matrix.

Schema 15 and the new environment contract prevent accidental reuse of v12
checkpoints or publication artifacts. Full production runs remain at the
configured 800-episode cap; development verification uses the separately
labelled, capped `scripts/diagnose_learning.py` and never consumes final-test
seeds or overwrites the manuscript outputs.

## Shared exploration and conditioning

The first 32 episodes cross four CLOB exploration modes (maximal submitted
volume at offset one, abstention, and two random fixed proposals) with four
auction modes (no order, persistent sales, persistent purchases, and
cancel/replace sales), twice. Random parameters use a dedicated stream derived
from the environment seed. Feasibility is enforced by the same action grid or
continuous adapter; evaluation never invokes this exploration. The enlarged
warm-up covers every combination before either learner starts optimization.
The per-phase replay minimum additionally protects minibatch diversity.
Zero-slope warm-up proposals use the canonical zero offset. Previously a
random nonzero offset could make the grid projection choose a positive-slope
order even in the designated abstention mode; this is fixed for all four methods.

All initial learning rates use the fixed factor `max(.1, 2**(-episode/90))`.
DDPG's actor starts at .0001 (the manuscript rate), while its critic and the
TD3/SAC networks start at .0003; DQN starts at .00015. Native SB3 optimizer
pre-step hooks apply the actor rate and the common gradient bound. No L2
critic regularization is active in the continuous methods; DQN uses the
AdamW decay described above. The rejected original-paper optimizer trial
remains an explicitly named diagnostic.

After relative-price conversion, replace the two auction weighted-quote
coordinates by the projected residual `I-z_hat` and displacement `delta`
defined above. This invertible transform retains the same 18 raw observations
and does not require H. It exposes inventory and price exposure to all four
learners without adding future information. The reference inventory curve and
feature transform are stored and restored with replay and RNG state.

The shaping-off treatment removes the manuscript's objective-changing reward
terms but retains the common invariant replay conditioning. It must be called
training on the economic objective with common numerical conditioning, not
an ablation of every form of intermediate reward transformation.

The default dataclass auction weight remains one for explicit legacy configs;
the active base sets .0001. All diagnostic parameter choices, including
rejected alternatives, are retained in the development ledger. Final-test
results are still required for publication claims about cross-seed rankings.

## Replications and historical scope

The publication matrix has ten fixed seeds (42, 7, 99, 123, 2024, 314, 577,
811, 1618, 2718), 440 runs and 100 economic test episodes per selected policy.
The final simulation RNG uses a fresh namespace, 19001, while preserving
matched environment seeds across policies and arms. Training RNG is unchanged.
Historical training and normalizer fitting sample whole rebased sessions
from all five stocks on the training dates only (50 sessions). Validation and
evaluation use the requested stock on their separate date partitions. The
v17/v18 test week has already been inspected: a fresh simulation RNG does not
make those historical dates an untouched holdout. Broader historical claims
require a subsequently frozen date block, not more resampling of those dates.
