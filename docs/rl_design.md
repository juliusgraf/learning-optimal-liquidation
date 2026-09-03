# Revised RL design

This document records the implemented learning contract. The resolved active
configuration is authoritative for experiment selection; the manuscript gives
the mathematical model, and `manuscript_update_notes.md` records any
author-facing changes still needed. This file explains how state, actions,
chronology, rewards, and evaluation criteria reach the learners.

## Physical clock

Synthetic and historical environments share one minute as their physical time
unit. The CLOB runs from minute 0 to auction opening at `tau_op=120`; the
closing auction then has 30 one-minute action intervals and clears at
`tau_cl=150`. CLOB `lambda0` values are intensities per minute, and auction
arrival/cancellation probabilities apply once per minute. The rough-Heston
Euler scheme converts its nonuniform minute grid to trading years with
`s_star=98,280` trading minutes per year. This affects calendar-time dynamics,
but not the number of simulator decisions or network input dimensions.

## Shared simulator contract

Rough-Heston and historical-midquote runs differ only in the exogenous
mid-price branch and run identity. They share the complete market and learner
configuration: `lambda0=1`, `V_inf=2`, `rho_lob=0.96`, exogenous book depth
200, strategic CLOB offset bound 12, auction flow, action semantics, centered
economic reward, validation design, and algorithm hyperparameters. Setting
overlays contain no simulator or learning overrides. The active historical
branch is therefore a historical mid-price replay inside the same simulated
CLOB/auction—not a separately calibrated simulator.

## Objective and Bellman targets

The finite-horizon learning objective is undiscounted. Every nonterminal target
uses Bellman factor one, including transitions across irregular CLOB time gaps.
The final auction transition stores the step reward and known terminal reward
separately and adds the latter exactly once, with no terminal network
bootstrap.

At the phase boundary, the last CLOB transition evaluates its next action with
the auction target network. DQN uses Double-DQN selection/evaluation when the
resolved config enables it; every Bellman maximization is masked by the next
state's admissible action set.

All four methods store rewards at the common scale `1e-3`. None clips rewards.
Each environment transition creates at most one eligible optimizer update from
that transition's phase. Repeated calls cannot manufacture extra updates.

## Common observation

Both phases use the same 18-coordinate input in manuscript order:

```text
(time, inventory, h_cl, s_mid, decision_index,
 depth_ask, depth_bid, top_ask, top_bid,
 n_mm, n_buy, n_sell, cancel_admissible,
 own_slope, own_weighted_quote, exogenous_slope,
 auction_imbalance, exogenous_weighted_quote)
```

Phase-inactive coordinates are zero. Auction quantities describe the currently
accepted exogenous proposals and active strategic schedules strictly before the
current action. The indicative price is the lagged value available under the
manuscript chronology.

Physical time in minutes and decision index are divided by `tau_cl`. All other continuous
features are standardized using statistics fitted on dedicated training-only
paths and then frozen. The same transformed observation reaches behavior
policies, replay, online networks, target networks, actors, and critics.
Normalizer state is embedded in every checkpoint and reused unchanged for
validation and test. In the `H_cl`-off ablation, the normalized `h_cl`
coordinate is set to zero while the network dimension remains 18.

## Discrete actions

The canonical five-coordinate representation is `(v, delta, K, ell, c)`.

CLOB actions consist of one canonical no-order action `(0,0)` and every
integer `v=1..30`, `delta=0..12`, for 391 actions. The state mask enforces
`v <= floor(inventory)`; zero volume always has zero offset. A strategic CLOB
order lives only for the current realized interval and is removed before the
next snapshot/carry-over.

Auction actions use the full manuscript lattice `beta*{0,...,K_max}`, with
`beta=1` and `K_max=32`, and the manuscript local action
`ell in [-B_max,B_max]=[-10,10]`. At decision time the simulator derives

```text
b_t = round_half_up((H_t^cl-S_{tau_op}^mid)/alpha) + ell_t,
```

and masks a local action if the resulting absolute `b_t` is outside the
ambient band `[-B_inf,B_inf]=[-150,150]`. The
same `B_inf=150` support sets the exogenous quote bounds `M_1=-B_inf` and
`M_2=B_inf`; cross-configuration validation requires the two uses of `B_inf`
to agree. `B_max=10` is the local manuscript-action half-width. With
cancellation enabled the grid has
`2 + 2*32*21 = 1,346` actions. Without cancellation it has
`1 + 32*21 = 673`. Zero slope has `ell=0`. Cancellation removes every live
strategic schedule submitted strictly before the current action; the new
current action is never canceled by its own bit. Without cancellation, a new
positive-slope action leaves every earlier schedule live; with cancellation,
it first removes all earlier schedules and may then submit a replacement.
Equal DQN Q-values choose the first action in lexicographic order.

Auction inventory is not clipped or bounded by a hidden liquidation
constraint. Two-sided learned schedules may buy. Clearing and accounting use
the actual signed pro-rata fill.

## Continuous proposal relaxation

DDPG, TD3, and SAC output tanh-normalized raw proposals in `[-1,1]`. The CLOB
actor has two coordinates. The auction actor has three coordinates when
cancellation is enabled and two otherwise. The environment adapter alone maps
these proposals to the same market action semantics used by DQN:

- CLOB volume/offset are half-up rounded, inventory-admissible, and canonical at
  zero volume;
- auction slope/local `ell` are half-up rounded, the nonnegative-reference condition
  is enforced, and the cancellation threshold is applied only when admissible.

Consequently all four learned methods use the same 21 local integer templates.
They are indicative-centred when `actions.auction_anchor=indicative` (the H-on
arms) and frozen-auction-open-mid-centred when it is `frozen_mid` (the H-off
arms), while every executed order still carries an absolute frozen-mid `b`.
DQN and the projected continuous actors share all 33 integer
slope levels `0,...,32`; they differ in raw proposal parameterization, not
executable slope support. A continuous proposal is clipped to the nearest ambient boundary;
a discrete proposal outside that boundary is masked.

Replay stores the committed normalized proposal, not a reconstructed market
action. DDPG/TD3 exploration noise and TD3 target smoothing are applied in
normalized proposal coordinates before projection. Projection statistics are
saved as evaluation diagnostics.

## Two-phase networks and replay

Each method has separate CLOB and auction function approximators and replay
buffers because the action spaces differ structurally. The observation size is
identical. Both settings use two 128-unit hidden layers and phase-specific
replay warm-ups of 2,000 CLOB and 512 auction transitions. Phase-local target updates occur only after a successful update from
that phase. Replay retains the next phase and admissibility mask so the CLOB to
auction junction is explicit.

The environment therefore changes its exposed Gymnasium `action_space` at the
CLOB-to-auction boundary. The repository's two-head episode loop and
continuous adapter handle that transition explicitly. Generic Gymnasium
wrappers that assume one stationary action space are not supported without an
additional fixed-space wrapper.

DQN uses masked epsilon-greedy exploration, uniform over admissible actions,
with the same probability in both phases. The active linear schedule is
`epsilon=1` for episodes 0--49, decays to `0.01` at episode 650, and remains
there. Each phase uses a coordinate-conditioned Q network rather than one
independent output parameter vector per discrete action. The normalized CLOB
coordinates are `(v/V_max,delta/L_max)` and the normalized auction coordinates
are `(K/(beta*K_max),ell/B_max,c)`. A two-layer state trunk produces a scalar
value and a rank-32 query; a shared two-layer action encoder produces the
rank-32 action embedding. The score is

```text
Q(s,a) = V(s) + <query(s), embedding(a)>/sqrt(32)
         + w_prior * 1{a is not the canonical no-op}.
```

Greedy selection and masked Double-DQN targets evaluate the complete exact
grid. Replay updates evaluate only the sampled action, avoiding an unnecessary
batch-by-grid tensor. This architecture shares statistical evidence across
the structured coordinates of the 1,346-action auction lattice without
relaxing or coarsening that lattice. Checkpoints record architecture identifier
`coordinate_conditioned_bilinear_v1` and reject checkpoints from the earlier
independent-output head. They also validate the simulator, reward, action-grid,
feature, Bellman, and DQN-hyperparameter contract, so a same-shaped tensor
checkpoint cannot be loaded under different economic or action semantics.

For the auction network, the state value/query heads start at zero and the
trainable non-no-op coefficient starts at `-0.02` in replay-scaled reward
units. The canonical no-order action is therefore the unique initial greedy
auction action. This is an initialization prior, not an admissibility or
safety constraint, and gradient updates can reverse it.
For exact reproduction, DDPG, TD3, and SAC likewise initialize their auction
actors at the projected no-order action `(K,ell,c)=(0,0,0)`: the auction output
weights are zero and the output biases are the inverse-`tanh` coordinates that
project to that action. SAC applies the same rule to its mean head and starts
its auction log-standard-deviation bias at `-2.5`. These are initialization
priors only; the full admissible action set remains available from episode
zero.
DDPG uses one critic, TD3 two critics and delayed actor updates, and SAC twin
critics plus entropy regularization. Their algorithm-specific hyperparameters
are resolved from `configs/algo/*.yaml`; the environment/reward/normalization
contract is shared.

## Reward and accounting separation

Headline training has shaping disabled. With `lambda_inv=2` and cancellation
fee parameter `d=0.1`, it uses actual CLOB execution cash, actual rationed
auction cash, the residual inventory mark, cancellation fees, and the terminal
inventory penalty. Initial-inventory centering removes the policy-invariant
constant `S0*I0` incrementally. Thus the complete headline training return,
periodic validation return, and final evaluation return all equal
`risk_adjusted_pnl` to numerical tolerance.

The exact manuscript three-regime shaped reward is retained in the two
explicit `shaping_on` treatments with `q=1` and exact cancellation clawback.
For an auction order submitted at decision `s`, define

```text
x_s   = K_s * H_s * (H_s - S_s)
phi_s = x_s + f_a(x_s).
```

If `L_t` is the set of strategic schedules live immediately before action
`t`, the implemented interim reward is

```text
r_t = phi_t - d_t*c_t - c_t * sum(phi_s for s in L_t).
```

The subtraction uses each order's original signed credit, not a revaluation at
the current indicative price. Cancellation occurs before the current schedule
is inserted, so a replacement does not cancel itself. Multiple schedules may
accumulate between cancellations. Pathwise over the undiscounted auction, each
submitted schedule's credit is either retained if the schedule survives to
clearing or exactly reversed by the first later cancel-all action, less all
cancellation fees. Centering remains the same policy-invariant adjustment in
these treatments. Validation and final evaluation always instantiate a
separate economic-only reward contract and record CLOB cash, terminal auction
cash, cancellation fees, residual mark, and the inventory penalty.

Primary policy selection and comparison use:

```text
pnl = CLOB cash + auction cash + residual mark
      - initial inventory value - cancellation fees
risk_adjusted_pnl = pnl - lambda_inv * I_final^2
```

No shaping term enters either field. With centering in the economic validation
and test environments, the episode return equals `risk_adjusted_pnl` to
numerical tolerance.

The shaped objective is not the economic objective. With `q=1`,
`phi_s=max(x_s,0)`. Under the indicative-centred local grid, any positive-slope
proposal below `H_s` receives positive fictive credit at submission regardless
of whether it later fills. Because `c=0` leaves earlier schedules live, a
policy can stack such schedules and retain every credit through clearing.
Clawback prevents canceled schedules from retaining their original credits;
it cannot reverse credits for schedules deliberately left live. The `q=1`
terminal correction also neutralizes negative purchase-side auction cash in
the shaped training signal. Consequently the shaped treatment can reward live
order accumulation and purchases while economic risk-adjusted PnL deteriorates.
This is an objective mismatch, not a centering or evaluation-accounting error.
Economic validation still ranks eligible checkpoints, and benchmark claims
require held-out confirmation.

## Validation, testing, and comparison

The normalizer uses training-only episodes. The risk-neutral AS benchmark
derives `A=lambda0/gamma_m` from the configured flow and uses a separately
seeded order-book impact regression for `k`; no volatility parameter is
calibrated because `gamma=0`. Periodic validation replays the policy under the
economic-only reward on a fixed validation seed set. A candidate becomes
reportable only after the CLOB and
auction learners reach `5,000` and `2,000` optimizer updates, respectively.
The headline DQN exposes the complete admissible auction grid from episode
zero and uses the same masked epsilon-greedy probability in both phases; there
is no forced no-order curriculum. Before both gates pass, validation scores
are diagnostics: they neither create `best.pt` nor consume patience. The first
eligible validation initializes the race, after
which ordinary joint-policy patience applies. The untrained policy's economic
score is saved only as a diagnostic reference and is never eligible for
selection. The active configuration sets
`checkpoint_require_initial_improvement=false`, so every mature validation is
reportable even when it scores below that initial reference. `best.pt`
maximizes mean `risk_adjusted_pnl` among mature candidates; `best_mature.pt`
is retained as redundant diagnostic provenance. A run fails selection only if
its budget never reaches the phase-maturity requirements. Final evaluation
uses the same economic-only contract on a separate fixed test seed set and
defaults to `best.pt`.

The two phase networks are not checkpointed independently. The CLOB Bellman
target bootstraps from the auction network at the junction, while the auction
policy is evaluated under inventories generated by the CLOB policy. Splicing
phase networks selected at different episodes would therefore evaluate a pair
that was never jointly validated and can create an out-of-distribution phase
boundary. Phase-specific maturity with joint economic selection preserves the
coupled control problem without changing its rewards or transition law.

Within each test episode, the learned policy, initial network, AS, and TWAP see
the same policy-independent realized CLOB tape, midprice path, and auction
proposal stream. Saved policy-minus-benchmark curves therefore use paired
common random numbers. Cross-seed confidence intervals are formed from paired
seed-level differences. Historical cross-asset outputs include currency and
basis-point forms.

The matched treatment set is:

1. observed `H_cl` feature and H-centred action anchor absent, shaping absent;
2. observed `H_cl` feature and H-centred action anchor present, shaping absent;
3. observed `H_cl` feature and H-centred action anchor absent, shaping present
   (so the H-derived reward channel intentionally remains);
4. observed `H_cl` feature and H-centred action anchor present, shaping present;
5. no auction, with both the `H_cl` feature and all auction-derived shaping
   disabled;
6. observed `H_cl`, H-centred action anchor, shaping absent, and no strategic
   cancellation.

The canonical synthetic headline runs are reused for cell 2 rather than
retrained under an ablation alias. Cell 4 is the explicit
`ablation_h_on_shaping_on` treatment.

The primary information comparison uses the two shaping-off arms. The paired
cross-treatment report additionally gives the H/anchor bundle effect at both
shaping levels, the shaping effect at both H/anchor levels, the full
auction-aware treatment versus the bundled no-auction comparator, and
cancellation on minus off. It pairs runs within algorithm and master seed and
requires exact evaluation-episode/environment-seed alignment before forming
seed-level differences.
