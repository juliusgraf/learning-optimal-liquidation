# Revised RL design

This document records the implemented learning contract. The revised
manuscript is authoritative for the model; this file explains how its state,
actions, chronology, rewards, and evaluation criteria reach the learners.

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

Physical time and decision index are divided by `tau_cl`. All other continuous
features are standardized using statistics fitted on dedicated training-only
paths and then frozen. The same transformed observation reaches behavior
policies, replay, online networks, target networks, actors, and critics.
Normalizer state is embedded in every checkpoint and reused unchanged for
validation and test. In the `H_cl`-off ablation, the normalized `h_cl`
coordinate is set to zero while the network dimension remains 18.

## Discrete actions

The canonical five-coordinate representation is `(v, delta, K, b, c)`.

CLOB actions consist of one canonical no-order action `(0,0)` and every
integer `v=1..30`, `delta=0..12`, for 391 actions. The state mask enforces
`v <= floor(inventory)`; zero volume always has zero offset. A strategic CLOB
order lives only for the current realized interval and is removed before the
next snapshot/carry-over.

Auction actions use `K=beta*k`, `k=0..10`, offsets `b=-25..25`, and a cancel-all
bit. With cancellation enabled the canonical grid has
`2 + 2*10*51 = 1,022` actions. Without cancellation it has
`1 + 10*51 = 511`. Zero slope has zero offset. Cancellation removes every live
strategic schedule submitted strictly before the current action; the new
current action is never canceled by its own bit. Equal DQN Q-values choose the
first action in lexicographic grid order.

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
- auction slope/offset are half-up rounded, the nonnegative-reference condition
  is enforced, and the cancellation threshold is applied only when admissible.

Replay stores the committed normalized proposal, not a reconstructed market
action. DDPG/TD3 exploration noise and TD3 target smoothing are applied in
normalized proposal coordinates before projection. Projection statistics are
saved as evaluation diagnostics.

## Two-phase networks and replay

Each method has separate CLOB and auction function approximators and replay
buffers because the action spaces differ structurally. The observation size is
identical. Phase-local target updates occur only after a successful update from
that phase. Replay retains the next phase and admissibility mask so the CLOB to
auction junction is explicit.

DQN uses masked epsilon-greedy exploration, uniform over admissible actions.
DDPG uses one critic, TD3 two critics and delayed actor updates, and SAC twin
critics plus entropy regularization. Their algorithm-specific hyperparameters
are resolved from `configs/algo/*.yaml`; the environment/reward/normalization
contract is shared.

## Reward and accounting separation

Training uses the manuscript's three-regime shaped reward. Evaluation separately
records CLOB cash, terminal auction cash, cancellation fees, residual mark,
inventory penalty, and each shaping adjustment.

Primary policy selection and comparison use:

```text
pnl = CLOB cash + auction cash + residual mark
      - initial inventory value - cancellation fees
risk_adjusted_pnl = pnl - lambda_inv * I_final^2
```

No shaping term enters either field. With shaping disabled, the episode return
equals `risk_adjusted_pnl + initial inventory value` to numerical tolerance.

## Validation, testing, and comparison

The normalizer and benchmark calibration use training-only streams. Periodic
validation uses a fixed validation seed set and chooses `best.pt` exclusively
by mean `risk_adjusted_pnl`. Final evaluation uses a separate fixed test seed
set and defaults to `best.pt`.

Within each test episode, the learned policy, initial network, AS, and TWAP see
the same policy-independent realized CLOB tape, midprice path, and auction
proposal stream. Saved policy-minus-benchmark curves therefore use paired
common random numbers. Cross-seed confidence intervals are formed from paired
seed-level differences. Historical cross-asset outputs include currency and
basis-point forms.

The matched treatment set is:

1. `H_cl` absent, shaping absent;
2. `H_cl` present, shaping absent;
3. `H_cl` absent, shaping present;
4. `H_cl` present, shaping present;
5. no auction;
6. no strategic cancellation.

The primary information comparison uses the two shaping-off arms.
