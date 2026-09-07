# Simulator specification

The executable specification combines [base.yaml](../configs/base.yaml) with a
price-source, learner and optional treatment overlay. This document summarizes
the current model; [rl_design.md](rl_design.md) specifies learning and evaluation.

## Trading and settlement

An agent starts with 100 inventory units at a rebased midprice of 100. It can sell
through a continuous limit order book for 120 minutes, followed by 30 decisions
in a closing call auction. The tick is 0.01. Synthetic rough-Heston paths or
historical quote midpoints supply exogenous prices; both settings simulate order
flow and liquidity. Historical prices are frozen at the auction opening.

During the CLOB phase, each strategic sell order has same-price priority,
expires after one decision interval and cannot oversell inventory. Only residual
exogenous liquidity initializes the auction. The raw projected clearing estimate
is calibrated toward current midprice using frozen, training-fitted time-bin
weights before the policy observes it.

Auction actions submit signed linear schedules, with slopes on the grid
`0.25 * {0,...,32}` and local quote offsets on the tick grid. Multiple schedules
can remain live. Cancel-all removes earlier strategic schedules before inserting
the replacement; it incurs a fee. Observations use the lagged indicative price.
The last strategic action enters clearing without a further exogenous update.
H-off treatments anchor quotes to frozen midprice to avoid leaking the hidden
forecast through execution.

Settlement rounds the continuous clearing root half-up to the tick grid. The
agent's schedules are netted before pro-rata allocation. Actual fills may be
negative and final inventory is not clipped. Residual inventory is marked at
the frozen exogenous midprice, including when it is negative.

See [mdp.py](../src/lmm/env/mdp.py), [action_spaces.py](../src/lmm/env/action_spaces.py)
and the market modules under [src/lmm/market](../src/lmm/market).

## Objectives and accounting

Let C denote realized sale proceeds less purchase costs and cancellation fees,
I the final inventory, and M the frozen exogenous midprice. Economic PnL is
`C + M*I - S0*I0`; risk-adjusted PnL subtracts `lambda*I²`, with `lambda=0.01`.
Checkpoint selection and headline economic comparisons use the latter.
Implementation shortfall relative to initial midprice is the negative of PnL
under this residual-marking convention. It includes the marked value of unsold
or short inventory; it is not a completed-sale-only execution statistic.

Headline training instead uses a preference objective. CLOB proceeds are
attenuated when execution falls below the forecast. Auction submissions receive
fictive credit `omega * (x + q*max(-x,0))`, where
`x = K*H*(H-Sa)` and `omega=0.0001`. Cancellation reverses the exact previously
credited amount. Terminal purchase attenuation uses the same weight on actual
aggregate auction cash. Actual cash, fees, residual marking and inventory risk
are not multiplied by this weight. The headline has `q=0` and `k_star=10000`;
other preference terms remain active. These formulas are implemented in
[rewards.py](../src/lmm/env/rewards.py).

Initial-value centering subtracts `S0*I0` once from the environment return when
enabled. Replay conditioning then adds potential differences and a frozen
reference exposure's negative price return. Terminal potential is zero, so
potential differences sum to minus the initial potential. The reference
adjustment is common to policies sharing the same exogenous path and frozen
reference; it need not have zero historical expectation. Neither replay
adjustment enters reported training return or economic PnL. Raw cash-flow
controls disable centering and conditioning, so return aliases must not be
compared across arms. See [metric definitions](metrics_schema.md).

## Scope of the evidence

Learners receive reduced observations and deterministic input transforms; the
model makes no Markov-sufficiency claim for those observations. DQN enumerates
discrete actions, while DDPG, TD3 and SAC project continuous proposals onto the
executable grid. Phase-specific policies are selected jointly on economic
validation performance.

The study uses ten training seeds per cell, equal-seed summaries and pointwise
uncertainty intervals. Historical dates were inspected during development.
Historical prices inside this simulator do not establish performance against
actual exchange liquidity. See the [results index](README.md) for completed
protocols, adverse outcomes and treatment comparisons.
