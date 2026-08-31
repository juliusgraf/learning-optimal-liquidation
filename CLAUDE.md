# Project: revised Learning Market Making with Closing Auctions

## Source of truth and scope

The latest manuscript in `paper/main.tex` and the author's revision remarks
define the market, chronology, rewards, feature map, policy classes,
benchmarks, evaluation, and acceptance criteria. Do not edit `paper/` as part
of code work. The legacy implementation is retained only for audit and
characterization; it is not an implementation specification.

Revised runs use artifact schema 2 and the current environment-contract ID.
Never load checkpoints, replay, metrics, figures, tables, or result files from
an earlier contract. Write runs below `results/revision_v2/`.

## Environment contract

- Pre-sample independent CLOB Poisson paths and marks through auction open,
  build the complete strict random grid, and partition arrivals exactly once
  into `(t_i,t_{i+1}]`.
- A policy sees only the current time/index and current state; future arrival
  times and future grid points are simulator-internal.
- Refresh the exogenous CLOB snapshot before each action. A strategic CLOB
  order lasts one interval, receives priority at its level, and never enters
  carry-over.
- The final residual exogenous book reflects actual matching on
  `(t_n,tau_op]`. Recalibration replaces only the final Algorithm-1
  observation while retaining earlier moments.
- Freeze the midprice observed at auction open throughout the auction.
- At every auction decision, process all six current exogenous proposal
  indicators in order `B,D,J+,G+,J-,G-` before revealing the state. Apply each
  proposal transactionally; reject and restore if exogenous validity would
  fail.
- Exogenous validity requires total active slope at least `D_mu` and
  nonnegative aggregate intercept/imbalance. One initialized schedule is
  persistent and never cancellation-eligible.
- Market-order counters count accepted arrivals cumulatively. Cancellation
  zeros volume and never decrements a counter.
- The auction state/reward uses the indicative price available under the
  manuscript's lagged chronology. The final action is included in terminal
  clearing, and no event occurs after it.
- Strategic cancel-all removes every live prior strategic schedule, then the
  current action is submitted. It never cancels itself.
- Clearing uses a continuous root followed by half-up tick projection and
  terminal pro-rata allocation. Inventory/cash use actual signed fill; no
  hidden auction inventory bound or terminal clipping is permitted. Self-trade
  count is always zero.

## Features and actions

Both phases use the exact common 18-coordinate manuscript feature map:

```text
(t, I, H_cl, S_mid, i, L+, L-, V+1, V-1,
 M, N+, N-, C, A_own, B_own, G, imbalance, J)
```

Phase-inactive coordinates are zero. Normalize `t` and `i` by `tau_cl`; fit
all other continuous statistics on training paths only, freeze them, save them
with checkpoints, and reuse them for every behavior/replay/online/target path.
The `H_cl`-off treatment zeroes the normalized coordinate without changing
network dimensions.

The canonical action is `(v,delta,K,b,c)`:

- CLOB DQN: one `(0,0)` no-op plus integer `v=1..30`, `delta=0..12` (391).
- Auction DQN with cancellation: 1,022 actions.
- Auction DQN without cancellation: 511 actions.
- Zero volume/slope uses zero offset. Mask inadmissible CLOB inventory and
  cancel-all actions. Equal Q-values select the first lexicographic action.
- DDPG/TD3/SAC emit tanh proposals in `[-1,1]`; the adapter applies half-up
  snapping and state-dependent projection. Replay stores normalized proposals.

## Reward and accounting

Use

```text
f_c(u) = min(1, u_+ / (k_star*alpha))
f_a(x) = q*(-x)_+
```

CLOB reward is execution cash times `f_c`. Interim auction reward is
`x + f_a(x) - d_t*c_t`. Terminal reward uses actual aggregate rationed fill,
applies `f_a` once to aggregate auction cash, marks residual inventory at the
frozen auction-open mid, and subtracts `lambda*I_final^2`.

One switch disables all shaping while retaining economic cash and cancellation
fees. The baseline has no cancellation clawback; a clawback variant is an
ablation only.

Keep economic fields separate from shaping. Report:

```text
pnl = CLOB cash + auction cash + residual mark
      - initial inventory value - cancellation fees
risk_adjusted_pnl = pnl - lambda*I_final^2
```

Never include shaping in PnL. With shaping off, episode return equals
`risk_adjusted_pnl + initial inventory value`.

## Learning contract

- The finite-horizon objective is undiscounted; Bellman factor is one for every
  transition and method.
- Final terminal value is included exactly once. The final CLOB transition
  bootstraps from the auction target network.
- Use the same replay reward scale (`1e-3`) for DQN/DDPG/TD3/SAC and no reward
  clipping.
- Exactly one eligible optimizer update occurs per environment step.
- Maintain separate CLOB/auction networks and replay buffers with explicit
  cross-phase targets.
- Use private component RNG streams from one master seed; never use global
  NumPy/Python RNG state.

## Benchmarks

AS and TWAP are external stylized policies evaluated on the same exogenous
paths and matching/clearing engine. CLOB volume/time/rounding follow the
manuscript. At auction open, compute `S_tilde` as the average of mean and
maximum executed CLOB prices, with `H_cl` fallback. Submit once:

```text
min(q, min(z*q, beta*K_max) * (p-S_tilde)_+)
```

Do not project `S_tilde` into the learned offset range. Leave the capped
one-sided schedule active, never buy, and never exceed remaining inventory.
Benchmarks receive no shaping.

## Evaluation and outputs

- Baselines: 1,000 synthetic and 500 historical training episodes.
- Fit normalization/benchmark calibration on train only; select `best.pt` on
  validation `risk_adjusted_pnl`; evaluate fixed policies on 100 held-out test
  episodes.
- Use common random numbers for policy/benchmark pairs.
- Run the four `H_cl` x shaping arms, no-auction comparator, and no-cancellation
  sensitivity.
- Save PnL, risk-adjusted PnL, terminal and negative inventory diagnostics,
  phase cash, fees, actual signed fill, clearing/allocation/carry-over/proposal/
  displacement/action diagnostics, grids, splits, normalization, config,
  versions, seeds, checkpoint metric, and ablation label.
- Confidence intervals for policy comparisons are paired at seed level.
  Historical cross-asset outputs include currency and basis-point forms.
- Fixed-policy cumulative differences are never called regret.

The active historical CSV contains 20 frozen one-minute sessions with strictly
chronological 10/5/5 train/validation/test pools. Verify its sidecar digest and
split contract before use and copy that manifest into every run directory.

## Engineering

- Python 3.10+, `src/` layout, package `lmm`, YAML config overlays.
- Use `pytest -q tests/test_revision_acceptance.py` before any long run, then
  the full fast suite.
- Preserve unrelated user changes and never modify `paper/` during code work.
- Figures/tables are regenerated from saved outputs; output generation never
  steps an environment.
