# RL design: two-phase DQN for the market-making MDP

This document is the authoritative specification of the learning method
(ruling D9: neither the legacy NFQ code nor the paper's printed Section 4 is
authoritative). It is written so that Section 4 of the paper can be rewritten
directly from it. Implementation: `src/lmm/agents/dqn.py`, `src/lmm/rl/`,
`src/lmm/experiments/`. Hyperparameter values live in `configs/algo/dqn.yaml`
and are reproduced in the table below.

## 1. Setting and notation

The MDP is the paper's (`sec:MDP`) with the corrected clearing equations and
the binding conventions of CLAUDE.md. Time grid
0 = t_0 < … < t_n < τ_op = t_{n+1} < … < t_m < τ_cl = t_{m+1} with
n = τ_op − 1, m = τ_cl − 1 (defaults τ_op = 120, τ_cl = 150). Decisions are
taken at t ∈ {0,…,m}; t = τ_cl is terminal (clearing + terminal reward, no
action). The discount is χ ∈ (0,1] from config (`rl.chi` = 0.99, ruling D15).

The agent observes the pruned feature vectors of ruling D11
(`src/lmm/env/features.py`), which are **time-augmented** (both include a
normalized time coordinate):

- CLOB phase x ∈ R^8: (inv_norm, h_cl, s_mid, depth_ask_norm,
  depth_bid_norm, top_ask_norm, top_bid_norm, t_norm);
- auction phase x ∈ R^7: (inv_norm, h_cl, s_mid, n_mm_norm, n_buy_norm,
  n_sell_norm, t_norm)

(exact normalizations in `src/lmm/env/features.py`; lists configurable in
`configs/base.yaml`, `features:`).

Actions are the discrete grids of `src/lmm/env/action_spaces.py`:

- CLOB: {(0,0)} ∪ {1..30} × {1..12}, |A_clob| = 361 (volume, tick offset);
- auction: K^a ∈ {0} ∪ linspace(1, K_max, 10), offset ∈ {−12..12},
  c ∈ {0,1}, |A_auction| = 550. K^a = 0 is identified with abstaining.

Admissibility Adm(x) (CLAUDE.md, time-free): a¹ ≤ x¹ (volume ≤ inventory),
a² ≥ x¹⁰/α (structural on the grid), a⁵ ≤ C(x) (cancel-all only if a live
prior order with K^a > 0 exists). The env exposes the boolean mask
`env.action_mask()`; Adm(x) is never empty (the no-op/abstain action is
always admissible).

## 2. Two phase networks (design choice)

Two independent Q-networks are trained: Q_φ for the CLOB phase
(R^8 → R^361) and Q_ψ for the auction phase (R^7 → R^550), each with its own
target network, optimizer and replay buffer. This is a documented DESIGN
CHOICE — the phases have structurally different state and action spaces — not
paper fidelity. Both are MLPs (`src/lmm/rl/networks.py::mlp`): linear layers
of widths `hidden_layers` with ReLU activations and a linear output head; one
output unit per discrete action.

## 3. Masked ε-greedy behavior policy

At state x with admissibility mask M = Adm(x) (ruling: **masking, not
projection** — the stored action always equals the executed action, AUDIT
N12):

- with probability ε: a ~ Uniform({a : M_a = 1});
- otherwise: a = argmax_{a : M_a = 1} Q(x, a), realized by setting
  Q(x, a) = −∞ on inadmissible entries before the argmax.

Both branches draw only from the admissible set; the env additionally rejects
inadmissible submissions with an error (defense in depth). Evaluation mode is
fully greedy (ε = 0, `torch.no_grad`, networks in eval mode) and consumes no
exploration randomness, so interleaved evaluations do not perturb the
training RNG streams.

ε follows the exponential schedule (`src/lmm/rl/schedules.py`): per episode e,

    ε(e) = ε_start                                   for e < W (warmup),
    ε(e) = clip(ε_start · exp(−r (e − W)), ε_end, ε_start)  otherwise,
    r = −ln(ε_end / ε_start) / D,

so ε reaches ε_end exactly at episode W + D and stays there. Defaults:
ε_start = 1.0, ε_end = 0.01, W = 100, D = 300.

## 4. Replay

One uniform FIFO ring buffer per phase (`src/lmm/rl/replay.py`), capacity
`buffer_size` each. A stored transition is (x, a, r̃, x′, done, junction,
M′) where r̃ = `reward_scale` · r (scaling applies inside replay only;
reported metrics stay in paper units), M′ = Adm(x′) over the NEXT phase's
grid, and `junction` flags CLOB transitions whose next state is the auction
open. M′ must be stored because the auction cancel-admissibility C(x′) is not
recoverable from the pruned features. Sampling is uniform WITH replacement
from a buffer-private seeded generator.

## 5. Bellman targets, junction, terminal

For a minibatch row (x, a, r̃, x′, done, junction, M′):

    y = r̃ + χ · (1 − done) · max_{a′ : M′_{a′} = 1} Q_target^{phase(x′)}(x′, a′),

where the **target network is chosen by the phase of x′**:

- ordinary CLOB row → CLOB target Q_φ⁻;
- junction row (x CLOB, x′ auction open) → AUCTION target Q_ψ⁻;
- auction row → auction target Q_ψ⁻.

Terminal (item 1, no fold): τ_cl carries no decision, so its value is the
KNOWN terminal reward, V(x_{τ_cl}) ≡ r_{τ_cl}. The final auction transition
(t = t_m) is stored with done = 1, the STEP reward r_{t_m} (the terminal
reward is NOT folded into it), and the terminal payoff g = r_{τ_cl} carried
alongside (replay `terminal_value`); the Bellman target bootstraps from that
known value,
    y_{t_m} = r_{t_m} + χ · r_{τ_cl}
(one factor χ, since τ_cl = t_m + 1), which equals Q*_{t_m}(x_{t_m}, a_{t_m})
exactly. Before 2026-06-23 the code FOLDED r_{t_m} + r_{τ_cl} into the stored
reward with zero bootstrap, discounting r_{τ_cl} at χ^0 relative to t_m —
exact only for χ = 1, off by (1−χ)r_{τ_cl} otherwise. The REPORTED discounted
return already re-discounts r_{τ_cl} at χ^{τ_cl}
(see `docs/metrics_schema.md`), now consistent with the training target.

Hand-computed target tests (ordinary / junction / masked max / terminal):
`tests/test_dqn.py::test_bellman_targets_hand_computed`.

## 6. Loss, optimization, target networks

- Loss: Huber (`SmoothL1Loss`) between Q(x, a) and y (config `loss`; `mse`
  optional). TD errors and gradient norms are logged per update.
- Optimizer: Adam, learning rate `lr`, one optimizer per phase network.
- Gradient clipping: global norm `grad_clip_norm`.
- Target networks: hard copy every `target_update_interval` environment
  steps (Mnih-style); optional Polyak averaging with `target_soft_tau`
  (mutually exclusive: soft updates run every step when set).

## 7. Update schedule

Updates are **per environment step** (NOT the legacy per-episode full-buffer
refit): after every `update_every` env steps, each phase buffer holding at
least `min_buffer` transitions contributes `updates_per_env_step` gradient
steps on minibatches of `batch_size`. The env-step counter is global across
phases.

## 8. Pseudocode

```
initialize Q_φ, Q_ψ, targets Q_φ⁻ ← Q_φ, Q_ψ⁻ ← Q_ψ, buffers B_clob, B_auct
save initial checkpoint (the "initial-DQN" baseline)
for episode e = 0 … E−1:
    ε ← schedule(e);  s ← env.reset(seed drawn from the env_train stream)
    while not terminal:
        M ← Adm(x);  a ← masked ε-greedy(Q_phase(x), M, ε)
        (x′, r, done) ← env.step(a);  M′ ← Adm(x′) unless done
        push (x, a, reward_scale·r, x′, done, junction, M′) into B_phase(x)
        every update_every env steps:
            for each B with |B| ≥ min_buffer: gradient step(s) on Huber(Q, y)
        every target_update_interval env steps: hard-sync both targets
    every eval_interval_episodes: greedy eval on the fixed env_eval seed list;
        save best checkpoint on improvement
    every checkpoint_interval_episodes: save resumable checkpoint
save final checkpoint
```

**Reported policy (early stopping).** `evaluate.py` reports `best.pt` by
default — the best-`env_eval`-validation checkpoint, disjoint from the
`env_final_eval` test seeds. This is standard model selection; it discards
training that *degraded* the policy (e.g. a late TD3 collapse) and is robust
across tickers/seeds without per-algo episode-budget tuning. `--checkpoint
final` selects the last-episode model; the full eval curve (`metrics.csv`
`eval_return_mean`) is reported so instability stays visible. For credible
reporting across the RNG/init lottery, aggregate over several master seeds with
IQM + bootstrap CIs (`scripts/run_multiseed.sh`; `docs/metrics_schema.md`).

## 9. Hyperparameters (defaults; `configs/algo/dqn.yaml`)

| name | value | meaning |
|---|---|---|
| buffer_size | 50000 | replay capacity per phase |
| min_buffer | 5000 | learning starts (per phase) |
| batch_size | 128 | minibatch size |
| lr | 1.5e-4 | Adam learning rate (lowered from 3e-4, 2026-06-15: calmer updates curb the overestimation-driven late-eval decay) |
| loss | huber | TD loss |
| grad_clip_norm | 1.0 | global gradient-norm clip |
| hidden_layers | [64, 64] | MLP widths per phase network (widened from [16,16]) |
| double_q | true | Double-DQN target (van Hasselt et al. 2016): online-net argmax, target-net eval; reduces overestimation. Code default False (vanilla Mnih-2015) |
| target_update_interval | 1000 | hard target sync (env steps; ignored when soft_tau set) |
| target_soft_tau | 0.0025 | Polyak coefficient (soft target updates) |
| updates_per_env_step | 1 | gradient steps per eligible update |
| update_every | 1 | env steps between updates |
| reward_scale | 1e-3 | replay-only reward scaling (paper rewards are O(1e3-1e6); 1e-3 keeps Bellman targets O(1-100). Reported metrics stay in paper units) |
| epsilon_start / end | 1.0 / 0.01 | ε-schedule endpoints |
| epsilon_warmup_episodes | 100 | episodes at ε_start |
| epsilon_decay_episodes | 600 | episodes from start to end (≈0.6×episodes; synthetic 1000-ep budget). Historical (500 ep) overrides to 300 via scripts/run_historical_dqn.sh |
| eval_interval_episodes | 100 | greedy-eval cadence |
| eval_n_seeds | 24 | seeds per periodic eval (best.pt selection; disjoint from the final-eval test seeds) |
| final_eval_n_seeds | 100 | seeds for evaluate.py |
| checkpoint_interval_episodes | 100 | resumable-checkpoint cadence |
| activation | relu | MLP activation |
| device | cpu | torch device |

χ = 0.99 lives in `rl.chi` (shared by all algorithms, ruling D15).

## 10. Evaluation protocol, CRN, regret

`evaluate.py` evaluates {trained DQN, initial-DQN, AS, TWAP} on K episodes
with **common random numbers**: one shared env seed per episode for all
policies, drawn from the dedicated `env_final_eval` stream (disjoint from
training). All policies run through the same episode loop
(`src/lmm/rl/loops.py::run_episode`) on envs built from the same resolved
config — in particular under ONE reward definition (AUDIT C.4). Reported
returns are UNDISCOUNTED episode sums; the χ-discounted V_0 estimate is
recorded alongside.

`regret.py` computes the paper's pseudo-regret (the only part of printed
Section 4 that is kept):

    PRegret(T) = Σ_{e=1}^{E} ( V_0^{π_benchmark}(x_{0,e}) − V_0^{π_e}(x_{0,e}) ),
    T = (m + 2) E,

estimated per episode with CRN; V_0 is χ-discounted by default
(`--returns undiscounted` matches the reported returns instead).

## 11. The initial-DQN baseline

`checkpoints/initial.pt` is written BEFORE any training: the freshly
initialized networks under the greedy masked policy. It serves as the
untrained-policy reference in evaluation and regret plots
(`evaluate.py` policy name `initial`).

## 12. Seeding and determinism (ruling D10)

One master seed (written to `seed.txt`) spawns one private
`numpy.random.Generator` per component via `np.random.SeedSequence.spawn`
(`src/lmm/utils/seeding.py`; canonical component list
`src/lmm/rl/loops.py::SEED_COMPONENTS`): per-episode env seeds (train / eval
/ final-eval streams), exploration, per-phase replay sampling, AS
calibration. torch is seeded from the same master seed with deterministic
algorithms enabled. No global `np.random.*` / `random.*` is ever touched.
Same config + seed ⇒ bit-identical `metrics.csv` (modulo the wall-clock
column), asserted by
`tests/test_agent_env_contract.py::test_end_to_end_pipeline_and_bit_identical_determinism`.
Checkpoints store all counters and RNG states (exploration, torch, replay)
so `--resume` continues the same streams.
