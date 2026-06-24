# Continuous-action extension: DDPG, TD3, SAC

This document specifies the **continuous-control relaxation** of the paper's
discrete market-making MDP (ruling D9: the learning method is designed per
standard RL literature, not paper fidelity). It is a *separate, clearly
labeled* relaxation — the discrete two-phase DQN of `docs/rl_design.md` and the
paper's discrete action sets are **unchanged**. Everything mathematical that
differs from the discrete setting is recorded here so that the author can phrase
the extension correctly in the paper.

Implementation: `src/lmm/env/action_spaces.py`
(`ContinuousActionAdapter`, `continuous_action_specs`), `src/lmm/rl/networks.py`
(`DeterministicActor`, `Critic`, `SquashedGaussianActor`),
`src/lmm/rl/continuous_replay.py` (`ContinuousReplayBuffer`),
`src/lmm/agents/continuous_base.py` (`ContinuousActorCriticAgent`),
`src/lmm/agents/{ddpg,td3,sac}.py`. Hyperparameters live in
`configs/algo/{ddpg,td3,sac}.yaml` and are reproduced in §8.

The MDP itself — state, dynamics, the corrected clearing equations, the
three-regime reward, the discount χ, the time grid, the cross-phase junction,
and the terminal fold — is **identical** to the discrete setting. Only the
*action set* and the *policy class* change.

## 1. From a finite action set to a continuous one

### 1.1 Discrete sets (unchanged baseline)

Per phase (`src/lmm/env/action_spaces.py`), the discrete grids are

- **CLOB**: `A_clob = {(0,0)} ∪ {1,…,V} × {δ_min,…,δ_max}` — a volume `v` and an
  integer book-level offset `δ` (defaults `V = 30`, `δ ∈ {1,…,12}`, 361 actions);
- **auction**: `A_auct = ({0} ∪ linspace(1, K_max, 10)) × {−o_max,…,o_max} ×
  {0,1}` — a supply slope `K^a`, an integer price-tick offset `o`, and a scalar
  cancel-all flag `c` (defaults `K_max = 10·I₀/V = 33.3̄`, `o ∈ {−12,…,12}`,
  `c ∈ {0,1}`, 550 actions). `K^a = 0` is identified with abstaining.

The order the agent submits at decision time `t` is, in both phases, fed to the
**same** env transition. The env already treats `v` and `K^a` as continuous
quantities (they enter execution `E_t` and the linear clearing equation
continuously); only `δ` and `o` are inherently discrete — `δ` indexes a book
*level* and `o` indexes a price *tick* `α·(k_mid + o)`.

### 1.2 Continuous Box relaxation

We replace each grid by a continuous box `𝒜 ⊂ ℝ^{d}`:

- **CLOB**: `a = (v, δ) ∈ [0, V] × [δ_min, δ_max] ⊂ ℝ²`.
- **auction** (`continuous_cancel: threshold`, default):
  `a = (K^a, s_off, c_logit) ∈ [0, K_max] × [−o_max, o_max] × [0, 1] ⊂ ℝ³`.
- **auction** (`continuous_cancel: never`):
  `a = (K^a, s_off) ∈ [0, K_max] × [−o_max, o_max] ⊂ ℝ²` (no cancellation in
  continuous control; `c ≡ 0`).

### 1.3 Projection / snapping map Π (env-side, `ContinuousActionAdapter`)

A continuous action is mapped to an executable order. Write `I_t` for the
current inventory and `k̄ = ⌊S^mid_{frozen}/α⌋` for the frozen mid tick.

**CLOB.**
```
v   ← clip(v, 0, min(V, I_t))          # inventory + bound PROJECTION (continuous)
δ̂  ← clip(round(δ), δ_min, δ_max)      # tick SNAPPING (integer book level)
order = ClobAction(volume = v, delta = δ̂)
```
`v` is projected, not snapped: fractional volumes are admissible and the env
executes them exactly (the discrete grid merely happened to sample integers).
`δ` must be snapped because the execution model walks integer book levels.

**Auction.**
```
K^a ← clip(K^a, 0, K_max)               # PROJECTION (continuous slope; NOT snapped)
ô   ← clip(round(s_off), −o_max, o_max) # tick SNAPPING (integer price tick)
S^a  = α·(k̄ + ô)
c    = 1{c_logit > 0.5} · 1{cancel-admissible at x}   # threshold THEN admissibility mask
order = AuctionAction(K_a = K^a, offset = ô, cancel = c)
```
`K^a` is **continuous** (author decision, this revision): it enters the linear
clearing equation continuously, so — symmetric with the CLOB volume `v` — it is
projected to its bound but never snapped to a grid. This is what gives
DDPG/TD3/SAC genuine continuous control in the auction.

**Identical dynamics.** Execution `E_t`, the clearing solves, and all three
reward regimes are computed on `Π(a)` — the *same* transition function as the
discrete env, evaluated at a (possibly off-grid) point. Only the policy that
proposes `a` changes. Consequently, a continuous action whose coordinates equal
a discrete grid point produces **bit-identically the same transition and
reward** as the discrete env on the same seed (asserted in
`tests/test_continuous_adapter.py`).

### 1.4 Admissibility Adm(x) under the relaxation

In the discrete setting `Adm(x)` is a *mask* over a finite set; greedy action
selection is a masked `argmax`. Under the relaxation `Adm(x)` becomes a
*projection onto a continuous feasible set*:

- `a¹ ≤ x¹` (volume ≤ inventory): the `v ← min(v, I_t)` projection.
- `a² ≥ x¹⁰/α` (quote at/above the mid tick): structural — `δ_min ≥ 0` and the
  box lower bound enforce `δ ≥ 0`, so `S^•_t = α·(k_mid + δ̂) ≥ S^mid`.
- `a⁵ ≤ C(x)` (cancel-all only if a live prior `K^a > 0` order exists): the
  `c = … · 1{cancel-admissible}` mask. `C(x) = 0` throughout the CLOB phase and
  at the auction open, so `c ≡ 0` there automatically.

**The cancel discontinuity (`threshold` mode).** The map
`c_logit ↦ 1{c_logit > 0.5}·1{cancel-admissible}` is discontinuous in `c_logit`
(at `0.5`) and in the state (at the cancel-admissibility boundary). The actor's
`c_logit` is a continuous control whose effect on the dynamics is piecewise
constant. We store the **continuous** `c_logit` in replay (see §3); when the
next state cannot cancel, the bootstrap target clamps the target action's
`c_logit` to the no-cancel region (§5). The `continuous_cancel: never` mode
removes the dimension entirely and is free of this discontinuity — use it if a
fully continuous, gradient-clean action space is required for the auction.

## 2. What carries over from the theory, and what does not

The following discrete-MDP statements **do not carry over verbatim** and must be
rephrased for the continuous extension:

1. **Finiteness.** With `𝒜` a box, the per-step action set is *uncountable*. The
   MDP is no longer a finite MDP; results that invoke finiteness do not apply
   as stated.
2. **Puterman Thm 6.2.10 (existence of an optimal deterministic stationary
   policy via a per-state `argmax` over a finite action set).** The theorem's
   hypotheses (finite action sets, attainment of the max by enumeration) are not
   met. Under standard regularity (compact `𝒜`, continuous reward and
   transition kernel) an optimal stationary policy still exists, but it is *not*
   obtained by enumerating actions; DDPG/TD3/SAC approximate it by policy
   gradient on a function approximator. The admissibility set `Adm(x)` is no
   longer a finite mask but a closed convex box intersected with the inventory
   constraint and the (discontinuous) cancel constraint.
3. **Greedy `argmax` policy / value iteration over actions.** Replaced by an
   actor network: a deterministic `μ_θ(x)` (DDPG/TD3) or a stochastic
   `π_θ(·|x)` (SAC). The "greedy" target `max_{a'} Q(x',a')` becomes
   `Q(x', μ_{θ'}(x'))` (deterministic) or an expectation under `π_θ`
   (entropy-regularized).
4. **Regret / `PRegret(T)`.** The definition is unchanged (it compares
   `V₀^{benchmark}` to `V₀^{π}` per episode under common random numbers); only
   the policy class producing `π` changes. The same `regret.py` is reused.

Everything else — the corrected clearing Eqs. (1)–(2), the θ recursion, the
three-regime reward (D4/D5), the terminal reward (D3/D8), the H_cl timing
(D1/D2), the discount χ, the time grid, and the two-phase structure — is
**unchanged**: the relaxation is purely a change of action set and policy class.

## 3. Replay and the stored action

We use **two replay buffers**, one per phase
(`src/lmm/rl/continuous_replay.py::ContinuousReplayBuffer`), mirroring the DQN
split. A stored transition is `(x, a, r̃, x', done, junction,
next_cancel_admissible)` where:

- `a` is the **projected continuous action** — i.e. `a` after the inventory and
  box *projection* of §1.3 but **before** the integer snapping of `δ`/`o` and
  the thresholding of `c`. Rationale: the critic `Q(x, a)` must see a *smooth*
  action argument so the deterministic/stochastic policy gradient is
  well-defined; storing the snapped/thresholded action would collapse the
  critic's action input back onto a discrete set and break differentiability.
  This is the standard "store the committed (squashed, projected) action" choice
  in continuous control. (The env still *executes* `Π(a)`; the discrepancy is
  the deliberate, documented snapping discretization.)
- `r̃ = clip(reward_scale · r, ±reward_clip)` (scaling AND optional symmetric
  clipping inside replay only; reported metrics stay in paper units). The
  paper's per-step fictive auction reward is unbounded and the
  deterministic/stochastic actors exploit it, so even after `reward_scale` the
  MSE critic targets diverged (DDPG `loss_auction`→1e18, SAC→1e26/`inf`);
  `reward_clip` bounds the regression target. Defaults `reward_scale = 1e-3`,
  `reward_clip = 8.0` (calibrated to the honest terminal-reward envelope ~5.5
  scaled; see `configs/algo/ddpg.yaml` and AUDIT §F.1; the discrete DQN keeps
  `reward_clip` off — it uses robust Huber and was not part of this instability).
- `junction` flags a CLOB transition whose next state is the auction open.
- `next_cancel_admissible` is the auction cancel-admissibility `C(x') > 0` of the
  next state (analogue of DQN's stored `next_mask`; needed because the pruned
  auction features do not encode `C(x')`). It is read by the bootstrap target
  (§5) to clamp the target action's `c_logit`.

Sampling is uniform with replacement from a buffer-private seeded
`numpy.random.Generator` (D10).

## 4. Networks and the two-phase split

Mirroring the DQN design choice (`docs/rl_design.md` §2), we train **per-phase**
actor/critic pairs — CLOB and auction have structurally different state/action
spaces. All bodies are MLPs (`src/lmm/rl/networks.py::mlp`).

- **Deterministic actor** `μ_θ : ℝ^{obs} → 𝒜` (DDPG, TD3):
  `μ_θ(x) = low + (high − low)·(tanh(MLP(x)) + 1)/2`, i.e. an MLP head squashed
  by `tanh` and affinely mapped into the box `[low, high]`.
- **Critic** `Q_φ : ℝ^{obs} × ℝ^{d} → ℝ`: MLP on the concatenation `[x, a]`.
  TD3/SAC use twin critics `Q_{φ₁}, Q_{φ₂}`.
- **Squashed-Gaussian actor** `π_θ` (SAC): heads `(m(x), ℓ(x))` give a Gaussian
  `u ~ 𝒩(m, exp(2·clip(ℓ, ℓ_min, ℓ_max)))`; the action is
  `a = low + (high − low)·(tanh(u) + 1)/2`, and the log-density carries the
  standard `tanh` change-of-variables correction
  `log π(a|x) = log 𝒩(u; m, σ) − Σ_i log(1 − tanh(u_i)² + ε) − Σ_i log((high_i −
  low_i)/2)`. Sampling uses the reparameterization (`rsample`).

Each phase has its own optimizer(s), target network(s), and replay buffer.
Targets are tracked by Polyak averaging `θ' ← (1−τ)θ' + τθ` every env step
(`target_soft_tau`).

## 5. Bellman targets, cross-phase junction, terminal fold

For a minibatch row `(x, a, r̃, x', done, junction, c_adm')` in the buffer of
`phase`, the **next phase** is

- `phase` for an ordinary row (`¬done ∧ ¬junction`),
- `auction` for a junction row (`¬done ∧ junction`; only in the CLOB buffer),
- none for a terminal row (`done`).

Let `χ` be the discount. The target action and value are taken in the **next
phase's** networks:

- **DDPG**: `a' = μ_{θ'}^{phase'}(x')`, `y = r̃ + χ·(1−done)·Q_{φ'}^{phase'}(x', a')`.
- **TD3**: `a' = clip(μ_{θ'}^{phase'}(x') + clip(𝒩(0, σ_tgt), −c, c), low, high)`
  (target-policy smoothing), `y = r̃ + χ·(1−done)·min_{i=1,2}
  Q_{φ'_i}^{phase'}(x', a')`.
- **SAC**: `(a', logp') ~ π_θ^{phase'}(·|x')` (current actor, no target actor),
  `y = r̃ + χ·(1−done)·(min_{i=1,2} Q_{φ'_i}^{phase'}(x', a') − α^{phase'}·logp')`.

**Cancel clamp.** When `phase' = auction` and `c_adm' = False`, the target
action's `c_logit'` is forced to `0` (the no-cancel region) before evaluating
the target critic, matching the env's masking of an inadmissible cancel at `x'`.
In `continuous_cancel: never` there is no `c` dimension and the clamp is a no-op.

**Terminal bootstrap (no fold).** Exactly as in the discrete design (item 1):
`τ_cl` carries no decision, so its value is the known terminal reward,
`V(x_{τ_cl}) ≡ r_{τ_cl}`. The final auction transition is stored with
`done = 1`, the step reward only, and `g = r_{τ_cl}` in `terminal_value`; the
target bootstraps `y = r̃_step + χ · g̃` (one `χ`, since `τ_cl = t_m + 1`),
matching `Q*_{t_m}` exactly (the earlier zero-bootstrap fold was exact only for
`χ = 1`). The reported discounted return re-discounts the terminal part at
`χ^{τ_cl}` (`docs/metrics_schema.md`), now consistent with the training target.

## 6. Update rules

Updates are **per environment step** (gated by `update_every`): once each phase
buffer holds ≥ `min_buffer` transitions, that phase runs `updates_per_env_step`
gradient steps on minibatches of `batch_size`. The env-step counter is global
across phases (same schedule as DQN).

Per phase and per gradient step:

1. **Critic(s).** Minimize `MSE(Q_{φ_i}(x, a), y)` for each critic `i`
   (`y` from §5; `y` uses `min` of the *twin target* critics for TD3/SAC).
   Gradients are clipped to global norm `grad_clip_norm`.
2. **Actor.**
   - DDPG: `L_μ = −E_x[Q_φ(x, μ_θ(x))]`.
   - TD3: `L_μ = −E_x[Q_{φ₁}(x, μ_θ(x))]`, applied every `policy_delay` critic
     updates (delayed updates); the target nets sync on the same delayed cadence.
   - SAC: `L_π = E_x[α·logp(ã|x) − min_i Q_{φ_i}(x, ã)]`, `ã ~ π_θ(·|x)`
     (reparameterized), every step.
3. **Temperature (SAC only).** With target entropy `H̄ = −dim(𝒜_phase)`
   (`−2` CLOB, `−3` auction in `threshold` mode / `−2` in `never`),
   `L_α = −E_x[logα·(logp(ã|x) + H̄)]` (optimized in `logα`); `α = exp(logα)`.
4. **Target sync.** Polyak `θ'_• ← (1−τ)θ'_• + τθ_•` (every step for DDPG/SAC;
   every `policy_delay` steps for TD3, alongside the delayed actor update).

## 7. Exploration, evaluation, seeding

- **DDPG / TD3 exploration.** Add action-space noise to `μ_θ(x)` then re-clip to
  the box: Gaussian `𝒩(0, exploration_noise_std·(high−low)/2)` (default), or an
  Ornstein–Uhlenbeck process per phase (`exploration_noise: ou`, state reset at
  episode start). Noise is drawn from the seeded numpy `exploration` generator.
- **SAC exploration** is intrinsic (stochastic policy); training samples
  `a ~ π_θ`, evaluation uses the deterministic mean action `tanh(m(x))` mapped to
  the box. No external noise (the `epsilon` metrics column is `0` for SAC).
- **Eval mode** (`eval_mode=True`) is deterministic and consumes no exploration
  randomness, so interleaved evaluation does not perturb training streams.
- **Seeding (D10).** One master seed spawns the per-component
  `numpy.random.Generator`s (`SEED_COMPONENTS`, unchanged): `exploration` drives
  DDPG/TD3 noise; `replay_clob`/`replay_auction` drive the two buffers. SAC's
  reparameterized sampling and all network initialization use the seeded torch
  global RNG (`torch.manual_seed(master)`, deterministic algorithms on). No
  global `numpy.random.*` / `random.*` is ever touched. Same config + seed ⇒
  bit-identical `metrics.csv` (modulo the wall-clock column).
- **Evaluation / regret.** The continuous agents run through the *same*
  `rl/loops.py::run_episode`, `experiments/evaluate.py` (CRN over the learned
  policy, its untrained "initial" snapshot, and the AS/TWAP benchmarks) and
  `experiments/regret.py` as the DQN; the learned policy's env is wrapped in the
  `ContinuousActionAdapter`, the benchmark envs are not.

## 8. Hyperparameters (`configs/algo/{ddpg,td3,sac}.yaml`)

Shared run/eval/checkpoint fields match the DQN config so `experiments/train.py`
drives every algorithm uniformly (`device`, `activation`, `reward_scale`,
`update_every`, `updates_per_env_step`, `eval_interval_episodes`, `eval_n_seeds`,
`final_eval_n_seeds`, `checkpoint_interval_episodes`, `buffer_size`,
`min_buffer`, `batch_size`, `grad_clip_norm`). `χ = 0.99` lives in `rl.chi`
(shared, ruling D15). Continuous-specific knobs:

| field | DDPG | TD3 | SAC | meaning |
|---|---|---|---|---|
| actor_lr | 1e-4 | 3e-4 | 3e-4 | Adam lr, actor |
| critic_lr | 3e-4 | 3e-4 | 3e-4 | Adam lr, critic(s). DDPG lowered 1e-3→3e-4 (2026-06-15) to fix the overestimation collapse — see AUDIT §F.3 |
| hidden_layers | [64,64] | [64,64] | [64,64] | MLP widths (actor & critic) |
| target_soft_tau | 0.0025 | 0.005 | 0.005 | Polyak coefficient τ (DDPG tightened 0.005→0.0025 with the retune, 2026-06-15) |
| exploration_noise | gaussian | gaussian | — | `gaussian`/`ou` (SAC: intrinsic) |
| exploration_noise_std | 0.1 | 0.1 | — | noise scale (× half-range) |
| target_noise_std | — | 0.2 | — | target-policy smoothing σ (× half-range) |
| target_noise_clip | — | 0.5 | — | smoothing clip c (× half-range) |
| policy_delay | — | 2 | — | delayed actor/target cadence |
| alpha_lr | — | — | 3e-4 | Adam lr, temperature |
| auto_alpha | — | — | true | auto-tune α to target entropy |
| target_entropy | — | — | auto | `−dim(𝒜_phase)` per phase |
| continuous_cancel | threshold | threshold | threshold | `threshold`/`never` (§1.4) |
| reward_scale | 1e-3 | 1e-3 | 1e-3 | replay-only reward scaling (paper rewards O(1e3–1e6)) |
| reward_clip | 8.0 | 8.0 | 8.0 | replay-only symmetric reward clip (bounds the fictive-auction-reward exploit; calibrated to the honest envelope, AUDIT §F.1) |

The continuous configs also normalize the raw ~100-valued price features
(`h_cl_norm`, `s_mid_norm`: centered at the initial mid `S0`, divided by
`features.price_norm_scale`, clipped to `±features.price_norm_clip`) so they do
not dominate the O(1) features and a spiking per-step `H_cl` cannot blow up the
network input; the discrete DQN keeps the legacy raw `h_cl`/`s_mid`. This is
observation-only — transitions, rewards, and reported metrics are unchanged.

These start from standard literature defaults (DDPG: Lillicrap et al. 2016; TD3:
Fujimoto et al. 2018; SAC: Haarnoja et al. 2018 with automatic temperature) and
are the single source of truth (configs/, ruling D6 convention). DDPG was
retuned (2026-06-15): `critic_lr` 1e-3→3e-4 and `target_soft_tau` 0.005→0.0025
to remove the deterministic-policy overestimation collapse (final-policy eval
0.6k→15.8k, stable), and `eval_n_seeds` 8→24 for a less noisy best.pt selection
— the structural cure is clipped double-Q (= TD3), this is the within-DDPG fix
(AUDIT §F.3). TD3/SAC are unchanged (already stable).
