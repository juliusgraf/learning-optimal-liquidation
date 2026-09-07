# PARAMS_FROM_CODE.md — Historical parameter extraction (superseded)

> **Historical audit only.** This file records the inputs used to construct an
> earlier configuration and is not a source of truth for the active project.
> Resolve current numerical values from `configs/base.yaml`, one setting
> overlay, one learner overlay, and any requested treatment overlay.

Per author rulings D6–D7, the values **actually passed at the legacy code's
instantiation sites** were the source of truth for the Phase 2 extraction:

- Synthetic instantiation: `main.py:1417-1422` (`MarketEmulator(...)` at module level).
- Historical instantiation: `data.py:1386-1398` (inside `train_and_evaluate_one_asset`,
  called from `data.py:1670-1679` with `tau_op=120, tau_cl=150, T=150, episodes=1000`).
- Constructor defaults are **not** authoritative (they are recorded here because several
  are wildly different from the instantiation values, which is itself a finding).
- The paper table `tab:params_generative` and the NFQ table (Sec. 4) are reference only
  (rulings D6, D9).

Flags: ⚠ = the three code columns disagree (constructor default vs synthetic vs
historical); ✗ = paper table disagrees with the binding code value; † = dead/unused.

## 1. `MarketEmulator` constructor parameters

| Param (meaning / paper symbol) | ctor default `main.py:121-127` | ctor default `data.py:78-83` | synthetic `main.py:1417-1422` | historical `data.py:1386-1398` | paper table (ref) | Flags / notes |
|---|---|---|---|---|---|---|
| `tau_op` (τ_op) | 120 | **60** | **120** | **120** | 120 | ⚠ defaults differ between files; instantiations agree |
| `tau_cl` (τ_cl) | 150 | **70** | **150** | **150** | 150 | ⚠ defaults differ; instantiations agree |
| `T` (physical session length; `dt = T/tau_cl`) | 150 | 70 | **150** | **150** | — | `dt = 1.0` at both sites; `T` is redundant given `tau_cl` |
| `I` (`I_max`, I₀) | 100 | 100 | **100** | **100** | 100 | |
| `V` (`V_max`, max order volume V) | 5000 | 5000 | **30** | **30** | 30 | default 167× larger than used |
| `L` (`L_max_clob`) | 10 | 10 | 12 | 12 | — | † assigned, never read anywhere |
| `Lc` (CLOB book depth, levels per side) | 8 | 8 | **12** | **12** | — | no explicit paper symbol (paper's L is the random depth) |
| `La` (max # exogenous auction MMs) | 5 | 5 | **12** | **12** | — | extra capacity cap, not in the paper (see AUDIT N8) |
| `L_max_auction` (`L_max`, cap on N^ζ and ν-array size; paper 𝒩) | 100 | 100 | **100** | **100** | — | |
| `lambda_param` (terminal inventory penalty λ) | 0.005 | 0.005 | **0.5** | **0.5** | 0.5 | default 100× smaller than used |
| `kappa` (CLOB reward slope = 1/(k\*α)) | 0.1 | 0.1 | **0.1** | **0.1** | k\*=1000 | κ=0.1 ⇔ k\*α=10 ⇔ k\*=1000 at α=0.01 — consistent |
| `q` (wrong-side penalty q) | 1.0 | 1.0 | **1.0** | **1.0** | 1 | |
| `d` (cancellation cost unit d) | 1.0 | 1.0 | **0.1** | **0.1** | 0.1 | default 10× larger than used |
| `gamma` (**Algorithm-1 smoothing τ**) | 0.5 | 0.5 | **0.95** | **0.99** | τ=0.95 | ⚠⚠ three-way divergence. Historical passes the **RL discount** (`gamma=0.99` function arg, `data.py:1355`) into the smoothing slot (`data.py:1391`) — a name-collision bug. **RESOLVED by ruling D15 (AUDIT §D, Q1): configs pin τ = 0.95 (smoothing) and χ = 0.99 (discount) in BOTH settings; the historical 0.99 smoothing is not reproduced.** |
| `v_m` (Pareto scale v_m) | 1000 | 1000 | **2.0** | **2.0** | 2 | default 500× larger than used |
| `pareto_gamma` (Pareto shape γ_m) | 2.0 | 2.0 | **2.5** | **2.5** | 2.5 | |
| `poisson_rate` (CLOB MO intensity λ₀) | 0.5 | 0.5 | **1.0** | **60.0** | 1 | ⚠ 60× divergence: historical uses 60 arrivals/side/step (1-minute bars ⇒ ~1/s), synthetic 1/step. ✗ paper table shows only λ₀=1 |
| `seed` | None | None | None | None | — | per-episode seeding via `reset(seed=...)` |
| `V_top_max` (top-of-book scale V_∞) | 50000.0 | 50000.0 | **15.0** | **15.0** | 15 | default 3333× larger than used |
| `lambda_decision` | 1.0 | 1.0 | 1.0 (default) | 1.0 (default) | — | † never read |
| `alpha` (tick size α) | **1.0** | **0.01** | **0.01** | **0.01** | 0.01 | ⚠ defaults differ between files; instantiations agree |
| `beta_a` (Beta shape a) | 2.0 | 2.0 | **2.0** | **2.0** | 2 | |
| `beta_b` (Beta shape b) | 5.0 | 5.0 | **5.0** | **5.0** | 5 | |
| `depth_decay` (geometric decay ρ) | 0.6 | 0.6 | **0.5** | **0.5** | 1/2 | |
| `price_band_ticks` (exogenous auction quote band ⇒ M₁=−band, M₂=+band) | 5.0 | 5.0 | **10.0** | **10.0** | M₁=−10, M₂=10 | |
| `sigma_mid` | 0.1 | 0.1 | 0.1 | — (default) | — | † only used by a commented-out Bachelier line (`main.py:567`) |
| `mid_rh_H` (rough-Heston H) | 0.1 | n/a | **0.1** | n/a | 0.1 | |
| `mid_rh_v0` (V₀) | **0.01** | n/a | **0.02** | n/a | 0.02 | ⚠ default ≠ instantiation |
| `mid_rh_theta` (θ) | **0.01** | n/a | **0.04** | n/a | 0.04 | ⚠ default ≠ instantiation |
| `mid_rh_kappa` (mean-reversion λ in the paper's notation) | 0.3 | n/a | **0.3** | n/a | 0.3 | |
| `mid_rh_xi` (vol-of-vol ν) | 0.3 | n/a | **0.3** | n/a | 0.3 | |
| `mid_rh_rho` (correlation ρ) | −0.7 | n/a | **−0.7** | n/a | −0.7 | |
| `mid_price_path` | n/a | None (required) | n/a | `data.csv` rows 0..119 of the symbol | — | historical only; length must equal `tau_op` (`data.py:1378-1379`) |

## 2. Hard-coded model literals (become explicit config parameters in Phase 2)

| Quantity (paper symbol) | Value | Location (main.py / data.py) | Paper table | Notes |
|---|---|---|---|---|
| p₁ (new exogenous MM arrival prob.) | 0.3 | 592 / 475 | 0.3 | ruled config param (D6) |
| Exogenous MM slope law (U₁, U₂) | K ~ U(0.1, 2.0) | 593 / 476 | 0.1, 2.0 | hardcoded literal |
| Exogenous MM quote law (M₁, M₂) | S ~ S_mid(frozen) + α·U{−10..10} | 594 / 477 | −10, 10 | band from `price_band_ticks`; mid frozen at τ_op (verified) |
| p₂ (exogenous MM cancellation prob.) | 0.2 | 597 / 480 | 0.2 | |
| p₃ (new taker arrival prob., per side) | 0.3 | 601, 609 / 484, 492 | 0.3 | independent per side |
| p₄ (taker cancellation prob., per side) | Bernoulli(0.1) × fair coin = **0.05 effective** | 617-625 / 500-508 | 0.1 ✗ | ruling D7: implement single Bernoulli(0.05); paper table's 0.1 is the outer coin only |
| Initial mid price S₀ | 100.0 / `mid_price_path[0]` (=100 in committed CSV) | 263 / 169-172 | 100 | |
| H₀ (Algorithm-1 initial value) | = initial mid | 264 / 176 | 100 | matches ruling "H₀ = initial mid = 100" |
| Inventory clip bound | ±`I_max` | 555, 694 / 440, 575 | — | to be removed per D8 (`numerical_guard` instead) |
| Numeric guards | 1e-12 (vol), 1e-9 (agent residual), 1e-6 (depth) | 379, 487, 537 / 279, 378, 425 | — | |
| `B` (price-grid half-width in `_get_state`) | 10 | 335-337 / 249 | — | † vestigial (X15 grid version commented out) |
| `SECONDS_PER_YEAR` | 252 × 6.5 × 3600 | 25-33 / n/a | — | rough-Heston physical-time scaling; 1 grid unit = 1 s |

## 3. Action-space constants

| Quantity | Synthetic `main.py:1424-1435` | Historical `data.py:1400-1409` | Paper (ref) | Notes |
|---|---|---|---|---|
| CLOB volume grid A¹ | {0} ∪ {1..30} (= `V_max`) | same | v ≤ 𝒱 | |
| CLOB depth grid A² | δ ∈ {1..12} (+ the (0,0) no-op) | same | k ≥ k_mid | δ is a *book-level offset*; δ=12 aliases δ=11 via the clamp `main.py:446` (AUDIT N6) |
| # CLOB actions | 361 | 361 | — | 1 + 30×12 |
| Auction slope grid A³ | {0} ∪ linspace(1, K_MAX, 10), K_MAX = 10·I₀/30 ≈ 33.33 (spacing ≈ 3.593) | same | β·{0..𝒦}, β=3.33, 𝒦=10 ✗ | paper grid β=3.33 not matched; table non-authoritative |
| Auction price grid A⁴ | S = S_mid + off·α, off ∈ {−12..12} | same | S^a ∈ α·ℕ | offset grid around (frozen) mid; S not snapped to absolute α·ℕ (AUDIT N4) |
| Cancel A⁵ | {0, 1} scalar at policy level, expanded to a per-time vector at the env interface | same | scalar c_t | D4 |
| # auction actions | 550 | 550 | — | 11×25×2 |

## 4. RL / training constants (recorded per D9 for the design doc; not authoritative)

| Quantity | Synthetic (main.py) | Historical (data.py) | Paper NFQ table (ref) | Notes |
|---|---|---|---|---|
| Episodes E | 2000 (`main.py:1438`) | 1000 (`data.py:1677`) | 2000 / 1000 (hist.) | matches paper text |
| Learning rate η | 3e-4 (`main.py:1439`) | 3e-4 (`data.py:1354`) | 3e-4 | Adam |
| Discount χ | 0.99 (`GAMMA`, `main.py:1440`) | 0.99 (`data.py:1355`) | 0.99 | **confirmed at instantiation per D6**; also leaks into the H_cl smoothing in data.py (see §1 `gamma`) |
| Buffer size N̄ | 50,000 (`main.py:1487`) | 50,000 (`data.py:1093`) | 50,000 | per-phase deques |
| Min buffer N̲ | 5,000 (`main.py:1488`) | 5,000 (`data.py:1094`) | 5,000 | |
| Epochs per episode M | 3 (`main.py:1489`) | 3 (`data.py:1095`) | 3 | full-buffer refit per episode |
| Batch size B | 128 (`main.py:1490`) | 128 (`data.py:1096`) | 128 | |
| `REWARD_SCALE` | 1 (`main.py:1593`) | 1 (`data.py:1356`) | — | |
| ε schedule | exp decay 1.0 → 0.01, warmup 100 episodes (`main.py:1597-1602`) | same formula (`data.py:1430-1436`) | same | data.py module-level copy (`data.py:1191`) has `total=None` default → would crash; dead |
| Network | MLP **2** hidden layers × 16, ReLU (`main.py:1460-1467`) | same (`data.py:1082-1091`) | "three width-16 hidden layers" ✗ | paper text wrong about its own code; Section 4 non-authoritative anyway |
| Input dims | 8 (CLOB) / 7 (auction) | same | — | see AUDIT A.7 for exact feature lists |
| Loss / clip | SmoothL1 (Huber), grad-norm clip 1.0 | same | Huber | |
| Target update | hard copy after every episode's fit (`main.py:1834-1835`) | same + no-op soft update (`data.py:1552-1555`) | hard per episode | |
| Eval cadence | every 100 eps × 8 seeds; final 100 seeds | same | — | |
| Seeds | MASTER 42, EVAL 120, FINAL_EVAL 200 (`main.py:1410, 1653-1666`) | 42 / 120 / 200 (`data.py:1364-1366`) | — | D10: seeding scheme untrusted, rebuilt fresh |

## 5. Benchmark calibration constants

| Quantity | Synthetic (main.py) | Historical (data.py) | Paper | Notes |
|---|---|---|---|---|
| AS A | λ₀/γ_m = 1.0/2.5 = 0.4 (`main.py:1938`) | 60/2.5 = 24 (`data.py:1561`) | A = λ₀/γ_m | |
| AS k | γ_m · K̂ (`main.py:1939`) | same (`data.py:1562`) | "k = αK" | code uses the **Pareto exponent** (= AS-paper's α), not the tick size; see AUDIT A.9 |
| K̂ regression | LS of ln Q = K̂·Δp, `n_samples=10000` (`main.py:1886-1939`) | same (`data.py:1232-1280, 1562`) | least squares, 5000 samples ✗ | paper says 5,000 samples; code passes 10,000 |
| AS γ (risk aversion) | 0 (`main.py:1940`) | 0 (`data.py:1563`) | γ → 0 | |
| AS σ | std of **log**-returns pooled over all training mid paths / √dt (`main.py:1942-1954`) | std of log-returns of the single fixed path / √dt (`data.py:1557-1559`) | "std of the mid price" | AS model is arithmetic BM; code uses log-returns (≈ relative at S≈100) |
| AS horizon T | τ_op − 1 = 119 (`main.py:1958`) | same (`data.py:1565`) | t_n | |
| TWAP δ | min of the δ grid = 1 (`delta_mode="min"`) | same | δ_t = 1 | |
| TWAP volume | ⌈q/(T−t+1)⌉ (`main.py:964-967`) | same (`data.py:850-853`) | ⌈q_t/(T−t+1)⌉ | |
| Auction heuristic z | 10 (`K_big = 10.0*q_left`, `main.py:1048`) | same (`data.py:939, 1034`) | z = 10 | |
| Auction heuristic S̃ | 0.5·(max + mean) of executed CLOB prices | same | same | legacy supply is **linear**, not (p−S̃)₊ — AUDIT N7, resolved by ruling D16: new code uses the one-sided (p−S̃)₊ curve |
| Dust threshold | q_left < 1e-2 ⇒ no auction order | same | — | |

## 6. Biggest divergences between the three code columns (flag summary)

1. **`gamma` (Algorithm-1 smoothing): 0.5 (default) vs 0.95 (synthetic) vs 0.99 (historical)** —
   and the historical value is an accidental reuse of the RL discount χ. This was the single
   most consequential parameter ambiguity; **resolved by author ruling D15 (AUDIT §D, Q1):
   τ = 0.95 smoothing and χ = 0.99 discount in both settings.** Configs carry
   `algo1.tau = 0.95` everywhere; the legacy historical smoothing of 0.99 is a bug whose
   behavior is preserved only in the characterization tests.
2. **`poisson_rate` λ₀: 0.5 (default) vs 1.0 (synthetic) vs 60.0 (historical)** — changes the
   CLOB decision grid qualitatively (synthetic episodes skip decision times and have
   variable length 106–119 steps; historical episodes are always exactly 149 steps).
3. **Constructor defaults are systematically stale**: `V` 5000→30, `v_m` 1000→2,
   `V_top_max` 50000→15, `lambda_param` 0.005→0.5, `d` 1.0→0.1, `Lc` 8→12, `La` 5→12,
   `depth_decay` 0.6→0.5, `price_band_ticks` 5→10, `alpha` 1.0 (main)→0.01, and the
   data.py defaults `tau_op=60/tau_cl=70` belong to an older experiment. Phase 2 configs
   must be seeded **only** from the instantiation columns above.
4. **Episodes: 2000 (synthetic) vs 1000 (historical)** — intentional, matches the paper text.
5. `pareto_gamma` 2.0→2.5 and rough-Heston `v0/theta` defaults 0.01→0.02/0.04: defaults
   again stale vs instantiation (instantiation matches the paper's calibration cited from
   Abi Jaber–El Euch).
