# AUDIT.md — Phase 1 audit of the legacy code against the paper

**Inputs read in full:** `paper/main.tex` (787 lines), `main.py` (2,373 lines, synthetic /
rough-Heston experiment), `data.py` (1,686 lines, historical S&P 500 experiment).
**Binding conventions:** CLAUDE.md (author rulings D1–D14, corrected Eqs. (1)–(2),
corrected θ recursion, scalar cancel-all with admissibility constraint C(x)).
**Nothing outside `audit/` and `tests/audit/` was modified.**

### Status of the committed `paper/main.tex` relative to the author corrections

The committed tex already contains most of the author's fixes: the corrected
`−Σ_ζ Σ_i ζ ν` sign in Eq. (1) (tex L163) and Eq. (2) (tex L170), the matching `+Σ ζν`
in Prop. linear (L214), the corrected max-form θ recursion with `c_{t_{j-1}}` (L144),
the scalar cancel with cost `d_j = (j−n−1)d` (L144), the C(x) admissibility rule (L336),
and the `(s−n)` superscript in Z_{τcl} (L220-221). Residual stale bits in the tex (the
CLAUDE.md versions prevail, per instructions — do not "fix back"):

- The terminal reward (tex L358, L361-364, L380-384) still writes `θ_{τcl}^{(s)}` with
  `s ∈ {n+1..m}`; the corrected superscript is `(s−n)`.
- Algorithm 2 line 16 (tex L588) conditions a **new** taker arrival on "at least one
  market taker is present" — the legacy code (sensibly) does not implement this
  precondition; presence is required only for cancellations (see A.8).
- `tab:params_generative` p₄ = 0.1 (D7: effective rate is 0.05) and the NFQ-table
  τ = 0.95 / "three hidden layers" (non-authoritative per D6/D9).

Line references below are to `main.py` unless prefixed `data.py:`; the two files are
~95% duplicated, and the corresponding data.py lines are given as `(d:NNN)` where the
logic is shared.

---

## A. Correspondence table

Status legend: **match** = implements the (corrected) paper object;
**discrepancy** = deviates, see register B; **missing** = paper object absent from code;
**extra** = code behavior with no paper counterpart.

### A.1 Time grid and phases

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| Grid 0=t₀<…<t_n<τ_op=t_{n+1}<…<t_m<τ_cl; n=τ_op−1, m=τ_cl−1 (sec:math, sec:MDP) | `current_time` int/float; τ_op=120, τ_cl=150 at both instantiation sites | match | n=119, m=149 |
| CLOB decisions at t ∈ {0,…,n} | CLOB step `main.py:434-583` (d:328-466); phase switch when `current_time ≥ tau_op−1` `main.py:573-580` (d:456-463) | **discrepancy (N1)** | The switch fires when time *reaches* 119=n, so the agent **never acts at t_n**. CLOB decisions are t ∈ {0,…,118} only (verified: historical episodes are always 119 CLOB + 30 auction = 149 decisions; paper grid gives 150 + terminal). |
| Event-driven grid hat_t_i (alg:generative_model line 2; assump:presence) | `main.py:453-460, 516-535` (d:346-353, 407-424): `target = min(max(⌊t⌋+1, max(τ⁺,τ⁻)), τ_op−1)` with τ^ζ ~ first exponential arrival | match | Exact transcription. With λ₀=1 (synthetic) the grid genuinely skips/has fractional times (observed 76–89 CLOB decisions per episode); with λ₀=60 (historical) it is always the integer grid. |
| Auction decisions at t ∈ {n+1,…,m} | `main.py:585-666` (d:468-547): `t = int(current_time)`, incremented by 1 | match | Integer grid 120..149. |
| t = τ_cl terminal: no action, clearing + terminal reward only | `main.py:668-705` (d:549-586) | match (design choice) | Clearing runs inside the t_m=149 step; terminal reward is **folded into the t_m transition** with `done=True` (consistent with the documented Phase-4 equivalence). The post-terminal state (time 150) is returned but never acted on. |
| Auction phase reset at τ_op (ledgers cleared, time set to 120) | `main.py:574-580` (d:457-463) | match | N⁺,N⁻, ν-arrays, exogenous supply reset at auction open. |

### A.2 CLOB phase: processes, execution, inventory

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| Mid price S^mid_t = α k^mid_t | `mid_price` init 100.0 `main.py:263`; rough-Heston update `main.py:566`; historical path `data.py:137-142, 451` | discrepancy (N4, minor) | Mid is a float, never snapped to the α-grid; `k_mid = ⌊mid/α⌋` in `step` (`main.py:447`) but `k0 = round(mid/α)` in the Algorithm-1 snapshot (`main.py:383`) — inconsistent rounding (N3). |
| N^ζ Poisson(λ₀) per side; ζ=+ ask/buy convention | exponential interarrivals at rate `poisson_rate·dt`, `main.py:459-460, 527-535` (d:352-353, 416-424) | match | Independent per side; thinned per step segment. (CLOB N^ζ are not state-visible, per paper.) |
| Market-order volumes ν = min(Pareto(v_m,γ_m), V) | `_sample_mo_volume` `main.py:193-196` (d:154-157) | match | Inverse-CDF Pareto, capped at V=30. |
| Exogenous book V^{ζ,1} ~ V_∞·Beta(β_a,β_b), geometric decay ρ, refresh each step | `_refresh_order_book` `main.py:183-191` (d:145-152); refresh call `main.py:563` (d:448) | match | 12 levels/side (`Lc`); level j=0 sits *at* the mid price on both sides (convention). Refresh happens after the Algorithm-1 snapshot — correct order per D2. |
| Book depth L^ζ_t = inf{j: V^{ζ,j}=0} | `main.py:537-538` (d:425-426) | match | First level ≤ 1e-6, else Lc. |
| Agent limit order (k_t, v_t), sells on ask side, δ_t = k_t−k_mid ≥ 0 | `main.py:436-451` (d:330-344) | match | δ is snapped: `j_agent = clamp(⌊δ⌋, 0, Lc−1)` so action δ=12 aliases δ=11 (N6); price = α(k_mid+j_agent) ≥ mid, so Adm's a² ≥ x¹⁰/α holds structurally. |
| E_t = max(0, min(v_t, Σ_{N⁺_{t-}}^{N⁺_t} ν − Σ_{j<δ} V^{+,j})) with agent priority at her level | `process_buy_order` `main.py:462-502` (d:355-393) | match | Sequential MO processing against the static intra-step book ≡ the aggregate formula: levels j<δ consumed first, **agent filled with priority over exogenous volume at her own level** (`main.py:479-488`), remainder walks deeper. |
| I_{t+Δt} = I_t − E_t; v_t ≤ I_t | clamp `main.py:439-440`; decrement `main.py:484` | match | Plus an inert `np.clip(I, ±I_max)` each CLOB step (`main.py:555`) that can never bind during the CLOB phase (sell-only). |
| Assumption ass:small_investors (MOs always execute) | MOs walk the book; any residual beyond Lc levels evaporates | match | |
| Sell MOs consume the bid side (no agent interaction) | `process_sell_order` `main.py:504-514` (d:395-405) | match | |

### A.3 Algorithm 1 (hypothetical clearing price, CLOB phase)

| Paper object (alg:hyp_clearing_price) | Code location | Status | Notes |
|---|---|---|---|
| Snapshot standing orders O_i (post-flow, incl. agent's unexecuted remainder, pre-refresh) | call site `main.py:540` (d:428); snapshot `main.py:382-404` (d:282-300) | match (per D2's computation spec) | Both book sides; agent residual added at her level (`main.py:400-404`). Runs after the flow, before `_refresh_order_book` — exactly D2's "end of step t over the post-flow standing book". |
| ê_i^k (mean level volume), ς̂_i^k (mean squared) | running sums + snapshot counter `main.py:170-172, 406-418` (d:126-128, 302-314) | match | Levels absent from a snapshot implicitly contribute 0 (sum unchanged, count incremented) — consistent with the algorithm's indicator sums. |
| K̂_i^k = (2ê − ς̂/ê)/α | `main.py:419` (d:315) | match + legacy clamp | `max(0, ·)` clamp on K̂ — keep per spec, add a log counter in the new code. |
| Solve Σ K̂(αk − p) = 0 ⇒ S̃ = ΣK̂αk / ΣK̂ | `main.py:421-426` (d:317-322) | match | |
| H_{t_i} ← H_{t_{i−1}} + τ(S̃ − H) ; skip when ΣK̂=0 | `main.py:424-427` (d:320-322) | match | Smoothing constant = `env.gamma` = **0.95 (synthetic) / 0.99 (historical)**, constructor default 0.5 — resolved by ruling D15: τ = 0.95 in both settings (Section D, Q1). |
| H₀ = configured initial value | `main.py:263-264` (d:176): H₀ = initial mid = 100 | match | Per ruling: H₀ = initial mid = 100. |
| Output is H_{t+1}, input to the time-(t+1) state and reward (D2) | reward at `main.py:545` uses the value updated at `main.py:540` **in the same step** | **discrepancy (D2)** | The time-t reward uses end-of-t information; must be lagged one step. The *state* seen at t+1 correctly equals the end-of-t value. |

### A.4 Clearing: corrected Eqs. (1)–(2), Prop. linear, θ, Z, I_τcl

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| Corrected Eq. (2) estimate at decision time t_j (end-of-t_{j−1} info, θ_{t_j}) | `main.py:639-658` (d:522-540) | **discrepancy (D1)** | Legacy computes it at the **end of step t_j**: includes time-t_j exogenous events (sampled `main.py:591-625`), the agent's **own time-t_j order** (`for s in range(tau_op, t+1)` — includes s=t, `main.py:645`), and the time-t_j cancel (applied `main.py:627-632` *before* the solve). Verified: flipping c_t changes the time-t H_cl and reward. |
| Corrected Eq. (1) at τ_cl (end-of-t_m info incl. final order and c_{t_m}) | `main.py:668-686` (d:549-567) | match | Terminal timing is correct per D1's terminal clause: exogenous events of step t_m, agent's final order, c_{t_m} all enter; θ_{t_{m+1}} embedding c_{t_m} is realized by the live/dead ledger. Fallback `clearing_price = H_cl` if aggregate slope is 0 (extra; N9). |
| Prop. linear closed form p* = [ΣK_iS_i + Σ(1−θ)K^aS^a + (Σν⁺−Σν⁻)] / [ΣK_i + Σ(1−θ)K^a] | `main.py:640-658, 669-686` | match (arithmetic), **naming inverted (D3)** | Verified numerically: adding volume to `market_buy_volumes` (indexed by `N_minus`) **raises** p*; adding to `market_sell_volumes` (indexed by `N_plus`) lowers it. Economics = corrected paper; but the counter named `N_plus` (state X7) counts **sell** MOs. Resolution: rename, do not re-derive. |
| θ recursion (corrected): θ_{t_j} = max(θ_{t_{j−1}}, c_{t_{j−1}}Σe_k), predictable | live/dead ledger `agent_order_active` `main.py:277-279, 627-632`; state X9 `main.py:300-311` | state: match; in-step timing: **discrepancy (D1/D3)** | The state X9 at decision time t reflects cancels through c_{t−1} (predictable ✓, equals corrected θ_t). The cancel **scope** also matches: c at step t deactivates orders submitted at τ_op..t−1 (`range(tau_op, t)`), so the final t_m order can never be cancelled ✓ and c_{n+1} is vacuous ✓. What deviates: c_t takes effect *within* step t (enters the time-t estimate), whereas corrected θ defers it to t+1. |
| Z_τcl = Σ(1−θ)K^a(S_cl − S^a) | `executed_final` `main.py:688-691` (d:569-572) | match | |
| I_τcl = I_τop − Z_τcl (inventory frozen during auction) | `main.py:693-695` (d:574-576); auction steps never touch inventory | **discrepancy (D8)** | `np.clip(I_final, −I_max, I_max)` at `main.py:694` (d:575). Freezing during the auction ✓. |
| Theorem th:clearing / Corollary | n/a (analytic) | match-by-construction | Code always uses the linear closed form (Prop. linear); exogenous auction supplies are linear by construction. |
| Invariants (buy raises p*; t_{j−1} order live at t_j; final order uncancellable; c_{t_j} leaves time-t_j estimate unchanged) | — | partially violated | Buy-raises ✓ (verified). Final order ✓. **t_{j−1} order can be dead in the time-t_j estimate** and **c_{t_j} changes the time-t_j estimate** — both consequences of the D1 timing bug; fixed by the corrected caching. |

### A.5 Rewards (three regimes)

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| CLOB: r_t = S•_t E_t f_c(k*α − (H_t − S•_t)), f_c(u)=(u)₊/(k*α) | `main.py:542-545` (d:430-432): `S·E·max(0, 1 − κ·max(0, H−S))` | **discrepancy (D5)** | κ=0.1 ≡ 1/(k*α) with k*α=10 (k*=1000 at α=0.01, consistent with the table). The inner `max(0,·)` **clamps the multiplier at 1** for S• > H_cl; the paper's f_c exceeds 1 there. Identical otherwise. Also uses the same-step H (D2). |
| Auction: r_t = K^aH(H−S^a) + f_a(K^aH(H−S^a)) − d_t c_t, f_a(u)=−q(−u)₊ | `main.py:660-663` (d:542-544) | fictive + wrong-side terms: match; cancel cost: **discrepancy (D4)**; H timing: **discrepancy (D1)** | `−q·max(0,−u)` ≡ f_a ✓. Cost charged is `d · cancel_count` where `cancel_count` = #(ones in the expanded c-vector over [τ_op, t)). Because the legacy policy layer always expands a scalar cancel to ones on the whole range (`main.py:1758-1761`), this *numerically equals* d·(t−τ_op)·c_t = d_t·c_t — but it is structurally a per-order count, it is charged even when **no live order exists** (vacuous cancel), and a non-legacy driver could submit arbitrary vectors. H is the look-ahead value (D1). |
| d_t = (t−n−1)·d | emergent (see above), not structural | discrepancy (D4) | Resolution: scalar c_t, structural cost d_t·c_t, admissibility c_t ≤ C(x). |
| Terminal: ΣK^aS_cl(S_cl−S^a)(1−θ) − λ\|I\|² + Σf_a(K^aS_cl(S_cl−S^a)(1−θ)) | `main.py:698-704` (d:579-585) | match (form) | `executed_final·S_cl ≡ ΣK^aS_cl(S_cl−S^a)` over live orders ✓; wrong-side sum per live order ✓; λ=0.5. Only deviation: clipped I (D8). Folded into the t_m reward (`reward += terminal_reward`, `main.py:704`). |
| Reward at t=0 uses H₀ | uses end-of-step-0 H (D2) | discrepancy (D2) | |

### A.6 MDP state X¹..X¹⁷ vs `_get_state` (`main.py:286-375`, d:201-276)

| Paper | Code | Status | Notes |
|---|---|---|---|
| X¹ = I_t | `X1 = inventory` (L292) | match | |
| X² = Z·1{t=τcl} | `X2 = last_executed if t == tau_cl` (L293) | match | Nonzero only in the never-acted-on post-terminal state. |
| X³ = H^cl_t | `X3 = H_cl` (L294) | match (value), timing per D1/D2 | At decision time t the cached value is the end-of-(t−1) estimate — correct; only the *reward* uses the wrong vintage. At the first auction step t_{n+1}, X³ = Algorithm 1's last CLOB output (consistent with D1's "end of step t_n"). |
| X⁴,X⁵ = L^±·1{t≤τop−1} | L295-296 | match | Zeroed in auction ✓. |
| X⁶ = M_t·1{t≥τop} | `len(active_supply)` (L297) | match | M capped by `La`=12 (extra, N8). |
| X⁷ = N⁺_{t-}, X⁸ = N⁻_{t-} | `N_plus`, `N_minus` (L298-299) | **naming inverted (D3)** | Predictable ✓ (counts as of end of t−1). But `N_plus` counts **sell** MOs (its volumes lower p*), so paper-X⁷ is code-`N_minus`. Rename only. |
| X⁹ = θ_t·1{t≥τop} | L300-311 | match (semantics), **discrepancy (N5: encoding)** | Derived flags `order existed ∧ not active` ≡ θ_t (predictable ✓). Encoding inconsistent: length 31 in auction (`range(tau_op,tau_cl)` + appended 0) vs length 151 in CLOB (`[0]*(tau_cl+1)`); paper says ℝ^{m−n}=ℝ³⁰. Evidence the fixed-length device is error-prone (D11). |
| X¹⁰ = S^mid | L312 | match | Frozen at τ_op during the auction (verified) — consistent with the model. |
| X¹¹,X¹² = (ν^{±,i}) | L313-318 | **naming inverted (D3)** | X11 = `market_sell_volumes[:N_plus]` — paper's ν⁺ array is code's `market_buy_volumes`. Padded to L_max=100. |
| X¹³,X¹⁴ = (V^{±,j}) | L319-334 | match | Lc entries, zeroed above depth. |
| X¹⁵ = ((K^i,S^i))_i | L335-359 (d:249-260) | **discrepancy (N5: bug)** | Present orders append two **scalars** (K_i, S_i); absent slots append a **list** `[0.0, 0.0]` → heterogeneous, length-unstable vector (10 scalars vs mixed). An earlier supply-curve-grid version is commented out (L339-349, only in main.py). Never consumed by features — latent. |
| X¹⁶ = S^a(t), X¹⁷ = K^a(t) (predictable: entries up to t−1) | L360-368 | match (semantics), encoding note | `tau_op ≤ s < t` — strict, so the PREREQUISITE predictable indexing already holds ✓. Length τ_cl+1=151 vs paper's m−n+1 (N5). |
| Efficient internal state + `paper_state()` accessor (D11) | `_get_state` materializes the full dict **every step in the hot path** | extra/inefficiency | D11 resolution: internal ledgers + `paper_state()` for tests only + pruned feature accessors. |

### A.7 Pruned RL features (these exact lists become the defaults, ruling D11)

| Phase | Legacy definition | Exact feature list |
|---|---|---|
| CLOB | `feat_clob` `main.py:1443-1450` ≡ `data.py:1065-1072` | `[X1/I_max, X3 (H_cl, raw), X10 (S_mid, raw), X4/Lc, X5/Lc, X13[0]/V_max, X14[0]/V_max, t_norm]` with `t_norm = clip(t/(τ_op−1), 0, 1)` — 8 dims |
| Auction | `feat_auction` `main.py:1452-1457` ≡ `data.py:1074-1079` | `[X1/I_max, X3 (raw), X10 (raw), X6/La, X7/L_max, X8/L_max, t_norm]` with `t_norm = clip((t−τ_op)/(τ_cl−τ_op), 0, 1)` — 7 dims |

Notes: (i) the paper's pruned auction list (tex L614) additionally includes
`Z_t·1{t=τcl}` and `θ_t`; **legacy includes neither** — no Z and no θ-summary feature
(CLAUDE.md's "theta-summary" guess is corrected here). (ii) X7/X8 carry the inverted
naming (X7 is the sell-MO count). (iii) H_cl and S_mid enter unnormalized (~100) next
to normalized features.

### A.8 Actions, admissibility, Algorithm 2 auction events

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| A¹,A² grids | `CLOB_ACTIONS` `main.py:1424-1427` (d:1400-1402): {(0,0)} ∪ {1..30}×{1..12}, 361 actions | match | δ=12 aliases 11 (N6). |
| A³,A⁴,A⁵ grids | `AUCT_ACTIONS` `main.py:1429-1435` (d:1404-1409): K ∈ {0}∪linspace(1, 33.33, 10), off ∈ {−12..12}, c ∈ {0,1}; 550 actions | match (grids are config choices) | K grid ≠ paper's β-grid (table non-authoritative). K=0 ≡ abstain ✓ (order recorded only `if K_a > 0`, `main.py:634-637`) — matches the adopted convention. |
| Adm: a¹ ≤ x¹ | projection `v = min(v, s['X1'])` at selection `main.py:1706, 1619` + env clamp `main.py:439-440` | **discrepancy (RL-side)** | Projection, not masking; the replay stores the *unprojected* action index while the executed action differs (see C). New code: mask in argmax and in random draws. |
| Adm: a² ≥ x¹⁰/α | structural (δ ≥ 0 ⇒ price ≥ mid) | match | |
| Adm: a⁵ ≤ C(x) (cancel only if a live prior K>0 order exists) | none | **missing (D4)** | Legacy permits (and charges) vacuous cancel-alls at any auction time. |
| Exploration sampled from Adm(x) | uniform over the full grid `main.py:1704, 1755` | missing | Phase 4 requirement. |
| p₁=0.3 new exogenous MM; K~U(0.1,2.0); S ~ S^mid_{τop} + α·U{−10..10} | `main.py:591-595` (d:474-478) | match | Mid frozen at τ_op ✓ (verified). Arrival suppressed when `len(active_supply) ≥ La=12` (extra, N8). |
| p₂=0.2 MM cancellation | `main.py:596-599` (d:479-482) | match | Random index removed. |
| p₃=0.3 new taker per side (independent) | `main.py:600-615` (d:483-498) | match | No "presence" precondition for *new* arrivals (tex Alg 2 L588 says otherwise; code is the sensible reading — flag for the author's tex, no code change). N^ζ capped at L_max=100 (never binds: ~9 expected arrivals). |
| p₄ taker cancellation | `main.py:616-625` (d:499-508): `Bernoulli(0.1)` × fair coin, per side | match (D7: ≡ Bernoulli(0.05)) | Settled; new code implements `Bernoulli(p4=0.05)` directly and documents the legacy construction. Cancelled ν set to 0, N unchanged ✓ (paper: "volume can be set to zero"). |

### A.9 Benchmarks (Sec. 5)

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| v_q(t) = Σ_{j≤q} (Ae⁻¹(T−t))^j/j! | `ASBenchmark._build_M/_solve_v` `main.py:775-796` (d:656-677) | match | Backward `expm(−M·dt)` recursion with γ=0 (α_AS=0, η=A/e) reproduces the closed form; terminal v(T,·)=1 asserted (`main.py:803`). The γ>0 machinery (`main.py:736-747`) is present but used only with γ=0. |
| δ^{a,*}(t,q) = (1/(αk))[1 + ln(v_q/v_{q−1})], integer-rounded | `_compute_deltas` `main.py:810-823`; nearest-grid-δ selection `main.py:879-885` | match | `inv_k_per_tick = 1/(kα)`, `const_ticks = 1/(kα)` for γ=0; δ in ticks; nearest grid point (paper: closest integer). q=0 ⇒ no-op action (paper Remark rem:as_is_mid would quote the mid; benign deviation — no volume either way). |
| Discrete-time approximation: v_t = q_t (expose whole inventory) | `main.py:882-884`: among same-δ actions with v ≤ q, take max v | match | Capped by grid max 30 < I₀=100 early on. |
| A = λ₀/γ_m | `main.py:1938` (d:1561) | match | |
| k calibration: "k = αK", KΔp = ln Q by least squares | `estimate_K_from_env` `main.py:1886-1935` (d:1232-1280); `AS_k = pareto_shape · K̂` `main.py:1939` (d:1562) | match (with a notation caveat) | The regression `K̂ = Σ(lnQ·Δp)/Σ(Δp²)` is LS for lnQ = K̂·Δp ✓. The multiplier is γ_m (the Pareto exponent) — this is the Avellaneda–Stoikov *α* (power-law exponent), **not** the tick size; the paper text "k = αK" collides with the paper's own α=tick. Code follows AS original. n_samples=10000 (paper says 5000). `estimate_K_from_env` mutates the book via `_refresh_order_book` (benign: every episode re-resets). |
| σ calibration | `main.py:1942-1954` (pooled training-path log-returns /√dt); `data.py:1557-1559` (single fixed path) | match-ish | AS assumes arithmetic BM; code uses log-returns (≈ relative, S≈100). Copies differ (ensemble vs single path) — intentional given fixed historical path. |
| Auction heuristic: single order at auction open, supply z·q_{τop}(p−S̃)₊, z=10, S̃ = ½(mean+max executed CLOB price) | `main.py:1040-1052, 1277-1308` (d:928-951, 1022-1047) | **discrepancy (N7)** | z=10 ✓, S̃ ✓, single order at open ✓ (subsequent steps submit K=0 no-ops, `main.py:1062-1069`). But the submitted supply is **linear** K(p−S̃) — the (·)₊ hockey-stick is not representable in the linear clearing solver, so the benchmark can be filled as a *buyer* when S_cl < S̃. Resolved by ruling D16: implement the one-sided (·)₊ curve (Section D, Q2). Dust threshold q<1e-2 ⇒ no order. |
| TWAP: v_t = ⌈q_t/(T−t+1)⌉ at δ=1 | `TWAPBenchmark.choose_clob_action_index` `main.py:953-981` (d:839-870); `twap_delta = min(grid) = 1` | match | data.py adds tolerance matching + fallback pools (d:855-863) — behaviorally equal on this grid (D12). |
| TWAP auction phase = same heuristic | `run_twap_benchmark_episodes` `main.py:1129-1153` | match | Identical code path. |

### A.10 Objective, discounting, regret, reward-definition uniformity

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| J(π) = E[Σ χ^t r_t], χ ∈ (0,1] | `GAMMA = 0.99` `main.py:1440` (d:1355 default), used in all Bellman targets (`main.py:1530, 1568`) | match | **χ = 0.99 confirmed at both instantiation sites (D6).** |
| Reported "returns" | undiscounted episode sums everywhere (`cum_r += r`) | match (convention) | Per CLAUDE.md: state this in captions/metadata. |
| PRegret(T) = Σ_e (V₀^bench(x₀ₑ) − V₀^{πₑ}(x₀ₑ)), T=(m+2)E | `main.py:2030-2039` | match (estimator) | Per-episode benchmark return minus the **training** episode return (π_e = behavior policy incl. ε-exploration — consistent with "policy at the beginning of episode e" up to exploration noise), single-sample value estimates, **common random numbers**: benchmarks replay the same `EPISODE_SEEDS[e]` (`main.py:2031-2032`); the exogenous event stream is policy-independent (verified: all env draws occur unconditionally or on exogenous-only conditions). |
| `benchmark_penalty_override` usage (which policies saw which reward) | manager `main.py:714-723` (d:595-604); call sites `main.py:985, 1079, 1201, 2031-2032 (defaults), 2169, 2179`; `data.py:874, 972, 1585-1598` | extra (dead-but-dangerous) | **Every call site passes `ignore_wrong_side=False, ignore_cancel=False`** (or the function defaults, which are False), so q and d were **never disabled**: in the committed code, NFQ, AS and TWAP were all evaluated under the *same* reward definition (q=1, d=0.1). The context manager's *own* defaults are `disable=True` — a footgun if ever called bare. Note benchmarks structurally never cancel (their c-vector is `[0.0]*30`, whose indices never reach 120..149 — N11), and the wrong-side penalty does bind on their auction order. Resolution: delete the override in the rewrite; one reward definition for all policies (see C.4). |
| Periodic-eval comparison plot | `main.py:1839-1844` (NFQ on `EVAL_SEEDS`) vs `main.py:1989-2012` (benchmarks on trailing `EPISODE_SEEDS` windows) | extra (flaw) | The `eval_returns_vs_benchmark.png` curves compare across **different seed sets** (not CRN). The final-eval table (`FINAL_EVAL_SEEDS` for all four policies, `main.py:2169-2200`) and the regret plot are seed-matched ✓. |

### A.11 Rough Heston (sub:rough; ruling D14: keep exactly, vectorize only)

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| Kernel K(u) = u^{H−1/2}/Γ(H+1/2) | `main.py:160, 205-208` | match | |
| V-recursion with (V)₊ truncation, full-history sum | `main.py:238-249` | match | O(n²) Python loop — primary hotspot (D14). |
| Correlated log-price update (ρ dW_v + √(1−ρ²) dW⊥), Y drift −½V₊dt | `main.py:220-233` | match | Note: when V₊=0 the price increment is skipped but the Gaussians are still drawn (stream stable). |
| Physical-time scaling | `main.py:25-33, 557-566`: dt_phys = grid-dt × (T/τ_cl) seconds, /SECONDS_PER_YEAR | match | 1 grid unit = 1 s; session = 150 s ≈ "2 min continuous + 30 s auction" ✓. |
| Params H=0.1, ρ=−0.7, V₀=0.02, θ=0.04, κ=0.3, ξ=0.3; S₀=100 | `main.py:1421-1422, 263` | match | Constructor defaults v0/θ = 0.01/0.01 are stale (see PARAMS). |
| Mid follows rough Heston only during CLOB | update only in the continuous branch (`main.py:566`) | match | Mid (and hence exogenous auction quotes) frozen at τ_op. |

### A.12 Historical protocol (sub:historical; ruling D13)

| Paper object | Code location | Status | Notes |
|---|---|---|---|
| Data: S&P 500 1-min mids, Dec 31 2025, 2:30–5:00 pm EST, normalized to 100 | `data.csv`: 200 rows × 10 tickers, first row ≡ 100.0, timestamps `2025-12-31 14:30:00+00:00`…(+00:00 label, EST per D13) | match (D13) | Committed CSV = frozen experimental input. File holds ~200 minutes; only rows 0..119 are consumed. |
| Episode construction | `data.py:1654-1666`: `mid_path = df[sym][0:120]`; `data.py:1378-1379` requires len == τ_op; `_update_mid_price_from_path` `data.py:137-142`: `mid_t = path[clip(⌊t⌋, 0, 119)]`, called only in the CLOB branch (d:451) | match | Row r ↦ mid at decision time t=r; **auction mid frozen at row 119** (verified). Same fixed path reused for every episode ✓ (paper: "same realized price path … for each episode"). `start_from_100=False` at the call site (data pre-normalized); the `normalize_start_100` loader (`data.py:23-53`) is **unused dead code** — the new loader implements `--normalize first=100` per D13. |
| Symbols / episodes | `data.py:1651` MSFT, JPM, PG, GOOGL, CAT; `episodes=1000` `data.py:1677` | match | Matches tab:dqn_results_full. σ̂ per symbol from the single path (`data.py:1557-1559`). |

---

## B. Discrepancy register with binding resolutions

Each item: code location, confirmed/corrected diagnosis, resolution to implement in
Phases 3–5.

### Implementation status (added Phase 8)

Every ruling D1–D20 is implemented and verified. Phase 8 re-ran the full suite
in a fresh venv (**278 not-slow + 22 slow tests passing**, plus the 10 legacy
characterization tests), exercised the whole train→evaluate→regret→figures→tables
pipeline end-to-end with `--smoke`, and confirmed a same-seed double run produces
a **bit-identical `metrics.csv`** (modulo the `wall_clock_s` timing column).
"Phase" below is the phase that landed the implementation; the per-ruling
diagnosis and binding resolution follow in the entries beneath this table.

| Ruling | Phase | Implemented in (`src/lmm/…`) | Enforced by (`tests/…`, `docs/…`) |
|---|---|---|---|
| D1 — auction H_cl cache (Eq. 2) | 3 | `market/clearing.py` (Eq2Cache), `env/mdp.py` | `test_timing_conventions`, `test_rewards` |
| D2 — CLOB H_cl one-step lag | 3 | `market/clob.py` (Algorithm 1), `env/mdp.py` | `test_timing_conventions`, `test_algorithm1` |
| D3 — corrected Eq. signs / N⁺ naming | 3 | `market/clearing.py`, `env/{mdp,features,rewards}.py` | `test_clearing`, `test_sign_conventions` |
| D4 — cancel semantics, cost, C(x) | 3 | `env/{rewards,action_spaces,mdp}.py` | `test_admissibility`, `test_rewards` |
| D5 — CLOB reward clamp removed | 3 | `env/rewards.py` | `test_rewards` |
| D6 — parameter source of truth | 1–2 | `configs/` ← `audit/PARAMS_FROM_CODE.md` | `test_skeleton` (config load) |
| D7 — taker cancel p4 = 0.05 | 3 | `market/auction.py` | `test_algorithm2` |
| D8 — no terminal inventory clip | 3 | `env/{rewards,mdp}.py` | `test_rewards`, `test_agent_env_contract` |
| D9 — textbook DQN (+ DDPG/TD3/SAC) | 4 (5: cont.) | `agents/*`, `rl/*` | `test_dqn`, `test_continuous_agents`; `docs/rl_design.md`, `docs/continuous_action_extension.md` |
| D10 — fresh per-component seeding | 2 | `utils/seeding.py` | `test_determinism` |
| D11 — internal state + `paper_state()` | 3 | `env/{mdp,features}.py` | `test_admissibility`, `test_agent_env_contract` |
| D12 — one library (no main/data dup) | 3 | `src/lmm` (whole package) | full suite |
| D13 — historical data loader | 6 | `data/load_yfinance_data.py`, `configs/historical_sp500.yaml` | `test_data_loader` |
| D14 — vectorized rough Heston/book | 3 | `market/{midprice,clob}.py` | `test_rough_heston` (allclose vs naive) |
| D15 — τ = 0.95, χ = 0.99 both settings | 2 | `configs/base.yaml`, `market/clearing.py` | `test_algorithm1` + config (cite D15) |
| D16 — one-sided benchmark curve | 3–4 | `market/clearing.py`, `agents/benchmarks.py` | `test_clearing`, `test_agent_env_contract` |
| D17 — degenerate fallback H_cl = S^mid | 3 | `market/clearing.py`, `env/mdp.py` | `test_clearing` |
| D18 — benchmark one-sided reward | 4 | `env/rewards.py`, `env/mdp.py` | `test_rewards` |
| D19 — S̃ nearest-tick snapping | 4 | `agents/benchmarks.py` | benchmark eval traces (`test_agent_env_contract`, smoke) |
| D20 — discounting conventions | 4 | `experiments/regret.py`, `agents/dqn.py` | `docs/metrics_schema.md`, `docs/rl_design.md` §5 |

### D1 (RULED) — Auction H_cl look-ahead. **Confirmed.**
`main.py:639-658` (d:522-540). The time-t estimate is computed after sampling time-t
exogenous events (`main.py:591-625`), after recording the agent's time-t order (the sum
runs `s ∈ [τ_op, t]`, `main.py:645`), and after applying c_t (`main.py:627-632`); it is
then used in the time-t reward (`main.py:663`). Verified: flipping c_t changes the
time-t H_cl (99.9197 vs 100.0091 in the audit experiment) and the reward.
**Resolution:** corrected Eq. (2); compute and cache at the END of step t−1 (orders
s ≤ t−1, exogenous state as of end of t−1, θ_t embedding c_{t−1}); step t reads the
cache for both state and reward. At t_{n+1} the cache holds Algorithm 1's last CLOB
output. Enforce with the four invariant tests listed in CLAUDE.md.

### D2 (RULED) — CLOB H_cl timing. **Confirmed.**
`main.py:540` updates H over the correct snapshot (post-flow standing book incl. the
agent's unexecuted remainder, before refresh at `main.py:563`) but the result feeds the
*same-step* reward (`main.py:545`). **Resolution:** the end-of-t update produces
H_{t+1}; the time-t reward uses the cached H_t (H₀ = 100 at t=0). One-step lag only —
the computation itself is keepable as-is.

### D3 (RULED) — Corrected Eqs. (1)–(2) signs; N_plus naming. **Confirmed.**
Arithmetic at `main.py:649-658, 678-686` already matches the corrected economics
(verified: net buy volume raises p*). The inversion is purely nominal: `N_plus`
(state X7) indexes `market_sell_volumes` — i.e. the code's "+" side is the paper's
ζ=− (sell) side. The internal comments/array names are self-consistent ("sell"
volumes lower the price); only the ± labels are swapped relative to the paper's
ζ=+ = buy convention. **Resolution:** rename (`N_plus`→ sell-count or swap to paper
convention), fix X7/X8/X11/X12 mapping, do **not** re-derive the clearing equation.
θ corrections per A.4; the (s−n) indexing is realized via the per-time ledger.

### D4 (RULED) — Cancellation semantics and cost. **Confirmed, with one numerical nuance.**
Vector expansion: the policy layer expands scalar cancel to a per-time vector
(`main.py:1758-1763`, d:1495-1501); the env consumes the vector (`main.py:627-632`).
Cost: `d · cancel_count` (`main.py:661-663`) where cancel_count counts ones over
[τ_op, t). **Nuance (corrected diagnosis):** because the legacy driver always sets all
t−τ_op entries, the realized charge equals d·(t−τ_op)·c_t = d_t·c_t — *numerically*
the paper's cost under legacy's own policy code. The structural deviations stand:
(i) it is a per-entry count, not a structural d_t·c_t (other drivers can be charged
differently); (ii) vacuous cancel-alls (no live K>0 order) are both allowed and
charged; (iii) the admissibility constraint c_t ≤ C(x) is absent. **Resolution:**
scalar c_t ∈ {0,1}, structural cost d_t·c_t with d_t = (t−n−1)d, admissibility
C(x) = max_i (1 − x^{9,(i)})·1{x^{17,(i)} > 0} enforced via masking; env ledger tracks
liveness, with a test asserting ledger ≡ C(paper_state()).

### D5 (RULED) — CLOB reward clamp. **Confirmed.**
`main.py:545` (d:432): multiplier `max(0, 1 − κ·max(0, H−S•))` equals the paper's f_c
for S• ≤ H_cl but clamps at 1 for S• > H_cl. **Resolution:** implement
f_c(u) = (u)₊/(k*α) exactly — multiplier (1 − (H_cl−S•)/(k*α))₊ with no upper clamp;
k* and α from config (legacy κ=0.1 ⇔ k*α=10 ⇔ k*=1000).

### D6 (RULED) — Parameter source of truth. **Done.**
See `audit/PARAMS_FROM_CODE.md` (constructor defaults vs `main.py:1417-1422` vs
`data.py:1386-1398`, plus hardcoded literals, action grids, RL constants, benchmark
calibration, and the divergence flags). Headline three-way divergences: Algorithm-1
smoothing γ (0.5/0.95/0.99 — resolved by ruling D15: τ=0.95) and λ₀ (0.5/1.0/60.0).

### D7 (RULED) — Taker cancellation probability. **Verified.**
`main.py:616-625` (d:499-508): outer `Bernoulli(0.1)` gated by an inner fair coin
(`rng.random() < 0.5`), independently per side ⇒ exact distribution Bernoulli(0.05)
per side. **Resolution (settled):** single `Bernoulli(p4=0.05)` per side; document the
legacy two-draw construction in a comment (note: collapsing two draws into one changes
the RNG stream, so seeded trajectories will differ from legacy — expected, pinned by
the characterization tests).

### D8 (RULED) — Terminal inventory clipping. **Confirmed.**
`main.py:694` (d:575): `I_final = np.clip(I − Z, −I_max, I_max)` before the λ|I|²
penalty; also an inert per-step clip at `main.py:555` (d:440). **Resolution:** remove
both; optional `numerical_guard` config (default OFF) with far-out thresholds, loud
logging, and a test that it never binds on seeded standard runs.

### D9 (RULED) — Legacy RL non-authoritative. **Recorded in Section C.**

### D10 (RULED) — Seeding untrusted. **Confirmed; flaws documented.**
- `main.py:1410-1415`: module-level `random.seed(42)`, `np.random.seed(42)`,
  `torch.manual_seed(42)`.
- `MarketEmulator.set_seed` (`main.py:176-181`) reseeds **global** `np.random` and
  `random` with the per-episode env seed on every `reset(seed=...)` — exploration
  (`random.random()` at `main.py:1704`) becomes a deterministic function of the env
  seed (duplicated seed streams across different generators), and any interleaved
  `reset` (e.g. the periodic eval at `main.py:1839-1844`) silently reseeds the global
  exploration stream. The modulo `s % (2**32−1)` is collision-prone in general (seeds
  are drawn from [0, 2³²−1), so it happens not to wrap here — fragile, not sound).
- `data.py:132-135` seeds only `self.rng` (no global reseed) and uses a dedicated
  `policy_rng = random.Random(master_seed)` (`data.py:1384`) — the two copies therefore
  have **different coupling between env noise and exploration noise** (D12).
- Seed streams for train/eval/final-eval come from independently-seeded `default_rng`
  instances (42/120/200), not `SeedSequence.spawn`.
**Resolution:** per CLAUDE.md — one `np.random.Generator` per component spawned from a
master `SeedSequence`; torch seeded from the same master with determinism flags; no
global `np.random.*`/`random.*` anywhere.

### D11 (RULED) — State representation. **Confirmed; bugs recorded.**
Evidence the fixed-length encoding is error-prone (see A.6): X15 mixes scalars and
lists (`main.py:350-359`); X9 has phase-dependent length (151 vs 31, vs paper's 30);
X16/X17 are length 151 vs paper's m−n+1; `_get_state` builds the full dict in the
training hot path every step. **Resolution:** efficient internal ledgers + lightweight
accessors; `paper_state()` materializing correctly-shaped X¹..X¹⁷ for tests/docs only;
the exact legacy pruned feature lists of A.7 as configurable defaults.

### D12 (RULED) — main.py/data.py duplication. **Divergences enumerated.**
1. Mid-price model: rough Heston (`main.py:198-250, 566`) vs fixed historical path
   frozen during the auction (`data.py:137-142, 451`); data.py drops all `mid_rh_*`
   params and methods but `reset` still references `_reset_rough_heston_state` in an
   unreachable branch (`data.py:180-183` — would raise AttributeError if reached).
2. `set_seed`: global reseeding (main) vs generator-only (data) — affects exploration
   coupling (see D10).
3. Constructor defaults: `tau_op/tau_cl/T` 120/150/150 vs 60/70/70; `alpha` 1.0 vs 0.01.
4. Instantiation: H_cl smoothing `gamma` 0.95 vs 0.99 (discount name-collision,
   `data.py:1391`); `poisson_rate` 1.0 vs 60.0; episodes 2000 vs 1000.
5. Exploration RNG: global `random` (`main.py:1704`) vs `policy_rng` (`data.py:1466`).
6. `run_eval_episode`: globals-based signature (main) vs explicit action lists (data);
   data.py guards the cancel slice with `if t > env.tau_op` (`data.py:1221`) while
   main.py and data.py's own `evaluate_policy` (`data.py:1312-1315`) do not (no-op
   difference: the slice is empty at t=τ_op).
7. TWAP δ matching: exact equality (`main.py:969`) vs 1e-12 tolerance + fallback pools
   (`data.py:855-863`) — equal on the integer grid.
8. data.py's `run_twap` clamps v ≤ X1 before stepping (`data.py:921-922`); main.py's
   does not (env clamps anyway).
9. `soft_update`: defined-never-called (`main.py:1495-1500`) vs called as a documented
   **no-op** after the hard update (`data.py:1552-1555`; commit history "Put back
   no-op soft update").
10. Module-level `epsilon_by_episode` in data.py has `total=None` (crashes if used;
    dead — the closure at `data.py:1430-1436` is the live one and equals main.py's).
11. Plot/trace machinery (EpisodeTracker filled, `plot_episode`,
    `run_benchmark_episode_trace`, all figures) only live in main.py; data.py keeps an
    unused EpisodeTracker and emits a CSV summary instead.
12. AS σ: pooled across training paths (main) vs the single fixed path (data).
13. data.py wraps the experiment in `train_and_evaluate_one_asset(...)` with explicit
    seeds; main.py runs at module import (not importable — see tests/audit loader).
**Resolution:** one library (`src/lmm`), one `MarketEmulator`, mid-price process and
all diverging knobs as config; both legacy files preserved under `legacy/` until the
final phase.

### D13 (RULED) — Historical data. **Documented** (A.12): row r ↦ mid at decision time
t=r for r ∈ [0,120); auction mid frozen at row 119; same path every episode; committed
CSV is pre-normalized (first row = 100); the unused `load_mid_matrix_from_csv` carries
the normalization flag the new loader will own (`--normalize first=100` default).

### D14 (RULED) — Performance hotspots. **Identified.**
1. Rough-Heston V-recursion: O(n²) pure-Python loop with per-step full-history sum
   (`main.py:238-249`). Resolution: vectorize the identical recursion with precomputed
   kernel weights; validate `allclose` against the naive loop on seeded paths. No
   scheme substitution.
2. Per-step Python book loops: `process_buy_order`/`process_sell_order`
   (`main.py:462-514`) and the Algorithm-1 per-level dict loops (`main.py:406-426`)
   run per event per step. Resolution: numpy arrays per side; same arithmetic.
3. NFQ per-episode full-buffer refits: target recomputation over the whole buffer +
   3 epochs of SGD over the whole buffer, every episode (`main.py:1502-1589`) —
   O(buffer) per episode, O(E²)-ish total until the 50k cap. Resolution: standard DQN
   per-step minibatch updates (Phase 4, D9).
4. Minor: `_get_state` full-dict materialization per step (D11);
   `estimate_K_from_env` 10k refresh+walk simulations (one-off; fine).

### New discrepancies found in Phase 1 (N-series)

| # | Finding (location) | Proposed resolution |
|---|---|---|
| N1 | **No decision at t_n = τ_op−1.** Phase switch at `current_time ≥ tau_op−1` (`main.py:573`, d:456) converts the arrival *at* 119 into the auction open; the agent acts at CLOB times 0..118 only (verified: historical episodes have exactly 119+30 decisions). Paper grid (and Algorithm 2's hat-grid, which has grid points at τ_op−1) gives a decision at t_n. | Implement decisions at t ∈ {0..n} per CLAUDE.md's grid; episode = m+2 steps. Characterization tests pin the legacy 149-decision behavior for comparison. |
| N2 | **Algorithm-1 smoothing constant conflict** (PARAMS §1): 0.5 (ctor) vs 0.95 (`main.py:1419`) vs 0.99 (`data.py:1391`, name-collision with discount χ). CLAUDE.md's parenthetical "legacy uses γ=0.5" contradicts the binding instantiation sites under D6's own rule. | **RESOLVED (ruling D15, Section D):** τ = 0.95 smoothing and χ = 0.99 discount in both settings; the historical 0.99 smoothing is a bug, not reproduced. |
| N3 | Mid-to-tick rounding inconsistent: `round(mid/α)` in the Algorithm-1 snapshot (`main.py:383`) vs `floor(mid/α)` for the agent's price (`main.py:447`). Off-by-one level when frac(mid/α) ≥ 0.5. | Pick one convention (floor) across the env; document; covered by Algorithm-1 unit tests. |
| N4 | Mid price never snapped to α·ℕ; S^a = X10 + off·α is an offset grid, not α·ℕ (`main.py:594, 1757`). | Keep the offset-grid action parameterization (config); snap emitted prices to the tick grid; document. |
| N5 | State-encoding bugs (D11 evidence): X15 scalar/list mixing (`main.py:350-359`); X9 length 151 (CLOB) vs 31 (auction) vs paper 30; X16/X17 length 151 vs m−n+1. | `paper_state()` emits correctly-shaped vectors; internal ledgers unaffected. |
| N6 | Action aliasing: δ=12 clamps to 11 (`main.py:446`), so two action indices are the same effective action; (0,0) is the only δ=0 action. | Define the δ grid as {0..Lc−1} (or {1..Lc−1}) explicitly in configs; no silent clamping. |
| N7 | AS/TWAP auction heuristic submits **linear** K(p−S̃) instead of the paper's z·q(p−S̃)₊ (`main.py:1048-1052`); benchmark can clear as a buyer. | **RESOLVED (ruling D16, Section D):** implement the one-sided (p−S̃)₊ curve for benchmarks (they only liquidate); clearing solve gains a two-case piecewise step for the benchmark order. |
| N8 | Exogenous auction MM arrivals suppressed when `len(active_supply) ≥ La=12` (`main.py:592`); no such cap in the paper. | Keep as explicit config cap (`La`), set from legacy value; document; (cap binds rarely: E[#MMs] ≈ 3–4). |
| N9 | Degenerate-clearing fallbacks: Eq. (2) slope=0 ⇒ H_cl = (frozen) mid (`main.py:657-658`); Eq. (1) slope=0 ⇒ S_cl = previous H_cl (`main.py:685-686`). Paper assumes existence. | **RESOLVED (ruling D17, Section D):** fallback is H_cl = S^mid in ALL degenerate cases, including the terminal clearing (changes the legacy terminal fallback); log when it binds. |
| N10 | Periodic-eval plot compares NFQ on `EVAL_SEEDS` vs benchmarks on trailing `EPISODE_SEEDS` windows (`main.py:1839-1844` vs `1989-2012`) — not seed-matched (final table and regret are matched). | New eval harness: same seed set for every policy in any reported comparison (CRN), per CLAUDE.md. |
| N11 | Benchmark cancel vector `[0.0]*30` is indexed by absolute time s ∈ [120, t) (`main.py:631`: `s < len(c_t)` always False) — works only by accident. | Scalar c_t in the new action interface kills the issue. |
| N12 | Replay stores the **unprojected** action index while the executed volume is projected v→min(v, I) (`main.py:1705-1706` vs buffer at `1721-1728`) — many indices alias to the same executed action with different stored labels. | Admissibility masking at selection (CLAUDE.md); stored action ≡ executed action. |
| N13 | Paper text says the Q-nets have *three* width-16 hidden layers (tex L614); legacy `DQN` has **two** (`main.py:1460-1467`). | Section 4 is non-authoritative (D9); record in docs/rl_design.md so the author's rewrite matches the new implementation. |

---

## C. RL record (legacy NFQ — recorded for the design doc; discarded per D9)

### C.1 What the legacy code actually does

**Networks.** Two MLPs (`DQN`, `main.py:1460-1474`): 2 hidden layers × 16 ReLU.
CLOB net: 8 → 361 Q-values; auction net: 7 → 550 Q-values; features per A.7
(time-augmented via t_norm). Target copies created and hard-synced at startup
(`main.py:1479-1480`). main.py also deep-copies the *pre-sync random target* into
`initial_clob_net`/`initial_auct_net` (`main.py:1476-1477`) but later loads
`initial_*_net_state` (the online nets' init, `main.py:1482-1483, 2194-2195`), so the
"Initial NFQ" baseline is the untrained online net, as intended.

**Acting.** ε-greedy over the full action grid, ε: 1.0 → 0.01 exponential with
100-episode warmup (`main.py:1597-1602`). Volume projection v ← min(v, X1) *after*
action selection (`main.py:1706`) — the stored index is unprojected (N12). Cancel
expansion to per-time vectors (`main.py:1758-1763`). Exploration RNG: global `random`
(main.py — reseeded every reset via `set_seed`, D10) vs `policy_rng` (data.py).

**Replay.** Two unbounded-until-50k deques, one per phase (`main.py:1487-1493`).
CLOB tuple: `(x₈, a_idx, r·scale, done, next_is_auct, x'₈ₒᵣ₇)`; auction tuple:
`(x₇, a_idx, r·scale, done, x'₇)`. The terminal auction transition carries the folded
step+terminal reward with `done=True` (zero bootstrap) — this is the junction
convention Phase 4 keeps.

**Fitting schedule (the NFQ part).** Once a buffer holds ≥ 5,000 transitions, after
*every episode*: recompute targets `y = r + χ(1−done)·max_a Q_target(x′)` for the
**entire buffer** in one no-grad pass (`main.py:1514-1530`), with the cross-phase
junction handled correctly — CLOB transitions whose `next_is_auct` flag is set
bootstrap from the **auction** target net (`main.py:1525-1528`); then run 3 epochs of
shuffled minibatch SGD (batch 128, Adam 3e-4, Huber, grad-norm clip 1.0) over the whole
buffer against those *frozen* targets (`main.py:1535-1551`); then hard-update both
target nets (`main.py:1834-1835`). data.py additionally calls a literal no-op
`soft_update(τ=0.01)` after the hard update (`data.py:1552-1555`).

**Dead code.** A per-step target `target = r + χ(1−done)·max Q_target(x′)` is computed
inside the env loop and never used (`main.py:1729-1744` CLOB, `1777-1786` auction) —
remnant of an online-DQN variant. Also the dead conditional
`feat_auction(s2) if not done else feat_auction(s2)` (`main.py:1767`).

**Evaluation.** Greedy (argmax, no ε) via `run_eval_episode` every 100 episodes on 8
fixed seeds (`EVAL_SEEDS`, master 120); final evaluation on 100 fixed seeds
(`FINAL_EVAL_SEEDS`, master 200) for Initial NFQ / Final NFQ / AS / TWAP — all four on
the same seeds (CRN ✓). Regret uses training returns vs benchmark returns on the
training seeds (CRN ✓). The *periodic* benchmark comparison plot is not seed-matched
(N10).

### C.2 Why it is discarded (D9)

1. **Cost:** per-episode full-buffer target recomputation + 3 full-buffer epochs ⇒
   work per episode grows linearly with the buffer until the 50k cap (reached around
   episode ~420 for CLOB, ~1670 for auction): ~O(episodes²) cumulative, and ~10³×
   more gradient steps per env step than standard DQN.
2. **Correctness hygiene:** unprojected stored actions (N12); no admissibility masking
   (vacuous cancels both explored and charged); global-RNG seeding coupling env and
   exploration streams (D10); dead per-step target code and no-op soft update inviting
   confusion; per-episode hard target updates give the targets a staleness of exactly
   one full refit (neither Mnih-style C-step nor Polyak); reward look-ahead (D1/D2)
   leaks within-step information into the reward signal the Q-function regresses on.
3. Neither the paper's Section 4 text nor this code is authoritative (author ruling);
   the replacement is a textbook DQN (Mnih 2015 conventions) per CLAUDE.md, with
   continuous-action extensions (DDPG/TD3/SAC) in a separate, clearly-labeled
   relaxation.

### C.3 MDP-side invariants the Phase 4 DQN must preserve

1. **State/features:** the exact pruned vectors of A.7 (8-dim CLOB, 7-dim auction,
   same normalizations and t_norm definitions) as configurable defaults; X3 must be
   the D1/D2-corrected cached H_cl.
2. **Actions:** the grids of A.8 (361 CLOB / 550 auction, scalar c) as configured
   defaults; admissibility via masking in both argmax and random draws
   (a¹ ≤ x¹, a² ≥ x¹⁰/α, a⁵ ≤ C(x)); document masking choice.
3. **Rewards:** corrected three-regime definitions (D4/D5 forms, D1/D2 timing,
   D8 no clipping); terminal reward folded into the final auction transition with
   done=True and zero bootstrap (documented equivalence).
4. **Timing:** decisions at t ∈ {0..n} ∪ {n+1..m} (fixing N1), terminal clearing at
   τ_cl from end-of-t_m information.
5. **Discounting:** χ = 0.99 in all Bellman targets; cross-phase junction — CLOB
   transitions entering the auction bootstrap from the auction target network.
6. **Evaluation:** undiscounted episode sums; common random numbers — identical env
   seed for learned policy and benchmark within an episode, identical seed sets for
   any reported comparison (fixes N10); regret per the paper's PRegret with the
   benchmark as π°.
7. **Single reward definition for all policies:** see C.4.

### C.4 `benchmark_penalty_override` — explicit resolution

Definition `main.py:714-723` (d:595-604) with *manager defaults* disable=True, but
every call site in both files passes (or defaults to) `False, False`:
`main.py:985, 1079, 1201, 2031-2032, 2169, 2179`; `data.py:874, 972, 1585-1598`.
**Therefore the published comparisons (final-eval table, regret curves, historical
table) were all generated under the single, common reward definition (q=1, d=0.1) for
NFQ, AS and TWAP alike — the comparison is valid on this axis.** The machinery is
nonetheless a footgun (calling the manager bare silently zeroes the penalties).
Resolution: delete the override entirely; the new evaluation harness uses one reward
definition for every policy, asserted in tests.

**Phase-4 closing note (implemented).** No per-policy reward machinery exists in the
new code: every policy (DQN, initial-DQN, AS, TWAP) is evaluated on envs built from
the SAME resolved config, so the single shared `RewardParams` applies identically to
all; any `reward.*` override therefore changes all policies at once (this realizes
the "explicit evaluation config flag" — default: no exemptions). The shared values
are recorded in `eval/metadata.yaml` and asserted equal to the resolved config by
`tests/test_agent_env_contract.py::test_end_to_end_pipeline_and_bit_identical_determinism`.
This is consistent with the published legacy runs (which, per the table above, never
disabled the penalties).

---

## D. Questions for the author — ALL ANSWERED

Q1–Q3 (Phase 1) are answered by rulings D15–D17 (2026-06-11); Q4–Q6 (Phase 4:
D16 sub-questions and the discounting conventions) are answered by rulings
D18–D20 (2026-06-12). All are binding for Phases 2–6.

**Q1 — Algorithm-1 smoothing constant. ANSWERED (ruling D15).** Context: the
instantiation sites pass `gamma=0.95` (synthetic, `main.py:1419`) and `gamma=0.99`
(historical, `data.py:1391` — the RL discount argument accidentally forwarded into
the smoothing slot), while the constructor default is 0.5.
**Ruling: χ = 0.99 is the RL discount factor and τ = 0.95 is the Algorithm-1
smoothing parameter, for BOTH the synthetic and the historical setting.** The
historical legacy run's smoothing of 0.99 is thereby ruled a bug, not to be
reproduced: Phase-2 configs pin `algo1.tau = 0.95` and `rl.chi = 0.99` everywhere
(regenerated historical results will intentionally differ from legacy on this axis;
the characterization tests keep pinning the legacy 0.99 behavior for comparison).

**Q2 — Benchmark auction supply curve. ANSWERED (ruling D16).** Context: the paper's
heuristic is one-sided, g̃_z(p) = z·q_{τop}(p − S̃)₊; legacy submits the two-sided
linear z·q_{τop}(p − S̃) (`main.py:1048-1052`), so the benchmark could be filled as a
buyer. **Ruling: benchmarks use the one-sided g̃_z(p) = z·q_{τop}(p − S̃)₊, because
benchmarks can only liquidate during the auction.** Implementation consequence for
Phases 3/5: the clearing solve for benchmark episodes must handle one hockey-stick
term on top of the linear aggregate — the LHS of Eq. (1)/(2) stays continuous and
nondecreasing in p, so the root is found by a two-case analysis: solve the linear
form excluding the benchmark order; if that root is ≤ S̃ it stands, otherwise re-solve
with the benchmark slope included (and the benchmark order contributes only above S̃).
The agent's own orders remain linear; Prop. linear is untouched for the agent.

**Q3 — Degenerate clearing fallbacks. ANSWERED (ruling D17).** Context: when the
aggregate slope is zero, Eq. (2)/(1) has no solution; legacy uses H_cl = S^mid for
the estimate (`main.py:657-658`) but the *previous H_cl* at the terminal
(`main.py:685-686`). **Ruling: use H_cl = S^mid as the fallback in ALL degenerate
cases** (estimate and terminal alike; S^mid is frozen at τ_op during the auction).
Log when the fallback binds.

**D17 refinement — float-safety slope guard (author, 2026-06-14).** The
exactly-zero test `ΣK <= 0` is replaced by a machine-scale guard
`ΣK <= _SLOPE_EPS` (`_SLOPE_EPS = 1e-8`, `src/lmm/market/clearing.py`) in the
linear solve, the hockey-stick solve, and Algorithm 1's `S_tilde` denominator:
an aggregate slope that should be exactly zero but carries floating-point
residue now reliably takes the S^mid fallback instead of producing a spurious
huge price. This is a NUMERICAL guard ONLY — economically small-but-real slopes
(`ΣK ~ 1e-2`, which legitimately yield large clearing prices) are NOT
regularized; that is faithful model behavior, and the RL-side instability from
chasing it is handled by the continuous agents' replay reward clipping
(`reward_clip`), not by altering the clearing. Pinned by
`tests/test_clearing.py::test_float_safety_guard_treats_sub_eps_slope_as_zero`
and `::test_economically_small_slope_is_not_regularized`. Normal seeded runs are
unaffected (the guard fires only below 1e-8, far below any realized slope), so
determinism goldens are unchanged.

### Phase-4 items — ANSWERED (rulings D18–D19, 2026-06-12)

New ambiguities that arose while implementing ruling D16 (benchmarks' one-sided
auction order); the most paper-faithful reading was implemented per CLAUDE.md and
the author has confirmed both.

**Q4 — Reward treatment of the one-sided benchmark order. ANSWERED (ruling D18).**
The paper's three-regime reward is written for linear curves K^a(p − S^a). For the
benchmark's hockey-stick z·q_{τop}(p − S̃)₊ (D16) the implemented reading replaces
the linear leg by the SUPPLIED VOLUME everywhere it appears: per-step reward
u = K^a·H_cl·(H_cl − S̃)₊, terminal contribution K^a·S_cl·(S_cl − S̃)₊, and
Z_{τcl} contribution K^a·(S_cl − S̃)₊. Consequently u ≥ 0 always, so the wrong-side
penalty f_a never binds for the one-sided order. **Ruling: confirmed — no
wrong-side penalty for the benchmark, because the benchmark is always selling.**
Implemented in `src/lmm/env/rewards.py` (`one_sided` arguments) and
`src/lmm/env/mdp.py::_terminal`; hand-checked in `tests/test_rewards.py` and
`tests/test_agent_env_contract.py`.

**Q5 — Tick-snapping of S̃ for the benchmark auction order. ANSWERED (ruling D19).**
S̃ (mean of the mean and max executed CLOB prices) is generally off-grid, while the
paper requires S^a ∈ αN. The env quotes auction orders as
α·(⌊S^mid_{τop}/α⌋ + offset) (AUDIT N4), so the benchmarks submit
offset = round(S̃/α) − ⌊S^mid_{τop}/α⌋, i.e. the executed quote is α·round(S̃/α) —
nearest-tick rounding of S̃. (Legacy passed the raw float S̃, off-grid.)
**Ruling: confirmed — the executed quote is the nearest-tick rounding of S̃.**
Implemented in `src/lmm/agents/benchmarks.py::_LiquidationBenchmark._auction_action`.

**Q6 — Discounting conventions for real-valued CLOB decision times. ANSWERED
(ruling D20, 2026-06-12).** The CLOB decision times t̂_i are real-valued with a
variable count per episode (Assumption assump:presence; ≈ 78 decisions over 120 time
units at λ₀ = 1), so "χ^t" is ambiguous between the grid TIME and the step INDEX.
Implemented pair: (i) the REPORTED discounted return (`return_disc`; the V_0
estimator in regret.py) uses the paper objective exactly — Σ χ^{t} r_t with t the
decision time, terminal at χ^{τ_cl} — applied identically to all policies under CRN;
(ii) the DQN Bellman targets use the standard one-step χ per transition (textbook
DQN, ruling D9). The mismatch affects only the training objective: relative to the
paper objective it up-weights auction/terminal rewards vs CLOB rewards by at most
χ^{−(τ_op − n_clob)} ≈ 0.99^{−42} ≈ 1.5, with no effect on any reported metric.
**Ruling: confirmed — keep both conventions as implemented; a semi-Markov χ^{Δt}
target would be non-standard complexity for a negligible policy difference.** Both
conventions are stated in `docs/metrics_schema.md` and `docs/rl_design.md` §5.

---

## E. Characterization tests

`tests/audit/test_legacy_characterization.py` (marker `@pytest.mark.legacy`; marker
registered in `tests/audit/conftest.py`). For the synthetic emulator (main.py, loaded
by exec-ing the source truncated before the module-level training script, since
main.py trains at import) and the historical emulator (data.py, importable), it runs
seeded episodes (seeds 101/202/303 + a cancel-at-135 variant) under fixed
deterministic policies and pins: total reward, final inventory, decision count, and a
SHA-256 hash of the H_cl trajectory (10-decimal rounding). These goldens pin current
behavior so later phases can show exactly which change altered which statistic.
Regeneration instructions are in the test file's docstring.

---

## Executive summary

1. Worst correctness issue: D1 — the auction reward at t uses an H_cl that already
   contains the agent's own time-t order, time-t exogenous events and c_t (verified:
   flipping c_t moves the time-t reward), letting the fictive reward be manipulated.
2. Second: D2 — the CLOB reward at t uses the end-of-t Algorithm-1 value instead of
   the cached end-of-(t−1) value; same one-step look-ahead family.
3. Third: D5 clamp — CLOB reward multiplier capped at 1, truncating the paper's
   premium for quoting above H_cl; and D8 — terminal inventory clipped at ±100 before
   the λ|I|² penalty.
4. D4: cancel cost numerically equals d_t·c_t under legacy's own driver, but vacuous
   cancel-alls are allowed and charged; the C(x) admissibility constraint is absent.
5. D3 confirmed as naming-only: clearing arithmetic already implements the corrected
   economics (buys raise p*); `N_plus`/X7/X11 label the sell side — rename only.
6. New N1: the agent never gets the paper's decision at t_n = 119; episodes have 149
   decisions, not 150 (+terminal).
7. New N12/A.8: no admissibility masking anywhere; volume is silently projected after
   selection and the replay stores the unprojected action.
8. benchmark_penalty_override: every call site passes False/False — all published
   comparisons used one common reward definition; the override is dead-but-dangerous.
9. Legacy NFQ: per-episode full-buffer refits (O(episodes²)), per-episode hard target
   syncs, dead per-step target code, no-op soft update — replaced by standard DQN (D9).
10. Seeding (D10): env reset reseeds *global* np.random/random with the env seed in
    main.py (exploration ≡ f(env seed)); data.py doesn't — the two experiments don't
    even share an exploration mechanism.
11. Params, biggest three-way divergence: Algorithm-1 smoothing γ = 0.5 (default) /
    0.95 (synthetic) / 0.99 (historical, accidental reuse of the discount) — resolved
    by ruling D15: τ = 0.95 and χ = 0.99 in both settings.
12. Second divergence: λ₀ (poisson_rate) = 0.5 (default) / 1.0 (synthetic) /
    60.0 (historical) — changes the decision grid qualitatively (synthetic episodes
    are 106–119 decisions long; historical always 149).
13. Third: constructor defaults are systematically stale (V 5000→30, v_m 1000→2,
    V_top_max 50000→15, λ 0.005→0.5, d 1.0→0.1, α 1.0→0.01 in main.py) — configs must
    be seeded from instantiation sites only.
14. Historical protocol pinned: mid path = data.csv rows 0..119 (pre-normalized to
    100), same path every episode, auction mid frozen at row 119.
15. The three Phase-1 questions are all ANSWERED (rulings D15–D17, Section D):
    τ = 0.95 / χ = 0.99 in both settings; benchmarks use the one-sided
    z·q(p−S̃)₊ curve; degenerate-clearing fallback is H_cl = S^mid in all cases.
    No open questions remain — everything else is covered by rulings D1–D17.

---

## F. Maintenance-phase parameter changes (post-Phase-8; FOR AUTHOR RATIFICATION)

These are maintenance/extension-mode changes made while stabilising the
continuous-action agents (DDPG/TD3/SAC) on the seeded `reproduce_all` run. They
do **not** touch the clearing math, the reward forms, the timing conventions, or
Algorithm 2's *structure*; one is a learner-side knob and one is a generative
*parameter* value (config is the source of truth per D6). The second changes the
auction's economic regime and is flagged here for the author to **ratify or
revise** when regenerating `tab:params_generative`.

### F.0 The auction clearing singularity (diagnosis)

The continuous agents' returns blew up to 1e9–1e11 (regret ≈ −1e11) on the
seeded run. Root cause is **not** a bug: the clearing price (corrected Prop.
linear) is `p* = N / D` with aggregate auction supply slope
`D = Σ_i K_i + Σ_s (1−θ) K^a` and `N = Σ K_i S_i + Σ(1−θ)K^a S^a + (Σν⁺−Σν⁻)`.
As `D → 0` the slope-weighted price terms vanish with it, but the **net taker
imbalance** `(Σν⁺−Σν⁻)` does not — so `p* → ∞`. `D → 0` requires **no exogenous
MM present (M=0) AND the agent's live slope ≈ 0** (all slopes are ≥ 0). The
continuous actor learns to set `K^a → 0` exactly in the `M=0` states, where the
fictive per-step auction reward `K^a·H_cl·(H_cl−S^a)` diverges. This is faithful
model behaviour (Prop. linear + ruling D17: economically-small slopes are NOT
regularised), so it is handled at the learner/parameter level, never at the
clearing.

### F.1 `reward_clip` 25 → 8 (continuous configs; learner-side, no model change)

`configs/algo/{ddpg,td3,sac}.yaml`. The replay-only reward clip (applied in
`continuous_base.py` AFTER `reward_scale`, to the value the critic regresses on;
reported returns use the raw env reward) was recalibrated to the honest reward
envelope. The largest *legitimate* per-transition reward is the terminal reward,
dominated by `λ·I_max² = 0.5·100² = 5000` plus bounded PnL (AS/TWAP reach ~5500
paper ≈ 5.5 scaled). clip=8 sits just above this — it never truncates the real
objective signal, yet bounds the critic target to the honest scale, killing the
1e9–1e11 target divergence. The old 25 left the target ~4.5× a legit terminal.
Pinned by `tests/test_reward_scaling.py::test_reward_clip_calibrated_to_honest_envelope`.
This is purely learner-side and does not affect the benchmarks or any reported
metric definition.

### F.2 `p1` 0.3 → 1.0, `p2` 0.2 → 0.0 (base.yaml; GENERATIVE PARAMETER — RATIFY)

`configs/base.yaml`. `p1` (new exogenous MM arrival prob) and `p2` (MM
cancellation prob). With `p1=1, p2=0` at least one exogenous MM is present at
every fresh auction solve (from `t_{n+2}`; the open `t_{n+1}` reads the cached
CLOB `H`), so `D ≥ Σ_i K_i > 0` always and the `D → 0` singularity cannot occur —
including when the agent zeroes its own `K^a` (it cannot remove the exogenous
MMs). Verified (learner-independent, abstain policy, 30 episodes):
degenerate-clearing fallbacks **313 → 0**, mean MMs/auction-step **1.84 → 9.48**,
`frac(M=0)` 36.9% → 3.2% (the residual is exactly the safe cached-`H` open step).
End-to-end (TD3 500-ep synthetic smoke, with F.1): singular episodes
(`|ret|>1e5`) **15.8% → 0%**, eval returns bounded O(1e4) vs spikes to 1.38e10,
median ≈ 10.6k ≈ AS.

**TRADE-OFF the author must weigh.** `p1`/`p2` are D6-pinned to the legacy
instantiation (0.3 / 0.2). Setting `p1=1, p2=0` is a deliberate config choice
(D6: config is source of truth; `tab:params_generative` regenerated from it),
NOT an author ruling. It **densifies auction liquidity** (≈9.5 MMs/step vs ≈1.8),
making the closing auction a thicker-book regime than the paper's thin-liquidity
calibration — which softens the very thin-liquidity phenomenon a closing-auction
model studies. Note also `K_min` (`U₁`) is still 0.1, so a lone early MM gives
only `D ≥ 0.1` (a weak floor); `|H_cl|` can still reach ~120 at the first fresh
solve (bounded, no catastrophe). Options for the author:
  (i) **Ratify** `p1=1, p2=0` (accept the denser-liquidity auction);
  (ii) **Revise** to a milder `p1`/`p2`/`U₁` that lowers — but does not
       eliminate — the singularity's frequency (it stays reachable for any
       `p1<1`), relying more on F.1 + robust reporting;
  (iii) Express guaranteed baseline liquidity **structurally** instead — a
        persistent reserve MM with fixed slope `K_reserve` (always present, never
        cancels) added to Algorithm 2 — which removes the singularity by
        construction and is economically explicit, but is a model-structure
        change (would be a new ruling, not a parameter).
The singularity is `imbalance / slope`: `p1`/`p2`/`U₁` change its **rate**, only
a structural floor (option iii, or a minimum agent slope `K^a ≥ 1` that removes
the `K^a=0 ≡ abstain` action) changes its **existence**.

### F.3 RL learning tuning (2026-06-15; learner-side only, no model change)

Diagnostic seed-42 runs showed all four algos *learn* a policy that beats
AS/TWAP, but vanilla DQN and DDPG **peak then decay/collapse** (overestimation),
so the reported `final.pt` was unreliable. All fixes below are learner-side and
do NOT touch the clearing math, reward forms, timing (D1/D2), θ, admissibility,
or the rough-Heston scheme. Validated single-seed (synthetic seed 42; historical
MSFT seed 42) — NOT multi-seed; the full reproduction (seed 42 × 5 tickers) is
the confirming run. None of these is an author ruling; they are D9 RL-design /
hyperparameter choices, recorded here and in `docs/rl_design.md` for the
Section-4 rewrite.

- **DQN Double-DQN** (`configs/algo/dqn.yaml` `double_q: true`; code default
  `False` = vanilla Mnih-2015): online-net action selection, target-net
  evaluation (van Hasselt et al. 2016). Reduces the max-bootstrap overestimation;
  `final.pt` ~doubled (4.8k→9.8k @600 ep) and the eval curve stabilised.
  Pinned by `tests/test_dqn.py::test_double_q_bellman_targets`.
- **DQN other knobs**: `lr` 3e-4→1.5e-4, `target_soft_tau` 0.005→0.0025,
  `hidden_layers` [16,16]→[64,64], `epsilon_decay_episodes` 600 (synthetic
  1000-ep; historical 500-ep overrides to 300 via `scripts/run_historical_dqn.sh`
  `-o`, the rule decay≈0.6×episodes), `eval_n_seeds` 8→24.
- **DDPG retune** (`configs/algo/ddpg.yaml`): `critic_lr` 1e-3→3e-4 (the
  aggressive 1e-3 critic drove the overestimation collapse; lowering it removed
  the collapse — `final.pt` 0.6k→15.8k, stable rising curve, synthetic AND
  historical), `target_soft_tau` 0.005→0.0025, `eval_n_seeds` 8→24. The
  structural cure is clipped double-Q (= TD3, already provided); this is the
  within-DDPG fix so DDPG stands as a non-collapsing baseline.
- **Episode counts**: synthetic 2000→1000, historical 1000→500
  (`configs/{synthetic_rough_heston,historical_sp500}.yaml`). All four plateau
  well before these budgets; historical needs fewer (lambda0=60 ⇒ ~2× decisions
  /ep and a fixed mid path ⇒ lower variance). TD3/SAC unchanged (already stable).
