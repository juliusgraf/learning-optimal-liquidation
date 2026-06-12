# Project: Code revision for "Learning Market Making with Closing Auctions" (Graf & Mastrolia)

## Mission
Rigorous code-only revision. The paper (LaTeX source in `paper/main.tex`) is the
mathematical ground truth for the market model, the MDP, Algorithm 1 (hypothetical
clearing price) and Algorithm 2 (generative stochastic market model), EXCEPT:
- Section 4 (`sec:learning`): the RL method AND its textual explanation are NOT
  authoritative (author ruling D9). They are replaced by a standard DQN plus
  continuous-action extensions (DDPG, TD3, SAC), designed per standard RL
  literature conventions; the author will rewrite Section 4 from our design doc.
- The printed Eqs. (1)-(2), the theta recursion, Prop. linear, and the terminal
  displays contain confirmed errata; the AUTHOR-CORRECTED versions below are
  binding and override the printed paper.
- The parameter table tab:params_generative was written AFTER the code and is
  NOT authoritative (rulings D6-D7): the values actually used at the legacy
  code's instantiation sites are the source of truth.
- All numerical results after Section 6.3 will be regenerated.
NEVER edit anything under `paper/`. The author updates the paper separately.
Prefer correctness, reproducibility, and clarity over preserving old
implementation choices. If a genuinely NEW ambiguity arises (not covered by the
rulings below), implement the most paper-faithful reading and record it in
`audit/AUDIT.md` under "Questions for the author" — do not silently choose.

## Paper map (labels in paper/main.tex)
- Sec. 2 `sec:marketmodel`: trading session, CLOB phase `sec:lob`, auction phase
  `sec:auction`, clearing rule `sec:clearing` with Eq. (1) `eq:hyp_cl_auction`,
  Eq. (2) `eq:hyp_cl_auction_estimation`, Theorem `th:clearing`,
  Proposition `prop:linear`, Algorithm 1 `alg:hyp_clearing_price` in `sec:proj`.
- Sec. 3 `sec:MDP`: state X^1..X^17, actions A^1..A^5, admissible set Adm(x),
  three-regime reward, objective (P) with discount chi in (0,1].
- Sec. 4 `sec:learning`: NOT authoritative (D9); only the MDP itself, chi, and
  the regret definition PRegret(T) are kept.
- Sec. 5 `sec:benchmark`: AS benchmark `section:AS` (closed-form v_q and
  delta^{a,*}), TWAP benchmark `section:TWAP`, shared auction heuristic
  (single order at auction open: supply z*q_{tau_op}(p - S_tilde)_+ with z=10,
  S_tilde = average of mean and max executed CLOB price).
- Sec. 6 `sec:numerics`: Algorithm 2 `alg:generative_model`; rough Heston
  scheme of Richard et al. in `sub:rough`; historical setting `sub:historical`.

## Binding mathematical conventions (author rulings D1-D14 are final)

### Time grid and phases
- Grid 0 = t_0 < ... < t_n < tau_op = t_{n+1} < ... < t_m < tau_cl = t_{m+1}.
  After the paper's index simplification: n = tau_op - 1, m = tau_cl - 1,
  episode horizon = m + 2 steps (decisions at 0..m; terminal clearing at tau_cl).
- CLOB decisions at t in {0,...,n}; auction decisions at t in {n+1,...,m};
  t = tau_cl is terminal: NO action, clearing + terminal reward only.
- The state X_t is what the agent observes BEFORE choosing A_t. In particular
  X^7_t = N^+_{t-}, X^8_t = N^-_{t-} (counts strictly before time t), and
  X^9_t = theta_t is predictable (see corrected recursion below).

### Sign conventions (BINDING)
- Limit orders: zeta = + denotes the ASK side (sell side), zeta = - the bid
  side. The agent sells on the ask side during the CLOB phase;
  delta_t = k_t - k_t_mid >= 0.
- Market orders: zeta = + denotes the BUY side (buy market orders consume
  opposite-side, i.e. ask, liquidity). N^+ counts BUYING market orders and
  N^- counts SELLING market orders, in BOTH phases. (Legacy code's clearing
  arithmetic already matches this economics; only its N_plus NAMING is
  inverted — rename, do not re-derive.)

### Corrected clearing equations (AUTHOR-CORRECTED; override the printed paper)
- theta recursion (corrected, ruling D3): theta_{t_n} = theta_{t_{n+1}} = 0 and
    theta_{t_j} = max( theta_{t_{j-1}},  c_{t_{j-1}} * sum_{k=1}^{j-n-2} e_k ),
  where e_k is the k-th unit vector in R^{m-n}, c in {0,1} is the scalar
  cancel-all action, and the order submitted at t_s has component index s-n.
  Semantics: c_{t_j} = 1, decided at t_j, cancels ALL agent orders submitted
  strictly before t_j (i.e. at t_{n+1}..t_{j-1}); this is reflected in
  theta_{t_{j+1}}. Hence theta_{t_j} is determined by actions up to t_{j-1}
  (predictable) and is legitimately part of the state X^9_{t_j}.
- Eq. (1), clearing at tau_cl = t_{m+1} (corrected):
    sum_{i=1}^{M_{t_m}} g_{i,t_m}(p)
    + sum_{s=n+1}^{m} (1 - theta_{t_{m+1}}^{(s-n)}) K^a_{t_s} (p - S^a_{t_s})
    - sum_{zeta in {+,-}} sum_{i=1}^{N^zeta_{t_m}} zeta * nu^{zeta,i}_{t_m} = 0.
- Eq. (2), estimate used at decision time t_j, j in {n+2,...,m} (corrected):
    sum_{i=1}^{M_{t_{j-1}}} g_{i,t_{j-1}}(p)
    + sum_{s=n+1}^{j-1} (1 - theta_{t_j}^{(s-n)}) K^a_{t_s} (p - S^a_{t_s})
    - sum_{zeta} sum_{i=1}^{N^zeta_{t_{j-1}}} zeta * nu^{zeta,i}_{t_{j-1}} = 0.
  Setting j = m+1 recovers Eq. (1) exactly (theta_{t_{m+1}} embeds c_{t_m}).
- Linear closed form (corrected Prop. linear), with all curves linear:
    p* = [ sum_i K_i S_i + sum_s (1-theta^{(s-n)}) K^a_s S^a_s
           + ( sum_i nu^{+,i} - sum_i nu^{-,i} ) ]
         / [ sum_i K_i + sum_s (1-theta^{(s-n)}) K^a_s ].
- Invariants to assert in tests: adding buy market volume weakly RAISES p*;
  adding sell market volume weakly LOWERS p*; the order submitted at t_{j-1}
  is always live in the time-t_j estimate; the final order at t_m can never be
  cancelled; changing c_{t_j} leaves the time-t_j estimate unchanged and
  changes the time-(t_{j+1}) estimate.
- Paper errata status (all resolved by the author; nothing is open here):
  the zeta*nu sign in Eqs. (1)-(2), the matching signs in Prop. linear and in
  phi(p) inside the proof of th:clearing, the theta_{t_{m+1}}^{(s-n)}
  indexing in Eq. (1)/Z_{tau_cl}/terminal reward, and the theta recursion are
  ALREADY FIXED in the author's latest tex. If paper/main.tex in this repo
  still shows the old versions, the corrected versions IN THIS FILE prevail —
  do not "fix back" toward a stale tex. The old vector-form constraint
  A^5_t <= 1 - theta_t is replaced by the AGREED scalar, TIME-FREE constraint
  (a function of the state alone, so Adm(x), greedy policies, and Puterman
  Thm 6.2.10 apply verbatim):
    c_t <= C(x) := max_{1 <= i <= m-n} ( 1 - x^{9,(i)} ) * 1{ x^{17,(i)} > 0 },
  i.e. a cancel-all is admissible iff at least one live prior order with
  POSITIVE slope K^a exists. The K > 0 indicator both excludes zero-padded
  (not-yet-submitted) entries — making any time truncation of the max
  unnecessary — and encodes the convention that submitting K^a_t = 0 is
  identified with abstaining (a zero-slope order contributes nothing to the
  clearing equation). Properties: (i) C(x) = 0 during the CLOB phase and at
  the auction open, so c_{t_{n+1}} = 0 follows automatically; (ii) each order
  is cancelled at most once by the max recursion regardless — the constraint
  only forbids vacuous-but-costly cancel-alls. PREREQUISITE (author has
  adopted this with the theta correction): the state histories X^16, X^17 use
  the PREDICTABLE indexing — at decision time t they contain entries for
  orders submitted up to t-1 only (the originally printed S^a(t) included the
  time-t action, which is not observable pre-action).

### H_cl timing (rulings D1-D2 — CRITICAL, enforce with tests)
- AUCTION (D1): the estimate H_cl used in the time-t_j state and reward is the
  solution of corrected Eq. (2), computed at the END of step t_{j-1}: exogenous
  limit orders and market orders as of end of t_{j-1}, agent orders s <= j-1,
  cancellation state theta_{t_j} (which embeds c_{t_{j-1}} and earlier).
  Nothing sampled or decided at t_j may enter it. Implementation: compute and
  CACHE at end of step t_{j-1}; step t_j reads the cache.
- CLOB (D2): Algorithm 1's update runs at the END of each step t, over the
  post-flow standing book at that moment (including any unexecuted remainder
  of the agent's time-t limit order, and BEFORE the next-step book refresh).
  Its output is H_{t+1}, the input to the agent's time-(t+1) state and reward.
  H_0 = configured initial value; the reward at t = 0 uses H_0.
- At the terminal time, S_cl solves corrected Eq. (1) using end-of-t_m
  information (which includes the agent's final t_m order and c_{t_m}).

### CLOB-phase reward (ruling D5: implement the paper EXACTLY)
  r_t(X_t, A_t) = S_t_bullet * E_t * f_c( k_star*alpha - (H_cl_t - S_t_bullet) ),
  with S_t_bullet = alpha * A^2_t, E_t the executed volume following the time-t
  action, H_cl_t the lagged Algorithm-1 value per D2, and
  f_c(u) = (u)_+ / (k_star*alpha). Do NOT clamp the multiplier at 1: for
  S_t_bullet > H_cl_t it exceeds 1 by definition (legacy clamps; that was
  wrong). Executed volume E_t per the paper's formula with agent execution
  priority at her own level.

### Auction-phase per-step reward (ruling D4: follow the paper)
  r_t = K^a_t * H_cl_t * (H_cl_t - S^a_t)
        + f_a( K^a_t * H_cl_t * (H_cl_t - S^a_t) )
        - d_t * c_t,
  with f_a(u) = -q * (-u)_+, d_t = (t - n - 1) * d (linear in the time index),
  and c_t in {0,1} the SCALAR cancel-all action (semantics above). The cost is
  d_t * c_t — NOT d times the number of orders cancelled (legacy was wrong).
  c_t = 1 is admissible only when at least one live prior order with
  K^a > 0 exists (see Admissibility), so the cost is charged exactly when a
  real cancel-all is exercised.

### Terminal reward at t = tau_cl (rulings D3, D8)
  Z_{tau_cl} = sum_{s=n+1}^{m} (1 - theta_{t_{m+1}}^{(s-n)}) K^a_{t_s} (S_cl - S^a_{t_s}),
  I_{tau_cl} = I_{tau_op} - Z_{tau_cl}  (inventory frozen during the auction),
  r_{tau_cl} = sum_s K^a_{t_s} * S_cl * (S_cl - S^a_{t_s}) * (1 - theta_{t_{m+1}}^{(s-n)})
               - lambda * |I_{tau_cl}|^2
               + sum_s f_a( K^a_{t_s} * S_cl * (S_cl - S^a_{t_s}) * (1 - theta_{t_{m+1}}^{(s-n)}) ).
  NO inventory clipping (D8). If a float-safety guard is genuinely needed,
  gate it behind config `numerical_guard` (default OFF), set thresholds far
  outside the economic range, log loudly when it binds, and add a test
  asserting it never binds on seeded standard runs.

### Admissibility (time-free, state-only — author's final choice)
  Adm(x) = { a : a^1 <= x^1 (volume <= inventory), a^2 >= x^10 / alpha (quote
  at or above mid), a^5 <= C(x) }, with
    C(x) = max_{1 <= i <= m-n} ( 1 - x^{9,(i)} ) * 1{ x^{17,(i)} > 0 },
  i.e. cancel-all only if a live prior order with K^a > 0 exists. C(x) is a
  function of the state alone (the K > 0 indicator excludes zero-padded
  entries, so no time index is needed); it equals 0 throughout the CLOB phase
  and at the auction open. Convention: submitting K^a_t = 0 is identified
  with abstaining. During the auction additionally K^a_t >= 0 and
  S^a_t in alpha*N. Exploration must sample from Adm(x): mask inadmissible
  actions in both the greedy argmax and the random draw (masking preferred
  over projection; document the choice). Implementation note: the env's
  internal ledger tracks liveness directly (equivalent to evaluating C on
  paper_state()); add a test asserting the two agree.

### Objective and discounting
  J(pi) = E[ sum_{t in T} chi^t r_t ]; chi from config (legacy used 0.99 —
  confirm at the instantiation site, per D6). Bellman targets use chi.
  Reported evaluation "returns" are the UNDISCOUNTED episode sum unless a
  table says otherwise (state in captions/metadata).
  Regret: PRegret(T) = sum_{e=1}^{E} ( V_0^{pi_benchmark}(x_{0,e}) - V_0^{pi_e}(x_{0,e}) ),
  T = (m+2)E, estimated per episode with common random numbers (same env seed
  for the learned policy and the benchmark within an episode).

### Algorithm 1 (hypothetical clearing price, CLOB phase) — exact spec
  Inputs: tick alpha > 0, smoothing tau in (0,1], H_0 (values from the LEGACY
  CODE instantiation per D6 — legacy uses gamma = 0.5 for the smoothing and
  H_0 = initial mid = 100; the paper table's 0.95 is NOT authoritative).
  At the END of each CLOB step t (ruling D2): snapshot standing orders O
  (price levels alpha*k, volumes v, including the agent's unexecuted
  remainder); update per-level running moments e_hat^k (mean level volume)
  and sigma_hat^k (mean squared level volume) over snapshots so far;
  K_hat^k = (2*e_hat - sigma_hat/e_hat)/alpha;
  S_tilde = sum_k K_hat*alpha*k / sum_k K_hat;
  H_{t+1} = H_t + tau*(S_tilde - H_t).
  K_hat < 0: replicate the legacy choice (clamp to 0) and log a counter;
  skip the smoothing update (keep previous H) when sum_k K_hat = 0.

### Algorithm 2 (generative market) — spec and parameter source (D6, D7)
  Structure per alg:generative_model: N^+, N^- independent Poisson(lambda_0)
  on [0, tau_op]; decision grid hat_t_i guaranteeing at least one new arrival
  per side per step (Assumption assump:presence); CLOB market-order sizes
  min(Pareto(v_m, gamma_m), V); top-of-book V^{zeta,1} ~ V_inf*Beta(beta_a,
  beta_b) with geometric depth decay rho; book refreshed each CLOB step.
  Auction per step: new exogenous MM with prob p1, K^i ~ U([U_1,U_2]),
  S^i ~ S^mid_{tau_op} + alpha*U({M_1..M_2}) (mid FROZEN at tau_op); MM
  cancellation with prob p2; new market taker per side with prob p3 each
  (independent); market-taker cancellation parameterized by p4.
  PARAMETER SOURCE OF TRUTH (rulings D6-D7): the values at the LEGACY CODE'S
  INSTANTIATION SITES (main.py for synthetic, data.py for historical) — not
  the constructor defaults, not tab:params_generative. Phase 1 extracts them
  into configs/; the author regenerates the paper table from configs later.
  The auction event literals in legacy become explicit config parameters:
  p1 = 0.3, p2 = 0.2, p3 = 0.3, and — AUTHOR RULING D7 — the taker-
  cancellation probability is the legacy EFFECTIVE rate p4 = 0.05 per side
  (legacy realized it as Bernoulli(0.1) gated by an extra fair coin, which is
  the same distribution; the new code implements a single Bernoulli(p4 = 0.05)
  directly and documents the legacy construction in a comment). This is
  settled; do not flag it as open.
  Rough Heston (ruling D14): KEEP the Richard et al. Euler scheme exactly as
  implemented (kernel K(u) = u^(H-1/2)/Gamma(H+1/2), (V)_+ truncation,
  correlated log-price update, physical-time scaling). Parameters from the
  legacy instantiation (H=0.1, rho=-0.7, v0=0.02, theta=0.04, kappa=0.3,
  xi=0.3). Performance work may only VECTORIZE the same recursion
  (precomputed kernel weights); validate vectorized == naive loop via
  allclose on seeded paths. No scheme substitution.

### State representation (ruling D11)
  The paper's X^1..X^17 (fixed-length zero-padded vectors) is a formal device.
  The env keeps an EFFICIENT internal representation (the same information:
  inventory, cached H_cl, book arrays, auction ledgers, theta, counters) and
  exposes: (a) lightweight accessors; (b) a `paper_state()` method
  materializing the X^1..X^17 dict for tests/documentation only (never in the
  training hot path); (c) the pruned feature vectors for RL (legacy pruning:
  CLOB (I, H_cl, L^+, L^-, S_mid, V^{+,1}, V^{-,1}, t-norm); auction
  (I, H_cl, M, N^+, N^-, theta-summary, S_mid, t-norm) — confirm exact legacy
  features in Phase 1 and keep them as the default, configurable).

### RL conventions (rulings D9, D10)
- Neither the legacy NFQ code nor the paper's Section 4 text is authoritative.
  Implement a textbook-standard DQN (Mnih et al. 2015 conventions) for the
  discrete setting: uniform replay, per-environment-step minibatch updates,
  target networks (hard update every C steps; soft-tau optional), epsilon-
  greedy with configured schedule, Huber loss, Adam, gradient clipping,
  eval mode (epsilon = 0), checkpointing, full seeding.
- Two phase networks (Q_phi CLOB, Q_psi auction) on time-augmented features
  are RETAINED as a documented DESIGN CHOICE (the phases have structurally
  different state/action spaces), not as paper fidelity. Cross-phase junction:
  a CLOB transition whose next state is in the auction bootstraps from the
  AUCTION target network; the terminal tau_cl reward is folded into the final
  auction transition with done=True and zero bootstrap (document equivalence).
- Produce docs/rl_design.md: precise architecture, update rules, schedules,
  pseudocode, and hyperparameters — written so the author can rewrite
  Section 4 of the paper directly from it.
- Seeding (D10): legacy seeding is untrusted; build fresh. ONE
  numpy.random.Generator per component, spawned from a master seed via
  np.random.SeedSequence.spawn; torch seeded from the same master seed with
  determinism flags set; NEVER global np.random.* or random.*. Same config +
  seed => bit-identical metrics across processes.
- Continuous-action extensions (DDPG/TD3/SAC) live in a SEPARATE, clearly
  labeled relaxation; the discrete DQN setting stays untouched; every
  mathematical change documented in docs/continuous_action_extension.md.

### Historical data (ruling D13)
  legacy data.csv IS real data: S&P 500 1-minute mid prices, Dec 31 2025,
  2:30-5:00 pm EST, normalized to 100 at session start. Treat the committed
  CSV as the frozen experimental input; the new loader regenerates it
  (--normalize first=100 default) and documents the construction.

## Engineering conventions
- Python >= 3.10, `src/` layout, package name `lmm`, `pyproject.toml`,
  type hints, docstrings citing paper labels and ruling numbers (e.g.
  "implements corrected Eq. (2); rulings D1, D3").
- No hard-coded experiment parameters in library code: YAML configs in
  `configs/` + CLI overrides (argparse + small dataclass loader).
- Every run writes results/<experiment_name>/<run_name>/ containing:
  config_resolved.yaml, seed.txt, git_sha.txt, metrics.csv (per-episode),
  eval/ (per-episode records), checkpoints/, logs/run.log, figures/, tables/.
  Figures and tables are ALWAYS regenerated from saved outputs by separate
  scripts (make_figures.py, make_tables.py).
- Tests: pytest in tests/; fast unit tests < 60 s total
  (`pytest -m "not slow"`); slow integration tests marked @pytest.mark.slow.
- Do not delete main.py / data.py until the final phase; keep them in
  legacy/ for the audit and behavioral comparison.
- Conventional commits per phase; gitignore results/.