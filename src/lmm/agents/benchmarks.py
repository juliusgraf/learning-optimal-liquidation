"""Benchmark policies (paper `sec:benchmark`): Avellaneda-Stoikov and TWAP.

Both share the auction heuristic: a SINGLE order at the auction open with the
one-sided supply z * q_{tau_op} * (p - S_tilde)_+, z = 10, S_tilde = average
of the mean and max executed CLOB prices (ruling D16: one-sided curve —
benchmarks only liquidate; the legacy linear curve could clear as a buyer,
AUDIT N7). Subsequent auction steps abstain (K^a = 0); dust threshold
q < dust_threshold => no order. One reward definition for all policies
(AUDIT C.4: the legacy penalty override is deleted).

Conventions (Phase 4 resolutions, recorded in audit/AUDIT.md):
- "executed CLOB price" = the agent's submitted price S^bullet_t on steps
  with E_t > 0 (legacy main.py:547-551), tracked via Transition.info;
- S_tilde is snapped to the tick grid through the auction offset
  (author-confirmed, ruling D19):
  offset = round(S_tilde/alpha) - floor(S_mid_frozen/alpha), so the env's
  quote alpha*(floor(S_mid/alpha) + offset) equals alpha*round(S_tilde/alpha);
- the one-sided order's reward uses the positive-part gap with no wrong-side
  penalty (author-confirmed, ruling D18; see env/rewards.py);
- benchmarks submit RAW ClobAction/AuctionAction values (off-grid allowed by
  the env); admissibility (v <= inventory, K >= 0) holds by construction.

Benchmarks run on the SAME env through the same episode loop as the DQN
(rl/loops.run_episode) — the structural CRN guarantee (ruling D10).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lmm.agents.base import Agent, Transition
from lmm.config import BenchmarkParams, ExperimentConfig
from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.market.clob import OrderBook, sample_mo_volume

__all__ = ["ASBenchmarkAgent", "TWAPBenchmarkAgent"]

_BOOK_EPS = 1e-9  # legacy estimate_K_from_env eps (main.py:1890)


class _LiquidationBenchmark(Agent):
    """Shared scaffolding: env binding, executed-price tracking, and the
    one-sided auction-open order (ruling D16)."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.params: BenchmarkParams = cfg.benchmark
        self._env = None
        self._exec_prices: list[float] = []
        self._auction_opened = False
        # AS closed-form horizon T = tau_op - 1 (legacy AS_T, main.py:2056):
        # the last CLOB decision time on the index grid.
        self.T_as = cfg.grid.tau_op - 1

    # -- lifecycle --------------------------------------------------------------

    def bind(self, env) -> None:
        self._env = env

    def start_episode(self, episode: int) -> None:
        self._exec_prices = []
        self._auction_opened = False

    def observe(self, transition: Transition) -> None:
        """Track executed CLOB prices (submitted price on steps with E_t > 0)
        for S_tilde; benchmarks do not learn."""
        tr = transition
        if tr.phase == "clob" and tr.info is not None and tr.info.get("E_t", 0.0) > 0.0:
            self._exec_prices.append(float(tr.info["S_bullet"]))

    def update(self) -> dict[str, float]:
        return {}

    # -- acting -------------------------------------------------------------------

    def act(self, obs: np.ndarray, mask: np.ndarray, phase: str, *, eval_mode: bool = False):
        if self._env is None:
            raise RuntimeError(f"{type(self).__name__} needs bind(env) before acting")
        if phase == "clob":
            return self._clob_action()
        return self._auction_action()

    def _clob_action(self) -> ClobAction:
        raise NotImplementedError

    def _auction_action(self) -> AuctionAction:
        """First auction decision: the single one-sided order
        z * q_{tau_op} * (p - S_tilde)_+ (D16); abstain afterwards, and
        entirely when q < dust_threshold or no CLOB execution happened
        (S_tilde undefined)."""
        if self._auction_opened:
            return AuctionAction(0.0, 0, 0)
        self._auction_opened = True
        env = self._env
        q = float(env.inventory)
        if q < self.params.dust_threshold or not self._exec_prices:
            return AuctionAction(0.0, 0, 0)
        s_tilde = 0.5 * (max(self._exec_prices) + sum(self._exec_prices) / len(self._exec_prices))
        alpha = env.grid.alpha
        offset = round(s_tilde / alpha) - math.floor(env.s_mid / alpha)
        return AuctionAction(self.params.z * q, int(offset), 0, one_sided=True)

    def _q_int(self) -> int:
        """Inventory index: int() truncation clipped to [0, I_max]."""
        return int(np.clip(int(self._env.inventory), 0, self.cfg.grid.I_max))

    def _t_idx(self) -> int:
        """Time index: floor of the (real-valued) CLOB decision time,
        clipped to [0, T_as]."""
        return int(np.clip(int(self._env.t), 0, self.T_as))

    # -- persistence (stateless policies) -------------------------------------------

    def save(self, path: str | Path) -> None:
        """No-op: the policy is a deterministic function of the config."""

    def load(self, path: str | Path) -> None:
        """No-op: nothing to restore."""


class ASBenchmarkAgent(_LiquidationBenchmark):
    """Avellaneda-Stoikov benchmark (`section:AS`; AUDIT A.9).

    Closed form (gamma -> 0): v_q(t) = sum_{j<=q} (A e^{-1} (T - t))^j / j!
    and delta^{a,*}(t, q) = (1/(alpha k)) [1 + ln(v_q / v_{q-1})] in TICKS
    (nearest-integer tick per Remark rem:as_is_mid, clipped to the delta
    grid); volume = full inventory exposure capped at V_max. Calibration:
    A = lambda_0 / gamma_m; k = gamma_m * K_hat with K_hat the least-squares
    coefficient of ln Q = K_hat * Delta p over ``as_n_samples`` simulated
    executions on refreshed exogenous books (legacy main.py:1887-1939);
    sigma = std(pooled log-returns, ddof=1)/sqrt(dt) over simulated mid
    paths per ``as_sigma_rule`` (sigma only enters for gamma > 0; recorded
    for completeness, as_gamma = 0 in all published runs).
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)
        self.A: float | None = None
        self.k: float | None = None
        self.sigma: float | None = None
        self.delta_ticks: np.ndarray | None = None  # (T_as+1, I_max+1)

    # -- calibration (AUDIT A.9; seeded streams per ruling D10) -------------------

    def calibrate(
        self,
        env,
        *,
        rng_k: np.random.Generator,
        rng_sigma: np.random.Generator,
    ) -> dict[str, float]:
        """Estimate (A, k, sigma) and precompute the delta^{a,*} table."""
        cfg = self.cfg
        p = cfg.clob_flow
        if cfg.benchmark.as_gamma != 0.0:
            raise NotImplementedError(
                "only the gamma -> 0 closed form is implemented (as_gamma = 0)"
            )
        self.A = p.lambda0 / p.gamma_m
        self.k = p.gamma_m * self._estimate_K_hat(rng_k)
        self.sigma = self._estimate_sigma(env, rng_sigma)
        self.delta_ticks = self._build_delta_table()
        return {"A": self.A, "k": self.k, "sigma": self.sigma}

    def _estimate_K_hat(self, rng: np.random.Generator) -> float:
        """Least squares of ln Q on Delta p over simulated executions:
        refresh an exogenous book, draw one market-order volume Q, walk the
        ask side; Delta p = alpha * (first surviving level). Skips samples
        that empty the book or leave the best level unchanged (legacy
        main.py:1887-1932 exactly)."""
        cfg = self.cfg
        p = cfg.clob_flow
        book = OrderBook(p, cfg.grid)
        ln_q: list[float] = []
        d_p: list[float] = []
        for _ in range(self.params.as_n_samples):
            book.refresh(rng)
            Q = sample_mo_volume(rng, p.v_m, p.gamma_m, p.V_max)
            ask = np.asarray(book.ask_volumes, dtype=float).copy()
            remain = Q
            for j in range(p.Lc):
                if remain <= _BOOK_EPS:
                    break
                take = min(remain, ask[j])
                ask[j] -= take
                remain -= take
            surviving = np.flatnonzero(ask > _BOOK_EPS)
            if len(surviving) == 0:
                continue  # book emptied: depth unidentified
            delta_p = cfg.grid.alpha * float(surviving[0])
            if delta_p <= 0.0:
                continue  # best level unchanged
            ln_q.append(math.log(Q))
            d_p.append(delta_p)
        if len(ln_q) < 2:
            raise RuntimeError("not enough samples to estimate K_hat")
        x = np.asarray(ln_q)
        y = np.asarray(d_p)
        return float(np.dot(x, y) / np.dot(y, y))

    def _estimate_sigma(self, env, rng: np.random.Generator) -> float:
        """Pooled std of log-returns over simulated mid paths on the integer
        CLOB grid 0..tau_op, ddof=1, scaled by 1/sqrt(dt). ``single_path``
        (historical, D13) uses one path; ``pooled_paths`` uses
        ``as_sigma_n_paths``."""
        g = self.cfg.grid
        n_paths = 1 if self.params.as_sigma_rule == "single_path" else self.params.as_sigma_n_paths
        rets: list[np.ndarray] = []
        for _ in range(n_paths):
            mid0 = env.midprice.reset(rng)
            path = [mid0] + [env.midprice.advance_to(float(t)) for t in range(1, g.tau_op + 1)]
            rets.append(np.diff(np.log(np.asarray(path))))
        dt = g.T_physical / g.tau_cl
        return float(np.std(np.concatenate(rets), ddof=1) / math.sqrt(dt))

    def _build_delta_table(self) -> np.ndarray:
        """delta^{a,*}(t, q) in ticks via the stable cumulative-logsumexp of
        the v_q(t) log-terms j ln(x) - ln(j!), x = A e^{-1} (T - t) (so the
        matrix-exponential solve of the legacy code is replaced by the
        paper's closed form; they agree — the generator matrix at gamma = 0
        is nilpotent with constant subdiagonal eta = A/e)."""
        I_max = self.cfg.grid.I_max
        inv_k_tick = 1.0 / (self.k * self.cfg.grid.alpha)
        j = np.arange(I_max + 1, dtype=float)
        log_fact = np.array([math.lgamma(jj + 1.0) for jj in j])
        table = np.full((self.T_as + 1, I_max + 1), np.inf)
        for t in range(self.T_as + 1):
            x = (self.A / math.e) * (self.T_as - t)
            if x <= 0.0:
                log_v = np.zeros(I_max + 1)  # v_q = 1 for all q
            else:
                log_terms = j * math.log(x) - log_fact
                log_v = np.logaddexp.accumulate(log_terms)
            table[t, 1:] = inv_k_tick * (1.0 + (log_v[1:] - log_v[:-1]))
        return table  # row q = 0 stays inf (never used: q <= 0 => no order)

    # -- policy ---------------------------------------------------------------------

    def _clob_action(self) -> ClobAction:
        if self.delta_ticks is None:
            raise RuntimeError("ASBenchmarkAgent.calibrate() must run before acting")
        q = self._q_int()
        if q <= 0:
            return ClobAction(0.0, 0)
        a = self.cfg.actions
        delta = int(
            np.clip(round(self.delta_ticks[self._t_idx(), q]), a.clob_delta_min, a.clob_delta_max)
        )
        volume = float(min(self._env.inventory, self.cfg.clob_flow.V_max))
        return ClobAction(volume, delta)

    # -- persistence (calibration is the only state) ----------------------------------

    def save(self, path: str | Path) -> None:
        if self.delta_ticks is None:
            raise RuntimeError("nothing to save: calibrate() has not run")
        np.savez(Path(path), A=self.A, k=self.k, sigma=self.sigma, delta_ticks=self.delta_ticks)

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path))
        self.A = float(data["A"])
        self.k = float(data["k"])
        self.sigma = float(data["sigma"])
        self.delta_ticks = data["delta_ticks"]


class TWAPBenchmarkAgent(_LiquidationBenchmark):
    """TWAP benchmark (`section:TWAP`): v_t = ceil(q_t / (T - t + 1)) capped
    at the inventory and V_max, quoted at delta = clob_delta_min (= 1 tick,
    ``twap_delta_mode = "min"``); same auction heuristic as AS."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)
        if cfg.benchmark.twap_delta_mode != "min":
            raise NotImplementedError("only twap_delta_mode = 'min' is implemented (config D6)")

    def _clob_action(self) -> ClobAction:
        q = float(self._env.inventory)
        if q <= 0.0:
            return ClobAction(0.0, 0)
        steps_left = max(1, self.T_as - self._t_idx() + 1)
        volume = float(min(math.ceil(q / steps_left), q, self.cfg.clob_flow.V_max))
        return ClobAction(volume, self.cfg.actions.clob_delta_min)
