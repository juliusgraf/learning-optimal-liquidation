"""Avellaneda--Stoikov and physical-time TWAP benchmark policies.

Both submit one capped positive-part auction schedule at the opening and then
abstain.  This benchmark schedule is intentionally outside the learned
two-sided action family and is cleared by the general monotone solver.

Conventions from the revised manuscript:
- "executed CLOB price" is the submitted price S^bullet_t on steps with
  E_t > 0, tracked via ``Transition.info``;
- S_tilde is the average of the mean and maximum executed CLOB prices, with
  H at auction open as the no-execution fallback;
- the benchmark's external one-sided schedule retains
  ``K*(p-S_tilde)_+``, is capped by remaining inventory, and is not projected
  onto the learned agent's indicative-centred policy-template grid;
- shaping is disabled for benchmarks, while genuine cancellation fees remain.

Benchmarks run on the SAME env through the same episode loop as the DQN
(rl/loops.run_episode) — the structural CRN guarantee (ruling D10).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lmm.agents.base import Agent, Transition
from lmm.config import BenchmarkParams, ExperimentConfig
from lmm.env.action_spaces import (
    AuctionAction,
    ClobAction,
    _BenchmarkAuctionAction,
    round_half_up,
)
from lmm.market.clob import OrderBook, sample_mo_volume

__all__ = ["ASBenchmarkAgent", "TWAPBenchmarkAgent"]

_BOOK_EPS = 1e-9  # legacy estimate_K_from_env eps (main.py:1890)


class _LiquidationBenchmark(Agent):
    """Shared scaffolding: env binding, executed-price tracking, and the
    capped auction-open schedule."""

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

    def _auction_action(self) -> AuctionAction | _BenchmarkAuctionAction:
        """Submit ``min(q, K*(p-S_tilde)_+)`` once at the auction open."""
        if self._auction_opened:
            return AuctionAction(0.0, 0, 0)
        self._auction_opened = True
        env = self._env
        q = float(env.inventory)
        if q <= 0.0:
            return AuctionAction(0.0, 0, 0)
        if self._exec_prices:
            s_tilde = 0.5 * (
                max(self._exec_prices) + sum(self._exec_prices) / len(self._exec_prices)
            )
        else:
            s_tilde = float(env.h_cl)
        K = min(self.params.z * q, self.cfg.actions.auction_K_grid_max)
        return _BenchmarkAuctionAction(
            K_a=float(K),
            reference_price=float(s_tilde),
            quantity_cap=float(q),
        )

    def _q_int(self) -> int:
        """AS inventory index ``max(1,ceil(q))`` for positive inventory."""
        q = float(self._env.inventory)
        if math.floor(q) < 1:
            return 0
        return int(np.clip(max(1, math.ceil(q)), 1, self.cfg.grid.I_max))

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
    executions on refreshed exogenous books. Avellaneda--Stoikov call the
    order-size tail exponent ``alpha``; this repository calls it ``gamma_m``
    because ``grid.alpha`` is already the price tick. Thus ``grid.alpha`` is
    not the multiplier in the calibration of k;
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
        tail_exponent = p.gamma_m
        self.A = p.lambda0 / tail_exponent
        self.k = tail_exponent * self._estimate_K_hat(rng_k)
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
        uses one path; ``pooled_paths`` samples ``as_sigma_n_paths`` from the
        training pool (or simulates that many synthetic paths)."""
        g = self.cfg.grid
        n_paths = 1 if self.params.as_sigma_rule == "single_path" else self.params.as_sigma_n_paths
        pooled_returns = env._sample_calibration_log_returns(rng, n_paths)
        # One integer interval is one configured physical clock unit (one
        # minute in every active setting), so sigma is per sqrt(clock unit).
        dt = g.physical_time_per_grid_unit
        return float(np.std(pooled_returns, ddof=1) / math.sqrt(dt))

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
        delta_raw = self._delta_at(float(self._env.t), q)
        delta = int(np.clip(round_half_up(delta_raw), 0, a.clob_delta_max))
        volume = float(
            min(math.floor(float(self._env.inventory)), self.cfg.actions.clob_volume_max)
        )
        return ClobAction(volume, delta)

    def _delta_at(self, t: float, q: int) -> float:
        """Evaluate the gamma->0 quote on physical time, without flooring t."""
        if self.A is None or self.k is None:
            raise RuntimeError("ASBenchmarkAgent.calibrate() must run before acting")
        x = (self.A / math.e) * max(self.T_as - float(t), 0.0)
        j = np.arange(q + 1, dtype=float)
        if x <= 0.0:
            log_v = np.zeros(q + 1)
        else:
            log_terms = j * math.log(x) - np.asarray(
                [math.lgamma(jj + 1.0) for jj in j]
            )
            log_v = np.logaddexp.accumulate(log_terms)
        return float(
            (1.0 + log_v[q] - log_v[q - 1])
            / (self.k * self.cfg.grid.alpha)
        )

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
    at the inventory and V_max, quoted at the manuscript best ask delta=1
    (the config retains the legacy label ``twap_delta_mode = "min"``); same
    auction heuristic as AS."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)
        if cfg.benchmark.twap_delta_mode != "min":
            raise NotImplementedError("only twap_delta_mode = 'min' is implemented (config D6)")

    def _clob_action(self) -> ClobAction:
        q = float(self._env.inventory)
        if q <= 0.0:
            return ClobAction(0.0, 0)
        floor_q = math.floor(q)
        if floor_q < 1:
            return ClobAction(0.0, 0)
        time_left = max(float(self.T_as) - float(self._env.t) + 1.0, 1.0)
        volume = float(
            min(
                self.cfg.actions.clob_volume_max,
                floor_q,
                math.ceil(q / time_left),
            )
        )
        return ClobAction(volume, 1)
