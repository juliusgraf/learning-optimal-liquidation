"""Auction phase: agent order ledger and exogenous flow (paper `sec:auction`,
Algorithm 2 lines 12-17; rulings D3 (theta), D4 (cancel), D6-D7 (params)).

Naming follows the PAPER sign convention (ruling D3): zeta = + is the BUY
side — buy takers (nu^+) RAISE the clearing price, N^+ counts buying market
orders. (The legacy code's `N_plus` counted sells; rename only, the clearing
arithmetic was already correct — AUDIT A.4/A.8.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import AuctionFlowParams, ClobFlowParams, GridParams
from lmm.market.clob import sample_mo_volume

__all__ = ["AgentOrderLedger", "AuctionEvents", "ExogenousAuctionFlow"]


class AgentOrderLedger:
    """Agent auction orders (K^a_{t_s}, S^a_{t_s}) with liveness tracking.

    Implements the CORRECTED theta recursion (ruling D3 / CLAUDE.md):
    theta_{t_n} = theta_{t_{n+1}} = 0 and
    theta_{t_j} = max(theta_{t_{j-1}}, c_{t_{j-1}} * sum_{k=1}^{j-n-2} e_k),
    i.e. a cancel-all decided at t_j kills all orders submitted strictly
    before t_j and is reflected in theta_{t_{j+1}} (predictable: theta_{t_j}
    is part of the state X^9_{t_j}). The internal live/dead flags are the
    direct realization; ``theta()`` materializes the paper vector. The env
    sequences ``apply_cancel_all(t)`` AFTER the time-t estimate has been
    cached, so the deferral to theta_{t+1} holds by construction; the final
    order at t_m is structurally uncancellable (cutoff s < t).

    Component i (0-based) holds the order decided at t_s with s = tau_op + i
    (paper component index s - n = i + 1), so vectors live in R^{m-n}.
    Convention: submitting K^a = 0 is identified with abstaining (no entry).
    X^16/X^17 use the predictable indexing — at decision time t they contain
    entries for orders submitted up to t-1 only (``history_at(t)``).
    """

    def __init__(self, grid: GridParams) -> None:
        self.grid = grid
        self.n_slots = grid.tau_cl - grid.tau_op  # m - n
        self.K = np.zeros(self.n_slots)
        self.S = np.zeros(self.n_slots)
        self.submitted = np.zeros(self.n_slots, dtype=bool)
        self.live = np.zeros(self.n_slots, dtype=bool)
        # Ruling D16: one-sided hockey-stick orders K (p - S)_+ (benchmarks
        # only liquidate); at most ONE may be live at any time.
        self.one_sided = np.zeros(self.n_slots, dtype=bool)

    def _slot(self, t: int) -> int:
        i = int(t) - self.grid.tau_op
        if not 0 <= i < self.n_slots:
            raise ValueError(f"auction time {t} outside [{self.grid.tau_op}, {self.grid.tau_cl - 1}]")
        return i

    def reset(self) -> None:
        """Clear all entries at auction open (tau_op)."""
        self.K[:] = 0.0
        self.S[:] = 0.0
        self.submitted[:] = False
        self.live[:] = False
        self.one_sided[:] = False

    def submit(self, t: int, K_a: float, S_a: float, one_sided: bool = False) -> None:
        """Record the order decided at auction time t (no-op if K_a == 0).

        ``one_sided=True`` records the benchmark hockey-stick K^a (p - S^a)_+
        (ruling D16); the clearing solver handles exactly one such order, so
        a second live one-sided order raises.
        """
        if K_a < 0.0:
            raise ValueError(f"K^a must be >= 0, got {K_a}")
        if K_a == 0.0:
            return  # abstain convention
        if one_sided and bool(np.any(self.live & self.one_sided)):
            raise ValueError("at most one live one-sided order is supported (ruling D16)")
        i = self._slot(t)
        self.K[i] = float(K_a)
        self.S[i] = float(S_a)
        self.submitted[i] = True
        self.live[i] = True
        self.one_sided[i] = one_sided

    def apply_cancel_all(self, t: int) -> None:
        """Apply c_t = 1: deactivate all orders submitted at times < t.

        Takes effect in theta_{t+1} / the time-(t+1) Eq. (2) estimate (D1) —
        the caller sequences this AFTER the time-t estimate is read.
        """
        i = self._slot(t)
        self.live[:i] = False

    def cancel_admissible(self) -> bool:
        """C(x) = max_i (1 - x^{9,(i)}) 1{x^{17,(i)} > 0} > 0 (CLAUDE.md):
        a cancel-all is admissible iff a live prior order with K^a > 0 exists.
        Agrees with C evaluated on ``paper_state()`` (asserted in tests):
        K = 0 slots are never submitted (abstain convention), so the live
        flags carry exactly the (1 - theta) 1{K > 0} information.
        """
        return bool(np.any(self.live & (self.K > 0.0)))

    def theta(self) -> np.ndarray:
        """Paper theta vector in R^{m-n} (component s-n for the order at t_s)."""
        return (self.submitted & ~self.live).astype(float)

    def live_orders(self) -> tuple[np.ndarray, np.ndarray]:
        """(K^a, S^a) arrays over currently live orders (for clearing solves)."""
        return self.K[self.live].copy(), self.S[self.live].copy()

    def live_orders_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None]:
        """(K_linear, S_linear, hockey) over live orders: the linear orders
        plus the single live one-sided order as ``(K, S)`` (None if absent;
        ruling D16). Input to ``solve_clearing``."""
        lin = self.live & ~self.one_sided
        hs = self.live & self.one_sided
        hockey: tuple[float, float] | None = None
        if np.any(hs):
            i = int(np.flatnonzero(hs)[0])
            hockey = (float(self.K[i]), float(self.S[i]))
        return self.K[lin].copy(), self.S[lin].copy(), hockey

    def history_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(X^16, X^17) = (S^a(t), K^a(t)): zero-padded vectors in R^{m-n}
        with entries for orders submitted at times s <= t-1 only
        (predictable indexing; CLAUDE.md PREREQUISITE)."""
        cut = int(np.clip(np.floor(t) - self.grid.tau_op, 0, self.n_slots))
        S = np.zeros(self.n_slots)
        K = np.zeros(self.n_slots)
        S[:cut] = self.S[:cut] * self.submitted[:cut]
        K[:cut] = self.K[:cut] * self.submitted[:cut]
        return S, K


@dataclass(frozen=True)
class AuctionEvents:
    """Exogenous events realized in one auction step (for the info dict)."""

    new_mm: bool
    mm_cancelled: bool
    new_buy_taker: bool
    new_sell_taker: bool
    buy_taker_cancelled: bool
    sell_taker_cancelled: bool


class ExogenousAuctionFlow:
    """Exogenous auction supply/demand per Algorithm 2 (rulings D6-D7).

    Per step: new exogenous MM with prob p1, K^i ~ U(K_min, K_max),
    S^i = alpha * (floor(S^mid_{tau_op}/alpha) + U{-band..band}) — the mid is
    FROZEN at tau_op and the quote is tick-snapped (AUDIT N4 resolution;
    legacy added the offset to the raw float mid); MM cancellation with prob
    p2; new market taker per side with prob p3 (independent); taker
    cancellation with prob p4 per side.

    Ruling D7: p4 = 0.05 as a SINGLE Bernoulli draw per side. Legacy realized
    the same distribution as Bernoulli(0.1) gated by an extra fair coin
    (main.py:616-625); collapsing the two draws changes the RNG stream, so
    seeded trajectories intentionally differ from legacy (pinned by the
    characterization tests; attributed in audit/BEHAVIOR_CHANGES.md).

    Fixed per-step draw order (policy-independent, CRN; ruling D10):
    p1-accept [K, offset] | p2-accept [index] | buy-p3 [volume] |
    sell-p3 [volume] | buy-p4 [index] | sell-p4 [index] — acceptance
    uniforms always drawn; bracketed draws conditional on exogenous-only
    state. The buy side (paper zeta = +) is drawn FIRST (legacy drew its
    "sell" side first; pure stream relabeling, D3).

    MM arrivals are suppressed when the MM count reaches La (legacy cap kept
    as explicit config, AUDIT N8). Cancelled taker volumes are set to 0 with
    N^zeta unchanged (paper: "volume can be set to zero").
    """

    def __init__(
        self,
        params: AuctionFlowParams,
        grid: GridParams,
        clob_params: ClobFlowParams,
    ) -> None:
        self.params = params
        self.grid = grid
        # Taker volumes follow the CLOB market-order law min(Pareto(v_m,
        # gamma_m), V) (Algorithm 2; legacy main.py:600-615 reuses it inline).
        self.clob_params = clob_params
        self.frozen_mid: float = grid.S0
        self._k_mid_frozen: int = 0
        self.supply_K: list[float] = []
        self.supply_S: list[float] = []
        self.n_buy: int = 0
        self.n_sell: int = 0
        self.buy_volumes = np.zeros(params.L_max)
        self.sell_volumes = np.zeros(params.L_max)

    def reset(self, frozen_mid: float) -> None:
        """Clear ledgers at auction open; store the frozen mid for quote draws."""
        self.frozen_mid = float(frozen_mid)
        self._k_mid_frozen = int(np.floor(frozen_mid / self.grid.alpha))
        self.supply_K = []
        self.supply_S = []
        self.n_buy = 0
        self.n_sell = 0
        self.buy_volumes = np.zeros(self.params.L_max)
        self.sell_volumes = np.zeros(self.params.L_max)

    def step(self, rng: np.random.Generator) -> AuctionEvents:
        """Sample one auction step's exogenous events; update internal ledgers."""
        p = self.params
        cp = self.clob_params
        band = p.price_band_ticks

        new_mm = rng.random() < p.p1 and len(self.supply_K) < p.La
        if new_mm:
            K_new = float(rng.uniform(p.K_min, p.K_max))
            offset = int(rng.integers(-band, band + 1))
            S_new = self.grid.alpha * (self._k_mid_frozen + offset)
            self.supply_K.append(K_new)
            self.supply_S.append(S_new)

        mm_cancelled = rng.random() < p.p2 and len(self.supply_K) > 0
        if mm_cancelled:
            idx = int(rng.integers(0, len(self.supply_K)))
            self.supply_K.pop(idx)
            self.supply_S.pop(idx)

        new_buy = rng.random() < p.p3
        if new_buy:
            vol = sample_mo_volume(rng, cp.v_m, cp.gamma_m, cp.V_max)
            if self.n_buy < p.L_max:
                self.buy_volumes[self.n_buy] = vol
            self.n_buy = min(self.n_buy + 1, p.L_max)

        new_sell = rng.random() < p.p3
        if new_sell:
            vol = sample_mo_volume(rng, cp.v_m, cp.gamma_m, cp.V_max)
            if self.n_sell < p.L_max:
                self.sell_volumes[self.n_sell] = vol
            self.n_sell = min(self.n_sell + 1, p.L_max)

        # Ruling D7: single Bernoulli(p4 = 0.05) per side (legacy: outer
        # Bernoulli(0.1) gated by a fair coin — identical law, two draws).
        buy_cancelled = rng.random() < p.p4 and self.n_buy > 0
        if buy_cancelled:
            idx = int(rng.integers(0, self.n_buy))
            self.buy_volumes[idx] = 0.0

        sell_cancelled = rng.random() < p.p4 and self.n_sell > 0
        if sell_cancelled:
            idx = int(rng.integers(0, self.n_sell))
            self.sell_volumes[idx] = 0.0

        return AuctionEvents(
            new_mm=new_mm,
            mm_cancelled=mm_cancelled,
            new_buy_taker=new_buy,
            new_sell_taker=new_sell,
            buy_taker_cancelled=buy_cancelled,
            sell_taker_cancelled=sell_cancelled,
        )

    def supply_curves(self) -> tuple[np.ndarray, np.ndarray]:
        """(K^i, S^i) over active exogenous MMs (M_t entries)."""
        return np.asarray(self.supply_K, dtype=float), np.asarray(self.supply_S, dtype=float)

    def net_market_volume(self) -> float:
        """sum_i nu^{+,i} - sum_i nu^{-,i} (paper convention: buys positive)."""
        return float(
            np.sum(self.buy_volumes[: self.n_buy]) - np.sum(self.sell_volumes[: self.n_sell])
        )

    @property
    def n_mm(self) -> int:
        """M_t: number of active exogenous auction market makers."""
        return len(self.supply_K)

    # -- test-only injection hooks (timing/sign-convention tests) ------------

    def inject_market_maker(self, K: float, S: float) -> None:
        """TEST ONLY: append an exogenous MM outside the sampled flow."""
        self.supply_K.append(float(K))
        self.supply_S.append(float(S))

    def inject_taker(self, side: int, volume: float) -> None:
        """TEST ONLY: append a market taker (side = +1 buy / -1 sell)."""
        if side > 0:
            if self.n_buy >= self.params.L_max:
                raise ValueError("buy taker ledger full")
            self.buy_volumes[self.n_buy] = float(volume)
            self.n_buy += 1
        else:
            if self.n_sell >= self.params.L_max:
                raise ValueError("sell taker ledger full")
            self.sell_volumes[self.n_sell] = float(volume)
            self.n_sell += 1
