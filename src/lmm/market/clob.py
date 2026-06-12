"""CLOB book state, refresh, and market-order processing (paper `sec:lob`).

Sign convention (BINDING, ruling D3 / CLAUDE.md): zeta = + is the ASK side
for limit orders and the BUY side for market orders; buy market orders
consume ask liquidity. The agent sells on the ask side; delta_t >= 0.

Level/tick geometry: exogenous level j (0 <= j < Lc) sits at tick
k_mid + j on the ask side and k_mid - j on the bid side (level 0 at the mid
tick on BOTH sides — legacy convention, AUDIT A.2), with k_mid =
floor(S^mid/alpha) (one rounding convention everywhere, AUDIT N3).

Agent order semantics (AUDIT N6 — no silent clamping): the agent's level IS
the action's delta, possibly == Lc (one tick past the deepest refreshed
exogenous level). A buy market order consumes exogenous levels strictly
cheaper than the agent's tick first, then fills the AGENT WITH PRIORITY at
her own level, then the exogenous volume at her level, then deeper levels
(legacy `process_buy_order`, main.py:462-502 — sequential processing within
a step is equivalent to the paper's aggregate E_t formula, AUDIT A.2;
`executed_volume` below is that closed form, asserted equal in tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import ClobFlowParams, GridParams

__all__ = ["BookSnapshot", "OrderBook", "executed_volume", "sample_mo_volume"]


def sample_mo_volume(rng: np.random.Generator, v_m: float, gamma_m: float, V_max: float) -> float:
    """Market-order volume min(Pareto(v_m, gamma_m), V) via the inverse CDF
    (Algorithm 2; legacy `_sample_mo_volume`, main.py:193-196). Consumes
    exactly one uniform. Used for CLOB market orders AND auction takers."""
    u = rng.random()
    vol = v_m / ((1.0 - u) ** (1.0 / gamma_m))
    return float(min(vol, V_max))

# Volume below which a book level counts as empty (legacy depth rule 1e-6,
# agent-residual rule 1e-9; main.py:487, 537-538).
_DEPTH_EPS = 1e-6
_AGENT_EPS = 1e-9


@dataclass(frozen=True)
class BookSnapshot:
    """Post-flow standing book at the end of a CLOB step (Algorithm 1 input, D2).

    Includes the agent's unexecuted remainder at her level; taken BEFORE the
    next-step book refresh.
    """

    k_mid: int  # mid tick index (floor convention, AUDIT N3 resolution)
    ask_volumes: np.ndarray  # shape (Lc,), level j at tick k_mid + j
    bid_volumes: np.ndarray  # shape (Lc,), level j at tick k_mid - j
    agent_level: int | None  # agent's level offset on the ask side, if any
    agent_remaining: float  # unexecuted remainder of the agent's time-t order


class OrderBook:
    """Exogenous CLOB with the agent's single ask-side limit order.

    Top-of-book V^{zeta,1} ~ V_inf * Beta(beta_a, beta_b) with geometric
    depth decay rho, refreshed each CLOB step (Algorithm 2 lines 5-6;
    legacy `_refresh_order_book`, main.py:183-191).
    """

    def __init__(self, params: ClobFlowParams, grid: GridParams) -> None:
        self.params = params
        self.grid = grid
        self.k_mid: int = 0
        self.ask_volumes = np.zeros(params.Lc)
        self.bid_volumes = np.zeros(params.Lc)
        self._agent_level: int | None = None
        self._agent_volume: float = 0.0

    def refresh(self, rng: np.random.Generator) -> None:
        """Redraw both sides' volumes (legacy `_refresh_order_book`).

        Consumes exactly two Beta draws (ask first, then bid) — stream-stable.
        The tick anchor ``k_mid`` is NOT touched here: the refreshed levels
        belong to the NEXT decision time's mid, so the env sets ``k_mid`` =
        floor(S^mid_t / alpha) when the next action is processed (legacy
        recomputed it from the new mid at action time, main.py:447).
        """
        p = self.params
        v1a = float(p.V_inf * rng.beta(p.beta_a, p.beta_b))
        v1b = float(p.V_inf * rng.beta(p.beta_a, p.beta_b))
        decay = p.depth_decay ** np.arange(p.Lc)
        self.ask_volumes = v1a * decay
        self.bid_volumes = v1b * decay

    def depth(self, side: int) -> int:
        """Book depth L^zeta_t = inf{j+1 : V^{zeta,j} <= eps} (capped at Lc).

        ``side`` = +1 for the ask, -1 for the bid (paper zeta). Matches the
        legacy rule (main.py:537-538): depth = first level with volume
        <= 1e-6, else Lc.
        """
        vols = self.ask_volumes if side > 0 else self.bid_volumes
        empty = np.flatnonzero(vols <= _DEPTH_EPS)
        return int(empty[0] + 1) if empty.size else self.params.Lc

    # -- agent order ----------------------------------------------------------

    def place_agent_order(self, volume: float, level: int) -> None:
        """Record the agent's time-t ask-side limit order at ``level`` >= 0.

        ``level`` may equal or exceed Lc (a quote past the refreshed book,
        AUDIT N6); volume 0 == no order.
        """
        if level < 0:
            raise ValueError(f"agent level must be >= 0, got {level}")
        if volume > 0.0:
            self._agent_level = int(level)
            self._agent_volume = float(volume)
        else:
            self._agent_level = None
            self._agent_volume = 0.0

    def clear_agent_order(self) -> None:
        """Drop the agent's order (end of step; unfilled remainder lapses)."""
        self._agent_level = None
        self._agent_volume = 0.0

    @property
    def agent_remaining(self) -> float:
        return self._agent_volume if self._agent_level is not None else 0.0

    # -- market-order processing ----------------------------------------------

    def process_buy_market_order(self, volume: float) -> float:
        """Consume ask liquidity; returns the agent volume executed.

        Implements the walk of legacy `process_buy_order` (main.py:462-502):
        exogenous levels j < agent level first, then the agent (EXECUTION
        PRIORITY at her own level, Sec. 2.1.1), then the exogenous volume at
        her level, then deeper levels. With no agent order the whole book is
        walked from level 0.
        """
        remain = float(volume)
        lc = self.params.Lc
        j_agent = self._agent_level if self._agent_level is not None else 0

        for j in range(min(j_agent, lc)):
            if remain <= 0.0:
                break
            lvl = self.ask_volumes[j]
            if lvl <= 0.0:
                continue
            take = min(remain, lvl)
            self.ask_volumes[j] -= take
            remain -= take

        executed = 0.0
        if remain > 0.0 and self._agent_level is not None:
            take = min(remain, self._agent_volume)
            if take > 0.0:
                executed = take
                self._agent_volume -= take
                remain -= take
                if self._agent_volume <= _AGENT_EPS:
                    self._agent_level = None
                    self._agent_volume = 0.0

        if remain > 0.0:
            if j_agent < lc:
                take = min(remain, self.ask_volumes[j_agent])
                self.ask_volumes[j_agent] -= take
                remain -= take
            j = j_agent + 1
            while remain > 0.0 and j < lc:
                take = min(remain, self.ask_volumes[j])
                self.ask_volumes[j] -= take
                remain -= take
                j += 1
        # Residual beyond the book evaporates (Assumption ass:small_investors).
        return executed

    def process_sell_market_order(self, volume: float) -> None:
        """Consume bid liquidity (no agent interaction; the agent only sells)."""
        remain = float(volume)
        for j in range(self.params.Lc):
            if remain <= 0.0:
                break
            lvl = self.bid_volumes[j]
            if lvl <= 0.0:
                continue
            take = min(remain, lvl)
            self.bid_volumes[j] -= take
            remain -= take

    def snapshot(self) -> BookSnapshot:
        """Post-flow standing book for Algorithm 1 (end-of-step semantics, D2)."""
        return BookSnapshot(
            k_mid=self.k_mid,
            ask_volumes=self.ask_volumes.copy(),
            bid_volumes=self.bid_volumes.copy(),
            agent_level=self._agent_level,
            agent_remaining=self.agent_remaining,
        )


def executed_volume(
    agent_volume: float,
    agent_delta: int,
    ask_volumes: np.ndarray,
    incoming_buy_volumes: np.ndarray,
) -> float:
    """Aggregate E_t formula of the paper (exposed separately for unit tests).

    E_t = max(0, min(v_t, sum_i nu^{+,i} - sum_{j < delta} V^{+,j}))

    over one step's buy market orders and the START-of-step exogenous ask
    book. Sequential `OrderBook.process_buy_market_order` calls within a step
    realize exactly this quantity because exogenous volume priced below the
    agent is consumed before her on every order (AUDIT A.2); asserted in
    tests/test_sign_conventions.py.
    """
    total_buy = float(np.sum(incoming_buy_volumes))
    ahead = float(np.sum(np.asarray(ask_volumes)[: int(agent_delta)]))
    return max(0.0, min(float(agent_volume), total_buy - ahead))
