"""Continuous-limit-order-book mechanics for the revised simulator.

The exogenous book has levels ``j = 1, ..., L_max`` at
``k_mid + j`` (ask) and ``k_mid - j`` (bid).  The strategic liquidator may
quote at any offset ``delta = 0, ..., L_max`` and has priority over exogenous
ask volume at the same price.  A strategic order belongs to one decision
interval only; :meth:`OrderBook.clear_agent_order` expires any remainder.

``BookSnapshot`` deliberately separates the exogenous book from the optional
strategic remainder.  Algorithm 1 and the auction carry-over consume
``exogenous_volume_by_tick`` and therefore cannot accidentally include the
strategic order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import ClobFlowParams, GridParams

__all__ = [
    "BookSnapshot",
    "OrderBook",
    "executed_volume",
    "sample_mo_volume",
]


def sample_mo_volume(
    rng: np.random.Generator,
    v_m: float,
    gamma_m: float,
    V_max: float,
) -> float:
    """Draw ``min(Pareto(v_m, gamma_m), V_max)`` with one uniform draw."""
    u = float(rng.random())
    vol = v_m / ((1.0 - u) ** (1.0 / gamma_m))
    return float(min(vol, V_max))


_DEPTH_EPS = 1e-6
_AGENT_EPS = 1e-9


@dataclass(frozen=True)
class BookSnapshot:
    """A finite CLOB snapshot anchored at ``k_mid``.

    ``ask_volumes[j-1]`` is the exogenous volume at tick ``k_mid + j`` and
    ``bid_volumes[j-1]`` is the volume at ``k_mid - j``.  The optional agent
    fields are retained for execution diagnostics and backwards-compatible
    callers; the exogenous projection used by Algorithm 1 ignores them.
    """

    k_mid: int
    ask_volumes: np.ndarray
    bid_volumes: np.ndarray
    agent_level: int | None = None
    agent_remaining: float = 0.0

    def __post_init__(self) -> None:
        ask = np.asarray(self.ask_volumes, dtype=float).copy()
        bid = np.asarray(self.bid_volumes, dtype=float).copy()
        if ask.ndim != 1 or bid.ndim != 1:
            raise ValueError("book sides must be one-dimensional")
        if np.any(ask < 0.0) or np.any(bid < 0.0):
            raise ValueError("book volumes must be nonnegative")
        ask.setflags(write=False)
        bid.setflags(write=False)
        object.__setattr__(self, "ask_volumes", ask)
        object.__setattr__(self, "bid_volumes", bid)

    def exogenous_volume_by_tick(self, eps: float = 0.0) -> dict[int, float]:
        """Aggregate the exogenous multiset into ``tick -> real volume``.

        The strategic remainder is intentionally excluded.  Returning a map
        preserves multiplicity after aggregation and makes absent levels
        implicitly equal to zero in Algorithm 1's running moments.
        """
        out: dict[int, float] = {}
        for idx, volume in enumerate(self.ask_volumes):
            if volume > eps:
                tick = self.k_mid + idx + 1
                out[tick] = out.get(tick, 0.0) + float(volume)
        for idx, volume in enumerate(self.bid_volumes):
            if volume > eps:
                tick = self.k_mid - idx - 1
                out[tick] = out.get(tick, 0.0) + float(volume)
        return out

    def with_no_agent(self) -> "BookSnapshot":
        """Return the same exogenous snapshot without strategic diagnostics."""
        return BookSnapshot(self.k_mid, self.ask_volumes, self.bid_volumes)


class OrderBook:
    """Fresh real-valued exogenous depth plus one one-interval strategic ask."""

    def __init__(self, params: ClobFlowParams, grid: GridParams) -> None:
        self.params = params
        self.grid = grid
        self.k_mid: int = 0
        self.ask_volumes = np.zeros(params.Lc, dtype=float)
        self.bid_volumes = np.zeros(params.Lc, dtype=float)
        self._agent_level: int | None = None
        self._agent_volume: float = 0.0

    def refresh(self, rng: np.random.Generator) -> None:
        """Draw a fresh two-sided snapshot (ask draw first, then bid)."""
        p = self.params
        ask_top = float(p.V_inf * rng.beta(p.beta_a, p.beta_b))
        bid_top = float(p.V_inf * rng.beta(p.beta_a, p.beta_b))
        self.load_snapshot(self.k_mid, ask_top, bid_top)

    def load_snapshot(self, k_mid: int, ask_top: float, bid_top: float) -> None:
        """Install a pre-sampled exogenous snapshot for a decision time."""
        if ask_top < 0.0 or bid_top < 0.0:
            raise ValueError("top-of-book volumes must be nonnegative")
        decay = self.params.depth_decay ** np.arange(self.params.Lc, dtype=float)
        self.k_mid = int(k_mid)
        self.ask_volumes = float(ask_top) * decay
        self.bid_volumes = float(bid_top) * decay
        # A fresh decision snapshot must never inherit a strategic remainder.
        self.clear_agent_order()

    def depth(self, side: int) -> int:
        """Number of populated exogenous levels on ``side`` (ask ``+1``)."""
        vols = self.ask_volumes if side > 0 else self.bid_volumes
        empty = np.flatnonzero(vols <= _DEPTH_EPS)
        return int(empty[0]) if empty.size else self.params.Lc

    def place_agent_order(self, volume: float, level: int) -> None:
        """Place the strategic ask at ``delta=level`` for the current interval."""
        if volume < 0.0:
            raise ValueError(f"agent volume must be nonnegative, got {volume}")
        if not 0 <= int(level) <= self.params.Lc:
            raise ValueError(
                f"agent level must lie in [0, {self.params.Lc}], got {level}"
            )
        if volume > 0.0:
            self._agent_level = int(level)
            self._agent_volume = float(volume)
        else:
            self.clear_agent_order()

    def clear_agent_order(self) -> None:
        """Expire the strategic order and any unfilled remainder."""
        self._agent_level = None
        self._agent_volume = 0.0

    @property
    def agent_remaining(self) -> float:
        return self._agent_volume if self._agent_level is not None else 0.0

    @property
    def agent_level(self) -> int | None:
        return self._agent_level

    def process_buy_market_order(self, volume: float) -> float:
        """Consume asks low-to-high and return the strategic fill.

        At strategic offset ``delta``, exogenous levels ``1,...,delta-1`` are
        cheaper.  The agent executes next, before exogenous level ``delta``.
        For ``delta=0`` the agent is ahead of every exogenous ask level.
        """
        remain = max(float(volume), 0.0)
        if remain == 0.0:
            return 0.0

        if self._agent_level is None:
            self._consume_ask_range(remain, 0)
            return 0.0

        delta = self._agent_level
        ahead_count = max(delta - 1, 0)
        for idx in range(min(ahead_count, self.params.Lc)):
            if remain <= 0.0:
                break
            take = min(remain, float(self.ask_volumes[idx]))
            self.ask_volumes[idx] -= take
            remain -= take

        executed = min(remain, self._agent_volume)
        if executed > 0.0:
            self._agent_volume -= executed
            remain -= executed
            if self._agent_volume <= _AGENT_EPS:
                self._agent_volume = 0.0
                self._agent_level = None

        # At delta >= 1, exogenous array index delta-1 is the same price;
        # at delta == 0, all exogenous asks are strictly more expensive.
        start = max(delta - 1, 0)
        if remain > 0.0:
            self._consume_ask_range(remain, start)
        return float(executed)

    def _consume_ask_range(self, volume: float, start: int) -> float:
        """Consume exogenous asks from ``start`` and return unfilled volume."""
        remain = float(volume)
        for idx in range(max(0, start), self.params.Lc):
            if remain <= 0.0:
                break
            take = min(remain, float(self.ask_volumes[idx]))
            self.ask_volumes[idx] -= take
            remain -= take
        return remain

    def process_sell_market_order(self, volume: float) -> None:
        """Consume exogenous bids high-to-low (array order 1,2,...)."""
        remain = max(float(volume), 0.0)
        for idx in range(self.params.Lc):
            if remain <= 0.0:
                break
            take = min(remain, float(self.bid_volumes[idx]))
            self.bid_volumes[idx] -= take
            remain -= take

    def snapshot(self, include_agent: bool = True) -> BookSnapshot:
        """Return the current residual book; arrays are copied and immutable."""
        return BookSnapshot(
            k_mid=self.k_mid,
            ask_volumes=self.ask_volumes,
            bid_volumes=self.bid_volumes,
            agent_level=self._agent_level if include_agent else None,
            agent_remaining=self.agent_remaining if include_agent else 0.0,
        )

    def exogenous_snapshot(self) -> BookSnapshot:
        """Snapshot suitable for Algorithm 1 and final carry-over."""
        return self.snapshot(include_agent=False)

    def residual_exogenous_snapshot(self) -> BookSnapshot:
        """Alias emphasizing the post-final-interval carry-over boundary."""
        return self.exogenous_snapshot()


def executed_volume(
    agent_volume: float,
    agent_delta: int,
    ask_volumes: np.ndarray,
    incoming_buy_volumes: np.ndarray,
) -> float:
    """Closed-form strategic fill over one interval.

    ``ask_volumes[0]`` is level one, hence only the first ``delta-1`` levels
    are strictly cheaper than a strategic quote at offset ``delta``.
    """
    total_buy = float(np.sum(np.asarray(incoming_buy_volumes, dtype=float)))
    ahead_count = max(int(agent_delta) - 1, 0)
    ahead = float(np.sum(np.asarray(ask_volumes, dtype=float)[:ahead_count]))
    return max(0.0, min(float(agent_volume), total_buy - ahead))
