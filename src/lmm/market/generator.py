"""Episode-level exogenous market realization.

The revised simulator samples the two independent CLOB Poisson paths through
``tau_op`` *before* the controlled episode starts, constructs the complete
strict decision grid, and assigns every arrival to exactly one interval
``(t_i, t_{i+1}]``.  Volume marks and fresh CLOB snapshot tops are also
pre-sampled, so policy choices cannot perturb the exogenous random stream.

The full :class:`EpisodeRealization` is simulator-internal.  Environment-facing
methods return only the current snapshot or current interval; future grid
points and order flow need never enter a policy feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmm.config import AuctionFlowParams, ClobFlowParams, GridParams
from lmm.market.auction import AuctionEvents, ExogenousAuctionFlow
from lmm.market.clob import BookSnapshot, OrderBook, sample_mo_volume

__all__ = [
    "EpisodeGrid",
    "EpisodeRealization",
    "ClobStepFlow",
    "build_episode_grid",
    "sample_episode_realization",
    "MarketGenerator",
]


def _readonly(values, dtype=float) -> np.ndarray:
    out = np.asarray(values, dtype=dtype).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class EpisodeGrid:
    """Complete realized decision grid for one episode.

    ``clob_times`` contains ``t_0,...,t_n`` with final value ``tau_op-1``;
    ``auction_times`` contains exactly ``h=tau_cl-tau_op`` action times; and
    ``terminal_time`` is ``tau_cl``.  The random CLOB decision index is
    ``n`` and the final action index is ``m=n+h``.
    """

    clob_times: np.ndarray
    auction_times: np.ndarray
    terminal_time: float

    def __post_init__(self) -> None:
        clob = _readonly(self.clob_times)
        auction = _readonly(self.auction_times)
        all_times = np.concatenate((clob, auction, np.array([self.terminal_time])))
        if len(clob) == 0 or clob[0] != 0.0:
            raise ValueError("the realized grid must start at t_0=0")
        if np.any(np.diff(all_times) <= 0.0):
            raise ValueError("the complete realized grid must be strictly increasing")
        object.__setattr__(self, "clob_times", clob)
        object.__setattr__(self, "auction_times", auction)

    @property
    def n(self) -> int:
        return len(self.clob_times) - 1

    @property
    def h(self) -> int:
        return len(self.auction_times)

    @property
    def m(self) -> int:
        return self.n + self.h

    @property
    def decision_times(self) -> np.ndarray:
        return _readonly(np.concatenate((self.clob_times, self.auction_times)))

    @property
    def all_times(self) -> np.ndarray:
        return _readonly(
            np.concatenate((self.clob_times, self.auction_times, [self.terminal_time]))
        )


@dataclass(frozen=True)
class ClobStepFlow:
    """The exogenous arrivals and strategic execution in one CLOB interval."""

    decision_index: int
    t: float
    t_next: float
    buy_volumes: np.ndarray
    sell_volumes: np.ndarray
    n_buy: int
    n_sell: int
    executed_agent: float
    residual_exogenous: BookSnapshot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "buy_volumes", _readonly(self.buy_volumes))
        object.__setattr__(self, "sell_volumes", _readonly(self.sell_volumes))


@dataclass(frozen=True)
class EpisodeRealization:
    """Policy-independent exogenous tape for one episode (internal API)."""

    grid: EpisodeGrid
    buy_arrival_times: np.ndarray
    sell_arrival_times: np.ndarray
    buy_volumes: np.ndarray
    sell_volumes: np.ndarray
    buy_bounds: np.ndarray
    sell_bounds: np.ndarray
    ask_tops: np.ndarray
    bid_tops: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "buy_arrival_times",
            "sell_arrival_times",
            "buy_volumes",
            "sell_volumes",
            "ask_tops",
            "bid_tops",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "buy_bounds", _readonly(self.buy_bounds, dtype=int))
        object.__setattr__(self, "sell_bounds", _readonly(self.sell_bounds, dtype=int))
        n_decisions = len(self.grid.clob_times)
        if len(self.ask_tops) != n_decisions or len(self.bid_tops) != n_decisions:
            raise ValueError("one fresh exogenous snapshot is required per CLOB decision")
        if len(self.buy_bounds) != n_decisions + 1 or len(self.sell_bounds) != n_decisions + 1:
            raise ValueError("interval-bound arrays must have n_clob_decisions+1 entries")

    def interval_volumes(self, decision_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return only arrivals in the current ``(t_i,t_{i+1}]`` interval."""
        i = int(decision_index)
        if not 0 <= i < len(self.grid.clob_times):
            raise IndexError(f"CLOB decision index {i} out of range")
        b0, b1 = int(self.buy_bounds[i]), int(self.buy_bounds[i + 1])
        s0, s1 = int(self.sell_bounds[i]), int(self.sell_bounds[i + 1])
        return self.buy_volumes[b0:b1].copy(), self.sell_volumes[s0:s1].copy()

    def interval_times(self, decision_index: int) -> tuple[float, float]:
        i = int(decision_index)
        t = float(self.grid.clob_times[i])
        if i == self.grid.n:
            return t, float(self.grid.auction_times[0])
        return t, float(self.grid.clob_times[i + 1])


def _sample_poisson_times(
    rng: np.random.Generator,
    rate: float,
    horizon: float,
) -> np.ndarray:
    if rate <= 0.0:
        raise ValueError(f"Poisson rate must be positive, got {rate}")
    now = 0.0
    arrivals: list[float] = []
    while True:
        now += float(rng.exponential(scale=1.0 / rate))
        if now > horizon:
            break
        arrivals.append(now)
    return _readonly(arrivals)


def build_episode_grid(
    buy_arrival_times: np.ndarray,
    sell_arrival_times: np.ndarray,
    tau_op: int,
    tau_cl: int,
) -> EpisodeGrid:
    """Apply Definition ``def:timing`` to two realized counting processes."""
    if tau_op < 1 or tau_cl <= tau_op:
        raise ValueError("require 1 <= tau_op < tau_cl")
    buy = np.asarray(buy_arrival_times, dtype=float)
    sell = np.asarray(sell_arrival_times, dtype=float)
    if np.any(np.diff(buy) <= 0.0) or np.any(np.diff(sell) <= 0.0):
        raise ValueError("arrival paths must be strictly increasing")

    cap = float(tau_op - 1)
    times = [0.0]
    t = 0.0
    while t < cap:
        b_idx = int(np.searchsorted(buy, t, side="right"))
        s_idx = int(np.searchsorted(sell, t, side="right"))
        next_buy = float(buy[b_idx]) if b_idx < len(buy) else math.inf
        next_sell = float(sell[s_idx]) if s_idx < len(sell) else math.inf
        next_both = max(next_buy, next_sell)
        candidate = max(1.0 + math.floor(t), next_both)
        t_next = min(cap, candidate)
        if not t_next > t:
            raise RuntimeError("decision-grid construction failed to advance")
        times.append(float(t_next))
        t = float(t_next)

    auction = np.arange(tau_op, tau_cl, dtype=float)
    return EpisodeGrid(np.asarray(times), auction, float(tau_cl))


def _interval_bounds(arrivals: np.ndarray, grid: EpisodeGrid) -> np.ndarray:
    """Indices partitioning all arrivals into ``(t_i,t_{i+1}]`` exactly once."""
    endpoints = np.concatenate((grid.clob_times, [grid.auction_times[0]]))
    bounds = np.searchsorted(arrivals, endpoints, side="right").astype(int)
    if bounds[0] != 0 or bounds[-1] != len(arrivals):
        raise AssertionError("CLOB arrival path was not fully partitioned")
    return bounds


def sample_episode_realization(
    rng: np.random.Generator,
    clob_params: ClobFlowParams,
    grid_params: GridParams,
) -> EpisodeRealization:
    """Pre-sample a complete policy-independent CLOB episode tape."""
    buy_times = _sample_poisson_times(rng, clob_params.lambda0, grid_params.tau_op)
    sell_times = _sample_poisson_times(rng, clob_params.lambda0, grid_params.tau_op)
    episode_grid = build_episode_grid(
        buy_times, sell_times, grid_params.tau_op, grid_params.tau_cl
    )

    buy_volumes = np.asarray(
        [
            sample_mo_volume(rng, clob_params.v_m, clob_params.gamma_m, clob_params.V_max)
            for _ in buy_times
        ],
        dtype=float,
    )
    sell_volumes = np.asarray(
        [
            sample_mo_volume(rng, clob_params.v_m, clob_params.gamma_m, clob_params.V_max)
            for _ in sell_times
        ],
        dtype=float,
    )

    n_snapshots = len(episode_grid.clob_times)
    ask_tops = clob_params.V_inf * rng.beta(
        clob_params.beta_a, clob_params.beta_b, size=n_snapshots
    )
    bid_tops = clob_params.V_inf * rng.beta(
        clob_params.beta_a, clob_params.beta_b, size=n_snapshots
    )
    return EpisodeRealization(
        grid=episode_grid,
        buy_arrival_times=buy_times,
        sell_arrival_times=sell_times,
        buy_volumes=buy_volumes,
        sell_volumes=sell_volumes,
        buy_bounds=_interval_bounds(buy_times, episode_grid),
        sell_bounds=_interval_bounds(sell_times, episode_grid),
        ask_tops=ask_tops,
        bid_tops=bid_tops,
    )


class MarketGenerator:
    """Own the realized CLOB tape, current book, and exogenous auction flow."""

    def __init__(
        self,
        clob_params: ClobFlowParams,
        auction_params: AuctionFlowParams,
        grid: GridParams,
    ) -> None:
        self.clob_params = clob_params
        self.auction_params = auction_params
        self.grid = grid
        self.book = OrderBook(clob_params, grid)
        self.auction_flow = ExogenousAuctionFlow(auction_params, grid, clob_params)
        self._realization: EpisodeRealization | None = None
        self._prepared_clob_index: int | None = None

    def reset_episode(self, rng: np.random.Generator) -> EpisodeGrid:
        """Pre-sample a new episode and return its complete internal grid."""
        self._realization = sample_episode_realization(rng, self.clob_params, self.grid)
        self._prepared_clob_index = None
        return self._realization.grid

    @property
    def episode_grid(self) -> EpisodeGrid:
        if self._realization is None:
            raise RuntimeError("reset_episode(rng) must be called first")
        return self._realization.grid

    def prepare_clob_decision(self, decision_index: int, k_mid: int) -> BookSnapshot:
        """Install and reveal only decision ``i``'s fresh exogenous snapshot."""
        if self._realization is None:
            raise RuntimeError("reset_episode(rng) must be called first")
        i = int(decision_index)
        if not 0 <= i < len(self._realization.grid.clob_times):
            raise IndexError(f"CLOB decision index {i} out of range")
        if self._prepared_clob_index is not None and i != self._prepared_clob_index + 1:
            raise ValueError(
                f"CLOB snapshots must be revealed sequentially; prepared "
                f"{self._prepared_clob_index}, requested {i}"
            )
        self.book.load_snapshot(
            int(k_mid),
            float(self._realization.ask_tops[i]),
            float(self._realization.bid_tops[i]),
        )
        self._prepared_clob_index = i
        return self.book.exogenous_snapshot()

    def step_clob_index(self, decision_index: int) -> ClobStepFlow:
        """Apply only the current pre-sampled interval to the installed book."""
        if self._realization is None:
            raise RuntimeError("reset_episode(rng) must be called first")
        i = int(decision_index)
        if self._prepared_clob_index != i:
            raise RuntimeError("prepare_clob_decision(i, k_mid) must precede step_clob_index(i)")
        buy, sell = self._realization.interval_volumes(i)
        executed = 0.0
        # The generative specification indexes sides independently.  Applying
        # all buy then all sell orders is equivalent because they touch disjoint
        # exogenous sides and only buys interact with the strategic ask.
        for volume in buy:
            executed += self.book.process_buy_market_order(float(volume))
        for volume in sell:
            self.book.process_sell_market_order(float(volume))
        t, t_next = self._realization.interval_times(i)
        residual = self.book.residual_exogenous_snapshot() if i == self.episode_grid.n else None
        return ClobStepFlow(
            decision_index=i,
            t=t,
            t_next=t_next,
            buy_volumes=buy,
            sell_volumes=sell,
            n_buy=len(buy),
            n_sell=len(sell),
            executed_agent=executed,
            residual_exogenous=residual,
        )

    # ------------------------------------------------------------------
    # Compatibility shim for the old on-demand environment.  New integration
    # must use reset_episode -> prepare_clob_decision -> step_clob_index.
    # ------------------------------------------------------------------
    def step_clob(
        self,
        rng: np.random.Generator,
        t: float,
        final: bool,
    ) -> ClobStepFlow:
        """Legacy on-demand step retained while environment integration lands."""
        p = self.clob_params
        next_buy = float(t) + float(rng.exponential(scale=1.0 / p.lambda0))
        next_sell = float(t) + float(rng.exponential(scale=1.0 / p.lambda0))
        target = (
            float(self.grid.tau_op)
            if final
            else min(
                max(math.floor(t) + 1.0, max(next_buy, next_sell)),
                float(self.grid.tau_op - 1),
            )
        )
        buy: list[float] = []
        sell: list[float] = []
        executed = 0.0
        while min(next_buy, next_sell) <= target:
            if next_buy <= next_sell:
                now = next_buy
                volume = sample_mo_volume(rng, p.v_m, p.gamma_m, p.V_max)
                buy.append(volume)
                executed += self.book.process_buy_market_order(volume)
                next_buy = now + float(rng.exponential(scale=1.0 / p.lambda0))
            else:
                now = next_sell
                volume = sample_mo_volume(rng, p.v_m, p.gamma_m, p.V_max)
                sell.append(volume)
                self.book.process_sell_market_order(volume)
                next_sell = now + float(rng.exponential(scale=1.0 / p.lambda0))
        return ClobStepFlow(
            decision_index=-1,
            t=float(t),
            t_next=target,
            buy_volumes=np.asarray(buy),
            sell_volumes=np.asarray(sell),
            n_buy=len(buy),
            n_sell=len(sell),
            executed_agent=executed,
            residual_exogenous=self.book.residual_exogenous_snapshot() if final else None,
        )

    def step_auction(self, rng: np.random.Generator) -> AuctionEvents:
        """Apply one six-proposal auction update."""
        return self.auction_flow.step(rng)
