"""Auction phase: agent order ledger and exogenous flow (paper `sec:auction`,
Algorithm 2 lines 12-17; rulings D3 (theta), D4 (cancel), D6-D7 (params)).

Phase 3 fills in the bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import AuctionFlowParams, GridParams

__all__ = ["AgentOrderLedger", "ExogenousAuctionFlow"]


class AgentOrderLedger:
    """Agent auction orders (K^a_{t_s}, S^a_{t_s}) with liveness tracking.

    Implements the CORRECTED theta recursion (ruling D3 / CLAUDE.md):
    theta_{t_n} = theta_{t_{n+1}} = 0 and
    theta_{t_j} = max(theta_{t_{j-1}}, c_{t_{j-1}} * sum_{k=1}^{j-n-2} e_k),
    i.e. a cancel-all decided at t_j kills all orders submitted strictly
    before t_j and is reflected in theta_{t_{j+1}} (predictable: theta_{t_j}
    is part of the state X^9_{t_j}). The internal live/dead flags are the
    direct realization; ``theta()`` materializes the paper vector.

    Convention: submitting K^a = 0 is identified with abstaining (no entry).
    The final order at t_m can never be cancelled; X^16/X^17 use the
    predictable indexing (entries for orders submitted up to t-1 only).
    """

    def __init__(self, grid: GridParams) -> None:
        self.grid = grid

    def reset(self) -> None:
        """Clear all entries at auction open (tau_op)."""
        raise NotImplementedError("Phase 3")

    def submit(self, t: int, K_a: float, S_a: float) -> None:
        """Record the order decided at auction time t (no-op if K_a == 0)."""
        raise NotImplementedError("Phase 3")

    def apply_cancel_all(self, t: int) -> None:
        """Apply c_t = 1: deactivate all orders submitted at times < t.

        Takes effect in theta_{t+1} / the time-(t+1) Eq. (2) estimate (D1) —
        the caller sequences this AFTER the time-t estimate is read.
        """
        raise NotImplementedError("Phase 3")

    def cancel_admissible(self) -> bool:
        """C(x) = max_i (1 - x^{9,(i)}) 1{x^{17,(i)} > 0} > 0 (CLAUDE.md):
        a cancel-all is admissible iff a live prior order with K^a > 0 exists.
        Must agree with C evaluated on ``paper_state()`` (asserted in tests).
        """
        raise NotImplementedError("Phase 3")

    def theta(self) -> np.ndarray:
        """Paper theta vector in R^{m-n} (component s-n for the order at t_s)."""
        raise NotImplementedError("Phase 3")

    def live_orders(self) -> tuple[np.ndarray, np.ndarray]:
        """(K^a, S^a) arrays over currently live orders (for clearing solves)."""
        raise NotImplementedError("Phase 3")


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
    S^i ~ S^mid_{tau_op} + alpha * U{-band..band} (mid FROZEN at tau_op);
    MM cancellation with prob p2; new market taker per side with prob p3
    (independent); taker cancellation with prob p4 per side.

    Ruling D7: p4 = 0.05 as a SINGLE Bernoulli draw per side. Legacy realized
    the same distribution as Bernoulli(0.1) gated by an extra fair coin
    (main.py:616-625); collapsing the two draws changes the RNG stream, so
    seeded trajectories intentionally differ from legacy (pinned by the
    characterization tests).

    MM arrivals are suppressed when the MM count reaches La (legacy cap kept
    as explicit config, AUDIT N8). Cancelled taker volumes are set to 0 with
    N^zeta unchanged. Naming uses the PAPER convention (D3): buy takers
    (paper zeta = +) RAISE the clearing price.
    """

    def __init__(self, params: AuctionFlowParams, grid: GridParams) -> None:
        self.params = params
        self.grid = grid

    def reset(self, frozen_mid: float) -> None:
        """Clear ledgers at auction open; store the frozen mid for quote draws."""
        raise NotImplementedError("Phase 3")

    def step(self, rng: np.random.Generator) -> AuctionEvents:
        """Sample one auction step's exogenous events; update internal ledgers."""
        raise NotImplementedError("Phase 3")

    def supply_curves(self) -> tuple[np.ndarray, np.ndarray]:
        """(K^i, S^i) over active exogenous MMs (M_t entries)."""
        raise NotImplementedError("Phase 3")

    def net_market_volume(self) -> float:
        """sum_i nu^{+,i} - sum_i nu^{-,i} (paper convention: buys positive)."""
        raise NotImplementedError("Phase 3")
