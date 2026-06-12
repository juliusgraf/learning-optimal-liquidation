"""Algorithm 2 orchestration (`alg:generative_model`; rulings D6-D7).

Owns the Poisson market-order arrivals on [0, tau_op], the event-driven
decision grid hat_t_i guaranteeing at least one new arrival per side per step
(Assumption `assump:presence`), the per-step CLOB flow, and the auction-step
event loop. Decisions occur at t in {0,...,n} (fixing AUDIT N1: legacy never
acted at t_n).

Phase 3 fills in the bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import AuctionFlowParams, ClobFlowParams, GridParams
from lmm.market.auction import AuctionEvents, ExogenousAuctionFlow
from lmm.market.clob import OrderBook

__all__ = ["ClobStepFlow", "MarketGenerator"]


@dataclass(frozen=True)
class ClobStepFlow:
    """Exogenous market-order flow realized within one CLOB step."""

    buy_volumes: np.ndarray  # nu for the step's BUY MOs (paper zeta = +)
    sell_volumes: np.ndarray  # nu for the step's SELL MOs (paper zeta = -)
    n_buy: int  # increment to paper N^+
    n_sell: int  # increment to paper N^-


class MarketGenerator:
    """Generative stochastic market model (Algorithm 2).

    CLOB phase: N^+, N^- independent Poisson(lambda_0) per side (exponential
    interarrivals); market-order sizes min(Pareto(v_m, gamma_m), V); book
    refreshed each step. Auction phase: delegates to
    :class:`ExogenousAuctionFlow` (p1=0.3, p2=0.2, p3=0.3, p4=0.05 per D7).
    """

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
        self.auction_flow = ExogenousAuctionFlow(auction_params, grid)

    def reset(self, rng: np.random.Generator) -> None:
        """New episode: reset book, arrival clocks and auction ledgers."""
        raise NotImplementedError("Phase 3")

    def next_clob_decision_time(self, t: float) -> float:
        """Next hat_t_i: min(max(floor(t)+1, max(tau^+, tau^-)), tau_op - 1)
        with tau^zeta the first post-t arrival per side (assump:presence;
        legacy main.py:453-460). Guarantees a decision at t_n = tau_op - 1."""
        raise NotImplementedError("Phase 3")

    def step_clob(self, rng: np.random.Generator, t_from: float, t_to: float) -> ClobStepFlow:
        """Sample and apply the exogenous CLOB flow on (t_from, t_to]."""
        raise NotImplementedError("Phase 3")

    def step_auction(self, rng: np.random.Generator) -> AuctionEvents:
        """One auction step of exogenous events (Algorithm 2 lines 12-17)."""
        raise NotImplementedError("Phase 3")
