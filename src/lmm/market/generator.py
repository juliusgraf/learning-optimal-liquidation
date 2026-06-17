"""Algorithm 2 orchestration (`alg:generative_model`; rulings D6-D7).

Owns the Poisson market-order arrivals on [0, tau_op], the event-driven
decision grid hat_t_i guaranteeing at least one new arrival per side per step
(Assumption `assump:presence`), the per-step CLOB flow, and the auction-step
event loop. Decisions occur at t in {0,...,n} (fixing AUDIT N1: legacy never
acted at t_n = tau_op - 1); the step taken AT t_n runs the exogenous flow on
(t_n, tau_op] and the env then opens the auction.

Arrival construction (Algorithm 2 lines 1-2; legacy main.py:453-460,
516-535, KEPT with the following justification): at each decision time t the
two exponential clocks are REDRAWN, tau^zeta ~ t + Exp(1/lambda_0), and the
in-step event loop is seeded with these same draws, renewing the fired
side's clock after every processed order. By the memorylessness of the
exponential distribution, redrawing the residual clock at a (stopping) step
boundary leaves the law of the arrival sequence unchanged, so the realized
N^+, N^- are exactly two independent Poisson(lambda_0) processes on
[0, tau_op] — the paper's construction. The decision grid uses the SAME
first-arrival draws: hat_t_{i+1} = min(max(floor(hat_t_i) + 1,
max(tau^+, tau^-)), n), which guarantees at least one new arrival per side
within every uncapped step (Assumption assump:presence; steps truncated by
the cap at n can lack an arrival — legacy behavior, AUDIT A.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmm.config import AuctionFlowParams, ClobFlowParams, GridParams
from lmm.market.auction import AuctionEvents, ExogenousAuctionFlow
from lmm.market.clob import OrderBook, sample_mo_volume

__all__ = ["ClobStepFlow", "MarketGenerator"]


@dataclass(frozen=True)
class ClobStepFlow:
    """Exogenous market-order flow realized within one CLOB step.

    ``t_next`` is the next decision time hat_t (== tau_op for the final
    step); ``executed_agent`` is E_t, the agent volume filled by this step's
    buy market orders (paper Sec. 2.1.1).
    """

    t_next: float
    buy_volumes: np.ndarray  # nu for the step's BUY MOs (paper zeta = +)
    sell_volumes: np.ndarray  # nu for the step's SELL MOs (paper zeta = -)
    n_buy: int  # increment to paper N^+
    n_sell: int  # increment to paper N^-
    executed_agent: float  # E_t


class MarketGenerator:
    """Generative stochastic market model (Algorithm 2).

    CLOB phase: N^+, N^- independent Poisson(lambda_0) per side (exponential
    interarrivals, see module docstring); market-order sizes
    min(Pareto(v_m, gamma_m), V); the env refreshes the book each step.
    Auction phase: delegates to :class:`ExogenousAuctionFlow`
    (p1 = 0.3, p2 = 0.2, p3 = 0.3, p4 = 0.05 per ruling D7).
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
        self.auction_flow = ExogenousAuctionFlow(auction_params, grid, clob_params)

    def step_clob(self, rng: np.random.Generator, t: float, final: bool) -> ClobStepFlow:
        """Sample and apply one CLOB step's exogenous flow starting at
        decision time ``t``; the agent's order must already be on ``book``.

        Draw order (policy-independent, D10): tau^+ then tau^- (the first
        post-t arrival per side), then alternating (volume, next-interarrival)
        pairs for each processed order, earlier clock first (buy wins ties —
        legacy main.py:527-535).

        ``final=True`` (t = t_n): the flow runs on (t_n, tau_op] and
        ``t_next = tau_op``; otherwise ``t_next = min(max(floor(t) + 1,
        max(tau^+, tau^-)), n)`` with n = tau_op - 1.
        """
        p = self.clob_params
        lam = p.lambda0  # per grid unit (legacy lam_step = poisson_rate * dt, dt = 1)

        next_buy = t + rng.exponential(scale=1.0 / lam)
        next_sell = t + rng.exponential(scale=1.0 / lam)

        if final:
            target = float(self.grid.tau_op)
        else:
            target = min(
                max(math.floor(t) + 1.0, max(next_buy, next_sell)),
                float(self.grid.tau_op - 1),
            )

        buy_volumes: list[float] = []
        sell_volumes: list[float] = []
        executed = 0.0
        while min(next_buy, next_sell) <= target:
            if next_buy <= next_sell:
                now = next_buy
                vol = sample_mo_volume(rng, p.v_m, p.gamma_m, p.V_max)
                buy_volumes.append(vol)
                executed += self.book.process_buy_market_order(vol)
                next_buy = now + rng.exponential(scale=1.0 / lam)
            else:
                now = next_sell
                vol = sample_mo_volume(rng, p.v_m, p.gamma_m, p.V_max)
                sell_volumes.append(vol)
                self.book.process_sell_market_order(vol)
                next_sell = now + rng.exponential(scale=1.0 / lam)

        return ClobStepFlow(
            t_next=float(target),
            buy_volumes=np.asarray(buy_volumes),
            sell_volumes=np.asarray(sell_volumes),
            n_buy=len(buy_volumes),
            n_sell=len(sell_volumes),
            executed_agent=executed,
        )

    def step_auction(self, rng: np.random.Generator) -> AuctionEvents:
        """One auction step of exogenous events (Algorithm 2 lines 12-17)."""
        return self.auction_flow.step(rng)
