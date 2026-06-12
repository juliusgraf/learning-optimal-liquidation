"""CLOB book state, refresh, and market-order processing (paper `sec:lob`).

Sign convention (BINDING, ruling D3 / CLAUDE.md): zeta = + is the ASK side
for limit orders and the BUY side for market orders; buy market orders
consume ask liquidity. The agent sells on the ask side; delta_t >= 0.

Phase 3 fills in the bodies (numpy arrays per side; same arithmetic as
legacy `main.py:183-191, 462-514`, per ruling D14 item 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import ClobFlowParams, GridParams

__all__ = ["BookSnapshot", "OrderBook"]


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
    depth decay rho, refreshed each CLOB step (Algorithm 2 lines 5-6).
    """

    def __init__(self, params: ClobFlowParams, grid: GridParams) -> None:
        self.params = params
        self.grid = grid

    def refresh(self, rng: np.random.Generator, k_mid: int) -> None:
        """Redraw both sides around the current mid (legacy `_refresh_order_book`)."""
        raise NotImplementedError("Phase 3")

    def depth(self, side: int) -> int:
        """Book depth L^zeta_t = inf{j : V^{zeta,j} = 0} (capped at Lc)."""
        raise NotImplementedError("Phase 3")

    def place_agent_order(self, volume: float, delta: int) -> None:
        """Record the agent's time-t ask-side limit order at level delta >= 0."""
        raise NotImplementedError("Phase 3")

    def process_buy_market_order(self, volume: float) -> float:
        """Consume ask liquidity; returns the agent volume executed.

        Implements E_t with AGENT EXECUTION PRIORITY at her own price level
        (Sec. 2.1.1; legacy `process_buy_order` main.py:462-502): exogenous
        levels j < delta are consumed first, then the agent's order, then
        deeper exogenous volume.
        """
        raise NotImplementedError("Phase 3")

    def process_sell_market_order(self, volume: float) -> None:
        """Consume bid liquidity (no agent interaction; the agent only sells)."""
        raise NotImplementedError("Phase 3")

    def snapshot(self) -> BookSnapshot:
        """Post-flow standing book for Algorithm 1 (end-of-step semantics, D2)."""
        raise NotImplementedError("Phase 3")


def executed_volume(
    agent_volume: float,
    agent_delta: int,
    ask_volumes: np.ndarray,
    incoming_buy_volumes: np.ndarray,
) -> float:
    """Aggregate E_t formula of the paper (exposed separately for unit tests).

    E_t = max(0, min(v_t, sum(nu over the step's buy MOs)
                          - sum_{j < delta} V^{+,j})).
    Must agree with sequential `OrderBook.process_buy_market_order` calls.
    """
    raise NotImplementedError("Phase 3")
