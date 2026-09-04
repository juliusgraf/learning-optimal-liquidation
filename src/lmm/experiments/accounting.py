"""Economic episode accounting, separate from the shaped RL reward.

The resolved training reward may be the headline centered economic objective
or the three-regime shaped treatment objective in :mod:`lmm.env.rewards`.
Policy comparison always uses risk-adjusted marked-to-market PnL: full-session
PnL less the stipulated terminal inventory penalty. PnL already includes
actual cancellation fees. Positive signed quantity is a sale and negative
signed quantity is a purchase.

Residual inventory is *deemed liquidated* at the frozen auction-open mid.  The
reference is exogenous to the agent's auction orders, unlike the clearing
price, so an agent cannot improve the residual mark by moving ``S_cl``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = ["LiquidationAccounting", "compute_liquidation_accounting"]


@dataclass(frozen=True)
class LiquidationAccounting:
    """Cash-accounting decomposition for one completed episode."""

    initial_value: float
    auction_cash: float
    residual_liquidation_cash: float
    liquidation_pnl_gross: float
    liquidation_pnl_net: float
    economic_objective: float


def compute_liquidation_accounting(
    *,
    initial_inventory: float,
    initial_mid: float,
    clob_exec_qty: float,
    clob_cash: float,
    auction_exec_qty: float,
    auction_price: float,
    final_inventory: float,
    residual_liquidation_price: float,
    cancel_cost: float = 0.0,
    inventory_penalty: float = 0.0,
) -> LiquidationAccounting:
    """Return marked-to-market PnL and risk-adjusted PnL.

    The gross economic metric is

    ``C_clob + S_cl*Z + S_ref*I_final - S_0*I_0``.

    The ``S_ref*I_final`` term is the signed cash from a stipulated terminal
    close: sell a positive residual, or buy back a negative residual.  The
    ``liquidation_pnl_net`` subtracts cancellation costs. The primary economic
    objective additionally subtracts the terminal inventory penalty:

    ``economic_objective = PnL_gross - cancel_cost - inventory_penalty``.
    """
    values = (
        initial_inventory,
        initial_mid,
        clob_exec_qty,
        clob_cash,
        auction_exec_qty,
        auction_price,
        final_inventory,
        residual_liquidation_price,
        cancel_cost,
        inventory_penalty,
    )
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("liquidation accounting inputs must all be finite")
    if float(cancel_cost) < 0.0 or float(inventory_penalty) < 0.0:
        raise ValueError("economic costs must be non-negative")

    expected_final = float(initial_inventory) - float(clob_exec_qty) - float(auction_exec_qty)
    scale = max(1.0, abs(float(initial_inventory)), abs(float(clob_exec_qty)), abs(float(auction_exec_qty)))
    if not math.isclose(float(final_inventory), expected_final, rel_tol=1e-10, abs_tol=1e-9 * scale):
        raise ValueError(
            "inventory conservation failed: "
            f"I_final={final_inventory}, I0-Q_clob-Z={expected_final}"
        )

    initial_value = float(initial_inventory) * float(initial_mid)
    auction_cash = float(auction_exec_qty) * float(auction_price)
    residual_cash = float(final_inventory) * float(residual_liquidation_price)
    gross = float(clob_cash) + auction_cash + residual_cash - initial_value
    net = gross - float(cancel_cost)
    return LiquidationAccounting(
        initial_value=initial_value,
        auction_cash=auction_cash,
        residual_liquidation_cash=residual_cash,
        liquidation_pnl_gross=gross,
        liquidation_pnl_net=net,
        economic_objective=net - float(inventory_penalty),
    )
