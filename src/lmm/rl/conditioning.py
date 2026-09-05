"""An undiscounted, observable-state potential; no change to the objective."""
from __future__ import annotations

import numpy as np


def inventory_potential(observation, cfg, *, done=False):
    """Approximate current liquidation value from the common observed moments.

    Auction clearing here is only a control variate: it uses a continuous
    aggregate root and does not pretend to reproduce tick/pro-rata settlement.
    Actual rewards still come exclusively from the unmodified simulator.
    Phi(terminal)=0 implies sum(r + Phi(next)-Phi(now)) = sum(r)-Phi(initial).
    No future state, benchmark action, or target policy enters the potential.
    """
    if done:
        return 0.0
    x = np.asarray(observation, dtype=np.float64)
    inventory, mid = x[1], x[3]
    own_slope, own_quote, exo_slope, imbalance, exo_quote = x[13:18]
    quantity, displacement = 0.0, 0.0
    if x[0] >= cfg.grid.tau_op and exo_slope > 0:
        displacement = (
            own_quote - mid * own_slope + exo_quote - mid * exo_slope + imbalance
        ) / (own_slope + exo_slope)
        quantity = own_slope * displacement - (own_quote - mid * own_slope)
    # CLOB inventory can still be sold during many future intervals. Valuing
    # *all* of it at the terminal quadratic penalty creates a huge artificial
    # value baseline (lambda*I0^2) and swamps tick-sized execution advantages.
    # The terminal exposure estimate belongs to the auction phase only.
    remaining = inventory - quantity if x[0] >= cfg.grid.tau_op else 0.0
    if cfg.rl.learning_clob_inventory_potential and x[0] < cfg.grid.tau_op:
        # A deterministic depletion corridor makes deadline exposure visible
        # before the phase boundary, without a large initial value baseline.
        # This is a control variate, not a trading constraint or extra cost:
        # its increments telescope just like the rest of this potential.
        corridor = cfg.grid.I0 * (1-float(x[0])/cfg.grid.tau_op)
        remaining = max(float(inventory)-corridor, 0.)
    credit = 0.0
    if cfg.rl.learning_credit_baseline and cfg.reward.effective_auction_shaping:
        # Observable upper local submission-credit scale for the remaining
        # slots. Subtracting its decline keeps the auction critic from having
        # to learn a large gross-notional baseline before action differences.
        # Its initial value is a policy-independent constant, NOT extra PnL.
        slots = cfg.grid.tau_cl - max(float(x[0]), cfg.grid.tau_op)
        anchor = x[2] if cfg.rl.h_cl_feature_enabled else mid
        credit = cfg.reward.auction_shaping_weight * slots * cfg.actions.auction_K_grid_max * max(float(anchor), 0.) * cfg.grid.alpha * (cfg.actions.B_max + .5)
    return float((mid - cfg.grid.S0) * inventory + displacement * quantity
                 - cfg.reward.lambda_inv * remaining**2 + credit)
