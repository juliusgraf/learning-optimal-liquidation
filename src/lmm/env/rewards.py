"""Pure reward and shaping formulas for the three trading regimes."""

from __future__ import annotations

import math

__all__ = [
    "f_c",
    "f_a",
    "clob_reward",
    "auction_fictive_reward",
    "auction_reward",
    "terminal_reward",
]


def f_c(u: float, k_star: int, alpha: float) -> float:
    """``min(1, u_+/(k_star*alpha))``."""
    denominator = float(k_star) * float(alpha)
    if denominator <= 0.0:
        raise ValueError("k_star*alpha must be positive")
    return float(min(1.0, max(float(u), 0.0) / denominator))


def f_a(x: float, q: float) -> float:
    """Purchase-side attenuation ``q*(-x)_+``."""
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"q must lie in [0,1], got {q}")
    return float(q) * max(-float(x), 0.0)


def clob_reward(
    s_bullet: float,
    executed: float,
    h_cl: float,
    k_star: int,
    alpha: float,
    *,
    shaping_enabled: bool = True,
) -> float:
    """CLOB cash, optionally multiplied by the manuscript shaping factor."""
    cash = float(s_bullet) * float(executed)
    if not shaping_enabled:
        return cash
    tolerated = float(k_star) * float(alpha) - max(float(h_cl) - float(s_bullet), 0.0)
    return cash * f_c(tolerated, k_star, alpha)


def auction_fictive_reward(
    K_a: float,
    S_a: float,
    h_cl: float,
    q: float,
) -> float:
    """Return the manuscript interim auction shaping ``phi=x+f_a(x)``.

    This is a fictive training signal, not an exchange cash flow.  Keeping it
    separate lets the environment record the exact signed shaping amount
    credited to each live strategic schedule.  A later cancellation can then
    reverse that same amount without re-marking the old order at a newer
    indicative price.
    """
    x = float(K_a) * float(h_cl) * (float(h_cl) - float(S_a))
    return x + f_a(x, q)


def auction_reward(
    K_a: float,
    S_a: float,
    h_cl: float,
    q: float,
    d_t: float,
    cancel: int,
    *,
    shaping_enabled: bool = True,
    external_policy: bool = False,
    cancelled_interim_shaping: float = 0.0,
) -> float:
    """Interim shaping, cancellation clawback, and the actual scalar fee.

    With shaping disabled (and for external benchmark schedules), the entire
    fictive term ``x+f_a(x)`` is omitted while ``d_t*c_t`` remains.  When a
    cancellation reverses shaping previously credited to live orders,
    ``cancelled_interim_shaping`` is that exact signed sum.  It is subtracted
    once; for ``q=1`` it is nonnegative, while the signed definition remains
    correct for every permitted ``q``.
    """
    fee = float(d_t) * float(cancel)
    clawback = float(cancelled_interim_shaping)
    if not math.isfinite(clawback):
        raise ValueError("cancelled_interim_shaping must be finite")
    if not int(cancel) and clawback != 0.0:
        raise ValueError("nonzero shaping clawback requires cancel=1")
    if not shaping_enabled or external_policy:
        if clawback != 0.0:
            raise ValueError("cannot claw back shaping when auction shaping is disabled")
        return -fee
    return auction_fictive_reward(K_a, S_a, h_cl, q) - fee - clawback


def terminal_reward(
    s_cl: float,
    actual_agent_quantity: float,
    inventory_final: float,
    frozen_mid: float,
    lambda_inv: float,
    q: float,
    *,
    shaping_enabled: bool = True,
    external_policy: bool = False,
) -> float:
    """Terminal economic cash, residual mark, penalty, and aggregate shaping.

    Purchase attenuation is evaluated exactly once on the agent's *actual*
    aggregate pro-rata cash ``S_cl*Z``.  It is not evaluated schedule by
    schedule and therefore cannot create an artificial strategic self-trade.
    """
    cash = float(s_cl) * float(actual_agent_quantity)
    residual_mark = float(frozen_mid) * float(inventory_final)
    penalty = float(lambda_inv) * float(inventory_final) ** 2
    shaping = 0.0
    if shaping_enabled and not external_policy:
        shaping = f_a(cash, q)
    return cash - penalty + shaping + residual_mark
