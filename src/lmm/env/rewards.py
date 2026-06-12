"""Three-regime reward functions as pure closed forms (paper `sec:MDP`;
rulings D1, D2, D4, D5, D8).

Kept side-effect free so tests can assert hand-computed cases and the env
reward can be asserted equal to the formula applied to its own ``info``
diagnostics (tests/test_rewards.py).

H_cl timing is the CALLER's contract (env/mdp.py): every ``h_cl`` argument
below must be the cached end-of-previous-step value (rulings D1/D2).
"""

from __future__ import annotations

import numpy as np

__all__ = ["f_c", "f_a", "clob_reward", "auction_reward", "terminal_reward"]


def f_c(u: float, k_star: int, alpha: float) -> float:
    """f_c(u) = (u)_+ / (k* alpha) — NO clamp of the result at 1 (ruling D5):
    for S^bullet_t > H^cl_t the multiplier exceeds 1 by definition (the
    legacy clamp at 1, main.py:545, was wrong)."""
    return max(u, 0.0) / (k_star * alpha)


def f_a(u: float, q: float) -> float:
    """Wrong-side penalty f_a(u) = -q (-u)_+ (paper, auction reward)."""
    return -q * max(-u, 0.0)


def clob_reward(s_bullet: float, executed: float, h_cl: float, k_star: int, alpha: float) -> float:
    """CLOB regime (ruling D5, paper EXACTLY):

    r_t = S^bullet_t * E_t * f_c(k* alpha - (H^cl_t - S^bullet_t)),

    with S^bullet_t = alpha A^2_t the submitted price, E_t the executed
    volume following the time-t action, and H^cl_t the LAGGED Algorithm-1
    value (ruling D2; the reward at t = 0 uses H_0)."""
    return s_bullet * executed * f_c(k_star * alpha - (h_cl - s_bullet), k_star, alpha)


def auction_reward(
    K_a: float,
    S_a: float,
    h_cl: float,
    q: float,
    d_t: float,
    cancel: int,
    one_sided: bool = False,
) -> float:
    """Auction regime (ruling D4, paper):

    r_t = K^a_t H^cl_t (H^cl_t - S^a_t)
          + f_a(K^a_t H^cl_t (H^cl_t - S^a_t)) - d_t c_t,

    with d_t = (t - n - 1) d computed by the caller and c_t the SCALAR
    cancel-all (cost d_t * c_t — NOT d times the number of orders cancelled;
    legacy was wrong). H^cl_t is the end-of-(t-1) Eq. (2) cache (ruling D1).

    ``one_sided=True`` (benchmark hockey-stick, ruling D16): the supplied
    volume is K^a (p - S^a)_+, so the gap enters through its positive part —
    u = K^a H^cl (H^cl - S^a)_+ >= 0 and the f_a penalty never binds (the
    benchmark is always selling; author-confirmed, ruling D18)."""
    gap = h_cl - S_a
    if one_sided:
        gap = max(gap, 0.0)
    u = K_a * h_cl * gap
    return u + f_a(u, q) - d_t * float(cancel)


def terminal_reward(
    K_live: np.ndarray,
    S_live: np.ndarray,
    s_cl: float,
    inventory_final: float,
    lambda_inv: float,
    q: float,
    one_sided: np.ndarray | None = None,
) -> float:
    """Terminal regime at t = tau_cl (rulings D3, D8):

    r_{tau_cl} = sum_s K^a_s S_cl (S_cl - S^a_s) (1 - theta^{(s-n)})
                 - lambda |I_{tau_cl}|^2
                 + sum_s f_a(K^a_s S_cl (S_cl - S^a_s) (1 - theta^{(s-n)})),

    where ``K_live``/``S_live`` enumerate the live orders only (the
    (1 - theta) factors realized by the ledger) and ``inventory_final`` =
    I_{tau_op} - Z_{tau_cl} with NO clipping (ruling D8).

    ``one_sided`` (optional bool mask over the live orders, ruling D16):
    where True the order is the benchmark hockey-stick K^a (p - S^a)_+, so
    its gap enters through the positive part — the contribution is
    K^a S_cl (S_cl - S^a)_+ >= 0 and f_a never binds for it."""
    gap = s_cl - np.asarray(S_live, dtype=float)
    if one_sided is not None:
        gap = np.where(np.asarray(one_sided, dtype=bool), np.maximum(gap, 0.0), gap)
    u = np.asarray(K_live) * s_cl * gap
    wrong_side = float(sum(f_a(float(ui), q) for ui in u))
    return float(np.sum(u)) - lambda_inv * abs(inventory_final) ** 2 + wrong_side
