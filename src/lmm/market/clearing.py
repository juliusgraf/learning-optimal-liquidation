"""Clearing-price machinery (paper `sec:clearing`, `sec:proj`).

Implements:
- Algorithm 1 (`alg:hyp_clearing_price`) with end-of-step-t semantics (D2);
- the corrected Eq. (2) estimator with the end-of-(t-1) cache (D1);
- the corrected Eq. (1) / Prop. linear closed form, plus the two-case solve
  for the benchmarks' one-sided hockey-stick order (ruling D16) and a
  bracketed root-finder for general monotone curves (Theorem `th:clearing`).

Degenerate fallback (ruling D17): H_cl = S^mid in ALL zero-slope cases,
estimate and terminal alike; logged when it binds.

Phase 3 fills in the bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import Algo1Params, GridParams
from lmm.market.clob import BookSnapshot

__all__ = [
    "Algo1Estimator",
    "ClearingInputs",
    "Eq2Cache",
    "solve_linear_clearing",
    "solve_clearing_with_hockey_stick",
    "solve_monotone_clearing",
]


class Algo1Estimator:
    """Algorithm 1: hypothetical clearing price during the CLOB phase.

    At the END of each CLOB step t (ruling D2), over the post-flow standing
    book (incl. the agent's unexecuted remainder, before refresh): update
    per-level running moments e_hat^k (mean volume) and sigma_hat^k (mean
    squared volume); K_hat^k = (2 e_hat - sigma_hat/e_hat)/alpha clamped at 0
    (legacy choice, counted in ``n_khat_clamped``);
    S_tilde = sum_k K_hat alpha k / sum_k K_hat;
    H_{t+1} = H_t + tau (S_tilde - H_t), smoothing SKIPPED when
    sum_k K_hat = 0 (counted in ``n_zero_slope_skips``).
    H_0 = initial mid (= 100; D15: tau = 0.95 in both settings).
    The output is H_{t+1}: input to the agent's time-(t+1) state and reward;
    the reward at t = 0 uses H_0.
    """

    def __init__(self, params: Algo1Params, grid: GridParams) -> None:
        self.params = params
        self.grid = grid
        self.n_khat_clamped: int = 0
        self.n_zero_slope_skips: int = 0

    def reset(self, h0: float) -> None:
        """New episode: clear moments, set H = h0 (= initial mid)."""
        raise NotImplementedError("Phase 3")

    def update(self, snapshot: BookSnapshot) -> float:
        """End-of-step update over ``snapshot``; returns the new H (= H_{t+1})."""
        raise NotImplementedError("Phase 3")

    @property
    def h(self) -> float:
        """Current smoothed hypothetical clearing price H."""
        raise NotImplementedError("Phase 3")


@dataclass(frozen=True)
class ClearingInputs:
    """End-of-step inputs to corrected Eq. (1)/(2) (paper convention, D3)."""

    K_exo: np.ndarray  # exogenous MM slopes K^i
    S_exo: np.ndarray  # exogenous MM quotes S^i
    K_agent: np.ndarray  # live agent slopes (1 - theta) already applied
    S_agent: np.ndarray  # corresponding agent quotes
    net_market_volume: float  # sum nu^{+,i} - sum nu^{-,i} (buys positive)
    fallback_mid: float  # H_cl = S^mid fallback when slope is zero (D17)


class Eq2Cache:
    """Auction-phase H_cl estimate: corrected Eq. (2) with D1 caching.

    The estimate used in the time-t_j state and reward is computed at the END
    of step t_{j-1}: exogenous orders as of end of t_{j-1}, agent orders
    s <= j-1, cancellation state theta_{t_j} (embedding c_{t_{j-1}}). Nothing
    sampled or decided at t_j may enter it. At t_{n+1} the cache holds
    Algorithm 1's last CLOB output. Setting j = m+1 recovers Eq. (1).
    """

    def __init__(self, grid: GridParams) -> None:
        self.grid = grid
        self.n_degenerate_fallbacks: int = 0

    def reset(self, h_from_algo1: float) -> None:
        """Auction open: seed the cache with Algorithm 1's last CLOB output."""
        raise NotImplementedError("Phase 3")

    def recompute(self, inputs: ClearingInputs) -> float:
        """End-of-step t-1: solve corrected Eq. (2); cache and return the root."""
        raise NotImplementedError("Phase 3")

    def read(self) -> float:
        """The cached estimate, valid for the CURRENT decision time (D1)."""
        raise NotImplementedError("Phase 3")


def solve_linear_clearing(inputs: ClearingInputs) -> float:
    """Corrected Prop. linear closed form (all curves linear):

    p* = [sum K_i S_i + sum (1-theta) K^a S^a + (sum nu^+ - sum nu^-)]
         / [sum K_i + sum (1-theta) K^a].

    Invariants (asserted in tests): adding buy market volume weakly RAISES
    p*; sell volume weakly LOWERS it. Zero denominator => ``fallback_mid``
    (ruling D17), logged by the caller.
    """
    raise NotImplementedError("Phase 3")


def solve_clearing_with_hockey_stick(
    inputs: ClearingInputs,
    z_slope: float,
    s_tilde: float,
) -> float:
    """Clearing with one one-sided benchmark order z_slope * (p - s_tilde)_+
    on top of the linear aggregate (ruling D16; benchmarks only liquidate).

    Two-case solve: root of the linear form excluding the benchmark order; if
    it is <= s_tilde it stands, otherwise re-solve with the benchmark slope
    included. The LHS stays continuous and nondecreasing in p.
    """
    raise NotImplementedError("Phase 3")


def solve_monotone_clearing(
    excess_supply,  # Callable[[float], float], nondecreasing in p
    bracket: tuple[float, float],
    tol: float = 1e-10,
) -> float:
    """Bracketed root-finder for general monotone supply curves
    (Theorem `th:clearing` existence; bisection on a widening bracket)."""
    raise NotImplementedError("Phase 3")
