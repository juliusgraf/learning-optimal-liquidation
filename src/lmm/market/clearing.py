"""Clearing-price machinery (paper `sec:clearing`, `sec:proj`).

Implements:
- Algorithm 1 (`alg:hyp_clearing_price`) with end-of-step-t semantics (D2);
- the corrected Eq. (2) estimator with the end-of-(t-1) cache (D1);
- the corrected Eq. (1) / Prop. linear closed form, plus the two-case solve
  for the benchmarks' one-sided hockey-stick order (ruling D16) and a
  bracketed root-finder for general monotone curves (Theorem `th:clearing`).

Sign convention (D3, corrected equations): the excess-supply function is

    Phi(p) = sum_i g_i(p) + sum_s (1 - theta^{(s-n)}) K^a_s (p - S^a_s)
             - (sum_i nu^{+,i} - sum_i nu^{-,i}),

so ``net_market_volume`` (buys positive) enters the linear numerator with a
PLUS sign: buy market volume weakly raises p*, sell volume weakly lowers it
(invariants asserted in tests/test_sign_conventions.py).

Degenerate fallback (ruling D17): H_cl = S^mid in ALL zero-slope cases,
estimate and terminal alike; logged when it binds. A machine-scale
``_SLOPE_EPS`` float-safety guard treats an aggregate slope that should be
exactly zero (but for floating-point residue) as zero (author refinement,
2026-06-14); it does NOT regularize economically small slopes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from lmm.config import Algo1Params, GridParams
from lmm.market.clob import BookSnapshot

__all__ = [
    "Algo1Estimator",
    "ClearingInputs",
    "Eq2Cache",
    "solve_clearing",
    "solve_linear_clearing",
    "solve_clearing_with_hockey_stick",
    "solve_monotone_clearing",
]

logger = logging.getLogger(__name__)

_EPS = 1e-12  # legacy eps (main.py:379)

# Float-safety slope guard (ruling D17, author refinement 2026-06-14): an
# aggregate clearing slope ΣK that should be exactly zero can leave a tiny
# non-zero residue from floating-point cancellation; dividing by it would
# produce a spurious clearing price. We treat ΣK <= _SLOPE_EPS as zero and take
# the S^mid fallback. This is a NUMERICAL guard ONLY (cf. the legacy _EPS
# above), deliberately at machine scale: economically small-but-real slopes
# (ΣK ~ 1e-2, which legitimately produce large clearing prices) are NOT
# regularized — that is faithful model behavior, and RL-side instability from
# chasing it is handled by the agents' reward clipping, not here.
_SLOPE_EPS = 1e-8


class Algo1Estimator:
    """Algorithm 1: hypothetical clearing price during the CLOB phase.

    At the END of each CLOB step t (ruling D2), over the post-flow standing
    book (incl. the agent's unexecuted remainder, before refresh): update
    per-level running moments e_hat^k (mean volume) and sigma_hat^k (mean
    squared volume); K_hat^k = (2 e_hat - sigma_hat/e_hat)/alpha clamped at 0
    (legacy choice, counted in ``n_khat_clamped``);
    S_tilde = sum_k K_hat alpha k / sum_k K_hat;
    H_{t+1} = H_t + tau (S_tilde - H_t), smoothing SKIPPED when
    sum_k K_hat <= _SLOPE_EPS (counted in ``n_zero_slope_skips``;
    float-safety guard, was == 0).
    H_0 = initial mid (= 100); ruling D15: tau = 0.95 in both settings.
    The output is H_{t+1}: input to the agent's time-(t+1) state and reward;
    the reward at t = 0 uses H_0.

    Matches legacy ``_update_hyp_clearing_price_from_book``
    (main.py:377-427) except the snapshot's mid tick uses the env-wide
    floor convention (AUDIT N3; legacy rounded here but floored elsewhere).
    Moments are keyed by ABSOLUTE tick k; levels absent from a snapshot
    implicitly contribute 0 (the per-snapshot count is global).
    """

    def __init__(self, params: Algo1Params, grid: GridParams) -> None:
        self.params = params
        self.grid = grid
        self.n_khat_clamped: int = 0
        self.n_zero_slope_skips: int = 0
        self._mom_sum: defaultdict[int, float] = defaultdict(float)
        self._mom_sum_sq: defaultdict[int, float] = defaultdict(float)
        self._mom_count: int = 0
        self._h: float = grid.S0

    def reset(self, h0: float) -> None:
        """New episode: clear moments, set H = h0 (= initial mid)."""
        self._mom_sum.clear()
        self._mom_sum_sq.clear()
        self._mom_count = 0
        self._h = float(h0)
        self.n_khat_clamped = 0
        self.n_zero_slope_skips = 0

    def update(self, snapshot: BookSnapshot) -> float:
        """End-of-step update over ``snapshot``; returns the new H (= H_{t+1})."""
        alpha = self.grid.alpha
        k0 = snapshot.k_mid

        vol_by_k: dict[int, float] = {}
        for j, v in enumerate(snapshot.ask_volumes):
            if v > _EPS:
                k = k0 + j
                vol_by_k[k] = vol_by_k.get(k, 0.0) + float(v)
        for j, v in enumerate(snapshot.bid_volumes):
            if v > _EPS:
                k = k0 - j
                vol_by_k[k] = vol_by_k.get(k, 0.0) + float(v)
        if snapshot.agent_level is not None and snapshot.agent_remaining > _EPS:
            k = k0 + snapshot.agent_level
            vol_by_k[k] = vol_by_k.get(k, 0.0) + float(snapshot.agent_remaining)

        self._mom_count += 1
        for k, v in vol_by_k.items():
            self._mom_sum[k] += v
            self._mom_sum_sq[k] += v * v

        num = 0.0
        den = 0.0
        count = self._mom_count
        for k, s in self._mom_sum.items():
            e_hat = s / count
            if e_hat <= _EPS:
                continue
            sig_hat = self._mom_sum_sq[k] / count
            raw = (2.0 * e_hat - sig_hat / max(e_hat, _EPS)) / alpha
            if raw < 0.0:
                self.n_khat_clamped += 1
                logger.debug("Algorithm 1: K_hat clamped to 0 at tick %d (raw %.6g)", k, raw)
            k_hat = max(0.0, raw)
            if k_hat > 0.0:
                num += k_hat * (alpha * k)
                den += k_hat

        if den > _SLOPE_EPS:
            s_tilde = num / den
            self._h = self._h + self.params.tau * (s_tilde - self._h)
        else:
            # (Near-)zero aggregate K_hat: skip smoothing, keep H (same
            # float-safety guard as the clearing solves; was den > 0.0).
            self.n_zero_slope_skips += 1
        return self._h

    @property
    def h(self) -> float:
        """Current smoothed hypothetical clearing price H."""
        return self._h


@dataclass(frozen=True)
class ClearingInputs:
    """End-of-step inputs to corrected Eq. (1)/(2) (paper convention, D3).

    ``K_agent``/``S_agent`` contain only the LIVE agent orders — the
    (1 - theta^{(s-n)}) factors are realized by the ledger's live mask
    before these arrays are built. ``hockey`` is the single live one-sided
    benchmark order (K, S) contributing K (p - S)_+ (ruling D16; None for
    the RL agent, whose orders are all linear).
    """

    K_exo: np.ndarray  # exogenous MM slopes K^i
    S_exo: np.ndarray  # exogenous MM quotes S^i
    K_agent: np.ndarray  # live agent slopes (1 - theta) already applied
    S_agent: np.ndarray  # corresponding agent quotes
    net_market_volume: float  # sum nu^{+,i} - sum nu^{-,i} (buys positive)
    fallback_mid: float  # H_cl = S^mid fallback when slope is zero (D17)
    hockey: tuple[float, float] | None = None  # live one-sided (K, S) (D16)


def solve_clearing(inputs: ClearingInputs) -> tuple[float, bool]:
    """Dispatch on ``inputs.hockey``: the pure linear closed form, or the
    two-case solve with the single one-sided benchmark order (ruling D16)."""
    if inputs.hockey is None:
        return solve_linear_clearing(inputs)
    z_slope, s_tilde = inputs.hockey
    return solve_clearing_with_hockey_stick(inputs, z_slope, s_tilde)


def solve_linear_clearing(inputs: ClearingInputs) -> tuple[float, bool]:
    """Corrected Prop. linear closed form (all curves linear):

    p* = [sum K_i S_i + sum (1-theta) K^a S^a + (sum nu^+ - sum nu^-)]
         / [sum K_i + sum (1-theta) K^a].

    Invariants (asserted in tests): adding buy market volume weakly RAISES
    p*; sell volume weakly LOWERS it. (Near-)zero denominator => ``fallback_mid``
    (ruling D17 + the ``_SLOPE_EPS`` float-safety guard); the second return
    value flags the degenerate case so the caller can count and log it.
    """
    den = float(np.sum(inputs.K_exo)) + float(np.sum(inputs.K_agent))
    if den <= _SLOPE_EPS:
        return inputs.fallback_mid, True
    num = (
        float(inputs.K_exo @ inputs.S_exo)
        + float(inputs.K_agent @ inputs.S_agent)
        + inputs.net_market_volume
    )
    return num / den, False


class Eq2Cache:
    """Auction-phase H_cl estimate: corrected Eq. (2) with D1 caching.

    The estimate used in the time-t_j state and reward is computed at the END
    of step t_{j-1}: exogenous orders as of end of t_{j-1}, agent orders
    s <= j-1, cancellation state theta_{t_j} (embedding c_{t_{j-1}}). Nothing
    sampled or decided at t_j may enter it. At t_{n+1} the cache holds
    Algorithm 1's last CLOB output. Setting j = m+1 recovers Eq. (1) exactly
    (theta_{t_{m+1}} embeds c_{t_m}), so the terminal clearing price S_cl is
    the recompute performed at the end of step t_m.
    """

    def __init__(self, grid: GridParams) -> None:
        self.grid = grid
        self.n_degenerate_fallbacks: int = 0
        self._h: float = grid.S0

    def reset(self, h_from_algo1: float) -> None:
        """Auction open: seed the cache with Algorithm 1's last CLOB output."""
        self._h = float(h_from_algo1)
        self.n_degenerate_fallbacks = 0

    def recompute(self, inputs: ClearingInputs) -> float:
        """End-of-step t-1: solve corrected Eq. (2); cache and return the root."""
        root, degenerate = solve_clearing(inputs)
        if degenerate:
            self.n_degenerate_fallbacks += 1
            # DEBUG, not WARNING: this is an EXPECTED, designed fallback (D17)
            # that fires routinely whenever the auction has no live supply
            # curve (abstaining policy + no exogenous MM) -- e.g. the greedy
            # untrained baseline abstains, making ~25% of its auction steps
            # degenerate. It is already counted in ``n_degenerate_fallbacks``
            # (surfaced per-episode as metrics.csv ``n_degenerate_fallbacks``),
            # so a per-occurrence WARNING would only flood the console. Mirrors
            # the K_hat-clamp counter above, which is logged at DEBUG too.
            logger.debug(
                "Eq. (2)/(1) degenerate (zero aggregate slope): falling back to "
                "S^mid = %.6f (ruling D17)",
                inputs.fallback_mid,
            )
        self._h = root
        return self._h

    def read(self) -> float:
        """The cached estimate, valid for the CURRENT decision time (D1)."""
        return self._h


def solve_clearing_with_hockey_stick(
    inputs: ClearingInputs,
    z_slope: float,
    s_tilde: float,
) -> tuple[float, bool]:
    """Clearing with one one-sided benchmark order z_slope * (p - s_tilde)_+
    on top of the linear aggregate (ruling D16; benchmarks only liquidate).

    Two-case solve: root of the linear form excluding the benchmark order; if
    it is <= s_tilde it stands (the benchmark contributes nothing there),
    otherwise re-solve with the benchmark slope included. The LHS stays
    continuous and nondecreasing in p, so the two cases are exhaustive and
    consistent. Returns ``(p*, degenerate)`` like ``solve_linear_clearing``.
    """
    if z_slope < 0.0:
        raise ValueError(f"benchmark slope must be >= 0, got {z_slope}")
    root, degenerate = solve_linear_clearing(inputs)
    if not degenerate and root <= s_tilde:
        return root, False
    den = float(np.sum(inputs.K_exo)) + float(np.sum(inputs.K_agent)) + z_slope
    if den <= _SLOPE_EPS:
        return inputs.fallback_mid, True
    num = (
        float(inputs.K_exo @ inputs.S_exo)
        + float(inputs.K_agent @ inputs.S_agent)
        + z_slope * s_tilde
        + inputs.net_market_volume
    )
    root_with = num / den
    if root_with >= s_tilde:
        return root_with, False
    # Phi(s_tilde) > 0 >= Phi(linear root): with the linear part degenerate
    # (den - z_slope == 0) and net demand short of the benchmark's kink, no
    # root exists above s_tilde and the equation is flat below it (D17).
    return inputs.fallback_mid, True


def solve_monotone_clearing(
    excess_supply: Callable[[float], float],
    bracket: tuple[float, float],
    tol: float = 1e-10,
    max_widen: int = 200,
) -> float:
    """Bracketed root-finder for general monotone supply curves
    (Theorem `th:clearing` existence; bisection on a widening bracket).

    ``excess_supply`` must be continuous and nondecreasing in p. The initial
    ``bracket`` is widened geometrically until it straddles a sign change.
    """
    lo, hi = float(bracket[0]), float(bracket[1])
    if lo >= hi:
        raise ValueError(f"invalid bracket {bracket!r}")
    f_lo, f_hi = excess_supply(lo), excess_supply(hi)
    width = hi - lo
    n = 0
    while f_lo > 0.0 and n < max_widen:
        lo -= width
        width *= 2.0
        f_lo = excess_supply(lo)
        n += 1
    while f_hi < 0.0 and n < max_widen:
        hi += width
        width *= 2.0
        f_hi = excess_supply(hi)
        n += 1
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError("no sign change found; excess supply has no root")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if excess_supply(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
