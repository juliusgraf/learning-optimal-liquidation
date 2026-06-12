"""Clearing-solver tests (Theorem `th:clearing`, corrected Prop. linear,
rulings D16, D17).

The linear closed form must agree with the bracketed general monotone
root-finder to 1e-9 on random instances; the benchmarks' one-sided
hockey-stick order (D16) is solved by the documented two-case method and
checked against the general solver; all degenerate cases fall back to
S^mid (D17) with a logged counter.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from lmm.config import GridParams
from lmm.market.clearing import (
    ClearingInputs,
    Eq2Cache,
    solve_clearing_with_hockey_stick,
    solve_linear_clearing,
    solve_monotone_clearing,
)

FALLBACK = 100.0


def random_inputs(rng: np.random.Generator, net: float | None = None) -> ClearingInputs:
    n_exo, n_agent = int(rng.integers(1, 7)), int(rng.integers(0, 4))
    return ClearingInputs(
        K_exo=rng.uniform(0.1, 5.0, n_exo),
        S_exo=rng.uniform(95.0, 105.0, n_exo),
        K_agent=rng.uniform(0.1, 5.0, n_agent),
        S_agent=rng.uniform(95.0, 105.0, n_agent),
        net_market_volume=float(rng.uniform(-50.0, 50.0)) if net is None else net,
        fallback_mid=FALLBACK,
    )


def excess_supply(inputs: ClearingInputs, z: float = 0.0, s_tilde: float = 0.0):
    """Phi(p) for the general solver, built independently of the closed form."""

    def phi(p: float) -> float:
        out = float(np.sum(inputs.K_exo * (p - inputs.S_exo)))
        out += float(np.sum(inputs.K_agent * (p - inputs.S_agent)))
        out += z * max(p - s_tilde, 0.0)
        return out - inputs.net_market_volume

    return phi


# ---------------------------------------------------------------------------
# Linear closed form == general monotone solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(25))
def test_linear_equals_general_solver(trial):
    rng = np.random.default_rng(500 + trial)
    inputs = random_inputs(rng)
    p_lin, degenerate = solve_linear_clearing(inputs)
    assert not degenerate
    p_gen = solve_monotone_clearing(excess_supply(inputs), bracket=(90.0, 110.0))
    assert abs(p_lin - p_gen) < 1e-9
    # The root really clears the market.
    assert excess_supply(inputs)(p_lin) == pytest.approx(0.0, abs=1e-9)


def test_general_solver_widens_bracket():
    """Roots far outside the initial bracket are still found (geometric
    widening; Theorem th:clearing only needs monotone continuity)."""
    inputs = ClearingInputs(
        K_exo=np.array([0.5]),
        S_exo=np.array([100.0]),
        K_agent=np.array([]),
        S_agent=np.array([]),
        net_market_volume=200.0,  # root at 500
        fallback_mid=FALLBACK,
    )
    p = solve_monotone_clearing(excess_supply(inputs), bracket=(99.0, 101.0))
    assert p == pytest.approx(500.0, abs=1e-8)


def test_general_solver_rejects_bad_bracket():
    with pytest.raises(ValueError, match="invalid bracket"):
        solve_monotone_clearing(lambda p: p, bracket=(1.0, 1.0))


# ---------------------------------------------------------------------------
# D16: one-sided hockey-stick benchmark order, two-case solve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(25))
def test_hockey_stick_equals_general_solver(trial):
    rng = np.random.default_rng(900 + trial)
    inputs = random_inputs(rng)
    z = float(rng.uniform(0.5, 20.0))
    s_tilde = float(rng.uniform(95.0, 105.0))
    p_two_case, degenerate = solve_clearing_with_hockey_stick(inputs, z, s_tilde)
    assert not degenerate
    p_gen = solve_monotone_clearing(
        excess_supply(inputs, z=z, s_tilde=s_tilde), bracket=(90.0, 110.0)
    )
    assert abs(p_two_case - p_gen) < 1e-9


def test_hockey_stick_inactive_when_root_below_kink():
    rng = np.random.default_rng(1)
    inputs = random_inputs(rng)
    p_lin, _ = solve_linear_clearing(inputs)
    p, _ = solve_clearing_with_hockey_stick(inputs, z_slope=10.0, s_tilde=p_lin + 5.0)
    assert p == p_lin  # the benchmark order contributes nothing below its kink


def test_hockey_stick_active_lowers_price_above_kink():
    rng = np.random.default_rng(2)
    inputs = random_inputs(rng, net=40.0)
    p_lin, _ = solve_linear_clearing(inputs)
    s_tilde = p_lin - 1.0  # kink strictly below the linear root
    p, _ = solve_clearing_with_hockey_stick(inputs, z_slope=10.0, s_tilde=s_tilde)
    assert s_tilde <= p < p_lin  # extra supply above the kink lowers p*


def test_hockey_stick_alone_clears_positive_demand():
    """Linear part empty (degenerate alone) but the benchmark order absorbs
    the positive net demand: p* = s_tilde + net/z, no fallback."""
    inputs = ClearingInputs(
        K_exo=np.array([]),
        S_exo=np.array([]),
        K_agent=np.array([]),
        S_agent=np.array([]),
        net_market_volume=20.0,
        fallback_mid=FALLBACK,
    )
    p, degenerate = solve_clearing_with_hockey_stick(inputs, z_slope=4.0, s_tilde=100.0)
    assert not degenerate
    assert p == pytest.approx(105.0, rel=1e-12)


def test_hockey_stick_flat_below_kink_falls_back():
    """Linear part empty and net demand NEGATIVE: Phi is flat (= -net > 0
    is wrong side) below the kink, no root exists => S^mid fallback (D17)."""
    inputs = ClearingInputs(
        K_exo=np.array([]),
        S_exo=np.array([]),
        K_agent=np.array([]),
        S_agent=np.array([]),
        net_market_volume=-5.0,
        fallback_mid=FALLBACK,
    )
    p, degenerate = solve_clearing_with_hockey_stick(inputs, z_slope=4.0, s_tilde=100.0)
    assert degenerate and p == FALLBACK


def test_hockey_stick_zero_slope_is_linear():
    rng = np.random.default_rng(3)
    inputs = random_inputs(rng)
    p_lin, _ = solve_linear_clearing(inputs)
    p, _ = solve_clearing_with_hockey_stick(inputs, z_slope=0.0, s_tilde=90.0)
    assert p == p_lin


def test_hockey_stick_rejects_negative_slope():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="benchmark slope"):
        solve_clearing_with_hockey_stick(random_inputs(rng), z_slope=-1.0, s_tilde=100.0)


# ---------------------------------------------------------------------------
# D17: degenerate fallback = S^mid, counted and logged
# ---------------------------------------------------------------------------


def degenerate_inputs() -> ClearingInputs:
    return ClearingInputs(
        K_exo=np.array([]),
        S_exo=np.array([]),
        K_agent=np.array([]),
        S_agent=np.array([]),
        net_market_volume=0.0,
        fallback_mid=FALLBACK,
    )


def test_linear_degenerate_falls_back_to_mid():
    p, degenerate = solve_linear_clearing(degenerate_inputs())
    assert degenerate and p == FALLBACK


def test_eq2_cache_counts_and_logs_fallback(synthetic_cfg, caplog):
    cache = Eq2Cache(synthetic_cfg.grid)
    cache.reset(101.0)
    with caplog.at_level(logging.WARNING, logger="lmm.market.clearing"):
        h = cache.recompute(degenerate_inputs())
    assert h == FALLBACK
    assert cache.read() == FALLBACK
    assert cache.n_degenerate_fallbacks == 1
    assert any("D17" in rec.getMessage() for rec in caplog.records)
