"""Rough Heston mid-price tests (`sub:rough`; ruling D14).

The vectorized implementation may only restate the legacy recursion with
precomputed kernel weights; ``test_vectorized_equals_naive`` runs both the
naive per-step loop (kept as the private reference implementation) and the
vectorized version on seeded paths and asserts allclose.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lmm.config import RoughHestonParams
from lmm.market.midprice import RoughHestonMidPrice

from helpers import load_synthetic_cfg


@pytest.fixture(scope="module")
def cfg():
    return load_synthetic_cfg()


def fractional_grid(rng: np.random.Generator, n: int, t_max: float) -> list[float]:
    """A strictly increasing grid with fractional spacings (like the
    lambda0 = 1 event-driven decision grid)."""
    steps = rng.uniform(0.3, 2.5, size=n)
    grid = np.cumsum(steps)
    return list(grid[grid <= t_max])


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_vectorized_equals_naive(cfg, seed):
    """Ruling D14: the vectorization is exact (same draws, same arithmetic)."""
    p, g = cfg.midprice.rough_heston, cfg.grid
    vec = RoughHestonMidPrice(p, g, method="vectorized")
    naive = RoughHestonMidPrice(p, g, method="naive")

    grid = fractional_grid(np.random.default_rng(seed + 1000), 200, g.tau_cl)
    vec.reset(np.random.default_rng(seed))
    naive.reset(np.random.default_rng(seed))

    mids_v = [vec.advance_to(t) for t in grid]
    mids_n = [naive.advance_to(t) for t in grid]

    np.testing.assert_allclose(mids_v, mids_n, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(vec.variance_path, naive.variance_path, rtol=1e-12, atol=0.0)


def test_first_steps_match_independent_reference(cfg):
    """Two hand-rolled Euler steps of the Richard et al. scheme (independent
    of both implementations): kernel K(u) = u^(H-1/2)/Gamma(H+1/2), (V)_+
    truncation, correlated log-price update, physical-time scaling."""
    p, g = cfg.midprice.rough_heston, cfg.grid
    model = RoughHestonMidPrice(p, g)
    model.reset(np.random.default_rng(42))
    m1 = model.advance_to(1.0)
    m2 = model.advance_to(2.5)

    # Reference: replay the same Gaussian stream.
    rng = np.random.default_rng(42)
    dt_unit = 1.0 / p.time_units_per_year
    gamma_const = 1.0 / math.gamma(p.H + 0.5)

    def kernel(u):
        return gamma_const * u ** (p.H - 0.5)

    Y = math.log(g.S0)
    times, V, dW = [0.0], [p.v0], []
    for t_grid_prev, t_grid in [(0.0, 1.0), (1.0, 2.5)]:
        dt = (t_grid - t_grid_prev) * dt_unit
        z_v, z_perp = rng.normal(), rng.normal()
        dw_v, dw_perp = math.sqrt(dt) * z_v, math.sqrt(dt) * z_perp
        v_pos = max(V[-1], 0.0)
        if v_pos > 0.0:
            dw_s = p.rho * dw_v + math.sqrt(1.0 - p.rho**2) * dw_perp
            Y += -0.5 * v_pos * dt + math.sqrt(v_pos) * dw_s
        t_new = times[-1] + dt
        times.append(t_new)
        dW.append(dw_v)
        v_new = p.v0
        for i in range(len(times) - 1):
            vi = max(V[i], 0.0)
            v_new += kernel(t_new - times[i]) * (
                (p.theta - p.kappa * vi) * (times[i + 1] - times[i]) + p.xi * math.sqrt(vi) * dW[i]
            )
        V.append(v_new)
    np.testing.assert_allclose([m1, m2][-1], math.exp(Y), rtol=1e-12)
    np.testing.assert_allclose(model.variance_path, V, rtol=1e-12)


def test_draws_consumed_when_variance_truncated(cfg):
    """(V)_+ = 0 skips the price move but still consumes both Gaussians
    (legacy stream stability, AUDIT A.11)."""
    p0 = cfg.midprice.rough_heston
    p = RoughHestonParams(
        H=p0.H,
        rho_h=p0.rho_h,
        v0=0.0,
        theta=p0.theta,
        varsigma=p0.varsigma,
        nu=p0.nu,
        s_star=p0.s_star,
    )
    model = RoughHestonMidPrice(p, cfg.grid)
    rng = np.random.default_rng(5)
    model.reset(rng)
    mid = model.advance_to(1.0)
    # V_0 = 0 => Y unchanged on the first step (mid == exp(log(S0)), the
    # same float round-trip as legacy, which also re-exponentiates).
    assert mid == math.exp(math.log(cfg.grid.S0))
    # Exactly two normals were consumed:
    ref = np.random.default_rng(5)
    ref.normal(), ref.normal()
    assert rng.random() == ref.random()


def test_annualized_time_scaling(cfg):
    """Minute-clock model time uses bar(t)=t/s_star in trading years."""
    p, g = cfg.midprice.rough_heston, cfg.grid
    model = RoughHestonMidPrice(p, g)
    model.reset(np.random.default_rng(0))
    model.advance_to(1.0)
    expected_dt_years = 1.0 / p.time_units_per_year
    assert model._times[1] == pytest.approx(expected_dt_years, rel=1e-15)
    assert cfg.grid.time_unit == "minutes"
    assert p.s_star == pytest.approx(252 * 6.5 * 60)
