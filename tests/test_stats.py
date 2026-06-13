"""Phase 7 statistics helpers (lmm.experiments.stats)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sp

from lmm.experiments import stats


def test_bootstrap_ci_brackets_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=200)
    point, lo, hi = stats.bootstrap_ci(x, rng=123)
    assert math.isclose(point, float(np.mean(x)))
    assert lo <= point <= hi
    # Same seed => identical interval (figures/tables reproducibility).
    assert stats.bootstrap_ci(x, rng=123) == (point, lo, hi)


def test_bootstrap_ci_degenerate_small_sample():
    point, lo, hi = stats.bootstrap_ci(np.array([3.0]))
    assert point == lo == hi == 3.0


def test_bootstrap_cumulative_band_shape_and_monotone_width():
    per_ep = np.array([1.0, -2.0, 3.0, 0.5, -1.0, 2.0])
    lo, hi = stats.bootstrap_cumulative_band(per_ep, n_boot=500, rng=0)
    assert lo.shape == hi.shape == per_ep.shape
    assert np.all(hi >= lo - 1e-9)


def test_paired_t_matches_scipy_and_detects_difference():
    a = np.array([10.0, 12.0, 9.0, 11.0, 13.0, 8.0, 14.0, 10.5])
    b = a - 3.0 + np.array([0.1, -0.1, 0.2, 0.0, -0.2, 0.1, -0.1, 0.0])
    stat, p = stats.paired_t(a, b)
    ref = sp.ttest_rel(a, b)
    assert math.isclose(stat, float(ref.statistic), rel_tol=1e-12)
    assert math.isclose(p, float(ref.pvalue), rel_tol=1e-12)
    assert p < 0.05  # a is clearly larger than b


def test_wilcoxon_matches_scipy():
    a = np.array([10.0, 12.0, 9.0, 11.0, 13.0, 8.0, 14.0, 10.5])
    b = np.array([7.0, 9.0, 8.0, 8.0, 11.0, 6.0, 12.0, 9.0])
    stat, p = stats.wilcoxon(a, b)
    ref = sp.wilcoxon(a, b, zero_method="wilcox")
    assert math.isclose(stat, float(ref.statistic), rel_tol=1e-12)
    assert math.isclose(p, float(ref.pvalue), rel_tol=1e-12)


def test_tests_return_nan_on_degenerate_input():
    z = np.zeros(5)
    assert all(math.isnan(v) for v in stats.wilcoxon(z, z))
    assert all(math.isnan(v) for v in stats.paired_t(np.array([1.0]), np.array([0.0])))


def test_rel_improvement():
    assert stats.rel_improvement(150.0, 100.0) == pytest.approx(50.0)
    assert stats.rel_improvement(50.0, 100.0) == pytest.approx(-50.0)
    assert math.isnan(stats.rel_improvement(1.0, 0.0))


def test_bootstrap_improvement_ci_point_and_bracket():
    rng = np.random.default_rng(1)
    bench = rng.normal(100.0, 5.0, size=80)
    algo = bench + rng.normal(20.0, 5.0, size=80)
    point, lo, hi = stats.bootstrap_improvement_ci(algo, bench, rng=0)
    assert math.isclose(point, stats.rel_improvement(float(np.mean(algo)), float(np.mean(bench))))
    assert lo <= point <= hi
    assert lo > 0.0  # algo clearly beats bench
