"""Statistics helpers for Phase 7 figures and tables.

Bootstrap CIs use a SEEDED numpy generator so figures/tables are reproducible.
Paired significance tests on common-random-number (CRN) returns use scipy
(Wilcoxon signed-rank as the primary test — episode returns are non-Gaussian
because of the terminal inventory penalty — with the paired t-test secondary).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as _sp_stats

__all__ = [
    "bootstrap_ci",
    "bootstrap_cumulative_band",
    "bootstrap_improvement_ci",
    "paired_t",
    "wilcoxon",
    "std_error",
    "rel_improvement",
    "iqm",
    "iqm_ci",
]


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of ``statistic`` over ``values``.

    Returns ``(point, lo, hi)`` where ``point`` is the statistic on the full
    sample and ``[lo, hi]`` is the ``1 - alpha`` percentile interval. Degrades
    gracefully on tiny samples (``n < 2`` => zero-width interval at the point).
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    point = float(statistic(v)) if v.size else float("nan")
    if v.size < 2:
        return point, point, point
    gen = _as_rng(rng)
    idx = gen.integers(0, v.size, size=(n_boot, v.size))
    boot = np.array([statistic(v[row]) for row in idx], dtype=float)
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi


def bootstrap_cumulative_band(
    per_episode: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap band for a CUMULATIVE curve (e.g. cumulative regret).

    For each resample, draw ``E`` per-episode values with replacement and take
    the cumulative sum; the band is the per-index ``1 - alpha`` percentile
    envelope of the resampled cumulative curves. Returns ``(lo, hi)`` arrays of
    length ``E`` (point-wise; the observed cumsum is plotted separately).
    """
    x = np.asarray(per_episode, dtype=float)
    e = x.size
    if e < 2:
        c = np.cumsum(x) if e else np.zeros(0)
        return c.copy(), c.copy()
    gen = _as_rng(rng)
    curves = np.cumsum(x[gen.integers(0, e, size=(n_boot, e))], axis=1)
    lo = np.percentile(curves, 100.0 * alpha / 2.0, axis=0)
    hi = np.percentile(curves, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lo, hi


def bootstrap_improvement_ci(
    algo: np.ndarray,
    bench: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI of the relative improvement (%) of ``algo`` over ``bench``.

    ``algo``/``bench`` are PAIRED (CRN-aligned by episode). Resamples episode
    indices jointly and recomputes ``100*(mean_algo - mean_bench)/|mean_bench|``.
    Returns ``(point, lo, hi)``."""
    a = np.asarray(algo, dtype=float)
    b = np.asarray(bench, dtype=float)
    point = rel_improvement(float(np.mean(a)), float(np.mean(b))) if a.size else float("nan")
    if a.size < 2:
        return point, point, point
    gen = _as_rng(rng)
    idx = gen.integers(0, a.size, size=(n_boot, a.size))
    am = a[idx].mean(axis=1)
    bm = b[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = 100.0 * (am - bm) / np.abs(bm)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(vals, 100.0 * alpha / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t-test of ``a - b`` (scipy ``ttest_rel``).

    Returns ``(statistic, p_value)``; ``(nan, nan)`` if undefined (n < 2 or a
    constant difference)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.allclose(a - b, (a - b)[0]):
        return float("nan"), float("nan")
    res = _sp_stats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on the paired differences.

    Returns ``(statistic, p_value)``; ``(nan, nan)`` if undefined (all
    differences zero, or n too small)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if d.size < 1 or np.all(d == 0.0):
        return float("nan"), float("nan")
    try:
        res = _sp_stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return float("nan"), float("nan")
    return float(res.statistic), float(res.pvalue)


def std_error(values: np.ndarray) -> float:
    """Standard error of the mean (ddof=1)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return float("nan")
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def rel_improvement(policy_mean: float, benchmark_mean: float) -> float:
    """Relative improvement of ``policy`` over ``benchmark`` in percent:
    ``100 * (policy - benchmark) / |benchmark|`` (matches the paper tables)."""
    if benchmark_mean == 0.0 or np.isnan(benchmark_mean):
        return float("nan")
    return 100.0 * (policy_mean - benchmark_mean) / abs(benchmark_mean)


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: the mean after trimming the bottom and top 25%
    (``scipy.stats.trim_mean(v, 0.25)``). The robust central-tendency estimator
    recommended for high-variance RL aggregation (Agarwal et al. 2021,
    ``rliable``). NaNs dropped; small samples degrade to the plain mean (with
    n seeds < 4 no trimming occurs, so IQM == mean — wide CIs are expected)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    return float(_sp_stats.trim_mean(v, 0.25))


def iqm_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = 0,
) -> tuple[float, float, float]:
    """``(iqm, lo, hi)`` percentile-bootstrap CI of the interquartile mean
    across seeds (thin wrapper over :func:`bootstrap_ci` with ``statistic=iqm``)."""
    return bootstrap_ci(values, statistic=iqm, n_boot=n_boot, alpha=alpha, rng=rng)
