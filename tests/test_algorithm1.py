"""Algorithm 1 unit tests (`alg:hyp_clearing_price`; rulings D2, D15).

Feeds deterministic sequences of standing books into ``Algo1Estimator`` and
verifies e_hat, sigma_hat, K_hat, S_tilde and the smoothed H against a short
independent NumPy reference computed inside the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from lmm.market.clearing import Algo1Estimator
from lmm.market.clob import BookSnapshot

from helpers import load_synthetic_cfg


@pytest.fixture(scope="module")
def cfg():
    return load_synthetic_cfg()


def snap(k_mid, ask, bid, agent_level=None, agent_remaining=0.0):
    lc = 12
    a = np.zeros(lc)
    b = np.zeros(lc)
    a[: len(ask)] = ask
    b[: len(bid)] = bid
    return BookSnapshot(
        k_mid=k_mid, ask_volumes=a, bid_volumes=b,
        agent_level=agent_level, agent_remaining=agent_remaining,
    )


def reference_h(snapshots, alpha, tau, h0):
    """Independent ~20-line NumPy reference for Algorithm 1 (per-tick
    running moments over snapshots, K_hat clamp, S_tilde, smoothing)."""
    eps = 1e-12
    sums: dict[int, float] = {}
    sums_sq: dict[int, float] = {}
    h = h0
    hs = []
    for i, s in enumerate(snapshots, start=1):
        vol = {}
        for j, v in enumerate(s.ask_volumes):
            if v > eps:
                vol[s.k_mid + j] = vol.get(s.k_mid + j, 0.0) + v
        for j, v in enumerate(s.bid_volumes):
            if v > eps:
                vol[s.k_mid - j] = vol.get(s.k_mid - j, 0.0) + v
        if s.agent_level is not None and s.agent_remaining > eps:
            k = s.k_mid + s.agent_level
            vol[k] = vol.get(k, 0.0) + s.agent_remaining
        for k, v in vol.items():
            sums[k] = sums.get(k, 0.0) + v
            sums_sq[k] = sums_sq.get(k, 0.0) + v * v
        ks = np.array(sorted(sums))
        e = np.array([sums[k] / i for k in ks])
        sig = np.array([sums_sq[k] / i for k in ks])
        k_hat = np.where(e > eps, np.maximum(0.0, (2 * e - sig / np.maximum(e, eps)) / alpha), 0.0)
        if k_hat.sum() > 0:
            s_tilde = float((k_hat * alpha * ks).sum() / k_hat.sum())
            h = h + tau * (s_tilde - h)
        hs.append(h)
    return hs


def test_matches_independent_reference(cfg):
    alpha, tau, h0 = cfg.grid.alpha, cfg.algo1.tau, 100.0
    rng = np.random.default_rng(3)
    snapshots = []
    k_mid = 10_000
    for step in range(25):
        k_mid += int(rng.integers(-2, 3))  # drifting mid tick
        ask = rng.uniform(0.0, 8.0, size=12)
        bid = rng.uniform(0.0, 8.0, size=12)
        agent = (int(rng.integers(0, 13)), float(rng.uniform(0.0, 5.0)))
        snapshots.append(snap(k_mid, ask, bid, agent_level=agent[0], agent_remaining=agent[1]))

    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset(h0)
    hs = [est.update(s) for s in snapshots]
    np.testing.assert_allclose(hs, reference_h(snapshots, alpha, tau, h0), rtol=1e-12)


def test_two_snapshot_moments_by_hand(cfg):
    """Fully hand-computed two-snapshot case (single tick on each side)."""
    alpha, tau = cfg.grid.alpha, cfg.algo1.tau
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset(100.0)
    # Snapshot 1: ask level 0 only (tick 10000, vol 4); bid level 1 (tick 9999, vol 2).
    s1 = snap(10_000, [4.0], [0.0, 2.0])
    h1 = est.update(s1)
    # Both ticks observed once: K_hat = (2v - v^2/v)/alpha = v/alpha.
    k_a, k_b = 4.0 / alpha, 2.0 / alpha
    s_tilde = (k_a * alpha * 10_000 + k_b * alpha * 9_999) / (k_a + k_b)
    assert h1 == pytest.approx(100.0 + tau * (s_tilde - 100.0), rel=1e-14)
    # Snapshot 2: same ask tick with vol 6; bid tick absent (counts as 0).
    h2 = est.update(snap(10_000, [6.0], [0.0]))
    e_a, sig_a = (4 + 6) / 2, (16 + 36) / 2
    e_b, sig_b = 2 / 2, 4 / 2
    k_a = max(0.0, (2 * e_a - sig_a / e_a) / alpha)
    k_b = max(0.0, (2 * e_b - sig_b / e_b) / alpha)
    s_tilde2 = (k_a * alpha * 10_000 + k_b * alpha * 9_999) / (k_a + k_b)
    assert h2 == pytest.approx(h1 + tau * (s_tilde2 - h1), rel=1e-14)


def test_negative_khat_clamped_and_counted(cfg):
    """A tick observed once among n > 2 snapshots has 2 e^2 < sigma, so its
    K_hat clamps to 0 (legacy choice, kept with a logged counter). The count
    is GLOBAL over snapshots (legacy main.py:406-419), so at the third
    snapshot ALL THREE once-observed ticks clamp — including the current
    snapshot's own tick (e = 5/3, sigma = 25/3 => 2e - sigma/e = -5/3 < 0)."""
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset(100.0)
    est.update(snap(10_000, [5.0], []))  # tick 10000 observed
    est.update(snap(10_001, [5.0], []))  # tick 10001; 10000 absent: e=v/2, K_hat=0 exactly
    assert est.n_khat_clamped == 0
    est.update(snap(10_002, [5.0], []))  # n=3: ticks 10000, 10001, 10002 all clamp
    assert est.n_khat_clamped == 3


def test_zero_slope_skips_smoothing(cfg):
    """When sum K_hat = 0 the smoothing update is skipped (H unchanged)."""
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset(100.0)
    # Four snapshots, each with a single never-repeated tick: at update 4
    # every tick has one observation among n >= 3 snapshots => all clamped.
    est.update(snap(10_000, [3.0], []))
    est.update(snap(10_010, [3.0], []))
    est.update(snap(10_020, [3.0], []))
    h3 = est.h
    skips_before = est.n_zero_slope_skips
    h4 = est.update(snap(10_030, [3.0], []))
    assert h4 == h3  # H unchanged
    assert est.n_zero_slope_skips == skips_before + 1


def test_empty_snapshot_counts_and_skips(cfg):
    """An all-empty book increments the snapshot count and skips smoothing."""
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset(100.0)
    h = est.update(snap(10_000, [], []))
    assert h == 100.0
    assert est.n_zero_slope_skips == 1
    assert est._mom_count == 1


def test_agent_remainder_enters_snapshot(cfg):
    """The agent's unexecuted remainder is part of the standing book (D2),
    including at level Lc (one past the exogenous book, AUDIT N6)."""
    est_with = Algo1Estimator(cfg.algo1, cfg.grid)
    est_without = Algo1Estimator(cfg.algo1, cfg.grid)
    est_with.reset(100.0)
    est_without.reset(100.0)
    h_w = est_with.update(snap(10_000, [4.0], [3.0], agent_level=12, agent_remaining=9.0))
    h_wo = est_without.update(snap(10_000, [4.0], [3.0]))
    assert h_w != h_wo
    assert h_w > h_wo  # extra volume above the mid pulls S_tilde up
