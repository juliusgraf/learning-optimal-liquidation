"""Algorithm 1 tests for pre-action exogenous snapshots and final replacement."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import load_synthetic_cfg
from lmm.market.clearing import Algo1Estimator
from lmm.market.clob import BookSnapshot


@pytest.fixture(scope="module")
def cfg():
    return load_synthetic_cfg("grid.alpha=0.01")  # fixed 10,000-tick examples below


def snap(k_mid, ask=(), bid=(), *, agent_level=None, agent_remaining=0.0):
    asks = np.zeros(12)
    bids = np.zeros(12)
    asks[: len(ask)] = ask
    bids[: len(bid)] = bid
    return BookSnapshot(
        k_mid,
        asks,
        bids,
        agent_level=agent_level,
        agent_remaining=agent_remaining,
    )


def test_h0_is_explicit_and_observations_start_at_one(cfg):
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset(99.5)
    assert est.h == 99.5
    with pytest.raises(ValueError, match="observations start at i=1"):
        est.observe(0, snap(10_000, [1.0]))


def test_current_levels_use_zero_filled_decision_index_moments(cfg):
    alpha = cfg.grid.alpha
    eta = cfg.algo1.eta_H
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset(100.0)

    first = est.observe(1, snap(10_000, [4.0], [2.0]))
    # Ask index 0 is tick k_mid+1; bid index 0 is tick k_mid-1.
    expected_price = (4.0 * 100.01 + 2.0 * 99.99) / 6.0
    assert first.K_hat == pytest.approx({9_999: 2.0 / alpha, 10_001: 4.0 / alpha})
    assert first.H == pytest.approx(100.0 + eta * (expected_price - 100.0))

    second = est.observe(2, snap(10_000, [6.0]))
    # Algorithm 1 iterates only over K_i, the levels present now.  The absent
    # bid level contributes zero to moments but is not a candidate at i=2.
    e = (4.0 + 6.0) / 2.0
    second_moment = (4.0**2 + 6.0**2) / 2.0
    k = (2.0 * e - second_moment / e) / alpha
    assert second.e_hat == pytest.approx({10_001: e})
    assert second.varsigma_hat == pytest.approx({10_001: second_moment})
    assert second.K_hat == pytest.approx({10_001: k})
    assert second.s_tilde == pytest.approx(100.01)


def test_strategic_remainder_is_structurally_ignored(cfg):
    base = snap(10_000, [4.0], [3.0])
    with_agent = snap(
        10_000,
        [4.0],
        [3.0],
        agent_level=12,
        agent_remaining=1_000.0,
    )
    a = Algo1Estimator(cfg.algo1, cfg.grid)
    b = Algo1Estimator(cfg.algo1, cfg.grid)
    a.reset()
    b.reset()
    assert a.observe(1, base) == b.observe(1, with_agent)


def test_zero_slope_or_empty_current_book_skips_smoothing(cfg):
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset(100.0)
    est.observe(1, snap(10_000, [5.0]))
    est.observe(2, snap(10_000, [5.0]))
    before = est.h
    clamps = est.n_khat_clamped
    # A new tick seen once at i=3 has raw slope v*(2-i)/(i*alpha)<0.
    third = est.observe(3, snap(10_010, [5.0]))
    assert third.K_hat[10_011] == 0.0
    assert third.H == before
    assert est.n_khat_clamped == clamps + 1

    fourth = est.observe(4, snap(10_000))
    assert fourth.H == before
    assert est.n_zero_slope_skips >= 2


def test_observations_must_be_sequential(cfg):
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset()
    with pytest.raises(ValueError, match="expected 1, got 2"):
        est.observe(2, snap(10_000, [1.0]))


def test_final_residual_replaces_O_n_and_retains_earlier_history(cfg):
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset()
    est.observe(1, snap(10_000, [4.0]))
    est.observe(2, snap(10_000, [100.0, 8.0]))
    out = est.final_replacement_calibration(snap(10_000, [2.0]), n=2)
    e = 3.0
    second_moment = 10.0
    expected_k = (2.0 * e - second_moment / e) / cfg.grid.alpha
    assert out.Q_star == {10_001: 2.0}
    assert out.K_hat == pytest.approx({10_001: expected_k})
    assert 10_002 not in out.K_hat
