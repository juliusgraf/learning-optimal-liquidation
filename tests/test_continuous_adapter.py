"""ContinuousActionAdapter tests (Phase 5; docs/continuous_action_extension.md).

The headline guarantee: a continuous action equal to a discrete grid point
produces EXACTLY the same transition and reward as the raw (discrete) env on
the same seed — i.e. the relaxation only changes the policy class, not the env
dynamics. Plus targeted projection/snapping unit checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import load_synthetic_cfg, new_env
from lmm.env.action_spaces import AuctionAction, ClobAction, ContinuousActionAdapter, continuous_action_specs


@pytest.fixture(scope="module")
def cfg():
    return load_synthetic_cfg()


def _continuous_for(order):
    """The continuous vector equal to a discrete action object's coordinates."""
    if isinstance(order, ClobAction):
        return np.array([order.volume, float(order.delta)], dtype=np.float64)
    return np.array([order.K_a, float(order.offset), 0.9 if order.cancel else 0.1])


# -- grid-point equivalence (the headline guarantee) --------------------------


def test_continuous_grid_point_matches_discrete_transition_exactly(cfg):
    """Drive a raw env and an adapter-wrapped env in lockstep with the SAME
    admissible grid actions (continuous coordinates = the grid point); assert
    bit-identical obs, reward and terminal clearing for a full episode,
    including a cancel-all after a live auction order."""
    env_d = new_env(cfg)
    env_c = ContinuousActionAdapter(new_env(cfg), continuous_cancel="threshold")
    obs_d, _ = env_d.reset(seed=4242)
    obs_c, _ = env_c.reset(seed=4242)
    assert np.array_equal(obs_d, obs_c)

    done = False
    step = 0
    info_d = info_c = None
    while not done:
        if env_d.phase == "clob":
            # Admissible grid action (guard against late-episode low inventory).
            v = 4.0 if env_d.inventory >= 4.0 else 0.0
            order = ClobAction(v, 3)
        else:
            cancel = 0 if step % 4 == 0 else (1 if env_d.action_mask().all() else 0)
            order = AuctionAction(2.0, 3, cancel)
        obs_d, r_d, term_d, _, info_d = env_d.step(order)
        obs_c, r_c, term_c, _, info_c = env_c.step(_continuous_for(order))
        assert np.array_equal(obs_d, obs_c), f"obs differ at step {step}"
        assert r_d == r_c, f"reward differ at step {step}: {r_d!r} vs {r_c!r}"
        assert term_d == term_c
        done = term_d
        step += 1

    assert info_d["S_cl"] == info_c["S_cl"]
    assert info_d["Z"] == info_c["Z"]
    assert info_d["I_final"] == info_c["I_final"]
    assert info_d["terminal_reward"] == info_c["terminal_reward"]


# -- CLOB projection / snapping ------------------------------------------------


def test_clob_projection_and_delta_snapping(cfg):
    env = ContinuousActionAdapter(new_env(cfg))
    env.reset(seed=1)
    inv = env.inventory
    # volume projected to [0, min(V, inv)]; delta snapped by rounding, clipped.
    order, committed = env._project_clob(np.array([1000.0, 3.4]))
    assert order.volume == pytest.approx(min(cfg.actions.clob_volume_max, inv))
    assert order.delta == 3  # round(3.4)
    assert committed[0] == pytest.approx(order.volume)
    assert committed[1] == pytest.approx(3.4)  # committed stores the UNSNAPPED delta

    order, committed = env._project_clob(np.array([-5.0, 2.6]))
    assert order.volume == 0.0  # clipped at 0
    assert order.delta == 3  # round(2.6)
    # delta below the grid min is clipped up to delta_min in BOTH order and committed
    order, committed = env._project_clob(np.array([2.0, -7.0]))
    assert order.delta == cfg.actions.clob_delta_min
    assert committed[1] == pytest.approx(float(cfg.actions.clob_delta_min))


# -- auction projection / snapping --------------------------------------------


def _drive_to_auction(env, seed=7):
    env.reset(seed=seed)
    while env.phase == "clob":
        env.step(np.array([0.0, 1.0]))  # NOOP CLOB through the adapter


def test_auction_K_continuous_offset_snapped_cancel_threshold(cfg):
    env = ContinuousActionAdapter(new_env(cfg), continuous_cancel="threshold")
    _drive_to_auction(env)
    # K is continuous (not snapped); offset snapped; cancel inadmissible at open.
    order, committed = env._project_auction(np.array([1.234, 2.6, 0.9]))
    assert order.K_a == pytest.approx(1.234)  # NOT snapped to the grid
    assert order.offset == 3  # round(2.6)
    assert order.cancel == 0  # c_logit > 0.5 but cancel inadmissible at the open
    assert committed.shape == (3,)
    assert committed[0] == pytest.approx(1.234) and committed[2] == pytest.approx(0.9)
    # K clipped to [0, K_max]
    order, _ = env._project_auction(np.array([-1.0, 0.0, 0.0]))
    assert order.K_a == 0.0
    order, _ = env._project_auction(np.array([1e9, 0.0, 0.0]))
    assert order.K_a == pytest.approx(cfg.actions.auction_K_grid_max)


def test_auction_cancel_threshold_admissible_after_live_order(cfg):
    env = ContinuousActionAdapter(new_env(cfg), continuous_cancel="threshold")
    _drive_to_auction(env)
    env.step(np.array([2.0, 3.0, 0.1]))  # submit a live K>0 order (no cancel)
    # now a cancel-all is admissible: c_logit > 0.5 -> cancel = 1
    order, _ = env._project_auction(np.array([2.0, 3.0, 0.9]))
    assert order.cancel == 1
    order, _ = env._project_auction(np.array([2.0, 3.0, 0.3]))
    assert order.cancel == 0  # below threshold


def test_continuous_cancel_never_drops_cancel_dim(cfg):
    specs = continuous_action_specs(cfg, "never")
    assert specs["auction"].dim == 2 and specs["clob"].dim == 2
    env = ContinuousActionAdapter(new_env(cfg), continuous_cancel="never")
    _drive_to_auction(env)
    env.step(np.array([2.0, 3.0]))  # live order; cancel never available
    order, committed = env._project_auction(np.array([5.0, 1.0]))
    assert order.cancel == 0 and committed.shape == (2,)
    assert env.action_space.shape == (2,)


def test_offset_snapping_anchors_on_frozen_mid(cfg):
    """The executed quote S^a = alpha*(floor(S_mid_frozen/alpha) + round(off))
    matches the raw env's tick-snapping on the same offset."""
    env = ContinuousActionAdapter(new_env(cfg), continuous_cancel="threshold")
    _drive_to_auction(env)
    k_bar = math.floor(env.s_mid / cfg.grid.alpha)  # adapter delegates s_mid
    _, _, _, _, info = env.step(np.array([2.0, 4.3, 0.1]))
    assert info["S_a"] == pytest.approx(cfg.grid.alpha * (k_bar + 4))  # round(4.3) = 4
