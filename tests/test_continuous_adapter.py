"""Projection and discrete-grid equivalence for continuous-control actions."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest

from helpers import load_synthetic_cfg, new_env
from lmm.env.action_spaces import (
    AuctionAction,
    ClobAction,
    ContinuousActionAdapter,
    continuous_action_specs,
)


@pytest.fixture(scope="module")
def cfg():
    return load_synthetic_cfg()


def normalized_for(order, cfg, *, offset_center=0):
    if isinstance(order, ClobAction):
        return np.array(
            [
                2.0 * order.volume / cfg.actions.V_max - 1.0,
                2.0 * order.delta / cfg.actions.L_max - 1.0,
            ],
            dtype=np.float32,
        )
    return np.array(
        [
            2.0 * order.K_a / cfg.actions.auction_K_grid_max - 1.0,
            (order.offset - offset_center) / cfg.actions.auction_template_offset_max,
            1.0 if order.cancel else -1.0,
        ],
        dtype=np.float32,
    )


def drive_to_auction(env, seed=7):
    env.reset(seed=seed)
    while env.phase == "clob":
        env.step(np.array([-1.0, -1.0], dtype=np.float32))


def test_phase_specific_spaces_use_common_18_coordinate_observation(cfg):
    raw = new_env(cfg)
    obs, _ = raw.reset(seed=260828)
    assert isinstance(raw.observation_space, gymnasium.spaces.Box)
    assert obs.dtype == np.float32 and obs.shape == (18,)
    assert isinstance(raw.action_space, gymnasium.spaces.Discrete)

    relaxed = ContinuousActionAdapter(new_env(cfg))
    obs, _ = relaxed.reset(seed=260828)
    assert obs.shape == (18,)
    assert relaxed.action_space.shape == (2,)
    while relaxed.phase == "clob":
        obs, _, _, _, _ = relaxed.step(np.array([-1.0, -1.0]))
    assert obs.shape == (18,)
    assert relaxed.action_space.shape == (3,)


def test_discrete_grid_points_produce_identical_full_episode(cfg):
    discrete = new_env(cfg)
    relaxed = ContinuousActionAdapter(new_env(cfg))
    obs_d, _ = discrete.reset(seed=4242)
    obs_c, _ = relaxed.reset(seed=4242)
    np.testing.assert_array_equal(obs_d, obs_c)

    step = 0
    while True:
        offset_center = 0
        if discrete.phase == "clob":
            volume = 4.0 if discrete.inventory >= 4.0 else 0.0
            order = ClobAction(volume, 3 if volume else 0)
        else:
            cancel = int(discrete.cancel_admissible)
            offset_center = discrete._auction_offset_center_ticks()
            order = AuctionAction(2.0 * cfg.actions.beta, offset_center + 3, cancel)
        obs_d, reward_d, done_d, _, info_d = discrete.step(order)
        obs_c, reward_c, done_c, _, info_c = relaxed.step(
            normalized_for(order, cfg, offset_center=offset_center)
        )
        np.testing.assert_array_equal(obs_d, obs_c)
        assert reward_d == reward_c
        assert done_d == done_c
        if done_d:
            assert info_d["S_cl"] == info_c["S_cl"]
            assert info_d["Z"] == info_c["Z"]
            assert info_d["I_final"] == info_c["I_final"]
            break
        step += 1


def test_clob_projection_snaps_and_logs_inventory_projection(cfg):
    env = ContinuousActionAdapter(new_env(cfg))
    env.reset(seed=1)
    env.env._inventory = 2.5
    order, committed, diagnostics = env._project_clob(np.array([1.5, 0.1]))
    assert order.volume == 2.0  # floor(inventory), after nearest-integer proposal
    assert order.delta == 7
    np.testing.assert_allclose(committed, [1.0, 0.1], rtol=0.0, atol=2e-8)
    assert diagnostics["input_clipped"]
    assert diagnostics["inventory_projection"]

    order, committed, diagnostics = env._project_clob(np.array([-1.0, 1.0]))
    assert order == ClobAction(0.0, 0)  # canonical no-order delta
    assert not diagnostics["input_clipped"]


def test_auction_projection_snaps_slope_offset_and_cancel(cfg):
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    order, committed, diagnostics = env._project_auction(np.array([0.0, 0.1, 1.0]))
    assert order.K_a == pytest.approx(0.5 * cfg.actions.auction_K_grid_max)
    assert order.offset == env.env._auction_offset_center_ticks() + 1
    assert order.cancel == 0  # threshold positive, but cancellation is not admissible yet
    assert diagnostics["cancel_threshold_positive"]
    assert not diagnostics["cancel_executed"]
    np.testing.assert_allclose(committed, [0.0, 0.1, 1.0], rtol=0.0, atol=2e-8)

    env.step(np.array([0.0, 0.0, -1.0]))
    order, _, diagnostics = env._project_auction(np.array([0.0, 0.0, 0.0]))
    assert order.cancel == 1
    assert diagnostics["cancel_executed"]


def test_zero_slope_has_canonical_zero_offset(cfg):
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    order, _, _ = env._project_auction(np.array([-1.0, 1.0, -1.0]))
    assert order == AuctionAction(0.0, 0, 0)


def test_continuous_auction_offset_can_be_centered_on_indicative_price():
    cfg = load_synthetic_cfg(
        "actions.B_max=150",
        "actions.auction_offset_center=indicative",
        "actions.auction_local_offset_max=10",
    )
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    env.env._h_cache = env.s_mid + 37 * cfg.grid.alpha
    order, _, _ = env._project_auction(np.array([0.0, 0.5, -1.0]))
    assert order.offset == 42  # indicative center 37 plus local displacement 5


def test_no_cancel_treatment_uses_two_dimensional_auction_proposal():
    cfg = load_synthetic_cfg(
        "actions.auction_cancel_mode=never",
        "actions.auction_order_mode=multi",
    )
    specs = continuous_action_specs(cfg)
    assert specs["clob"].dim == 2 and specs["auction"].dim == 2
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    order, committed, diagnostics = env._project_auction(np.array([0.0, 0.0]))
    assert order.cancel == 0 and committed.shape == (2,)
    assert not diagnostics["cancel_threshold_positive"]


def test_no_auction_treatment_can_terminate_from_clob_phase():
    cfg = load_synthetic_cfg("experiment.auction_enabled=false")
    env = ContinuousActionAdapter(new_env(cfg))
    env.reset(seed=260831)

    terminated = False
    while not terminated:
        assert env.phase == "clob"
        _, _, terminated, truncated, _ = env.step(
            np.array([-1.0, -1.0], dtype=np.float32)
        )
        assert not truncated

    assert env.phase == "terminal"
    # A terminal state has no next action; retaining the last active box keeps
    # the Gym interface well-defined without inventing a terminal action space.
    assert env.action_space.shape == (2,)


def test_adapter_step_records_raw_committed_and_projected_actions(cfg):
    env = ContinuousActionAdapter(new_env(cfg))
    env.reset(seed=2)
    proposal = np.array([2.0, 0.25], dtype=np.float32)
    _, _, _, _, info = env.step(proposal)
    np.testing.assert_array_equal(info["raw_action_vec"], proposal)
    np.testing.assert_array_equal(info["proposal_action_vec"], [1.0, 0.25])
    assert len(info["projected_action_five"]) == 5
    assert info["projection_diagnostics"]["input_clipped"]
