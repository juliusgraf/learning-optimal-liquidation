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


def normalized_for(order, cfg):
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
            order.ell / cfg.actions.B_max,
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
        if discrete.phase == "clob":
            volume = 4.0 if discrete.inventory >= 4.0 else 0.0
            order = ClobAction(volume, 3 if volume else 0)
        else:
            cancel = int(discrete.cancel_admissible)
            order = AuctionAction(
                2.0 * cfg.actions.beta,
                3,
                cancel,
            )
        obs_d, reward_d, done_d, _, info_d = discrete.step(order)
        obs_c, reward_c, done_c, _, info_c = relaxed.step(
            normalized_for(order, cfg)
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
    assert order.ell == 1
    assert order.cancel == 0  # threshold positive, but cancellation is not admissible yet
    assert diagnostics["cancel_threshold_positive"]
    assert not diagnostics["cancel_executed"]
    np.testing.assert_allclose(committed, [0.0, 0.1, 1.0], rtol=0.0, atol=2e-8)

    env.step(np.array([0.0, 0.0, -1.0]))
    stacked, _, stacked_diag = env._project_auction(np.array([0.0, 0.0, -1.0]))
    assert stacked.K_a > 0.0 and stacked.cancel == 0
    assert not stacked_diag["cancel_executed"]
    order, _, diagnostics = env._project_auction(np.array([0.0, 0.0, 0.0]))
    assert order.cancel == 1
    assert diagnostics["cancel_executed"]


def test_zero_slope_has_canonical_zero_offset(cfg):
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    order, _, _ = env._project_auction(np.array([-1.0, 1.0, -1.0]))
    assert order == AuctionAction(0.0, 0, 0)


def test_continuous_auction_action_remains_local_when_indicative_price_moves():
    cfg = load_synthetic_cfg(
        "auction_flow.B_inf=150",
        "actions.B_inf=150",
        "actions.B_max=10",
    )
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    env.env._h_cache = env.s_mid + 37 * cfg.grid.alpha
    order, _, _ = env._project_auction(np.array([0.0, 0.5, -1.0]))
    assert order.ell == 5
    env.env._h_cache = env.s_mid - 41 * cfg.grid.alpha
    shifted_h_order, _, _ = env._project_auction(np.array([0.0, 0.5, -1.0]))
    assert shifted_h_order.ell == 5


def test_continuous_local_ell_resolves_to_absolute_b_inside_environment():
    cfg = load_synthetic_cfg(
        "actions.B_inf=150",
        "actions.B_max=10",
    )
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    env.env._h_cache = env.s_mid + 37 * cfg.grid.alpha
    order, _, _ = env._project_auction(np.array([0.0, 0.5, -1.0]))
    assert order.ell == 5
    _, _, _, _, info = env.step(np.array([0.0, 0.5, -1.0]))
    assert info["projected_action_five"][3] == 5
    assert info["action"].ell == 5
    assert info["executed_b"] == 42


def test_continuous_h_off_projection_uses_frozen_mid_anchor():
    cfg = load_synthetic_cfg(
        "rl.h_cl_feature_enabled=false",
        "actions.auction_anchor=frozen_mid",
        "auction_flow.B_inf=10",
        "actions.B_inf=10",
        "actions.B_max=10",
    )
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    env.env._h_cache = env.s_mid + 28 * cfg.grid.alpha

    # Under indicative centering this proposal would lie beyond B_inf. H-off
    # retains ell=10 because its anchor is the frozen-mid coordinate b=0.
    order, _, diagnostics = env._project_auction(
        np.array([0.0, 1.0, -1.0])
    )
    assert order.ell == 10
    assert not diagnostics["ell_admissibility_projection"]
    assert diagnostics["auction_anchor"] == "frozen_mid"
    assert diagnostics["auction_anchor_b"] == 0

    _, _, _, _, info = env.step(np.array([0.0, 1.0, -1.0]))
    assert info["executed_b"] == 10
    assert info["auction_anchor"] == "frozen_mid"
    assert info["auction_anchor_b"] == 0


def test_projection_uses_zero_slope_when_anchor_has_no_local_admissible_price():
    cfg = load_synthetic_cfg(
        "auction_flow.B_inf=10",
        "actions.B_inf=10",
        "actions.B_max=10",
    )
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    env.env._h_cache = env.s_mid + 25 * cfg.grid.alpha

    order, committed, diagnostics = env._project_auction(
        np.array([0.0, 0.0, -1.0])
    )
    assert order == AuctionAction(0.0, 0, 0)
    np.testing.assert_allclose(committed, [0.0, 0.0, -1.0])
    assert diagnostics["slope_admissibility_projection"]
    positive_slope = np.array(
        [a.K_a > 0.0 for a in env.env.auction_grid.actions], dtype=bool
    )
    assert not env.action_mask()[positive_slope].any()
    projected_index = env.env.auction_grid.actions.index(order)
    assert env.action_mask()[projected_index]

    # Cancellation remains an independent admissible coordinate. Create a live
    # schedule under an ordinary anchor, then move the anchor outside the band.
    env.env._h_cache = env.s_mid
    env.step(np.array([0.0, 0.0, -1.0]))
    assert env.env.cancel_admissible
    env.env._h_cache = env.s_mid - 25 * cfg.grid.alpha
    cancel_only, _, cancel_diagnostics = env._project_auction(
        np.array([0.0, 0.0, 1.0])
    )
    assert cancel_only == AuctionAction(0.0, 0, 1)
    assert cancel_diagnostics["slope_admissibility_projection"]


@pytest.mark.parametrize("anchor_sign", [-1, 1])
def test_projection_retains_the_single_boundary_quote(anchor_sign):
    cfg = load_synthetic_cfg(
        "auction_flow.B_inf=10",
        "actions.B_inf=10",
        "actions.B_max=10",
    )
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    # At |anchor|=B_inf+B_max, exactly one local offset remains admissible.
    env.env._h_cache = env.s_mid + anchor_sign * 20 * cfg.grid.alpha
    order, _, diagnostics = env._project_auction(
        np.array([0.0, float(anchor_sign), -1.0])
    )
    assert order.K_a > 0.0
    assert order.ell == -anchor_sign * cfg.actions.B_max
    assert not diagnostics["slope_admissibility_projection"]
    assert diagnostics["ell_admissibility_projection"]
    projected_index = env.env.auction_grid.actions.index(order)
    assert env.action_mask()[projected_index]


def test_no_cancel_projection_falls_back_to_no_order_at_extreme_anchor():
    cfg = load_synthetic_cfg(
        "auction_flow.B_inf=10",
        "actions.B_inf=10",
        "actions.B_max=10",
        "actions.auction_cancel_mode=never",
    )
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    assert env.action_space.shape == (2,)
    env.env._h_cache = env.s_mid - 21 * cfg.grid.alpha
    order, committed, diagnostics = env._project_auction(np.array([0.0, 0.0]))
    assert order == AuctionAction(0.0, 0, 0)
    np.testing.assert_allclose(committed, [0.0, 0.0])
    assert diagnostics["slope_admissibility_projection"]
    assert env.action_mask()[env.env.auction_grid.actions.index(order)]


def test_discrete_and_continuous_agents_share_full_slope_and_local_offset_grids(cfg):
    raw = new_env(cfg)
    dqn_offsets = sorted(
        {a.ell for a in raw.auction_grid.actions if a.K_a > 0.0}
    )
    dqn_slopes = sorted({a.K_a for a in raw.auction_grid.actions})
    assert dqn_offsets == list(range(-10, 11))
    assert dqn_slopes == [float(k) for k in range(33)]

    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    continuous_offsets = sorted(
        {
            env._project_auction(np.array([0.0, ell / 10.0, -1.0]))[0].ell
            for ell in range(-10, 11)
        }
    )
    continuous_slopes = sorted(
        {
            env._project_auction(
                np.array([-1.0 + 2.0 * k / 32.0, 0.0, -1.0])
            )[0].K_a
            for k in range(33)
        }
    )
    assert continuous_offsets == list(range(-10, 11))
    assert continuous_slopes == [float(k) for k in range(33)]


def test_no_cancel_treatment_uses_two_dimensional_auction_proposal():
    cfg = load_synthetic_cfg(
        "actions.auction_cancel_mode=never",
    )
    specs = continuous_action_specs(cfg)
    assert specs["clob"].dim == 2 and specs["auction"].dim == 2
    env = ContinuousActionAdapter(new_env(cfg))
    drive_to_auction(env)
    order, committed, diagnostics = env._project_auction(np.array([0.0, 0.0]))
    assert order.cancel == 0 and committed.shape == (2,)
    assert not diagnostics["cancel_threshold_positive"]


def test_no_auction_treatment_can_terminate_from_clob_phase():
    cfg = load_synthetic_cfg(
        "experiment.auction_enabled=false",
        "rl.h_cl_feature_enabled=false",
        "actions.auction_anchor=frozen_mid",
        "reward.shaping_enabled=false",
    )
    env = ContinuousActionAdapter(new_env(cfg))
    obs, _ = env.reset(seed=260831)
    h_index = cfg.features.clob.index("h_cl")
    assert obs[h_index] == 0.0

    terminated = False
    while not terminated:
        assert env.phase == "clob"
        obs, _, terminated, truncated, info = env.step(
            np.array([-1.0, -1.0], dtype=np.float32)
        )
        assert not truncated
        assert obs[h_index] == 0.0
        assert info["clob_shaping_adjustment"] == 0.0

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
