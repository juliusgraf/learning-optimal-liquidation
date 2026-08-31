"""Regression tests for the manuscript's unit Bellman factor."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import load_dqn_cfg
from lmm.experiments.train import make_agent
from lmm.rl.loops import _transition_discount
from lmm.rl.replay import ReplayBatch
from lmm.utils.seeding import seed_everything
from lmm.rl.loops import SEED_COMPONENTS


def test_transition_discount_is_one_for_every_legacy_mode():
    assert _transition_discount(0.99, "elapsed_time", 2.0, 4.5) == 1.0
    assert _transition_discount(0.25, "per_transition", 2.0, 4.5) == 1.0
    assert _transition_discount(1.0, "ignored", 2.0, 100.0) == 1.0
    with pytest.raises(ValueError, match="must increase"):
        _transition_discount(0.99, "elapsed_time", 2.0, 2.0)


def test_dqn_target_ignores_deprecated_row_discount():
    cfg = load_dqn_cfg("algo.hyperparams.double_q=false")
    seeds = seed_everything(1, SEED_COMPONENTS, seed_torch=True)
    agent = make_agent(cfg, seeds)
    n_actions = len(agent.clob_grid)
    obs_dim = len(cfg.features.clob)
    # Make every target Q exactly 2.
    with __import__("torch").no_grad():
        for p in agent.q_target["clob"].parameters():
            p.zero_()
        agent.q_target["clob"][-1].bias.fill_(2.0)
    batch = ReplayBatch(
        obs=np.zeros((2, obs_dim), np.float32),
        action=np.zeros(2, np.int64),
        reward=np.array([1.0, 1.0], np.float32),
        next_obs=np.zeros((2, obs_dim), np.float32),
        done=np.array([False, False]),
        junction=np.array([False, False]),
        next_mask=np.ones((2, n_actions), bool),
        discount=np.array([0.5, 0.9], np.float32),
    )
    y = agent.compute_targets("clob", batch).cpu().numpy()
    np.testing.assert_allclose(y, [3.0, 3.0], rtol=0, atol=1e-6)


def test_replay_normalizes_deprecated_transition_discount_to_one():
    cfg = load_dqn_cfg()
    seeds = seed_everything(2, SEED_COMPONENTS, seed_torch=True)
    agent = make_agent(cfg, seeds)
    buf = agent.replay["clob"]
    buf.add(
        np.zeros(len(cfg.features.clob), np.float32),
        0,
        0.0,
        np.zeros(len(cfg.features.clob), np.float32),
        False,
        False,
        np.ones(len(agent.clob_grid), bool),
        discount=0.876,
    )
    state = buf.state_dict()
    assert state["discount"][0] == 1.0
