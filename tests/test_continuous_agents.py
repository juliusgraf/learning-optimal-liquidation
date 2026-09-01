"""DDPG / TD3 / SAC unit + integration tests (Phase 5; ruling D9, D10).

Covers hand-derived Bellman targets (DDPG single critic, TD3 clipped-double-Q
min, SAC entropy term), the SAC temperature loss and target entropy, the
cancel clamp, continuous replay + checkpoint round-trips, and the slow
end-to-end smoke training + bit-identical determinism per algorithm. The
adapter equivalence lives in tests/test_continuous_adapter.py.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from helpers import load_algo_cfg, load_ddpg_cfg, load_sac_cfg, load_td3_cfg
from lmm.agents.base import Transition
from lmm.agents.ddpg import DDPGAgent
from lmm.agents.sac import SACAgent
from lmm.agents.td3 import TD3Agent
from lmm.experiments import train as train_mod
from lmm.env.features import FeatureNormalizer
from lmm.rl.continuous_replay import ContinuousReplayBuffer
from lmm.rl.loops import SEED_COMPONENTS
from lmm.utils.seeding import seed_everything


def make_cont_agent(cfg, cls, master_seed: int = 1234):
    seeds = seed_everything(master_seed, SEED_COMPONENTS, seed_torch=True)
    return cls(cfg, seeds)


def test_shared_no_cancel_mode_changes_sac_action_dim_and_target_entropy():
    cfg = load_sac_cfg(
        "actions.auction_cancel_mode=never",
        "actions.auction_order_mode=multi",
    )
    agent = make_cont_agent(cfg, SACAgent)
    assert agent._act_dim["auction"] == 2
    assert agent.target_entropy["auction"] == -3.0


def test_safe_auction_actor_still_projects_to_zero_with_wide_slope_band():
    cfg = load_ddpg_cfg(
        "actions.K_max=32",
        "algo.hyperparams.safe_auction_initialization=true",
    )
    agent = make_cont_agent(cfg, DDPGAgent)
    obs = torch.zeros((1, len(cfg.features.auction)), dtype=torch.float32)
    with torch.no_grad():
        raw = agent.actor["auction"](obs).squeeze(0).cpu().numpy()
    projected_k_coordinate = (raw[0] + 1.0) * cfg.actions.K_max / 2.0
    assert projected_k_coordinate < 0.5
    assert raw[2] < 0.0  # no cancellation request


def constant_critic(critic, value: float) -> None:
    """Make a Critic output ``value`` for EVERY (obs, action) (zero all
    weights/biases, set the output bias)."""
    with torch.no_grad():
        linears = [m for m in critic.net.modules() if isinstance(m, nn.Linear)]
        for m in linears:
            m.weight.zero_()
            m.bias.zero_()
        linears[-1].bias.fill_(float(value))


def cont_batch(obs_dim, act_dim, next_obs_dim, rows):
    """rows: list of (reward, done, junction, next_obs or None, next_cancel_adm)."""
    from lmm.rl.continuous_replay import ContinuousReplayBatch

    n = len(rows)
    next_obs = np.zeros((n, next_obs_dim), dtype=np.float32)
    for i, (_, _, _, nobs, _) in enumerate(rows):
        if nobs is not None:
            next_obs[i, : len(nobs)] = nobs
    return ContinuousReplayBatch(
        obs=np.zeros((n, obs_dim), dtype=np.float32),
        action=np.zeros((n, act_dim), dtype=np.float32),
        reward=np.array([r[0] for r in rows], dtype=np.float32),
        next_obs=next_obs,
        done=np.array([r[1] for r in rows], dtype=bool),
        junction=np.array([r[2] for r in rows], dtype=bool),
        next_cancel_admissible=np.array([r[4] for r in rows], dtype=bool),
    )


# -- hand-derived Bellman targets ---------------------------------------------


def test_ddpg_target_single_critic_and_junction():
    cfg = load_ddpg_cfg()
    agent = make_cont_agent(cfg, DDPGAgent)
    clob_dim, auc_dim = len(cfg.features.clob), len(cfg.features.auction)
    A, B = 3.0, -2.0  # constant CLOB-target / auction-target Q values
    for c in agent.critic_targets["clob"]:
        constant_critic(c, A)
    for c in agent.critic_targets["auction"]:
        constant_critic(c, B)
    batch = cont_batch(
        clob_dim, agent._act_dim["clob"], max(clob_dim, auc_dim),
        [
            (1.0, False, False, np.ones(clob_dim), False),  # ordinary CLOB -> A
            (0.5, False, True, np.ones(auc_dim), True),     # junction -> AUCTION B
            (-3.0, True, False, None, False),               # terminal -> r
        ],
    )
    y = agent.compute_targets("clob", batch).cpu().numpy()
    assert y[0] == pytest.approx(1.0 + A, abs=1e-5)
    assert y[1] == pytest.approx(0.5 + B, abs=1e-5)
    assert y[2] == pytest.approx(-3.0, abs=1e-6)


def test_td3_target_uses_min_of_twin_critics():
    cfg = load_td3_cfg()
    agent = make_cont_agent(cfg, TD3Agent)
    assert agent.n_critics == 2
    clob_dim, auc_dim = len(cfg.features.clob), len(cfg.features.auction)
    constant_critic(agent.critic_targets["clob"][0], 5.0)
    constant_critic(agent.critic_targets["clob"][1], 2.0)  # min = 2.0
    constant_critic(agent.critic_targets["auction"][0], -1.0)
    constant_critic(agent.critic_targets["auction"][1], 4.0)  # min = -1.0
    batch = cont_batch(
        clob_dim, agent._act_dim["clob"], max(clob_dim, auc_dim),
        [
            (1.0, False, False, np.ones(clob_dim), False),  # clob min(5,2)=2
            (0.0, False, True, np.ones(auc_dim), True),     # junction min(-1,4)=-1
            (7.0, True, False, None, False),                # terminal
        ],
    )
    y = agent.compute_targets("clob", batch).cpu().numpy()
    assert y[0] == pytest.approx(3.0, abs=1e-5)
    assert y[1] == pytest.approx(-1.0, abs=1e-5)
    assert y[2] == pytest.approx(7.0, abs=1e-6)


def test_sac_target_min_twin_plus_entropy_term():
    cfg = load_sac_cfg()
    agent = make_cont_agent(cfg, SACAgent)
    auc_dim = len(cfg.features.auction)
    c1, c2 = 1.0, 0.5  # min = 0.5
    constant_critic(agent.critic_targets["auction"][0], c1)
    constant_critic(agent.critic_targets["auction"][1], c2)
    alpha = 0.3
    with torch.no_grad():
        agent.log_alpha["auction"].fill_(math.log(alpha))
    rows = [
        (2.0, False, False, np.ones(auc_dim), False),
        (1.0, False, False, np.full(auc_dim, 0.5), False),
        (9.0, True, False, None, False),  # terminal
    ]
    batch = cont_batch(auc_dim, agent._act_dim["auction"], auc_dim, rows)
    torch.manual_seed(123)
    y = agent.compute_targets("auction", batch).cpu().numpy()
    # reproduce the actor sample on the two non-terminal rows (same torch seed).
    torch.manual_seed(123)
    nobs = torch.as_tensor(batch.next_obs[[0, 1], :auc_dim], dtype=torch.float32)
    _, logp = agent.actor["auction"](nobs)
    expected = batch.reward[[0, 1]] + min(c1, c2) - alpha * logp.detach().numpy()
    assert np.allclose(y[[0, 1]], expected, atol=1e-5)
    assert y[2] == pytest.approx(9.0, abs=1e-6)


def test_sac_temperature_loss_and_target_entropy():
    cfg = load_sac_cfg()
    agent = make_cont_agent(cfg, SACAgent)
    assert agent.target_entropy == {"clob": -2.0, "auction": -3.0}

    auc_dim = len(cfg.features.auction)
    log_alpha0 = 0.5
    with torch.no_grad():
        agent.log_alpha["auction"].fill_(log_alpha0)
    obs = torch.zeros((6, auc_dim), dtype=torch.float32)
    # capture the actor sample's log-prob BEFORE _update_actor steps the actor.
    torch.manual_seed(77)
    _, logp0 = agent.actor["auction"](obs)
    H = agent.target_entropy["auction"]
    expected_alpha_loss = float((-(log_alpha0 * (logp0.detach() + H))).mean())
    expected_entropy = float((-logp0.detach()).mean())

    torch.manual_seed(77)  # _update_actor recomputes the same sample (params unchanged)
    out = agent._update_actor("auction", obs)
    assert out["alpha_loss"] == pytest.approx(expected_alpha_loss, abs=1e-5)
    assert out["entropy"] == pytest.approx(expected_entropy, abs=1e-5)


def test_target_cancel_coordinate_is_common_proposal_not_algorithm_specific_clamp():
    cfg = load_ddpg_cfg()
    agent = make_cont_agent(cfg, DDPGAgent)
    obs = torch.zeros((2, len(cfg.features.auction)))
    cadm = np.array([True, False])
    out, _ = agent._target_next_action("auction", obs, cadm)
    # Actor/critic/replay use proposal u. Gamma_x, shared by every continuous
    # algorithm, applies the executable cancellation mask.
    assert torch.equal(out[0], out[1])


# -- continuous replay + checkpointing ----------------------------------------


def test_continuous_replay_round_trip_restores_stream():
    buf = ContinuousReplayBuffer(8, obs_dim=7, action_dim=3, next_obs_dim=7, rng=np.random.default_rng(0))
    for i in range(6):
        buf.add(
            obs=np.full(7, float(i)),
            action=np.full(3, -1.0 + 0.4 * i, dtype=float),
            reward=float(i),
            next_obs=np.full(7, float(i + 1)),
            done=False,
            junction=False,
            next_cancel_admissible=bool(i % 2),
        )
    _ = buf.sample(8)
    state = buf.state_dict()
    nxt = buf.sample(8)
    b2 = ContinuousReplayBuffer(8, 7, 3, 7, np.random.default_rng(99))
    b2.load_state_dict(state)
    nxt2 = b2.sample(8)
    assert np.array_equal(nxt.obs, nxt2.obs)
    assert np.array_equal(nxt.action, nxt2.action)
    assert np.array_equal(nxt.next_cancel_admissible, nxt2.next_cancel_admissible)


@pytest.mark.parametrize("algo,cls", [("ddpg", DDPGAgent), ("td3", TD3Agent), ("sac", SACAgent)])
def test_checkpoint_round_trip(algo, cls, tmp_path):
    cfg = load_algo_cfg(
        algo,
        "algo.hyperparams.min_buffer=8",
        "algo.hyperparams.min_buffer_clob=8",
        "algo.hyperparams.min_buffer_auction=8",
        "algo.hyperparams.batch_size=8",
    )
    agent = make_cont_agent(cfg, cls, master_seed=5)
    normalizer = FeatureNormalizer(cfg.grid.tau_cl).fit(
        np.vstack([np.zeros(18), np.ones(18)])
    )
    agent.set_feature_normalizer(normalizer)
    rng = np.random.default_rng(0)
    clob_dim, ad = len(cfg.features.clob), agent._act_dim["clob"]
    for i in range(40):
        agent.observe(
            Transition(
                obs=rng.normal(size=clob_dim),
                action=np.clip(rng.normal(size=ad), -1.0, 1.0),
                reward=float(rng.normal()),
                next_obs=rng.normal(size=clob_dim),
                done=False,
                phase="clob",
                next_phase="clob",
                next_mask=None,
            )
        )
        agent.update()
    path = tmp_path / "ckpt.pt"
    agent.save(path, include_replay=True)

    restored = make_cont_agent(cfg, cls, master_seed=6)  # different init/seeds
    restored.load(path)
    assert restored._env_steps == agent._env_steps
    for (ka, va), (kb, vb) in zip(
        agent.actor["clob"].state_dict().items(), restored.actor["clob"].state_dict().items()
    ):
        assert ka == kb and torch.equal(va, vb)
    if isinstance(agent, SACAgent):  # temperature restored
        assert torch.equal(agent.log_alpha["auction"], restored.log_alpha["auction"])
    # the replay sampling stream continues identically after restore
    ba = agent.replay["clob"].sample(8)
    bb = restored.replay["clob"].sample(8)
    assert np.array_equal(ba.obs, bb.obs) and np.array_equal(ba.action, bb.action)


# -- end-to-end smoke + determinism (slow) ------------------------------------


def _train(tmp_path, algo: str, run_name: str) -> str:
    argv = [
        "--config", "configs/base.yaml",
        "--config", "configs/synthetic_rough_heston.yaml",
        "--config", f"configs/algo/{algo}.yaml",
        "--run-name", run_name,
        "-o", f"experiment.results_root={tmp_path}",
        "-o", "experiment.episodes=4",
        "-o", "rl.normalizer_fit_episodes=1",
        "-o", "rl.validation_frequency_episodes=2",
        "-o", "rl.validation_size=2",
        "-o", "rl.validation_patience_evals=10",
        "-o", "rl.test_size=2",
        "-o", "algo.hyperparams.checkpoint_interval_episodes=4",
        "-o", "algo.hyperparams.min_buffer=150",
        "-o", "algo.hyperparams.min_buffer_clob=150",
        "-o", "algo.hyperparams.min_buffer_auction=90",
        "-o", "algo.hyperparams.batch_size=32",
    ]
    assert train_mod.main(argv) == 0
    return str(tmp_path / "synthetic_rough_heston" / run_name)


def _metrics_no_wall_clock(path) -> list[list[str]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][-1] == "wall_clock_s"
    return [row[:-1] for row in rows]


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ddpg", "td3", "sac"])
def test_smoke_training_writes_outputs(algo, tmp_path):
    run = _train(tmp_path, algo, f"smoke_{algo}")
    for rel in (
        "config_resolved.yaml",
        "seed.txt",
        "metrics.csv",
        "logs/run.log",
        "checkpoints/initial.pt",
        "checkpoints/final.pt",
    ):
        assert (tmp_path / "synthetic_rough_heston" / f"smoke_{algo}" / rel).exists(), rel
    rows = _metrics_no_wall_clock(f"{run}/metrics.csv")
    assert len(rows) == 1 + 4  # header + one row per episode


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ddpg", "td3", "sac"])
def test_training_is_bit_identical_with_fixed_seed(algo, tmp_path):
    run_a = _train(tmp_path, algo, "run_a")
    run_b = _train(tmp_path, algo, "run_b")
    assert _metrics_no_wall_clock(f"{run_a}/metrics.csv") == _metrics_no_wall_clock(f"{run_b}/metrics.csv")
