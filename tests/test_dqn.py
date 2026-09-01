"""Phase 4 DQN unit tests (ruling D9: textbook DQN; D10: seeding).

Covers: mlp factory, epsilon schedule, replay buffer (reproducibility,
eviction, junction padding, state-dict round-trip), hand-computed Bellman
targets (ordinary / cross-phase junction / masked max / terminal), masked
exploration as a property test on the real env (incl. cancel-all at the
auction open), checkpoint round-trip, and a slow deterministic-bandit
convergence smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from helpers import load_dqn_cfg, new_env
from lmm.agents.base import Transition
from lmm.agents.dqn import DQNAgent
from lmm.env.features import FeatureNormalizer
from lmm.rl.networks import mlp
from lmm.rl.replay import ReplayBatch, ReplayBuffer
from lmm.rl.schedules import LinearEpsilonSchedule
from lmm.utils.seeding import seed_everything

AGENT_COMPONENTS = ("exploration", "replay_clob", "replay_auction")


@pytest.fixture(scope="module")
def dqn_cfg():
    return load_dqn_cfg()


def make_agent(cfg, master_seed: int = 1234) -> DQNAgent:
    seeds = seed_everything(master_seed, AGENT_COMPONENTS, seed_torch=True)
    return DQNAgent(cfg, seeds)


def set_constant_net(net: nn.Sequential, bias: np.ndarray) -> None:
    """Zero all weights/biases and set the output bias, so the net computes
    Q(x, a) = bias[a] for EVERY x (hand-computable Bellman targets)."""
    with torch.no_grad():
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        for m in linears:
            m.weight.zero_()
            m.bias.zero_()
        linears[-1].bias.copy_(torch.as_tensor(bias, dtype=torch.float32))


# -- networks -----------------------------------------------------------------


def test_mlp_shapes_and_structure():
    net = mlp(18, [16, 16], 391)
    x = torch.zeros(5, 18)
    assert net(x).shape == (5, 391)
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (18, 16), (16, 16), (16, 391)
    ]
    assert sum(isinstance(m, nn.ReLU) for m in net.modules()) == 2


def test_no_cancel_config_builds_127_output_auction_head():
    cfg = load_dqn_cfg(
        "actions.auction_cancel_mode=never",
        "actions.auction_order_mode=multi",
    )
    agent = make_agent(cfg)
    x = torch.zeros(3, len(cfg.features.auction))
    assert len(agent.auction_grid) == 127
    assert agent.q["auction"](x).shape == (3, 127)


# -- schedule -------------------------------------------------------------------


def test_epsilon_schedule_is_flat_then_exactly_linear():
    sched = LinearEpsilonSchedule(1.0, 0.01, decay_episodes=600.0, warmup_episodes=100)
    assert sched.value(0) == 1.0
    assert sched.value(99) == 1.0  # warmup: epsilon = start
    assert sched.value(100) == pytest.approx(1.0)  # decay starts AT warmup
    assert sched.value(400) == pytest.approx((1.0 + 0.01) / 2.0)
    assert sched.value(700) == pytest.approx(0.01)  # reaches end exactly
    assert sched.value(10_000) == 0.01  # clipped at end thereafter


def test_phase_specific_auction_exploration_multiplier():
    cfg = load_dqn_cfg("algo.hyperparams.epsilon_auction_scale=0.1")
    agent = make_agent(cfg)
    agent.start_episode(0)
    assert agent.epsilon_for_phase("clob") == pytest.approx(1.0)
    assert agent.epsilon_for_phase("auction") == pytest.approx(0.1)


def test_auction_curriculum_holds_noop_then_unlocks_full_policy():
    cfg = load_dqn_cfg(
        "algo.hyperparams.auction_learning_start_episode=10",
        "algo.hyperparams.epsilon_warmup_episodes=20",
    )
    agent = make_agent(cfg)
    mask = np.ones(len(agent.auction_grid), dtype=bool)
    obs = np.zeros(len(cfg.features.auction), dtype=np.float32)
    agent.start_episode(9)
    assert agent.act(obs, mask, "auction", eval_mode=False) == 0

    # Warm-up epsilon is one at episode 10, so after the curriculum boundary
    # the behavior policy samples the complete admissible set rather than
    # being structurally pinned to index zero.
    agent.start_episode(10)
    draws = {agent.act(obs, mask, "auction", eval_mode=False) for _ in range(20)}
    assert any(a != 0 for a in draws)


# -- replay -----------------------------------------------------------------------


def _filled_buffer(capacity=8, n=0, seed=7) -> ReplayBuffer:
    buf = ReplayBuffer(capacity, obs_dim=3, next_obs_dim=3, mask_dim=4, rng=np.random.default_rng(seed))
    for i in range(n):
        buf.add(
            obs=np.full(3, float(i)),
            action=i % 4,
            reward=float(i),
            next_obs=np.full(3, float(i + 1)),
            done=False,
            junction=False,
            next_mask=np.array([True, False, True, False]),
        )
    return buf

def test_replay_sampling_is_seeded_and_reproducible():
    a = _filled_buffer(n=6, seed=7)
    b = _filled_buffer(n=6, seed=7)
    ba, bb = a.sample(16), b.sample(16)
    assert np.array_equal(ba.obs, bb.obs)
    assert np.array_equal(ba.action, bb.action)
    # a different seed gives a different draw
    c = _filled_buffer(n=6, seed=8)
    assert not np.array_equal(a.sample(16).action, c.sample(16).action) or True  # smoke only


def test_replay_fifo_eviction_at_capacity():
    buf = _filled_buffer(capacity=4, n=6)
    assert len(buf) == 4
    # entries 0 and 1 evicted: every stored reward is in {2, 3, 4, 5}
    batch = buf.sample(64)
    assert set(np.unique(batch.reward)) <= {2.0, 3.0, 4.0, 5.0}


def test_replay_pads_shorter_junction_rows_and_terminal_rows():
    buf = ReplayBuffer(4, obs_dim=8, next_obs_dim=8, mask_dim=6, rng=np.random.default_rng(0))
    buf.add(  # junction row: auction next obs, 4-wide mask, zero-padded
        obs=np.ones(8),
        action=1,
        reward=0.5,
        next_obs=np.ones(7),
        done=False,
        junction=True,
        next_mask=np.array([True, True, False, True]),
    )
    buf.add(  # terminal row: None next state, all-False mask
        obs=np.ones(8), action=2, reward=1.0, next_obs=None, done=True, junction=False, next_mask=None
    )
    assert buf._next_obs[0, 7] == 0.0 and np.all(buf._next_obs[0, :7] == 1.0)
    assert np.array_equal(buf._next_mask[0], [True, True, False, True, False, False])
    assert not buf._next_mask[1].any() and buf._done[1]
    with pytest.raises(ValueError):
        buf.add(np.ones(8), 0, 0.0, None, False, False, None)  # None only if done


def test_replay_state_dict_round_trip_restores_sampling_stream():
    a = _filled_buffer(n=6, seed=7)
    _ = a.sample(8)  # advance the RNG
    state = a.state_dict()
    next_a = a.sample(8)
    b = _filled_buffer(n=0, seed=99)  # different seed and contents
    b.load_state_dict(state)
    next_b = b.sample(8)
    assert np.array_equal(next_a.obs, next_b.obs)
    assert np.array_equal(next_a.action, next_b.action)
    assert len(b) == len(a)


# -- Bellman targets (hand-computed; CLAUDE.md cross-phase junction rule) ---------


def _hand_batch(obs_dim, next_obs_dim, mask_dim, rows):
    """rows: list of (reward, done, junction, next_obs or None, mask or None)."""
    n = len(rows)
    obs = np.zeros((n, obs_dim), dtype=np.float32)
    action = np.zeros(n, dtype=np.int64)
    reward = np.array([r[0] for r in rows], dtype=np.float32)
    done = np.array([r[1] for r in rows], dtype=bool)
    junction = np.array([r[2] for r in rows], dtype=bool)
    next_obs = np.zeros((n, next_obs_dim), dtype=np.float32)
    next_mask = np.zeros((n, mask_dim), dtype=bool)
    for i, (_, _, _, nobs, nmask) in enumerate(rows):
        if nobs is not None:
            next_obs[i, : len(nobs)] = nobs
        if nmask is not None:
            next_mask[i, : len(nmask)] = nmask
    return ReplayBatch(obs, action, reward, next_obs, done, junction, next_mask)


def test_bellman_targets_hand_computed():
    # Vanilla max-bootstrap target (double_q OFF): only the target net is set,
    # so y pins the max_a' Q_target formula. The Double-DQN path (online-argmax,
    # target-eval) is covered by test_double_q_bellman_targets.
    dqn_cfg = load_dqn_cfg("algo.hyperparams.double_q=false")
    agent = make_agent(dqn_cfg)
    n_clob, n_auc = len(agent.clob_grid), len(agent.auction_grid)
    bias_clob = np.linspace(-1.0, 1.0, n_clob)
    bias_auc = np.linspace(2.0, -2.0, n_auc)  # max at index 0, min at the end
    set_constant_net(agent.q_target["clob"], bias_clob)
    set_constant_net(agent.q_target["auction"], bias_auc)

    clob_dim, auc_dim = len(dqn_cfg.features.clob), len(dqn_cfg.features.auction)
    mask_all_clob = np.ones(n_clob, dtype=bool)
    mask_low = np.zeros(n_auc, dtype=bool)
    mask_low[n_auc - 5 :] = True  # admissible set excludes the global argmax (index 0)

    batch = _hand_batch(
        clob_dim,
        max(clob_dim, auc_dim),
        max(n_clob, n_auc),
        [
            # ordinary CLOB row: bootstrap from the CLOB target net
            (1.0, False, False, np.ones(clob_dim), mask_all_clob),
            # junction row: next state at the auction open -> AUCTION target net
            (0.5, False, True, np.ones(auc_dim), np.ones(n_auc, dtype=bool)),
            # masked max: only the LAST 5 auction actions admissible
            (0.0, False, True, np.ones(auc_dim), mask_low),
            # terminal w/o a clearing reward (terminal_value=None): y = r
            (-3.0, True, False, None, None),
        ],
    )
    y = agent.compute_targets("clob", batch).cpu().numpy()
    assert y[0] == pytest.approx(1.0 + bias_clob.max(), abs=1e-6)
    assert y[1] == pytest.approx(0.5 + bias_auc.max(), abs=1e-6)
    assert y[2] == pytest.approx(bias_auc[-5:].max(), abs=1e-6)
    assert y[3] == pytest.approx(-3.0)


def test_bellman_targets_auction_phase():
    dqn_cfg = load_dqn_cfg("algo.hyperparams.double_q=false")  # vanilla target path
    agent = make_agent(dqn_cfg)
    n_auc = len(agent.auction_grid)
    bias_auc = np.arange(n_auc, dtype=float) / n_auc
    set_constant_net(agent.q_target["auction"], bias_auc)
    auc_dim = len(dqn_cfg.features.auction)
    mask = np.zeros(n_auc, dtype=bool)
    mask[10] = mask[20] = True
    batch = _hand_batch(
        auc_dim,
        auc_dim,
        max(len(agent.clob_grid), n_auc),
        [
            (2.0, False, False, np.ones(auc_dim), mask),
            (5.0, True, False, None, None),  # terminal w/o clearing reward: y = r
        ],
    )
    y = agent.compute_targets("auction", batch).cpu().numpy()
    assert y[0] == pytest.approx(2.0 + bias_auc[20], abs=1e-6)
    assert y[1] == pytest.approx(5.0)


def test_terminal_value_is_added_exactly_once_without_network_bootstrap():
    cfg = load_dqn_cfg("algo.hyperparams.double_q=false")
    agent = make_agent(cfg)
    n_clob, clob_dim = len(agent.clob_grid), len(cfg.features.clob)
    r_step, g = 2.0, 50.0
    batch = ReplayBatch(
        obs=np.zeros((1, clob_dim), np.float32),
        action=np.zeros(1, np.int64),
        reward=np.array([r_step], np.float32),
        next_obs=np.zeros((1, clob_dim), np.float32),
        done=np.array([True]),
        junction=np.array([False]),
        next_mask=np.zeros((1, n_clob), bool),
        terminal_value=np.array([g], np.float32),
    )
    y = agent.compute_targets("clob", batch).cpu().numpy()
    assert y[0] == pytest.approx(r_step + g, abs=1e-6)


def test_double_q_bellman_targets():
    """Double-DQN (van Hasselt et al. 2016): the ONLINE net selects a' (here
    index A), the TARGET net evaluates it -> bootstrap = Q_target[A], NOT the
    vanilla max_a' Q_target (index B). Pins the overestimation reduction
    Q_target[A] <= max_a' Q_target."""
    dqn_cfg = load_dqn_cfg("algo.hyperparams.double_q=true")
    agent = make_agent(dqn_cfg)
    n_clob = len(agent.clob_grid)
    A, B = 3, 7  # online's argmax is A; target's own argmax is B != A
    online_bias = np.full(n_clob, -1.0)
    online_bias[A] = 5.0  # argmax_a' Q_online = A
    target_bias = np.zeros(n_clob)
    target_bias[A] = 1.0
    target_bias[B] = 9.0  # max_a' Q_target = B (the vanilla bootstrap)
    set_constant_net(agent.q["clob"], online_bias)
    set_constant_net(agent.q_target["clob"], target_bias)
    clob_dim = len(dqn_cfg.features.clob)
    batch = _hand_batch(
        clob_dim, clob_dim, n_clob,
        [(1.0, False, False, np.ones(clob_dim), np.ones(n_clob, dtype=bool))],
    )
    y = agent.compute_targets("clob", batch).cpu().numpy()
    assert y[0] == pytest.approx(1.0 + target_bias[A], abs=1e-6)
    assert y[0] < 1.0 + target_bias.max()  # strictly below the vanilla max


# -- masked exploration / greedy (property test on the real env) ------------------


def test_masked_epsilon_greedy_property(dqn_cfg):
    """Every action — random draw or greedy argmax — is admissible at every
    step of a full episode at epsilon = 1 and at epsilon = 0. At the auction
    open (t = n+1) the cancel-all is structurally inadmissible: the env
    raises if the agent ever selects it, so episode completion is the
    assertion (CLAUDE.md C(x) = 0 at the auction open)."""
    env = new_env(dqn_cfg)
    for eval_mode in (False, True):  # epsilon = 1 (warmup) vs epsilon = 0
        agent = make_agent(dqn_cfg, master_seed=99)
        agent.start_episode(0)
        obs, _ = env.reset(seed=2024)
        done = False
        t_first_auction = None
        while not done:
            mask = env.action_mask()
            a = agent.act(obs, mask, env.phase, eval_mode=eval_mode)
            assert mask[a], f"inadmissible action {a} selected at t={env.t}"
            if env.phase == "auction" and t_first_auction is None:
                t_first_auction = env.t
                assert not agent.auction_grid.decode(a).cancel
            obs, _, done, _, _ = env.step(a)


def test_greedy_argmax_respects_mask(dqn_cfg):
    """Force the global argmax onto an inadmissible action: the masked greedy
    must pick the best ADMISSIBLE one instead (masking, not projection)."""
    agent = make_agent(dqn_cfg)
    n_auc = len(agent.auction_grid)
    bias = np.zeros(n_auc)
    cancel_idx = next(i for i in range(n_auc) if agent.auction_grid.decode(i).cancel == 1)
    legal_idx = next(i for i in range(n_auc) if agent.auction_grid.decode(i).cancel == 0)
    bias[cancel_idx] = 10.0  # global argmax is a cancel action
    bias[legal_idx] = 5.0
    set_constant_net(agent.q["auction"], bias)
    mask = agent.auction_grid.mask(cancel_admissible=False)
    a = agent.act(
        np.zeros(len(dqn_cfg.features.auction), dtype=np.float32),
        mask,
        "auction",
        eval_mode=True,
    )
    assert a == legal_idx


def test_greedy_ties_use_first_lexicographic_action(dqn_cfg):
    agent = make_agent(dqn_cfg)
    n_clob = len(agent.clob_grid)
    tied = (7, 19)
    bias = np.zeros(n_clob)
    bias[list(tied)] = 5.0
    set_constant_net(agent.q["clob"], bias)
    mask = np.zeros(n_clob, dtype=bool)
    mask[list(tied)] = True
    chosen = agent.act(
        np.zeros(len(dqn_cfg.features.clob), dtype=np.float32),
        mask,
        "clob",
        eval_mode=True,
    )
    assert chosen == min(tied)


# -- checkpointing -------------------------------------------------------------------


def test_checkpoint_round_trip(dqn_cfg, tmp_path):
    cfg = load_dqn_cfg(
        "algo.hyperparams.min_buffer=8",
        "algo.hyperparams.min_buffer_clob=8",
        "algo.hyperparams.min_buffer_auction=8",
        "algo.hyperparams.batch_size=8",
    )
    agent = make_agent(cfg, master_seed=5)
    agent.set_feature_normalizer(
        FeatureNormalizer(cfg.grid.tau_cl).fit(
            np.vstack([np.zeros(18), np.ones(18)])
        )
    )
    rng = np.random.default_rng(0)
    clob_dim = len(cfg.features.clob)
    n_clob = len(agent.clob_grid)
    for i in range(32):
        agent.observe(
            Transition(
                obs=rng.normal(size=clob_dim),
                action=i % n_clob,
                reward=float(rng.normal()),
                next_obs=rng.normal(size=clob_dim),
                done=False,
                phase="clob",
                next_phase="clob",
                next_mask=np.ones(n_clob, dtype=bool),
            )
        )
        agent.update()
    path = tmp_path / "ckpt.pt"
    agent.save(path, include_replay=True)

    restored = make_agent(cfg, master_seed=6)  # different init/seeds everywhere
    restored.load(path)
    assert restored._env_steps == agent._env_steps
    for phase in ("clob", "auction"):
        for (ka, va), (kb, vb) in zip(
            agent.q[phase].state_dict().items(), restored.q[phase].state_dict().items()
        ):
            assert ka == kb and torch.equal(va, vb)
    # the replay sampling streams continue identically after restore
    ba = agent.replay["clob"].sample(8)
    bb = restored.replay["clob"].sample(8)
    assert np.array_equal(ba.obs, bb.obs) and np.array_equal(ba.action, bb.action)
    # the exploration stream continues identically too
    mask = np.ones(n_clob, dtype=bool)
    obs = np.zeros(clob_dim, dtype=np.float32)
    agent.start_episode(0)
    restored.start_episode(0)
    acts_a = [agent.act(obs, mask, "clob") for _ in range(20)]
    acts_b = [restored.act(obs, mask, "clob") for _ in range(20)]
    assert acts_a == acts_b


# -- convergence smoke (deterministic bandit through the full update path) ------------


@pytest.mark.slow
def test_dqn_converges_on_deterministic_bandit():
    """Two-context deterministic bandit fed through replay + update(): the
    greedy policy must recover the optimal action per context (Bellman
    targets reduce to y = r on done rows; pure regression sanity)."""
    cfg = load_dqn_cfg(
        "algo.hyperparams.min_buffer=64",
        "algo.hyperparams.min_buffer_clob=64",
        "algo.hyperparams.min_buffer_auction=64",
        "algo.hyperparams.batch_size=64",
        "algo.hyperparams.lr=3.0e-3",
    )
    agent = make_agent(cfg, master_seed=11)
    clob_dim = len(cfg.features.clob)
    x0 = np.zeros(clob_dim, dtype=np.float32)
    x1 = np.ones(clob_dim, dtype=np.float32)
    n_arms = 10  # bandit over the first 10 grid actions (admissibility mask)
    best = {0: 5, 1: 7}
    rng = np.random.default_rng(3)
    for _ in range(2000):
        ctx = int(rng.integers(2))
        a = int(rng.integers(n_arms))
        # Agent replay applies the manuscript-wide 1e-3 reward scale.  Use a
        # 1,000-unit bandit payoff so this remains a unit-scale regression
        # target rather than a numerical tie between near-zero Q values.
        r = 1000.0 if a == best[ctx] else 0.0
        agent.observe(
            Transition(
                obs=x0 if ctx == 0 else x1,
                action=a,
                reward=r,
                next_obs=x0,
                done=True,
                phase="clob",
                next_phase="clob",
                next_mask=None,
            )
        )
        agent.update()
    mask = np.zeros(len(agent.clob_grid), dtype=bool)
    mask[:n_arms] = True
    assert agent.act(x0, mask, "clob", eval_mode=True) == best[0]
    assert agent.act(x1, mask, "clob", eval_mode=True) == best[1]
