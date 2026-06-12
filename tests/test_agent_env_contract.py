"""Agent-env contract tests (Phase 4): benchmarks on the real env, D16
one-sided wiring through the env, CRN across policy envs, the shared reward
definition (AUDIT C.4), and the slow end-to-end train/evaluate/regret
pipeline with the bit-identical determinism check (ruling D10).
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import yaml

from helpers import NOOP_AUCTION, NOOP_CLOB, drive_to_auction, load_dqn_cfg, new_env
from lmm.agents.benchmarks import ASBenchmarkAgent, TWAPBenchmarkAgent
from lmm.env.action_spaces import AuctionAction
from lmm.experiments import evaluate as evaluate_mod
from lmm.experiments import regret as regret_mod
from lmm.experiments import train as train_mod
from lmm.rl.loops import run_episode

SMALL_OVERRIDES = (
    "benchmark.as_n_samples=500",
    "benchmark.as_sigma_n_paths=3",
)


@pytest.fixture(scope="module")
def cfg():
    return load_dqn_cfg(*SMALL_OVERRIDES)


@pytest.fixture(scope="module")
def quiet_cfg():
    """No exogenous auction flow at all: hand-computable clearing."""
    return load_dqn_cfg(
        "auction_flow.p1=0.0",
        "auction_flow.p2=0.0",
        "auction_flow.p3=0.0",
        "auction_flow.p4=0.0",
        *SMALL_OVERRIDES,
    )


def make_as_agent(cfg, env) -> ASBenchmarkAgent:
    agent = ASBenchmarkAgent(cfg)
    agent.bind(env)
    agent.calibrate(env, rng_k=np.random.default_rng(1), rng_sigma=np.random.default_rng(2))
    return agent


# -- benchmarks run full episodes on the real env ------------------------------


@pytest.mark.parametrize("name", ["as", "twap"])
def test_benchmarks_complete_episodes_without_error(cfg, name):
    env = new_env(cfg)
    if name == "as":
        agent = make_as_agent(cfg, env)
    else:
        agent = TWAPBenchmarkAgent(cfg)
        agent.bind(env)
    for episode, seed in enumerate((11, 12)):
        agent.start_episode(episode)
        res = run_episode(env, agent, seed, chi=cfg.rl.chi, train=False)
        # liquidation policies: inventory drawn down from I0, never negative
        # during the CLOB phase (Z can overshoot at the terminal, D16)
        assert res.n_steps == res.n_clob_steps + (cfg.grid.tau_cl - cfg.grid.tau_op)
        assert np.isfinite(res.return_undisc)
        assert np.isfinite(res.s_cl)


def test_as_delta_table_matches_closed_form(cfg):
    env = new_env(cfg)
    agent = make_as_agent(cfg, env)
    A, k = agent.A, agent.k
    T = agent.T_as
    alpha = cfg.grid.alpha
    # direct (unstable but fine at small q) evaluation of the closed form
    for t, q in ((0, 1), (10, 3), (T - 1, 7), (T, 2)):
        x = (A / math.e) * (T - t)
        v = [sum(x**j / math.factorial(j) for j in range(qq + 1)) for qq in (q - 1, q)]
        expected = (1.0 / (k * alpha)) * (1.0 + math.log(v[1] / v[0]))
        assert agent.delta_ticks[t, q] == pytest.approx(expected, rel=1e-12)
    assert np.isinf(agent.delta_ticks[0, 0])  # q = 0 row never used


def test_benchmark_abstains_in_auction_without_clob_executions(cfg):
    """No executed CLOB price => S_tilde undefined => no auction order
    (and no ValueError from the env)."""
    env = new_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    drive_to_auction(env, seed=5)  # NOOP CLOB actions: nothing executes
    assert agent._exec_prices == []
    a = agent.act(np.zeros(7), env.action_mask(), "auction")
    assert a.K_a == 0.0 and not a.one_sided
    # the auction-open flag also means later steps abstain
    assert agent.act(np.zeros(7), env.action_mask(), "auction").K_a == 0.0


# -- D16 wiring through the env -------------------------------------------------


def finish_auction(env, first_action):
    """Submit ``first_action`` at the auction open, NOOP afterwards; returns
    the terminal info dict."""
    _, r, done, _, info = env.step(first_action)
    while not done:
        _, r, done, _, info = env.step(NOOP_AUCTION)
    return info


def test_one_sided_order_inactive_above_s_tilde(quiet_cfg):
    """S_tilde above the clearing price: the one-sided order contributes
    NOTHING — S_cl equals the abstain case and Z = 0 (ruling D16)."""
    env_a, env_b = new_env(quiet_cfg), new_env(quiet_cfg)
    for env in (env_a, env_b):
        drive_to_auction(env, seed=77)
        env.generator.auction_flow.inject_market_maker(1.0, env.s_mid)
    info_a = finish_auction(env_a, AuctionAction(5.0, 50, 0, one_sided=True))
    info_b = finish_auction(env_b, NOOP_AUCTION)
    assert info_a["S_cl"] == pytest.approx(info_b["S_cl"])
    assert info_a["Z"] == pytest.approx(0.0)
    assert info_a["I_final"] == pytest.approx(env_a.grid.I0)  # nothing executed


def test_one_sided_order_active_below_s_tilde_hand_computed(quiet_cfg):
    """S_tilde below the clearing price: two-curve hand computation of S_cl,
    Z and the terminal reward (rulings D16, D3, D8)."""
    env = new_env(quiet_cfg)
    drive_to_auction(env, seed=77)
    s_exo = env.s_mid
    env.generator.auction_flow.inject_market_maker(1.0, s_exo)
    offset = -50
    K_b = 5.0
    s_tilde = env.grid.alpha * (math.floor(env.s_mid / env.grid.alpha) + offset)
    info = finish_auction(env, AuctionAction(K_b, offset, 0, one_sided=True))

    s_cl = (1.0 * s_exo + K_b * s_tilde) / (1.0 + K_b)
    assert s_cl > s_tilde  # the hockey stick is active at the root
    z = K_b * (s_cl - s_tilde)
    i_final = env.grid.I0 - z
    r = quiet_cfg.reward
    r_term = K_b * s_cl * (s_cl - s_tilde) - r.lambda_inv * i_final**2  # f_a inactive (u >= 0)
    assert info["S_cl"] == pytest.approx(s_cl, rel=1e-12)
    assert info["Z"] == pytest.approx(z, rel=1e-12)
    assert info["I_final"] == pytest.approx(i_final, rel=1e-12)
    assert info["terminal_reward"] == pytest.approx(r_term, rel=1e-12)


def test_one_sided_per_step_reward_uses_positive_part(quiet_cfg):
    """Submission-step reward of the one-sided order: K H (H - S~)_+ with no
    wrong-side penalty when H < S~ (Phase 4 resolution, AUDIT)."""
    env = new_env(quiet_cfg)
    drive_to_auction(env, seed=77)
    h = env.h_cl
    # quote far ABOVE H: (H - S~)_+ = 0 -> reward exactly 0
    _, reward, _, _, info = env.step(AuctionAction(5.0, 200, 0, one_sided=True))
    assert info["H_used"] == pytest.approx(h)
    assert reward == 0.0


def test_second_live_one_sided_order_rejected(quiet_cfg):
    env = new_env(quiet_cfg)
    drive_to_auction(env, seed=77)
    env.step(AuctionAction(5.0, -10, 0, one_sided=True))
    with pytest.raises(ValueError, match="one-sided"):
        env.step(AuctionAction(5.0, -10, 0, one_sided=True))


# -- CRN across policy envs (the evaluate.py coupling) -----------------------------


def test_crn_exogenous_trajectories_identical_across_policies():
    """Two envs reset with the same seed but driven by DIFFERENT liquidation
    policies see the same exogenous market (conditional cancel draws removed
    per the helpers.py CRN note)."""
    cfg = load_dqn_cfg(
        "auction_flow.p2=0.0",
        "auction_flow.p4=0.0",
        *SMALL_OVERRIDES,
    )
    env_twap, env_noop = new_env(cfg), new_env(cfg)

    twap = TWAPBenchmarkAgent(cfg)
    twap.bind(env_twap)
    twap.start_episode(0)
    res = run_episode(env_twap, twap, 31337, chi=cfg.rl.chi, train=False)

    drive_to_auction(env_noop, seed=31337)
    done = False
    while not done:
        _, _, done, _, _ = env_noop.step(NOOP_AUCTION)

    flow_a, flow_b = env_twap.generator.auction_flow, env_noop.generator.auction_flow
    assert flow_a.n_buy == flow_b.n_buy and flow_a.n_sell == flow_b.n_sell
    assert flow_a.frozen_mid == flow_b.frozen_mid
    K_a, S_a = flow_a.supply_curves()
    K_b, S_b = flow_b.supply_curves()
    assert np.array_equal(K_a, K_b) and np.array_equal(S_a, S_b)
    assert res.n_clob_steps > 0


# -- end-to-end pipeline (slow) ------------------------------------------------------


EXPECTED_FILES = (
    "config_resolved.yaml",
    "seed.txt",
    "git_sha.txt",
    "metrics.csv",
    "logs/run.log",
    "checkpoints/initial.pt",
    "checkpoints/best.pt",
    "checkpoints/final.pt",
    "checkpoints/ckpt_ep4.pt",
    "checkpoints/ckpt_ep4_trainstate.pt",
)


def _train(tmp_path, run_name: str) -> str:
    argv = [
        "--config", "configs/base.yaml",
        "--config", "configs/synthetic_rough_heston.yaml",
        "--config", "configs/algo/dqn.yaml",
        "--run-name", run_name,
        "-o", f"experiment.results_root={tmp_path}",
        "-o", "experiment.episodes=5",
        "-o", "algo.hyperparams.eval_interval_episodes=2",
        "-o", "algo.hyperparams.eval_n_seeds=2",
        "-o", "algo.hyperparams.checkpoint_interval_episodes=4",
        "-o", "algo.hyperparams.min_buffer=200",
        "-o", "algo.hyperparams.batch_size=32",
        "-o", "benchmark.as_n_samples=500",
        "-o", "benchmark.as_sigma_n_paths=3",
    ]
    assert train_mod.main(argv) == 0
    return str(tmp_path / "synthetic_rough_heston" / run_name)


def _read_metrics_excluding_wall_clock(path) -> list[list[str]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][-1] == "wall_clock_s"
    return [row[:-1] for row in rows]


@pytest.mark.slow
def test_end_to_end_pipeline_and_bit_identical_determinism(tmp_path):
    run_a = _train(tmp_path, "run_a")
    run_b = _train(tmp_path, "run_b")

    for rel in EXPECTED_FILES:
        assert (tmp_path / "synthetic_rough_heston" / "run_a" / rel).exists(), rel

    # D10: same config + seed => bit-identical metrics (modulo wall clock)
    rows_a = _read_metrics_excluding_wall_clock(f"{run_a}/metrics.csv")
    rows_b = _read_metrics_excluding_wall_clock(f"{run_b}/metrics.csv")
    assert rows_a == rows_b
    assert len(rows_a) == 1 + 5  # header + one row per episode

    # evaluation with CRN over all four policies
    assert evaluate_mod.main(["--run-dir", run_a, "--n-episodes", "2"]) == 0
    with open(f"{run_a}/eval/records.csv", newline="") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 2 * len(evaluate_mod.POLICIES)
    by_episode: dict[str, set[str]] = {}
    for row in records:
        by_episode.setdefault(row["episode"], set()).add(row["env_seed"])
        assert np.isfinite(float(row["return_undisc"]))
    for seeds in by_episode.values():
        assert len(seeds) == 1  # CRN: one shared env seed per episode

    # AUDIT C.4: one reward definition for all policies, recorded in metadata
    metadata = yaml.safe_load(open(f"{run_a}/eval/metadata.yaml"))
    resolved = yaml.safe_load(open(f"{run_a}/config_resolved.yaml"))
    assert metadata["reward_params_shared_by_all_policies"] == resolved["reward"]

    # regret joins on episode and asserts CRN seed equality internally
    for benchmark in ("as", "twap"):
        assert regret_mod.main(["--run-dir", run_a, "--benchmark", benchmark]) == 0
        with open(f"{run_a}/eval/regret_{benchmark}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        expected = sum(float(r["regret"]) for r in rows)
        assert float(rows[-1]["cum_regret"]) == pytest.approx(expected)


@pytest.mark.slow
def test_evaluate_is_reproducible(tmp_path):
    """Running evaluate.py twice on the same run dir gives identical records
    (the eval seed stream is a pure function of seed.txt; D10)."""
    run = _train(tmp_path, "run_c")
    assert evaluate_mod.main(["--run-dir", run, "--n-episodes", "2"]) == 0
    first = open(f"{run}/eval/records.csv").read()
    assert evaluate_mod.main(["--run-dir", run, "--n-episodes", "2"]) == 0
    assert open(f"{run}/eval/records.csv").read() == first
