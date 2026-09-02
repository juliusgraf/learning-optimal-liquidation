"""Episode-trace recorder: schema + determinism (Phase 7, rulings D9/D10).

The ``on_step`` hook must be a pure observer — a traced episode must produce a
byte-identical :class:`EpisodeResult` to an untraced one (no RNG consumed).
"""

from __future__ import annotations

import pandas as pd
from helpers import load_dqn_cfg

from lmm.agents.benchmarks import TWAPBenchmarkAgent
from lmm.env.mdp import make_env
from lmm.experiments.tracing import TRACE_COLUMNS, EpisodeTraceRecorder
from lmm.experiments.train import make_agent, wrap_env_for_agent
from lmm.rl.loops import SEED_COMPONENTS, run_episode
from lmm.utils.seeding import seed_everything


def _agent_and_env():
    cfg = load_dqn_cfg()
    seeds = seed_everything(0, SEED_COMPONENTS, seed_torch=True)
    env = wrap_env_for_agent(make_env(cfg), cfg)
    agent = make_agent(cfg, seeds)
    return cfg, env, agent


def test_on_step_does_not_perturb_episode():
    cfg, env, agent = _agent_and_env()
    res_plain = run_episode(env, agent, 314, chi=cfg.rl.chi, train=False)
    rec = EpisodeTraceRecorder()
    res_traced = run_episode(env, agent, 314, chi=cfg.rl.chi, train=False, on_step=rec)

    assert res_traced.return_undisc == res_plain.return_undisc
    assert res_traced.return_disc == res_plain.return_disc
    assert res_traced.n_steps == res_plain.n_steps == len(rec.rows)
    assert res_traced.s_cl == res_plain.s_cl
    assert res_traced.i_final == res_plain.i_final


def test_trace_schema_and_terminal_row(tmp_path):
    cfg, env, agent = _agent_and_env()
    rec = EpisodeTraceRecorder()
    run_episode(env, agent, 7, chi=cfg.rl.chi, train=False, on_step=rec)

    assert list(rec.rows[0].keys()) == TRACE_COLUMNS
    assert rec.rows[0]["phase"] == "clob"
    last = rec.rows[-1]
    assert last["phase"] == "auction" and last["is_terminal"] == 1
    assert last["S_cl"] != "" and last["terminal_reward"] != ""
    assert last["auction_anchor"] == cfg.actions.auction_anchor
    assert isinstance(last["auction_anchor_b"], int)
    assert last["auction_anchor_price"] != ""

    out = tmp_path / "trace.csv"
    rec.write(out)
    df = pd.read_csv(out)
    assert list(df.columns) == TRACE_COLUMNS
    assert (df["phase"] == "clob").any() and (df["phase"] == "auction").any()


def test_benchmark_private_schedule_records_diagnostic_absolute_offset():
    cfg = load_dqn_cfg()
    env = make_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    rec = EpisodeTraceRecorder()
    run_episode(env, agent, 19, chi=cfg.rl.chi, train=False, on_step=rec)

    auction_rows = [row for row in rec.rows if row["phase"] == "auction"]
    assert auction_rows
    assert all(isinstance(row["act_b"], int) for row in auction_rows)
    assert all(
        row["act_ell"] == "" or isinstance(row["act_ell"], int)
        for row in auction_rows
    )
