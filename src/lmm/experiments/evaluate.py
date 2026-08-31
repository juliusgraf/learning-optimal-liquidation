"""Frozen-policy evaluation with common random numbers (Phase 4).

Evaluates the checkpointed DQN, the untrained "initial" DQN, and the AS/TWAP
benchmarks on the SAME seed set (CRN: identical env seed for every policy
within an episode; fixes AUDIT N10), under ONE reward definition and one
bounded executable action envelope for all policies. The primary economic
outcome is liquidation P&L less cancellation costs and the terminal inventory
penalty. Undiscounted and discounted shaped returns are retained as learning
diagnostics; the latter is also consumed by
paired fixed-policy economic comparisons.

The eval seed stream is the dedicated ``env_final_eval`` component of the
run's seed bundle — disjoint from the training and periodic-eval streams by
construction (ruling D10). Writes eval/records.csv (one row per (policy,
episode)) and eval/metadata.yaml.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

from lmm.agents.benchmarks import ASBenchmarkAgent, TWAPBenchmarkAgent
from lmm.agents.base import ENVIRONMENT_CONTRACT
from lmm.config import load_config, to_dict
from lmm.env.mdp import make_env
from lmm.experiments.tracing import EpisodeTraceRecorder
from lmm.experiments.train import draw_seed, make_agent, wrap_env_for_agent
from lmm.rl.loops import SEED_COMPONENTS, EpisodeResult, run_episode
from lmm.utils.seeding import seed_everything

__all__ = ["build_parser", "main", "RECORD_COLUMNS", "POLICIES"]

# Reference labels; the learned label is resolved from ``cfg.algo.name``.
POLICIES = ("initial", "as", "twap")

RECORD_COLUMNS = [
    "policy",
    "episode",
    "env_seed",
    "return_undisc",
    "return_disc",
    "training_return",
    "clob_economic_cash",
    "auction_economic_cash",
    "cancellation_fees",
    "residual_mark",
    "terminal_penalty",
    "clob_shaping_adjustment",
    "auction_interim_shaping",
    "auction_terminal_shaping",
    "clob_reward_sum",
    "auction_step_reward_sum",
    "terminal_reward",
    "initial_mid",
    "initial_inventory",
    "clob_exec_qty",
    "clob_cash",
    "auction_exec_qty",
    "auction_cash",
    "cancel_cost",
    "residual_liquidation_price",
    "residual_liquidation_cash",
    "liquidation_pnl_gross",
    "liquidation_pnl_net",
    "inventory_penalty",
    "economic_objective",
    "pnl",
    "risk_adjusted_pnl",
    "pnl_per_initial_notional",
    "risk_adjusted_pnl_per_initial_notional",
    "pnl_bps",
    "risk_adjusted_pnl_bps",
    "negative_terminal_inventory",
    "negative_terminal_inventory_magnitude",
    "S_cl",
    "Z_tau_cl",
    "I_final",
    "H_at_tau_op",
    "cancel_count",
    "n_steps",
    "n_clob_steps",
    "n_degenerate_fallbacks",
    "agent_price_displacement",
    "leave_agent_out_price",
    "continuous_clearing_price",
    "rounding_residual",
    "Q_supply",
    "Q_demand",
    "rho_supply",
    "rho_demand",
    "carryover_slope",
    "fallback_used",
    "self_trade_count",
    "H_forecast_bias",
    "H_forecast_MAE",
    "H_forecast_RMSE",
    "H_MAE_improvement_vs_mid",
    "H_MAE_improvement_vs_open_mid",
]


def _record_row(policy: str, episode: int, res: EpisodeResult) -> list[str]:
    def fmt(v):
        return format(v, ".17g") if isinstance(v, float) else str(v)

    return [
        policy,
        str(episode),
        str(res.env_seed),
        fmt(res.return_undisc),
        fmt(res.return_disc),
        fmt(res.training_return),
        fmt(res.clob_economic_cash),
        fmt(res.auction_economic_cash),
        fmt(res.cancellation_fees),
        fmt(res.residual_mark),
        fmt(res.terminal_penalty),
        fmt(res.clob_shaping_adjustment),
        fmt(res.auction_interim_shaping),
        fmt(res.auction_terminal_shaping),
        fmt(res.clob_reward_sum),
        fmt(res.auction_step_reward_sum),
        fmt(res.terminal_reward),
        fmt(res.initial_mid),
        fmt(res.initial_inventory),
        fmt(res.clob_exec_qty),
        fmt(res.clob_cash),
        fmt(res.auction_exec_qty),
        fmt(res.auction_cash),
        fmt(res.cancel_cost),
        fmt(res.residual_liquidation_price),
        fmt(res.residual_liquidation_cash),
        fmt(res.liquidation_pnl_gross),
        fmt(res.liquidation_pnl_net),
        fmt(res.inventory_penalty),
        fmt(res.economic_objective),
        fmt(res.pnl),
        fmt(res.risk_adjusted_pnl),
        fmt(res.pnl_per_initial_notional),
        fmt(res.risk_adjusted_pnl_per_initial_notional),
        fmt(res.pnl_bps),
        fmt(res.risk_adjusted_pnl_bps),
        str(int(res.negative_terminal_inventory)),
        fmt(res.negative_terminal_inventory_magnitude),
        fmt(res.s_cl),
        fmt(res.z_tau_cl),
        fmt(res.i_final),
        fmt(res.h_at_tau_op),
        str(res.cancel_count),
        str(res.n_steps),
        str(res.n_clob_steps),
        str(res.n_degenerate_fallbacks),
        fmt(res.agent_price_displacement),
        fmt(res.leave_agent_out_price),
        fmt(res.continuous_clearing_price),
        fmt(res.rounding_residual),
        fmt(res.q_supply),
        fmt(res.q_demand),
        fmt(res.rho_supply),
        fmt(res.rho_demand),
        fmt(res.carryover_slope),
        str(int(res.fallback_used)),
        str(res.self_trade_count),
        fmt(res.h_forecast_bias),
        fmt(res.h_forecast_mae),
        fmt(res.h_forecast_rmse),
        fmt(res.h_mae_improvement_vs_mid),
        fmt(res.h_mae_improvement_vs_open_mid),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-evaluate",
        description="Evaluate a frozen policy vs benchmarks with CRN.",
    )
    parser.add_argument("--run-dir", required=True, help="training run directory")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="checkpoint to evaluate (without .pt). Default 'best' = early "
        "stopping: the best-validation snapshot (selected on the env_eval "
        "stream, disjoint from the env_final_eval test seeds). Pass 'final' "
        "for the last-episode checkpoint.",
    )
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="evaluation episodes (default: rl.test_size)")
    parser.add_argument("--symbol", default=None, help="historical setting: symbol to replay")
    parser.add_argument(
        "--trace-episodes", type=int, default=0,
        help="write per-step anatomy traces for the first N eval episodes "
        "(eval/traces/<policy>_ep<i>.csv) for make_figures; 0 = none",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    if cfg.algo is None:
        raise ValueError("resolved run config has no learned algorithm")
    learned_name = cfg.algo.name
    policies = (learned_name, *POLICIES)
    master_seed = int((run_dir / "seed.txt").read_text().strip())
    seeds = seed_everything(master_seed, SEED_COMPONENTS, seed_torch=True)
    n_episodes = (
        args.n_episodes
        if args.n_episodes is not None
        else int(cfg.rl.test_size)
    )

    selection_path = run_dir / "checkpoints" / "best_selection.yaml"
    selection = yaml.safe_load(selection_path.read_text()) if selection_path.exists() else None
    if args.checkpoint == "best":
        if not isinstance(selection, dict):
            raise ValueError(
                "best.pt has no best_selection.yaml provenance; retrain or explicitly "
                "evaluate a named non-best checkpoint"
            )
        if selection.get("metric") != cfg.rl.checkpoint_metric:
            raise ValueError(
                "best.pt selection metric does not match the resolved config: "
                f"selection={selection.get('metric')!r}, "
                f"config={cfg.rl.checkpoint_metric!r}"
            )

    # Disjoint final-eval seed stream (D10); the SAME list for every policy
    # is the CRN coupling (the env's exogenous draws are policy-independent).
    eval_rng = seeds.generators["env_final_eval"]
    eval_seeds = [draw_seed(eval_rng) for _ in range(n_episodes)]

    # One env per policy, all from the SAME resolved config => identical
    # RewardParams for all policies (AUDIT C.4; asserted in tests). The
    # learned-policy envs (dqn/initial) are wrapped in the continuous adapter
    # for DDPG/TD3/SAC runs; the benchmark envs stay raw (benchmarks submit
    # ClobAction/AuctionAction objects directly).
    learned = {learned_name, "initial"}
    benchmark_cfg = replace(
        cfg,
        reward=replace(
            cfg.reward,
            shaping_enabled=False,
            clawback_shaping=False,
        ),
    )
    envs = {
        p: (
            wrap_env_for_agent(
                make_env(cfg, symbol=args.symbol, data_split="test"), cfg
            )
            if p in learned
            else make_env(
                benchmark_cfg, symbol=args.symbol, data_split="test"
            )
        )
        for p in policies
    }

    learned_agent = make_agent(cfg, seeds)
    learned_agent.load(run_dir / "checkpoints" / f"{args.checkpoint}.pt")
    initial = make_agent(cfg, seeds)
    initial.load(run_dir / "checkpoints" / "initial.pt")
    as_agent = ASBenchmarkAgent(benchmark_cfg)
    # AS calibration is an estimated policy parameter.  It is fitted on the
    # training split and then frozen before the held-out test episodes below.
    as_calibration_env = make_env(
        benchmark_cfg, symbol=args.symbol, data_split="train"
    )
    calibration = as_agent.calibrate(
        as_calibration_env,
        rng_k=seeds.generators["as_calibration"],
        rng_sigma=seeds.generators["as_sigma_paths"],
    )
    twap = TWAPBenchmarkAgent(benchmark_cfg)
    agents = {
        learned_name: learned_agent,
        "initial": initial,
        "as": as_agent,
        "twap": twap,
    }
    for name, agent in agents.items():
        agent.bind(envs[name])

    records_path = run_dir / "eval" / "records.csv"
    grid_rows: list[dict] = []
    forecast_rows: list[dict] = []
    action_rows: list[dict] = []
    proposal_rows: list[dict] = []
    with records_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(RECORD_COLUMNS)
        for episode, env_seed in enumerate(eval_seeds):
            for name, agent in agents.items():
                agent.start_episode(episode)
                res = run_episode(envs[name], agent, env_seed, chi=cfg.rl.chi, train=False)
                writer.writerow(_record_row(name, episode, res))
                if name == learned_name:
                    grid_rows.append(
                        {
                            "episode": episode,
                            "env_seed": env_seed,
                            **res.realized_grid,
                        }
                    )
                for row in res.forecast_records:
                    forecast_rows.append(
                        {
                            "policy": name,
                            "episode": episode,
                            "env_seed": env_seed,
                            "opening_mid": res.residual_liquidation_price,
                            **row,
                        }
                    )
                for row in res.action_records:
                    action_rows.append(
                        {
                            "policy": name,
                            "episode": episode,
                            "env_seed": env_seed,
                            **row,
                        }
                    )
                event_names = sorted(
                    {
                        key.removeprefix("proposal_").rsplit("_", 1)[0]
                        for key in res.diagnostics
                        if key.startswith("proposal_")
                    }
                )
                for event_name in event_names:
                    counts = {
                        disposition: int(
                            res.diagnostics.get(
                                f"proposal_{event_name}_{disposition}", 0.0
                            )
                        )
                        for disposition in (
                            "proposed",
                            "accepted",
                            "rejected",
                            "ineligible",
                        )
                    }
                    proposed = counts["proposed"]
                    proposal_rows.append(
                        {
                            "policy": name,
                            "episode": episode,
                            "env_seed": env_seed,
                            "event": event_name,
                            **counts,
                            "acceptance_rate": (
                                counts["accepted"] / proposed if proposed else float("nan")
                            ),
                            "rejection_rate": (
                                counts["rejected"] / proposed if proposed else float("nan")
                            ),
                        }
                    )

    with (run_dir / "eval" / "realized_grids.jsonl").open("w") as f:
        for row in grid_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def write_dict_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                encoded = dict(row)
                for key, value in tuple(encoded.items()):
                    if isinstance(value, list):
                        encoded[key] = json.dumps(value, separators=(",", ":"))
                writer.writerow(encoded)

    write_dict_rows(
        run_dir / "eval" / "h_forecasts.csv",
        forecast_rows,
        [
            "policy", "episode", "env_seed", "time", "decision_index", "phase",
            "time_to_close", "h_cl", "s_mid", "opening_mid", "s_cl", "h_error", "mid_error",
        ],
    )
    write_dict_rows(
        run_dir / "eval" / "action_diagnostics.csv",
        action_rows,
        [
            "policy", "episode", "env_seed", "time", "decision_index", "phase",
            "raw_action", "raw_proposal", "projected_action_five",
            "executed_volume", "executed_delta", "executed_K_a", "executed_offset",
            "executed_cancel", "input_clipped", "bound_saturation_count",
            "rounded_coordinate_count", "inventory_projection",
            "offset_admissibility_projection", "cancel_threshold_positive",
            "cancel_executed",
        ],
    )
    action_summary_rows: list[dict] = []
    for policy in policies:
        for phase in ("clob", "auction"):
            rows = [
                row
                for row in action_rows
                if row["policy"] == policy and row["phase"] == phase
            ]
            continuous = [row for row in rows if "raw_proposal" in row]
            n = len(continuous)
            if n:
                mappings: dict[tuple[float, ...], set[tuple[float, ...]]] = {}
                for row in continuous:
                    projected = tuple(float(x) for x in row["projected_action_five"])
                    proposal = tuple(round(float(x), 8) for x in row["raw_proposal"])
                    mappings.setdefault(projected, set()).add(proposal)
                collision_buckets = {
                    projected
                    for projected, proposals in mappings.items()
                    if len(proposals) > 1
                }
                collision_steps = sum(
                    1
                    for row in continuous
                    if tuple(float(x) for x in row["projected_action_five"])
                    in collision_buckets
                )

                def frequency(key: str) -> float:
                    return float(np.mean([bool(row.get(key, False)) for row in continuous]))

                threshold_n = sum(
                    bool(row.get("cancel_threshold_positive", False))
                    for row in continuous
                )
                cancel_n = sum(
                    bool(row.get("cancel_executed", False)) for row in continuous
                )
                action_summary_rows.append(
                    {
                        "policy": policy,
                        "phase": phase,
                        "n_projected_actions": n,
                        "input_clipping_frequency": frequency("input_clipped"),
                        "bound_saturation_action_frequency": float(
                            np.mean(
                                [
                                    int(row.get("bound_saturation_count", 0)) > 0
                                    for row in continuous
                                ]
                            )
                        ),
                        "mean_bound_saturated_coordinates": float(
                            np.mean(
                                [
                                    int(row.get("bound_saturation_count", 0))
                                    for row in continuous
                                ]
                            )
                        ),
                        "rounding_action_frequency": float(
                            np.mean(
                                [
                                    int(row.get("rounded_coordinate_count", 0)) > 0
                                    for row in continuous
                                ]
                            )
                        ),
                        "mean_rounded_coordinates": float(
                            np.mean(
                                [
                                    int(row.get("rounded_coordinate_count", 0))
                                    for row in continuous
                                ]
                            )
                        ),
                        "inventory_projection_frequency": frequency(
                            "inventory_projection"
                        ),
                        "offset_admissibility_projection_frequency": frequency(
                            "offset_admissibility_projection"
                        ),
                        "cancel_threshold_positive_count": threshold_n,
                        "cancel_executed_count": cancel_n,
                        "cancel_execution_rate_given_positive_threshold": (
                            cancel_n / threshold_n if threshold_n else float("nan")
                        ),
                        "unique_projected_actions": len(mappings),
                        "many_to_one_projection_bucket_count": len(collision_buckets),
                        "many_to_one_projection_step_frequency": collision_steps / n,
                    }
                )
            else:
                action_summary_rows.append(
                    {
                        "policy": policy,
                        "phase": phase,
                        "n_projected_actions": 0,
                    }
                )
    write_dict_rows(
        run_dir / "eval" / "action_diagnostics_summary.csv",
        action_summary_rows,
        [
            "policy", "phase", "n_projected_actions",
            "input_clipping_frequency", "bound_saturation_action_frequency",
            "mean_bound_saturated_coordinates", "rounding_action_frequency",
            "mean_rounded_coordinates", "inventory_projection_frequency",
            "offset_admissibility_projection_frequency",
            "cancel_threshold_positive_count", "cancel_executed_count",
            "cancel_execution_rate_given_positive_threshold",
            "unique_projected_actions", "many_to_one_projection_bucket_count",
            "many_to_one_projection_step_frequency",
        ],
    )
    write_dict_rows(
        run_dir / "eval" / "proposal_diagnostics.csv",
        proposal_rows,
        [
            "policy", "episode", "env_seed", "event", "proposed", "accepted",
            "rejected", "ineligible", "acceptance_rate", "rejection_rate",
        ],
    )
    forecast_summary_rows: list[dict] = []
    for policy in policies:
        rows = [row for row in forecast_rows if row["policy"] == policy]
        buckets = sorted(
            {int(float(row["time_to_close"]) // 10) * 10 for row in rows},
            reverse=True,
        )
        for bucket in buckets:
            selected = [
                row
                for row in rows
                if int(float(row["time_to_close"]) // 10) * 10 == bucket
            ]
            h_error = np.asarray([float(row["h_error"]) for row in selected])
            mid_error = np.asarray([float(row["mid_error"]) for row in selected])
            opening_error = np.asarray(
                [float(row["opening_mid"]) - float(row["s_cl"]) for row in selected]
            )
            forecast_summary_rows.append(
                {
                    "policy": policy,
                    "time_to_close_bin_lo": bucket,
                    "time_to_close_bin_hi": bucket + 10,
                    "n": len(selected),
                    "signed_bias": float(np.mean(h_error)),
                    "mae": float(np.mean(np.abs(h_error))),
                    "rmse": float(np.sqrt(np.mean(h_error**2))),
                    "mae_improvement_vs_contemporaneous_mid": float(
                        np.mean(np.abs(mid_error) - np.abs(h_error))
                    ),
                    "mae_improvement_vs_opening_mid": float(
                        np.mean(np.abs(opening_error) - np.abs(h_error))
                    ),
                }
            )
    write_dict_rows(
        run_dir / "eval" / "h_forecast_summary.csv",
        forecast_summary_rows,
        [
            "policy", "time_to_close_bin_lo", "time_to_close_bin_hi", "n",
            "signed_bias", "mae", "rmse",
            "mae_improvement_vs_contemporaneous_mid",
            "mae_improvement_vs_opening_mid",
        ],
    )

    # Per-step anatomy traces (Phase 7): replay the FIRST n_trace eval seeds
    # for every policy with a recorder attached. The env resets by seed so this
    # reproduces the records.csv trajectories exactly (frozen/greedy policies
    # consume no exploration RNG; benchmarks reset per episode in start_episode).
    n_trace = min(args.trace_episodes, n_episodes)
    if n_trace > 0:
        traces_dir = run_dir / "eval" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        for episode in range(n_trace):
            env_seed = eval_seeds[episode]
            for name, agent in agents.items():
                agent.start_episode(episode)
                rec = EpisodeTraceRecorder()
                run_episode(
                    envs[name], agent, env_seed, chi=cfg.rl.chi, train=False, on_step=rec
                )
                rec.write(traces_dir / f"{name}_ep{episode}.csv")

    metadata = {
        "master_seed": master_seed,
        "checkpoint": str(run_dir / "checkpoints" / f"{args.checkpoint}.pt"),
        "early_stopping": args.checkpoint == "best",
        "n_episodes": n_episodes,
        "policies": list(policies),
        "learned_policy_label": learned_name,
        "evaluation_episode_seeds": eval_seeds,
        "crn": "identical env seed per episode across all policies (env_final_eval stream)",
        "primary_outcome": "risk_adjusted_pnl",
        "liquidation_pnl_formula": "clob_cash + S_cl*Z + S_mid_tau_op*I_final - S_mid_0*I_0",
        "economic_objective_formula": "liquidation_pnl_gross - cancel_cost - inventory_penalty",
        "auction_cancel_mode": cfg.actions.auction_cancel_mode,
        "residual_inventory_convention": "deemed liquidated at the frozen, policy-independent auction-open mid",
        "return_convention": "return_undisc/return_disc are shaped training returns; "
        f"discount mode={cfg.rl.discount_mode}, chi={cfg.rl.chi}",
        "checkpoint_selection": selection,
        "learned_reward_params": to_dict(cfg.reward),
        "benchmark_reward_params": to_dict(benchmark_cfg.reward),
        "benchmarks_use_economic_rewards_only": True,
        "as_calibration": {k: float(v) for k, v in calibration.items()},
        "trace_episodes": n_trace,
        "setting": cfg.experiment.name,
        "symbol": args.symbol,  # historical ticker replayed (None for synthetic)
        "artifact_schema_version": cfg.experiment.artifact_schema_version,
        "environment_contract": ENVIRONMENT_CONTRACT,
        "dqn_equal_q_tie_breaking": "first action in lexicographic grid order",
        "ablation_label": cfg.experiment.ablation_label,
        "auction_enabled": cfg.experiment.auction_enabled,
        "normalization_state_file": str(run_dir / "feature_normalizer.yaml"),
        "historical_data_split": (
            to_dict(cfg.midprice.historical)
            if cfg.midprice.historical is not None
            else None
        ),
        "policy_evaluation_split": "test",
        "checkpoint_selection_split": "validation",
        "normalization_and_benchmark_calibration_split": "train",
    }
    (run_dir / "eval" / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    print(f"wrote {records_path} ({n_episodes} episodes x {len(policies)} policies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
