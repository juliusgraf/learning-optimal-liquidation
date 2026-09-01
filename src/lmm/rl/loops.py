"""Shared episode runner (Phase 4): ONE loop drives every policy.

Running the DQN, the benchmarks and the initial-DQN baseline through the
same function is the structural common-random-numbers guarantee (ruling D10,
AUDIT N10): the env's exogenous draws are policy-independent, so two policies
given the same env seed see the same market.

The revised objective is undiscounted.  Every transition uses Bellman factor
one, including the irregular CLOB clock and the terminal clearing reward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from lmm.agents.base import BELLMAN_FACTOR, Agent, Transition
from lmm.env.mdp import MarketMakingEnv
from lmm.experiments.accounting import compute_liquidation_accounting

__all__ = ["SEED_COMPONENTS", "EpisodeResult", "run_episode"]

# Canonical seed-bundle component list (ruling D10). Streams are a function
# of the SORTED full component set (utils/seeding.py), so every entry point
# (train / evaluate / tests) must use THIS list even when it consumes only a
# subset — otherwise the shared streams shift.
SEED_COMPONENTS = (
    "normalizer_env",
    "normalizer_policy",
    "env_train",
    "env_eval",
    "env_final_eval",
    "exploration",
    "replay_clob",
    "replay_auction",
    "as_calibration",
    "as_sigma_paths",
)


@dataclass
class EpisodeResult:
    """Per-episode record (one metrics.csv row is built from this)."""

    env_seed: int
    return_undisc: float = 0.0
    return_disc: float = 0.0
    training_return: float = 0.0
    clob_reward_sum: float = 0.0
    auction_step_reward_sum: float = 0.0
    terminal_reward: float = 0.0
    clob_economic_cash: float = 0.0
    auction_economic_cash: float = 0.0
    cancellation_fees: float = 0.0
    residual_mark: float = 0.0
    terminal_penalty: float = 0.0
    clob_shaping_adjustment: float = 0.0
    auction_interim_shaping: float = 0.0
    auction_shaping_clawback: float = 0.0
    auction_terminal_shaping: float = 0.0
    reward_baseline_adjustment: float = 0.0
    initial_mid: float = float("nan")
    initial_inventory: float = float("nan")
    clob_exec_qty: float = 0.0
    clob_cash: float = 0.0
    auction_exec_qty: float = 0.0
    auction_cash: float = 0.0
    cancel_cost: float = 0.0
    residual_liquidation_price: float = float("nan")
    residual_liquidation_cash: float = float("nan")
    liquidation_pnl_gross: float = float("nan")
    liquidation_pnl_net: float = float("nan")
    inventory_penalty: float = float("nan")
    economic_objective: float = float("nan")
    pnl: float = float("nan")
    risk_adjusted_pnl: float = float("nan")
    pnl_per_initial_notional: float = float("nan")
    risk_adjusted_pnl_per_initial_notional: float = float("nan")
    pnl_bps: float = float("nan")
    risk_adjusted_pnl_bps: float = float("nan")
    negative_terminal_inventory: bool = False
    negative_terminal_inventory_magnitude: float = 0.0
    s_cl: float = float("nan")
    z_tau_cl: float = float("nan")
    i_final: float = float("nan")
    h_at_tau_op: float = float("nan")
    cancel_count: int = 0
    n_steps: int = 0
    n_clob_steps: int = 0
    n_degenerate_fallbacks: int = 0
    agent_price_displacement: float = float("nan")
    leave_agent_out_price: float = float("nan")
    continuous_clearing_price: float = float("nan")
    rounding_residual: float = float("nan")
    q_supply: float = float("nan")
    q_demand: float = float("nan")
    rho_supply: float = float("nan")
    rho_demand: float = float("nan")
    carryover_slope: float = float("nan")
    fallback_used: bool = False
    self_trade_count: int = 0
    h_forecast_bias: float = float("nan")
    h_forecast_mae: float = float("nan")
    h_forecast_rmse: float = float("nan")
    h_mae_improvement_vs_mid: float = float("nan")
    h_mae_improvement_vs_open_mid: float = float("nan")
    realized_grid: dict[str, list[float] | float] = field(default_factory=dict)
    forecast_records: list[dict[str, float | int | str]] = field(default_factory=list)
    action_records: list[dict[str, object]] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)


def _transition_discount(chi: float, mode: str, t: float, t_next: float) -> float:
    """Compatibility helper returning the manuscript's unit Bellman factor.

    ``chi`` and ``mode`` are intentionally ignored; they remain in the
    signature so external callers do not need to change in lockstep.
    """
    dt = float(t_next) - float(t)
    if not dt > 0.0:
        raise ValueError(f"transition time must increase, got t={t}, t_next={t_next}")
    return BELLMAN_FACTOR


def run_episode(
    env: MarketMakingEnv,
    agent: Agent,
    env_seed: int,
    *,
    chi: float,
    train: bool,
    on_step: Callable[..., None] | None = None,
) -> EpisodeResult:
    """Play one full episode; in train mode the agent observes every
    transition and update() runs after each env step (the per-env-step
    schedule and gating live inside the agent). Eval mode (train=False) is
    greedy, stores nothing, and consumes no exploration randomness.

    ``on_step`` is an OPTIONAL per-step collector for episode-anatomy traces
    (Phase 7): when not None it is called once per step AFTER the step's
    bookkeeping with ``(step_idx, phase, t_decision, reward, cum_reward, info,
    env)`` — all pure reads (no RNG), so the trajectory is byte-identical to a
    run with ``on_step=None``. The default None path is the training/eval hot
    path and is unchanged."""
    agent.set_train(train)
    res = EpisodeResult(env_seed=env_seed)
    diag_lists: dict[str, list[float]] = {}

    raw_obs, _ = env.reset(seed=env_seed)
    obs = agent.preprocess_observation(raw_obs)
    grid = env.episode_grid
    res.realized_grid = {
        "time_unit": env.grid.time_unit,
        "clob_times": [float(x) for x in grid.clob_times],
        "auction_times": [float(x) for x in grid.auction_times],
        "terminal_time": float(grid.terminal_time),
    }
    res.initial_mid = float(env.s_mid)
    res.initial_inventory = float(env.inventory)
    done = False
    while not done:
        phase = env.phase
        t_decision = env.t
        res.forecast_records.append(
            {
                "time": float(t_decision),
                "decision_index": int(env.decision_index),
                "phase": phase,
                "h_cl": float(env.h_cl),
                "s_mid": float(env.s_mid),
            }
        )
        mask = env.action_mask()
        a = agent.act(obs, mask, phase, eval_mode=not train)
        raw_next_obs, reward, done, _, info = env.step(a)
        next_obs = agent.preprocess_observation(raw_next_obs)
        next_mask = None if done else env.action_mask()
        transition_discount = BELLMAN_FACTOR

        agent.observe(
            Transition(
                obs=obs,
                action=a,
                reward=reward,
                next_obs=next_obs,
                done=done,
                phase=phase,
                next_phase=env.phase,
                next_mask=next_mask,
                info=info,
                discount=transition_discount,
            )
        )
        if train:
            for k, v in agent.update(phase=phase).items():
                diag_lists.setdefault(k, []).append(v)

        res.n_steps += 1
        action_record: dict[str, object] = {
            "time": float(t_decision),
            "decision_index": int(info["decision_index"]),
            "phase": phase,
        }
        executed_action = info.get("action")
        for attr in ("volume", "delta", "K_a", "offset", "cancel"):
            if executed_action is not None and hasattr(executed_action, attr):
                action_record[f"executed_{attr}"] = float(getattr(executed_action, attr))
        if "raw_action_vec" in info:
            action_record["raw_action"] = [
                float(x) for x in np.asarray(info["raw_action_vec"]).reshape(-1)
            ]
        if "proposal_action_vec" in info:
            action_record["raw_proposal"] = [
                float(x) for x in np.asarray(info["proposal_action_vec"]).reshape(-1)
            ]
        if "projected_action_five" in info:
            action_record["projected_action_five"] = [
                float(x) for x in info["projected_action_five"]
            ]
        for key, value in info.get("projection_diagnostics", {}).items():
            action_record[key] = value
        res.action_records.append(action_record)
        res.return_undisc += reward
        res.training_return += reward
        r_term = float(info["terminal_reward"]) if done else 0.0
        # Kept as a schema-compatible alias; under factor one it is identical.
        res.return_disc += reward
        res.clob_economic_cash += float(info.get("clob_economic_cash", 0.0))
        res.auction_economic_cash += float(info.get("auction_economic_cash", 0.0))
        res.cancellation_fees += float(info.get("cancellation_fee", 0.0))
        res.residual_mark += float(info.get("residual_mark", 0.0))
        res.terminal_penalty += float(info.get("terminal_penalty", 0.0))
        res.clob_shaping_adjustment += float(
            info.get("clob_shaping_adjustment", 0.0)
        )
        step_clawback = float(info.get("auction_shaping_clawback", 0.0))
        res.auction_interim_shaping += float(
            info.get("auction_interim_shaping", 0.0)
        ) - step_clawback
        res.auction_shaping_clawback += step_clawback
        res.auction_terminal_shaping += float(
            info.get("auction_terminal_shaping", 0.0)
        )
        res.reward_baseline_adjustment += float(
            info.get("reward_baseline_adjustment", 0.0)
        )
        if phase == "clob":
            res.n_clob_steps += 1
            res.clob_reward_sum += reward
            qty = float(info["E_t"])
            res.clob_exec_qty += qty
            res.clob_cash += float(info["clob_economic_cash"])
        else:
            if res.n_steps == res.n_clob_steps + 1:
                res.h_at_tau_op = info["H_used"]  # first auction decision (t = tau_op)
            res.cancel_count += int(info["action"].cancel)
            res.cancel_cost += float(info["cancellation_fee"])
            res.auction_step_reward_sum += reward
        if done:
            res.terminal_reward = r_term
            if phase == "clob":
                res.clob_reward_sum -= r_term
            else:
                res.auction_step_reward_sum -= r_term
            res.s_cl = info["S_cl"]
            res.z_tau_cl = info["Z"]
            res.i_final = info["I_final"]
            res.auction_exec_qty = res.z_tau_cl
            res.residual_liquidation_price = float(env.s_mid)
            res.inventory_penalty = float(info["terminal_penalty"])
            accounting = compute_liquidation_accounting(
                initial_inventory=res.initial_inventory,
                initial_mid=res.initial_mid,
                clob_exec_qty=res.clob_exec_qty,
                clob_cash=res.clob_cash,
                auction_exec_qty=res.auction_exec_qty,
                auction_price=res.s_cl,
                final_inventory=res.i_final,
                residual_liquidation_price=res.residual_liquidation_price,
                cancel_cost=res.cancel_cost,
                inventory_penalty=res.inventory_penalty,
            )
            res.auction_cash = accounting.auction_cash
            res.residual_liquidation_cash = accounting.residual_liquidation_cash
            res.liquidation_pnl_gross = accounting.liquidation_pnl_gross
            res.liquidation_pnl_net = accounting.liquidation_pnl_net
            res.economic_objective = accounting.economic_objective
            res.pnl = accounting.liquidation_pnl_net
            res.risk_adjusted_pnl = accounting.economic_objective
            initial_notional = res.initial_mid * res.initial_inventory
            if initial_notional <= 0.0:
                raise ValueError("initial notional must be positive")
            res.pnl_per_initial_notional = res.pnl / initial_notional
            res.risk_adjusted_pnl_per_initial_notional = (
                res.risk_adjusted_pnl / initial_notional
            )
            res.pnl_bps = 10_000.0 * res.pnl_per_initial_notional
            res.risk_adjusted_pnl_bps = (
                10_000.0 * res.risk_adjusted_pnl_per_initial_notional
            )
            res.negative_terminal_inventory = res.i_final < 0.0
            res.negative_terminal_inventory_magnitude = max(-res.i_final, 0.0)
            if not np.isclose(res.clob_cash, res.clob_economic_cash):
                raise AssertionError("CLOB economic cash decomposition drifted")
            if not np.isclose(res.auction_cash, res.auction_economic_cash):
                raise AssertionError("auction economic cash decomposition drifted")
            if not np.isclose(res.cancel_cost, res.cancellation_fees):
                raise AssertionError("cancellation-fee decomposition drifted")
            if not np.isclose(res.residual_liquidation_cash, res.residual_mark):
                raise AssertionError("residual-mark decomposition drifted")
            res.agent_price_displacement = float(info["agent_price_displacement"])
            res.leave_agent_out_price = float(info["leave_agent_out_tick_price"])
            res.continuous_clearing_price = float(info["continuous_price"])
            res.rounding_residual = float(info["clearing_residual"])
            res.q_supply = float(info["Q_supply"])
            res.q_demand = float(info["Q_demand"])
            res.rho_supply = float(info["rho_supply"])
            res.rho_demand = float(info["rho_demand"])
            res.carryover_slope = float(info["carryover_slope"])
            res.fallback_used = bool(info["fallback_used"])
            res.self_trade_count = int(info["self_trade_count"])
            if res.self_trade_count != 0:
                raise AssertionError("strategic self-trade count must remain zero")
            for event_name, dispositions in info["proposal_counts"].items():
                for disposition, value in dispositions.items():
                    res.diagnostics[
                        f"proposal_{event_name}_{disposition}"
                    ] = float(value)

            target = float(res.s_cl)
            opening_mid = float(env.s_mid)
            h_errors = np.asarray(
                [float(row["h_cl"]) - target for row in res.forecast_records]
            )
            mid_abs = np.asarray(
                [abs(float(row["s_mid"]) - target) for row in res.forecast_records]
            )
            h_abs = np.abs(h_errors)
            res.h_forecast_bias = float(np.mean(h_errors))
            res.h_forecast_mae = float(np.mean(h_abs))
            res.h_forecast_rmse = float(np.sqrt(np.mean(h_errors**2)))
            res.h_mae_improvement_vs_mid = float(np.mean(mid_abs - h_abs))
            res.h_mae_improvement_vs_open_mid = float(
                np.mean(abs(opening_mid - target) - h_abs)
            )
            for row in res.forecast_records:
                row["s_cl"] = target
                row["time_to_close"] = float(env.grid.tau_cl - float(row["time"]))
                row["h_error"] = float(row["h_cl"]) - target
                row["mid_error"] = float(row["s_mid"]) - target
        if on_step is not None:
            on_step(res.n_steps - 1, phase, t_decision, reward, res.return_undisc, info, env)
        obs = next_obs

    res.n_degenerate_fallbacks = env.eq2.n_degenerate_fallbacks
    for k, values in diag_lists.items():
        if k.startswith("n_grad_steps"):
            res.diagnostics[k] = float(sum(values))
        elif k.startswith("td_abs_max"):
            res.diagnostics[k] = float(max(values))
        else:
            res.diagnostics[k] = float(sum(values) / len(values))
    return res
