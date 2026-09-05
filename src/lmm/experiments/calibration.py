"""Audit the units and incentive scales of a resolved numerical calibration."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lmm.config import ExperimentConfig, load_config


def calibration_scales(cfg: ExperimentConfig) -> dict[str, float | str | None]:
    s, alpha = cfg.grid.S0, cfg.grid.alpha
    weight = cfg.reward.auction_shaping_weight
    return {
        'interpretation': 'dimensional diagnostics, not estimates of exchange liquidity',
        'model_tick_bps': 10000*alpha/s,
        'model_best_quoted_spread_bps': 20000*alpha/s,
        'clob_zero_reward_gap': cfg.reward.k_star*alpha,
        'clob_clipping_distance_ticks': cfg.reward.k_star,
        'clob_clipping_distance_bps': 10000*cfg.reward.k_star*alpha/s,
        'clob_clipping_distance_fraction_of_S0': cfg.reward.k_star*alpha/s,
        'clob_penalty_per_unit_price_gap_at_S0': s/(cfg.reward.k_star*alpha),
        'clob_deduction_per_unit_at_one_tick_gap_and_S0': s/cfg.reward.k_star,
        'purchase_subsidy_per_unit_at_S0': weight*cfg.reward.q*s,
        'auction_fictive_sale_credit_per_unit_at_S0': weight*s,
        'auction_local_excess_sale_stationary_point': weight*s/(2*cfg.reward.lambda_inv) if cfg.reward.lambda_inv else None,
        'inventory_penalty_one_unit_in_ticks': cfg.reward.lambda_inv/alpha,
        'ten_percent_residual_penalty_bps_of_initial_notional': 10000*cfg.reward.lambda_inv*(.1*cfg.grid.I0)**2/(s*cfg.grid.I0),
        'marginal_inventory_penalty_at_ten_percent_residual_ticks': 2*cfg.reward.lambda_inv*.1*cfg.grid.I0/alpha,
        'auction_slope_quantum': cfg.actions.beta,
        'auction_slope_max_per_schedule': cfg.actions.auction_K_grid_max,
        'nominal_min_slope_one_tick_quantity': cfg.actions.beta*alpha,
        'nominal_max_schedule_local_quantity': cfg.actions.auction_K_grid_max*alpha*cfg.actions.B_max,
        'nominal_max_schedule_fraction_of_initial_inventory': cfg.actions.auction_K_grid_max*alpha*cfg.actions.B_max/cfg.grid.I0,
        'nominal_all_slots_local_quantity': cfg.grid.h*cfg.actions.auction_K_grid_max*alpha*cfg.actions.B_max,
        'maximum_cancellation_cost_bps_of_initial_notional': 10000*(cfg.grid.h-1)*cfg.reward.d/(s*cfg.grid.I0),
        'expected_clob_buy_volume_unconditional': cfg.clob_flow.lambda0*cfg.grid.tau_op*pareto_capped_mean(cfg.clob_flow.v_m,cfg.clob_flow.gamma_m,cfg.clob_flow.V_max),
    }


def pareto_capped_mean(scale: float, exponent: float, cap: float) -> float:
    """E[min(X,cap)] from the Pareto survival integral, including shape 1."""
    if not all(math.isfinite(x) and x > 0 for x in (scale, exponent, cap)):
        raise ValueError('Pareto scale, exponent and cap must be finite and positive')
    if cap <= scale:
        return cap
    log_ratio = math.log(cap/scale)
    if exponent == 1:
        return scale*(1+log_ratio)
    return scale*(1+math.expm1((1-exponent)*log_ratio)/(1-exponent))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(calibration_scales(load_config(*args.config)), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result+'\n')
    print(result)


if __name__ == '__main__':
    main()
