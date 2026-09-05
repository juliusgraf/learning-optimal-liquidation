"""Dimensional audit guards, including finite-cap Pareto edge cases."""
from dataclasses import replace
import json
import math

import pytest

from lmm.config import ConfigError, load_config
from lmm.experiments.calibration import calibration_scales, pareto_capped_mean
from lmm.env.rewards import clob_reward


def test_capped_pareto_survival_integral_at_and_below_shape_one():
    assert pareto_capped_mean(2, 1, 30) == pytest.approx(2+2*math.log(15))
    assert pareto_capped_mean(2, .5, 8) == pytest.approx(6)
    assert pareto_capped_mean(2, 2.5, 1) == 1
    assert pareto_capped_mean(2, 1+1e-10, 30) == pytest.approx(2+2*math.log(15))


def test_no_quadratic_penalty_has_no_finite_local_stationary_point():
    cfg = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml')
    cfg = replace(cfg, reward=replace(cfg.reward, lambda_inv=0))
    scales = calibration_scales(cfg)
    assert scales['auction_local_excess_sale_stationary_point'] is None
    json.dumps(scales, allow_nan=False)


def test_k_star_is_tick_count_but_local_deduction_depends_on_price_level():
    cfg = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml')
    scales = calibration_scales(cfg)
    assert scales['clob_clipping_distance_ticks'] == 10000
    assert scales['clob_clipping_distance_fraction_of_S0'] == 1
    # One tick below the forecast deducts one tick at the reference price.
    cash = 100.
    assert cash-clob_reward(100, 1, 100.01, 10000, .01) == pytest.approx(.01)
    # A visually smaller clipping parameter magnifies the local deduction.
    assert cash-clob_reward(100, 1, 100.01, 100, .01) == pytest.approx(1.)
    # Price-unit conversion preserves the multiplier and scales rewards.
    assert clob_reward(500, 2, 500.5, 10000, .05) == pytest.approx(5*clob_reward(100, 2, 100.1, 10000, .01))


@pytest.mark.parametrize('half_life', ['.nan', '.inf', '-1'])
def test_invalid_learning_rate_schedule_rejected(half_life):
    with pytest.raises(ConfigError, match='half life'):
        load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml',
                    overrides=[f'rl.learning_rate_half_life_episodes={half_life}'])
