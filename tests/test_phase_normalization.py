"""Independent phase statistics, training-only fitting and checkpoint integrity."""
import copy

import numpy as np
import pytest

from lmm.env.features import FeatureNormalizer
from lmm.config import load_config


def observations():
    rows = np.random.default_rng(517).normal(size=(12, 18))
    rows[:6, 0] = [0, 10, 30, 60, 90, 119]
    rows[6:, 0] = [120, 125, 130, 135, 140, 149]
    rows[:6, 1] = [100, 80, 50, 30, 10, 2]
    rows[6:, 1] = [0, 1, 2, 1, 0, 2]
    rows[:6, 13:18] = 0
    rows[6:, 13:18] = [0, 0, 1, 0, 100]
    return rows


def test_phase_scaling_uses_only_its_own_training_rows_and_keeps_time_convention():
    rows = observations()
    norm = FeatureNormalizer(150, phase_normalization=True, tau_op=120).fit(rows)
    out = norm.transform(rows)
    for selected in (slice(0, 6), slice(6, 12)):
        expected = (rows[selected, 1]-rows[selected, 1].mean())/rows[selected, 1].std()
        np.testing.assert_allclose(out[selected, 1], expected, rtol=1e-6)
    np.testing.assert_allclose(out[:, 0], rows[:, 0]/150, rtol=1e-6)
    before = copy.deepcopy(norm.state_dict())
    norm.transform(rows*1.001)
    assert norm.state_dict() == before
    with pytest.raises(RuntimeError, match='frozen'):
        norm.update(rows)


def test_phase_state_roundtrip_and_unseen_phase_rejection():
    rows = observations()
    norm = FeatureNormalizer(150, phase_normalization=True, tau_op=120).fit(rows)
    restored = FeatureNormalizer(150)
    restored.load_state_dict(norm.state_dict())
    np.testing.assert_array_equal(restored.transform(rows), norm.transform(rows))
    clob = FeatureNormalizer(150, phase_normalization=True, tau_op=120).fit(rows[:6])
    with pytest.raises(ValueError, match='no training observations'):
        clob.transform(rows[6])


def test_inventory_compression_is_invertible_and_leaves_clob_information_unchanged():
    rows = observations()
    rows[:, 3] = 100
    rows[:6, 13:18] = 0
    rows[6:, 13] = 2
    rows[6:, 14] = 199.8
    rows[6:, 15] = 10
    rows[6:, 16] = 0
    rows[6:, 17] = 1000
    rows[6:, 1] = [-100, -2, 0, 1, 10, 100]
    linear = FeatureNormalizer(150, relative_prices=True, auction_exposure_features=True)
    compressed = FeatureNormalizer(150, relative_prices=True, auction_exposure_features=True,
                                   auction_inventory_asinh=True, inventory_scale=2.)
    a, b = linear._coordinates(rows), compressed._coordinates(rows)
    np.testing.assert_array_equal(a[:6], b[:6])
    np.testing.assert_allclose(2*np.sinh(b[6:, [1,14]]), a[6:, [1,14]], atol=1e-12)
    compressed.fit(rows)
    restored = FeatureNormalizer(150)
    restored.load_state_dict(compressed.state_dict())
    np.testing.assert_array_equal(compressed.transform(rows), restored.transform(rows))


@pytest.mark.parametrize('key,value', [('tau_op', 150), ('phase_scale', [[0]*18]*2),
                                     ('phase_count', [1.5, 2]), ('phase_mean', [[float('nan')]*18]*2)])
def test_corrupt_phase_state_rejected(key, value):
    norm = FeatureNormalizer(150, phase_normalization=True, tau_op=120).fit(observations())
    state = norm.state_dict()
    state[key] = value
    with pytest.raises(ValueError):
        FeatureNormalizer(150).load_state_dict(state)


@pytest.mark.parametrize('treatment,enabled', [('representation_pooled', False), ('representation_phase_asinh', True)])
def test_representation_treatments_match_all_algorithms_without_changing_economics(treatment, enabled):
    for algo in ('dqn','ddpg','td3','sac'):
        paths = ['configs/base.yaml','configs/synthetic_rough_heston.yaml',f'configs/algo/{algo}.yaml']
        original = load_config(*paths)
        matched = load_config(*paths, f'configs/treatment/{treatment}.yaml')
        assert matched.rl.phase_normalization is enabled
        assert matched.rl.auction_inventory_asinh is enabled
        for field in ('grid','actions','reward','clob_flow','auction_flow','midprice','algo','features'):
            assert getattr(matched, field) == getattr(original, field)


@pytest.mark.parametrize('algo', ['dqn','ddpg','td3','sac'])
def test_no_auction_terminal_observation_uses_the_visited_clob_phase(algo):
    from lmm.experiments.train import make_agent, fit_feature_normalizer, wrap_env_for_agent
    from lmm.env.mdp import make_env
    from lmm.rl.loops import run_episode, SEED_COMPONENTS
    from lmm.utils.seeding import seed_everything
    cfg = load_config('configs/base.yaml','configs/synthetic_rough_heston.yaml',
        f'configs/algo/{algo}.yaml','configs/treatment/no_auction.yaml',
        overrides=['rl.phase_normalization=true','rl.auction_inventory_asinh=true',
                   'rl.normalizer_fit_episodes=4','algo.hyperparams.hidden_layers=[16,16]'])
    seeds = seed_everything(429, SEED_COMPONENTS)
    agent = make_agent(cfg, seeds)
    normalizer, _ = fit_feature_normalizer(cfg, seeds)
    agent.set_feature_normalizer(normalizer)
    assert normalizer.phase_count[0] > 0 and normalizer.phase_count[1] == 0
    state = normalizer.state_dict()
    result = run_episode(wrap_env_for_agent(make_env(cfg),cfg),agent,309,chi=1.,train=True)
    assert np.isfinite(result.risk_adjusted_pnl)
    assert result.auction_exec_qty == 0
    assert normalizer.state_dict() == state
