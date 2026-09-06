"""Cash-flow-only means no hidden intermediate credit on the replay path."""
from dataclasses import replace
import json

import numpy as np
import pytest

from lmm.config import load_config
from lmm.env.mdp import make_env
from lmm.experiments import cashflow_comparison as C
from lmm.rl.loops import run_episode


def config(algo='dqn'):
    return load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml',
                       f'configs/algo/{algo}.yaml', f'configs/treatment/{C.ARM}.yaml')


def test_cashflows_reach_replay_without_intermediate_reward_transformations():
    cfg = config()
    cfg = replace(cfg, rl=replace(cfg.rl, structured_warmup_episodes=0))

    class Probe:
        def __init__(self):
            self.transitions = []
            self.rng = np.random.default_rng(123)
        def set_train(self, value): pass
        def preprocess_observation(self, x): return x
        def act(self, obs, mask, phase, **kwargs):
            return int(self.rng.choice(np.flatnonzero(mask)))
        def observe(self, transition): self.transitions.append(transition)
        def update(self, **kwargs): return {}

    agent = Probe()
    result = run_episode(make_env(cfg), agent, 81635, chi=1., train=True)
    for tr in agent.transitions:
        info = tr.info
        expected = (info['clob_economic_cash'] + info['auction_economic_cash']
                    + info['residual_mark'] - info['terminal_penalty'] - info['cancellation_fee'])
        assert tr.reward == pytest.approx(expected, abs=1e-9)
        assert info['reward_baseline_adjustment'] == 0
        if tr.phase == 'auction' and not tr.done:
            assert tr.reward == pytest.approx(-info['cancellation_fee'])
    assert sum(t.reward for t in agent.transitions) == pytest.approx(result.replay_return_unscaled)
    assert result.training_return == pytest.approx(result.replay_return_unscaled)
    assert result.training_return - result.initial_mid*result.initial_inventory == pytest.approx(result.economic_objective)
    assert result.potential_adjustment == result.market_baseline_adjustment == 0
    assert result.clob_exec_qty > 0
    assert any(t.phase == 'auction' and not t.done for t in agent.transitions)


@pytest.mark.parametrize('algo', ['dqn', 'ddpg', 'td3', 'sac'])
def test_reward_ablation_preserves_forecast_action_set_and_market_path(algo):
    raw = config(algo)
    headline = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml',
                           f'configs/algo/{algo}.yaml')
    assert raw.actions == headline.actions
    assert raw.algo == headline.algo
    a, b = make_env(raw), make_env(headline)
    x, _ = a.reset(seed=815); y, _ = b.reset(seed=815)
    np.testing.assert_array_equal(x, y)
    rng = np.random.default_rng(321)
    while True:
        np.testing.assert_array_equal(a.action_mask(), b.action_mask())
        action = int(rng.choice(np.flatnonzero(a.action_mask())))
        x, _, done, _, left = a.step(action)
        y, _, other_done, _, right = b.step(action)
        np.testing.assert_array_equal(x, y)
        assert done == other_done
        for field in ('clob_economic_cash', 'auction_economic_cash', 'residual_mark',
                      'terminal_penalty', 'cancellation_fee'):
            assert left[field] == right[field]
        if done: break


def test_followup_dry_run_adds_only_40_controls_without_writing(tmp_path, capsys):
    root = tmp_path / 'followup'
    assert C.main(['--dry-run', '--root', str(root)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['runs'] == 40
    assert all(cmd[-2] == C.ARM for cmd in plan['commands'])
    assert not root.exists()
    assert len(C.jobs_for([91031], smoke=True)) == 8
