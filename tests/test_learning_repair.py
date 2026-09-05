"""Behavioral guards for objective-preserving conditioning and SB3 integration."""
from dataclasses import replace
from collections import deque

import numpy as np
import pytest
import torch

from lmm.agents.base import Transition
from lmm.config import load_config, economic_evaluation_config
from lmm.env.features import FeatureNormalizer
from lmm.env.mdp import make_env
from lmm.experiments.train import make_agent, wrap_env_for_agent, fit_feature_normalizer
from lmm.rl.conditioning import inventory_potential
from lmm.rl.loops import SEED_COMPONENTS, run_episode, _accumulate_transition
from lmm.utils.seeding import seed_everything
from lmm.rl.schedules import learning_rate_factor
from lmm.rl.exploration import PersistentWarmup


def config(algo):
    return load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml',
                       f'configs/algo/{algo}.yaml')


def checkpoint_normalizer(cfg):
    """Small synthetic statistics with the policy's actual feature contract."""
    rows = np.zeros((2,18))
    rows[1,0] = cfg.grid.tau_op
    return FeatureNormalizer(cfg.grid.tau_cl, relative_prices=cfg.rl.relative_price_features,
        auction_exposure_features=cfg.rl.auction_exposure_features,
        phase_normalization=cfg.rl.phase_normalization, tau_op=cfg.grid.tau_op,
        auction_inventory_asinh=cfg.rl.auction_inventory_asinh,
        inventory_scale=cfg.grid.alpha/cfg.reward.lambda_inv).fit(rows)


@pytest.mark.parametrize('shaping', [True, False])
@pytest.mark.parametrize('clob_risk', [True, False])
def test_potential_preserves_manuscript_return_pathwise(shaping, clob_risk):
    cfg = config('dqn')
    cfg = replace(cfg, reward=replace(cfg.reward, shaping_enabled=shaping))
    cfg = replace(cfg, rl=replace(cfg.rl, learning_clob_inventory_potential=clob_risk))
    env = make_env(cfg)
    raw, _ = env.reset(seed=13)
    first = inventory_potential(raw, cfg)
    phi, total, conditioned = first, 0., 0.
    rng = np.random.default_rng(51)
    done = False
    while not done:
        action = rng.choice(np.flatnonzero(env.action_mask()))
        raw, reward, done, _, _ = env.step(int(action))
        nxt = inventory_potential(raw, cfg, done=done)
        total += reward
        conditioned += reward + nxt - phi
        phi = nxt
    assert phi == 0.0
    assert conditioned == pytest.approx(total - first, abs=1e-8)
    assert first == 0.0  # no huge lambda*I0^2 baseline


def test_relative_prices_expose_spread_and_schedule_moments():
    x = np.zeros(18)
    x[2:4] = [100.12, 100]
    x[13:18] = [2, 200.08, 10, 3, 999.8]
    n = FeatureNormalizer(150, relative_prices=True)
    centered = n._coordinates(x)[0]
    assert centered[2] == pytest.approx(.12)
    assert centered[14] == pytest.approx(.08)
    assert centered[17] == pytest.approx(-.2)
    y = x.copy(); y[2:4] += 200; y[14] += 200*x[13]; y[17] += 200*x[15]
    translated = n._coordinates(y)[0]
    assert translated[[2, 14, 17]] == pytest.approx(centered[[2, 14, 17]])


def test_fixed_learning_rate_decay_preserves_initial_rate_and_positive_floor():
    cfg = config('ddpg')
    rl = replace(cfg.rl, learning_rate_half_life_episodes=60, learning_rate_min_fraction=.1)
    assert [learning_rate_factor(rl, e) for e in (0, 60, 120, 180, 800)] == pytest.approx([1,.5,.25,.125,.1])


def test_auction_exposure_coordinates_are_invertible_and_do_not_use_H():
    x = np.zeros(18); x[0]=125; x[1]=20; x[2:4]=[100.4,100]
    x[13:18]=[100,9990,300,2,30010]
    n=FeatureNormalizer(150, relative_prices=True, auction_exposure_features=True)
    y=n._coordinates(x)[0]
    residual, displacement=y[14],y[17]
    own_relative=x[13]*displacement-x[1]+residual
    exo_relative=x[15]*displacement+x[1]-residual-x[16]
    assert own_relative == pytest.approx(x[14]-x[3]*x[13])
    assert exo_relative == pytest.approx(x[17]-x[3]*x[15])
    x[2]=0
    assert n._coordinates(x)[0][[14,17]] == pytest.approx([residual, displacement])
    clob=x.copy(); clob[13:18]=0
    assert np.all(n._coordinates(clob)[0][13:18] == 0)


@pytest.mark.parametrize('q', [0., .5])
def test_draft_auction_credit_scale_changes_only_shaping_and_exact_clawbacks(q):
    cfg = config('dqn')
    cfg = replace(cfg, reward=replace(cfg.reward, q=q, auction_shaping_weight=1.))
    other = replace(cfg, reward=replace(cfg.reward, auction_shaping_weight=.0001))
    env, scaled = make_env(cfg), make_env(other)
    env.reset(seed=812);scaled.reset(seed=812)
    rng = np.random.default_rng(14)
    total_difference = original_shape = 0.
    clawbacks = 0
    while True:
        a = int(rng.choice(np.flatnonzero(env.action_mask())))
        _, r, done, _, info = env.step(a)
        _, r_scaled, done_scaled, _, other_info = scaled.step(a)
        total_difference += r-r_scaled
        original_shape += info.get('auction_interim_shaping', 0)-info.get('auction_shaping_clawback', 0)+info.get('auction_terminal_shaping', 0)
        clawbacks += abs(info.get('auction_shaping_clawback', 0)) > 0
        assert done == done_scaled
        assert env.inventory == scaled.inventory
        for key in ('auction_economic_cash','terminal_penalty','cancellation_fee'):
            assert info.get(key) == other_info.get(key)
        if done:
            break
    assert clawbacks > 0
    assert total_difference == pytest.approx((1-.0001)*original_shape)


def test_credit_potential_preserves_return_up_to_known_initial_constant():
    cfg = config('dqn')
    cfg = replace(cfg, rl=replace(cfg.rl, learning_credit_baseline=True))
    env = make_env(cfg)
    obs, _ = env.reset(seed=15)
    initial = previous = inventory_potential(obs, cfg)
    adjustment = 0.
    assert initial > 0
    while True:
        obs, _, done, _, _ = env.step(0)
        current = inventory_potential(obs, cfg, done=done)
        adjustment += current-previous
        previous = current
        if done:
            break
    assert adjustment == pytest.approx(-initial)


def test_market_control_variate_is_policy_independent_and_absent_from_evaluation():
    cfg=config('dqn')
    cfg=replace(cfg, rl=replace(cfg.rl, market_return_control_variate=True,
                               market_control_reference='linear', structured_warmup_episodes=0))
    class Probe:
        def __init__(self, trade): self.trade=trade
        def set_train(self, value): pass
        def preprocess_observation(self, x): return x
        def act(self, obs, mask, phase, **kwargs):
            return int(np.flatnonzero(mask)[-1]) if self.trade else 0
        def observe(self, tr): pass
        def update(self, **kwargs): return {}
    results=[run_episode(make_env(cfg),Probe(trade),72,chi=1.,train=True) for trade in [False,True]]
    assert results[0].market_baseline_adjustment == pytest.approx(results[1].market_baseline_adjustment)
    assert abs(results[0].market_baseline_adjustment) > .01
    for result in results:
        assert result.replay_return_unscaled == pytest.approx(result.training_return+result.potential_adjustment+result.market_baseline_adjustment)
    evaluated=run_episode(make_env(cfg),Probe(True),72,chi=1.,train=False)
    assert evaluated.market_baseline_adjustment == 0
    assert evaluated.pnl == pytest.approx(results[1].pnl)
    assert evaluated.training_return == pytest.approx(results[1].training_return)


@pytest.mark.parametrize('setting,symbol', [('synthetic_rough_heston', None),
                                          ('historical_sp500_midquotes', 'MSFT')])
def test_calibrated_reference_is_frozen_restorable_and_policy_independent(setting, symbol):
    cfg = load_config('configs/base.yaml', f'configs/{setting}.yaml', 'configs/algo/dqn.yaml',
                      overrides=['rl.normalizer_fit_episodes=4', 'rl.market_return_control_variate=true',
                                 'rl.market_control_reference=calibration', 'rl.structured_warmup_episodes=0'])
    normalizer, used = fit_feature_normalizer(cfg, seed_everything(773, SEED_COMPONENTS), symbol=symbol)
    assert len(used) == 4
    assert normalizer.inventory_reference_times == list(range(cfg.grid.tau_op+1))
    assert normalizer.inventory_reference_values[0] == cfg.grid.I0
    assert np.all(np.diff(normalizer.inventory_reference_values) <= 1e-12)
    state = normalizer.state_dict()
    restored = FeatureNormalizer(cfg.grid.tau_cl)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    with pytest.raises(RuntimeError, match='frozen'):
        restored.set_inventory_reference([0, 120], [100, 0])
    class Probe:
        _feature_normalizer = restored
        def __init__(self, trade): self.trade = trade
        def set_train(self, value): pass
        def preprocess_observation(self, x): return x
        def act(self, obs, mask, phase, **kwargs):
            return int(np.flatnonzero(mask)[-1]) if self.trade else 0
        def observe(self, tr): pass
        def update(self, **kwargs): return {}
    results = [run_episode(make_env(cfg, symbol=symbol, data_split='validation'), Probe(trade),
                           92, chi=1., train=True) for trade in [False, True]]
    assert results[0].market_baseline_adjustment == pytest.approx(results[1].market_baseline_adjustment)
    assert restored.state_dict() == state


@pytest.mark.parametrize('times,values', [([0, 2, 1], [3, 2, 1]), ([0, 1], [2]),
                                         ([0, 1], [2, float('nan')]), (None, [2, 1])])
def test_invalid_reference_checkpoint_rejected(times, values):
    normalizer = FeatureNormalizer(150).fit(np.zeros((2, 18)))
    state = {**normalizer.state_dict(), 'inventory_reference_times': times,
             'inventory_reference_values': values}
    with pytest.raises(ValueError, match='reference'):
        FeatureNormalizer(150).load_state_dict(state)


@pytest.mark.parametrize('episode', [0, 2, 3, 4, 6, 8, 10, 15])
def test_warmup_discrete_and_continuous_execution_match(episode):
    dcfg, ccfg = config('dqn'), config('td3')
    discrete, continuous = make_env(dcfg), wrap_env_for_agent(make_env(ccfg), ccfg)
    dw, cw = PersistentWarmup(dcfg, 118, episode), PersistentWarmup(ccfg, 118, episode)
    x, _ = discrete.reset(seed=88)
    y, _ = continuous.reset(seed=88)
    while True:
        a = dw.act(x, discrete.action_mask(), discrete.phase)
        b = cw.act(y, continuous.action_mask(), continuous.phase)
        x, r, done, _, info = discrete.step(a)
        y, s, other_done, _, other = continuous.step(b)
        np.testing.assert_array_equal(x, y)
        assert r == s and done == other_done
        if done: break
    # The CLOB mode and auction mode are crossed, not confounded.
    if episode//4 == 0: assert dw.volume == dcfg.actions.V_max
    if episode//4 == 1: assert dw.volume == 0
    if episode%4 == 0: assert dw.K == 0
    if episode%4 == 2: assert dw.ell >= 0 and dw.cancel == 0


def test_warmup_ends_at_fixed_episode_and_is_absent_from_evaluation():
    cfg = config('dqn')
    cfg = replace(cfg, rl=replace(cfg.rl, structured_warmup_episodes=4,
                                 market_return_control_variate=False))
    class Probe:
        def __init__(self, episode): self._episode, self.calls = episode, 0
        def set_train(self, value): pass
        def preprocess_observation(self, x): return x
        def act(self, *args, **kwargs): self.calls += 1; return 0
        def observe(self, tr): pass
        def update(self, **kwargs): return {}
    for episode, train, expected in [(0, True, False), (4, True, True), (0, False, True)]:
        agent = Probe(episode)
        result = run_episode(make_env(cfg), agent, 99, chi=1., train=train)
        assert bool(agent.calls) == expected
        assert result.replay_return_unscaled == pytest.approx(
            result.training_return+result.potential_adjustment+result.market_baseline_adjustment)


def test_n_step_phase_boundary_and_terminal_preserve_first_action_and_one_G():
    q = deque()
    obs = np.zeros(18, dtype=np.float32)
    def tr(i, reward, phase='clob', nxt='clob', done=False):
        return Transition(obs+i, i, reward, obs+i+1, done, phase, nxt,
                          info={'proposal_action_vec': np.array([i, 0]),
                                'terminal_reward': 7 if done else 0})
    assert _accumulate_transition(q, tr(0, 1), 5) == []
    assert _accumulate_transition(q, tr(1, 2), 5) == []
    rows = _accumulate_transition(q, tr(2, 3, nxt='auction'), 5)
    assert [r.reward for r in rows] == [6, 5, 3]
    assert [r.action for r in rows] == [0, 1, 2]
    assert all(not r.done and r.next_phase == 'auction' for r in rows)
    assert not q
    assert _accumulate_transition(q, tr(3, 4, 'auction', 'auction'), 5) == []
    rows = _accumulate_transition(q, tr(4, 12, 'auction', 'terminal', True), 5)
    assert [r.reward for r in rows] == [16, 12]  # 12 already contains terminal G=7
    assert [r.info['terminal_reward'] for r in rows] == [7, 7]
    assert rows[0].info['proposal_action_vec'].tolist() == [3, 0]
    assert not q


@pytest.mark.parametrize('algo', ['dqn', 'ddpg', 'td3', 'sac'])
def test_active_headline_and_economic_evaluation_contract(algo):
    cfg = config(algo)
    assert cfg.reward.effective_clob_shaping and cfg.reward.effective_auction_shaping
    assert cfg.reward.q == 0 and cfg.reward.learning_potential
    assert cfg.grid.alpha == .01 and cfg.actions.auction_K_grid_max == 8
    assert cfg.reward.auction_shaping_weight == .0001
    assert cfg.reward.k_star == 10000 and cfg.reward.lambda_inv == .01
    assert cfg.algo.hyperparams['reward_scale'] == 1.
    assert cfg.algo.hyperparams['hidden_layers'] == [256, 256]
    assert cfg.rl.n_step == 1
    if algo != 'dqn':
        assert cfg.algo.backend == 'sb3'
    evaluation = economic_evaluation_config(cfg)
    assert not evaluation.reward.effective_clob_shaping
    assert not evaluation.reward.effective_auction_shaping
    assert not evaluation.reward.learning_potential
    assert evaluation.reward.lambda_inv == cfg.reward.lambda_inv


@pytest.mark.parametrize('algo', ['ddpg', 'td3', 'sac'])
def test_sb3_junction_and_terminal_targets_are_not_double_counted(algo):
    cfg = config(algo)
    a = make_agent(cfg, seed_everything(5, SEED_COMPONENTS))
    b = a.replay['clob']
    obs = np.zeros((1, 18), dtype=np.float32)
    b.continuation = lambda obs: torch.full((len(obs), 1), 7., device=obs.device)
    b.add(obs, obs, np.zeros((1,2)), np.array([3.]), np.array([False]), [dict(junction=True)])
    sample = b.sample(4)
    assert sample.rewards.flatten().tolist() == [10.]*4
    assert sample.dones.flatten().tolist() == [1.]*4
    # Final rows already contain r_step + G, and never call continuation.
    t = a.replay['auction']
    t.add(obs, obs, np.zeros((1,3)), np.array([11.]), np.array([True]), [{}])
    sample = t.sample(4)
    assert sample.rewards.flatten().tolist() == [11.]*4
    assert sample.dones.flatten().tolist() == [1.]*4


@pytest.mark.parametrize('algo', ['ddpg', 'td3', 'sac'])
def test_sb3_save_load_policy_and_replay(tmp_path, algo):
    cfg = config(algo)
    seeds = seed_everything(34, SEED_COMPONENTS)
    a = make_agent(cfg, seeds)
    n = checkpoint_normalizer(cfg)
    a.set_feature_normalizer(n)
    obs = np.zeros((1,18), dtype=np.float32)
    a.replay['clob'].add(obs, obs, np.zeros((1,2)), np.array([1.]), np.array([True]), [{}])
    before = a.act(obs[0], np.ones(2, bool), 'clob', eval_mode=True)
    path = tmp_path / 'model.pt'
    a.save(path, include_replay=True)
    restored = make_agent(cfg, seed_everything(99, SEED_COMPONENTS))
    restored.load(path)
    assert restored.act(obs[0], np.ones(2, bool), 'clob', eval_mode=True) == pytest.approx(before)
    assert len(restored.replay['clob']) == 1
    assert restored.replay['clob'].sample(2).rewards.flatten().tolist() == [1.,1.]

    # Optimizer hooks are not serialized by Torch. Loading must reinstall the
    # actual bound, not merely preserve its configuration value.
    critic = restored.models['clob'].critic
    for parameter in critic.parameters():
        parameter.grad = torch.full_like(parameter, 100.)
    critic.optimizer.step()
    # Float32 reduction of a large constant gradient has measurable summation
    # error; check the actual bound with a double-precision reference norm.
    norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten().double() for p in critic.parameters()]))
    assert norm <= cfg.algo.hyperparams['grad_clip_norm'] + 1e-6
    assert restored.models['clob']._lmm_grad_norms['critic'] > 1.


@pytest.mark.parametrize('algo', ['ddpg', 'td3', 'sac'])
def test_sb3_native_update_and_resume_preserve_next_update(tmp_path, algo):
    cfg = config(algo)
    cfg = replace(cfg, algo=replace(cfg.algo, hyperparams={
        **cfg.algo.hyperparams, 'batch_size': 4,
        'min_buffer_clob': 4, 'min_buffer_auction': 4,
    }))
    a = make_agent(cfg, seed_everything(34, SEED_COMPONENTS))
    a.set_feature_normalizer(checkpoint_normalizer(cfg))
    rng = np.random.default_rng(45)
    for phase in a.PHASES:
        for i in range(8):
            obs = rng.normal(size=18).astype(np.float32)
            a.observe(Transition(obs, 0, float(i), obs + .1, True, phase, 'terminal',
                                 info={'proposal_action_vec': np.zeros(a.models[phase].action_space.shape)}))
        assert np.isfinite(a.update(phase)[f'loss_{phase}'])
    path = tmp_path / 'resume.pt'
    a.save(path, include_replay=True)
    a._pending_update_phase = 'auction'
    expected = a.update('auction')
    weights = {k: v.clone() for k, v in a.models['auction'].policy.state_dict().items()}
    b = make_agent(cfg, seed_everything(99, SEED_COMPONENTS))
    b.load(path)
    b._pending_update_phase = 'auction'
    assert b.update('auction') == pytest.approx(expected)
    for key, value in b.models['auction'].policy.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_optional_optimizer_settings_survive_native_rate_reset_and_load(tmp_path):
    cfg = config('ddpg')
    cfg = replace(cfg, rl=replace(cfg.rl, learning_rate_half_life_episodes=90),
                  algo=replace(cfg.algo, hyperparams={**cfg.algo.hyperparams,
                      'actor_learning_rate': 1e-4, 'critic_weight_decay': .01}))
    agent = make_agent(cfg, seed_everything(18, SEED_COMPONENTS))
    agent.set_feature_normalizer(checkpoint_normalizer(cfg))
    agent.start_episode(90)
    path = tmp_path/'optimizer.pt'
    agent.save(path)
    restored = make_agent(cfg, seed_everything(19, SEED_COMPONENTS))
    restored.load(path)
    model = restored.models['auction']
    model.lr_schedule = lambda _: .0005
    model._update_learning_rate([model.actor.optimizer, model.critic.optimizer])
    for p in model.actor.parameters(): p.grad = torch.ones_like(p)
    model.actor.optimizer.step()
    assert model.actor.optimizer.param_groups[0]['lr'] == pytest.approx(5e-5)
    assert model.critic.optimizer.param_groups[0]['weight_decay'] == .01


@pytest.mark.parametrize('algo', ['dqn', 'ddpg', 'td3', 'sac'])
@pytest.mark.parametrize('phase_normalization', [False, True])
def test_enabled_conditioning_and_warmup_resume_exact_episode(tmp_path, algo, phase_normalization):
    cfg = config(algo)
    cfg = replace(cfg, rl=replace(cfg.rl, normalizer_fit_episodes=4,
                                 phase_normalization=phase_normalization,
                                 auction_inventory_asinh=phase_normalization),
                  algo=replace(cfg.algo, hyperparams={**cfg.algo.hyperparams,
                      'min_buffer_clob': 4, 'min_buffer_auction': 4, 'batch_size': 4,
                      'hidden_layers': [16, 16]}))
    seeds = seed_everything(842, SEED_COMPONENTS)
    original = make_agent(cfg, seeds)
    normalizer, _ = fit_feature_normalizer(cfg, seeds)
    original.set_feature_normalizer(normalizer)
    original.start_episode(0)
    run_episode(wrap_env_for_agent(make_env(cfg), cfg), original, 301, chi=1., train=True)
    path = tmp_path/'resume.pt'
    original.save(path, include_replay=True)
    original.start_episode(1)
    expected = run_episode(wrap_env_for_agent(make_env(cfg), cfg), original, 302, chi=1., train=True)
    restored = make_agent(cfg, seed_everything(942, SEED_COMPONENTS))
    restored.load(path)
    restored.start_episode(1)
    actual = run_episode(wrap_env_for_agent(make_env(cfg), cfg), restored, 302, chi=1., train=True)
    for field in ('training_return', 'pnl', 'risk_adjusted_pnl', 'market_baseline_adjustment',
                  'replay_return_unscaled', 'potential_adjustment', 'diagnostics', 'action_records'):
        assert getattr(actual, field) == getattr(expected, field)
    for phase in ['clob', 'auction']:
        a = original.q[phase] if algo == 'dqn' else original.models[phase].policy
        b = restored.q[phase] if algo == 'dqn' else restored.models[phase].policy
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
