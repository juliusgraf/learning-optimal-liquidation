"""Regression checks for numerical stability, warm-up and training-only pooling."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from helpers import load_dqn_cfg, load_historical_cfg
from lmm.agents.base import Transition
from lmm.agents.dqn import DQNAgent
from lmm.experiments.train import make_agent
from lmm.config import load_config
from lmm.market import midprice
from lmm.rl.loops import SEED_COMPONENTS
from lmm.utils.seeding import seed_everything


def test_normalized_q_does_not_amplify_large_observation_magnitudes():
    cfg = load_dqn_cfg('algo.hyperparams.layer_norm=true')
    agent = DQNAgent(cfg, seed_everything(871, SEED_COMPONENTS))
    q = agent.q['auction']
    observations = torch.randn(16, 18)
    ordinary = q(observations)
    extreme = q(observations*1e6)
    # Linear output heads stay unrestricted, but finite fixed weights no
    # longer extrapolate Q values linearly with unobserved inventory scale.
    assert torch.isfinite(extreme).all()
    assert float(extreme.detach().abs().max()) < 10*max(1., float(ordinary.detach().abs().max()))
    assert q.action_encoder(q.action_coordinates).abs().max() <= 1
    actions = torch.arange(16)
    torch.testing.assert_close(q(observations, actions), ordinary[torch.arange(16), actions])
    assert any(isinstance(m, nn.LayerNorm) for m in q.state_trunk)


@pytest.mark.parametrize('algo', ['dqn','ddpg','td3','sac'])
def test_both_phases_wait_for_complete_behavior_warmup(algo):
    cfg = load_config('configs/base.yaml','configs/synthetic_rough_heston.yaml',f'configs/algo/{algo}.yaml',
        overrides=['rl.structured_warmup_episodes=2','rl.learning_starts_after_warmup=true',
                   'algo.hyperparams.min_buffer_clob=1','algo.hyperparams.min_buffer_auction=1',
                   'algo.hyperparams.batch_size=1'])
    agent = make_agent(cfg, seed_everything(877, SEED_COMPONENTS))
    for phase in ['clob','auction']:
        action = 0 if algo == 'dqn' else np.zeros(agent.models[phase].action_space.shape, np.float32)
        tr = Transition(np.zeros(18,np.float32), action, 1., np.zeros(18,np.float32), True,
                        phase, phase, None, {'proposal_action_vec': action})
        agent.start_episode(1)
        agent.observe(tr)
        assert agent.update(phase) == {}
        agent.start_episode(2)
        agent.observe(tr)
        assert agent.update(phase)[f'n_grad_steps_{phase}'] == 1
        assert agent.update(phase) == {}


def test_historical_pooling_never_pools_validation_or_test(monkeypatch):
    cfg = load_historical_cfg('midprice.historical.training_pool=all_symbols')
    seen = []
    def paths(params, repo_root, split, horizon):
        seen.append(split)
        return {s: np.full((2,121), 100+i) for i,s in enumerate(params.symbols)}
    monkeypatch.setattr(midprice, '_load_regularized_mid_paths', paths)
    train = midprice.build_midprice(cfg, symbol='MSFT', data_split='train')
    validation = midprice.build_midprice(cfg, symbol='MSFT', data_split='validation')
    test = midprice.build_midprice(cfg, symbol='MSFT', data_split='test')
    assert train._paths.shape == (10,121)
    assert validation._paths.shape == test._paths.shape == (2,121)
    assert np.all(validation._paths == 100) and np.all(test._paths == 100)
    assert seen == ['train','validation','test']


def test_new_test_namespace_keeps_pairing_and_changes_only_test_stream():
    from lmm.experiments.evaluate import final_evaluation_rng
    cfg = load_dqn_cfg('rl.test_seed_namespace=18001')
    old = replace(cfg, rl=replace(cfg.rl, test_seed_namespace=0))
    a = seed_everything(42, SEED_COMPONENTS)
    b = seed_everything(42, SEED_COMPONENTS)
    new = final_evaluation_rng(cfg, a).integers(0,2**31-1,100)
    paired = final_evaluation_rng(cfg, b).integers(0,2**31-1,100)
    prior = final_evaluation_rng(old, a).integers(0,2**31-1,100)
    np.testing.assert_array_equal(new, paired)
    assert not set(new).intersection(prior)
    np.testing.assert_array_equal(a.generators['env_train'].integers(0,2**31-1,100),
                                  b.generators['env_train'].integers(0,2**31-1,100))


@pytest.mark.parametrize('algo', ['dqn','ddpg','td3','sac'])
def test_no_order_warmup_really_submits_no_order_for_every_random_offset(algo):
    from lmm.rl.exploration import PersistentWarmup
    from lmm.env.action_spaces import AuctionActionGrid
    cfg = load_config('configs/base.yaml','configs/synthetic_rough_heston.yaml',f'configs/algo/{algo}.yaml')
    grid = AuctionActionGrid(cfg.actions)
    mask = np.ones(len(grid), dtype=bool)
    obs = np.zeros(18, dtype=np.float32)
    for seed in range(64):
        for episode in range(0,32,4):
            warmup = PersistentWarmup(cfg, seed, episode)
            action = warmup.act(obs, mask, 'auction')
            if algo == 'dqn':
                assert action == 0
            else:
                np.testing.assert_array_equal(action, [-1.,0.,-1.])


def test_control_exploration_covers_wait_cancel_and_full_order_grid():
    cfg = load_dqn_cfg('algo.hyperparams.auction_control_exploration_probability=.5')
    agent = DQNAgent(cfg, seed_everything(991, SEED_COMPONENTS))
    mask = np.ones(len(agent.auction_grid), dtype=bool)
    choices = np.array([agent.act(np.zeros(18,np.float32),mask,'auction') for _ in range(12000)])
    assert .23 < (choices==0).mean() < .27
    assert .23 < (choices==1).mean() < .27
    assert len(np.unique(choices[choices>1])) > 1300
    mask[1] = False
    masked = [agent.act(np.zeros(18,np.float32),mask,'auction') for _ in range(1000)]
    assert 1 not in masked
    # Evaluation remains the ordinary exact masked Q argmax, with no fallback.
    obs = np.zeros(18,np.float32)
    with torch.no_grad():
        expected = agent.q['auction'](torch.from_numpy(obs)[None])[0].masked_fill(~torch.from_numpy(mask),-torch.inf).argmax().item()
    assert agent.act(obs,mask,'auction',eval_mode=True) == expected


def test_reference_value_cannot_drift_through_the_action_encoder():
    cfg = load_dqn_cfg('algo.hyperparams.reference_centered_phases=[auction]')
    q = DQNAgent(cfg, seed_everything(919,SEED_COMPONENTS)).q['auction']
    obs = torch.randn(8,18)
    zero = torch.zeros(8,dtype=torch.int64)
    torch.testing.assert_close(q(obs,zero),q.value_head(q.state_trunk(obs)).squeeze(-1))
    q(obs,zero).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in q.action_encoder.parameters())
    assert torch.count_nonzero(q.query_head.weight.grad)==0
    torch.testing.assert_close(q(obs)[:,0],q(obs,zero))


def test_known_fee_q_uses_exact_time_cost_without_changing_bellman_reward():
    cfg = load_dqn_cfg('algo.hyperparams.known_auction_fee=true')
    agent = DQNAgent(cfg, seed_everything(919,SEED_COMPONENTS))
    q = agent.q['auction']
    for p in q.parameters():
        p.data.zero_()
    obs = torch.zeros(2,18)
    obs[:,0] = torch.tensor([cfg.grid.tau_op,cfg.grid.tau_cl-1])/cfg.grid.tau_cl
    actual = q(obs)
    torch.testing.assert_close(actual[:,0],torch.zeros(2))
    expected = torch.tensor([0.,-29*cfg.reward.d*agent.hp.reward_scale])
    torch.testing.assert_close(actual[:,1],expected)
    torch.testing.assert_close(q(obs,torch.ones(2,dtype=torch.int64)),expected)
    # A sufficiently valuable learned advantage can still choose cancellation.
    with torch.no_grad():
        q.nonnoop_prior.fill_(1.)
    assert q(obs)[1,1] > q(obs)[1,0]
