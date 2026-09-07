"""Causal controls and native-library critic conditioning."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from lmm.config import load_config
from lmm.env.features import FeatureNormalizer
from lmm.experiments.train import make_agent
from lmm.rl.conditioning import inventory_potential
from lmm.rl.loops import SEED_COMPONENTS
from lmm.utils.seeding import seed_everything


def config(*overrides):
    return load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml',
                       'configs/algo/ddpg.yaml', overrides=list(overrides))


def test_forecast_information_and_anchor_are_independent():
    for visible in ('true', 'false'):
        for anchor in ('indicative', 'frozen_mid'):
            cfg = config(f'rl.h_cl_feature_enabled={visible}', f'actions.auction_anchor={anchor}')
            assert cfg.actions.auction_anchor == anchor
            assert cfg.rl.h_cl_feature_enabled == (visible == 'true')


def test_auction_credit_is_immediate_risk_aware_and_terminal_corrected():
    cfg = config()
    x = np.zeros(18); x[0] = 120; x[1] = 10; x[2:4] = 100
    x[15] = 100; x[17] = 10000
    after = x.copy(); after[13] = 8; after[14] = 8*99.9
    dense = inventory_potential(after, cfg)-inventory_potential(x, cfg)
    assert dense > 0  # a sale reduces projected terminal exposure
    sparse = replace(cfg, rl=replace(cfg.rl, learning_auction_inventory_potential=False))
    assert inventory_potential(after, sparse)-inventory_potential(x, sparse) == 0
    flat, sold = x.copy(), after.copy(); flat[1] = sold[1] = 0
    assert inventory_potential(sold, cfg)-inventory_potential(flat, cfg) < 0
    path = [x, after, sold]
    phi = [inventory_potential(s, cfg) for s in path]+[inventory_potential(sold, cfg, done=True)]
    assert sum(np.diff(phi)) == pytest.approx(-phi[0])


def test_forecast_reliability_changes_signal_without_changing_market_or_settlement():
    from lmm.config import economic_evaluation_config
    from lmm.env.mdp import make_env
    cfg = economic_evaluation_config(config('actions.auction_anchor=frozen_mid',
        'algo1.clob_forecast_weights=[]'))
    revised = replace(cfg, algo1=replace(cfg.algo1, clob_forecast_weights=(0.,.2,.3,.4)))
    raw, adjusted = make_env(cfg), make_env(revised)
    raw.reset(seed=19717); adjusted.reset(seed=19717)
    done=False; signal_changed=False
    while not done:
        x,r,done,_,info=raw.step(0)
        y,s,other_done,_,other=adjusted.step(0)
        assert done == other_done
        assert r == pytest.approx(s,abs=1e-10)
        np.testing.assert_allclose(np.delete(x,2), np.delete(y,2),rtol=0,atol=1e-10)
        if raw.phase=='clob' and raw.t>0:
            w=revised.algo1.clob_forecast_weights[min(3,int(4*raw.t/cfg.grid.tau_op))]
            assert y[2] == pytest.approx(y[3]+w*(x[2]-x[3]))
            signal_changed |= abs(y[2]-x[2])>1e-6
    assert signal_changed
    assert info['S_cl'] == other['S_cl']


def test_layernorm_keeps_native_ddpg_and_checkpoint_update(tmp_path):
    cfg = config('algo.hyperparams.critic_layer_norm=true',
                 'algo.hyperparams.hidden_layers=[16,16]',
                 'algo.hyperparams.batch_size=4', 'algo.hyperparams.min_buffer_clob=4')
    a = make_agent(cfg, seed_everything(701, SEED_COMPONENTS))
    rows = np.zeros((2,18)); rows[1,0] = 120; rows[1,15] = 1
    a.set_feature_normalizer(FeatureNormalizer(150, relative_prices=True,
        auction_exposure_features=True, phase_normalization=True, tau_op=120,
        auction_inventory_asinh=True, inventory_scale=1).fit(rows))
    a.start_episode(40)
    model = a.models['clob']
    assert model.critic.n_critics == 1 and model.policy_delay == 1 and model.target_noise_clip == 0
    assert sum(isinstance(m,torch.nn.LayerNorm) for m in model.critic.modules()) == 2
    assert not any(isinstance(m,torch.nn.LayerNorm) for m in model.actor.modules())
    for online,target in zip(model.critic.parameters(),model.critic_target.parameters()):
        torch.testing.assert_close(online,target)
    b = a.replay['clob']
    for i in range(8):
        b.add(np.full((1,18),i,dtype=np.float32),np.zeros((1,18)),np.zeros((1,2)),
              np.array([i/10]),np.array([True]),[{}])
    a._pending_update_phase = 'clob'; assert np.isfinite(a.update()['loss_clob'])
    a.save(tmp_path/'model.pt',include_replay=True)
    a._pending_update_phase = 'clob'; expected = a.update()
    restored = make_agent(cfg,seed_everything(702,SEED_COMPONENTS)); restored.load(tmp_path/'model.pt')
    restored._pending_update_phase = 'clob'
    assert restored.update() == pytest.approx(expected)
    for left,right in zip(a.models['clob'].policy.parameters(), restored.models['clob'].policy.parameters()):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
