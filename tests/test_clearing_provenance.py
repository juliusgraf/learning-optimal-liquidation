"""Mechanism isolation for checkpoints, reporting, and prepared rerun commands."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from lmm.config import load_config, LEGACY_CLEARING, VOLUME_MAX_CLEARING, save_resolved
from lmm.env.features import FeatureNormalizer
from lmm.experiments.plotting import collect_runs
from lmm.experiments.train import make_agent
from lmm.rl.loops import SEED_COMPONENTS
from lmm.utils.seeding import seed_everything


@pytest.mark.parametrize('algo,backend', [('dqn', 'native'), ('ddpg', 'native'),
    ('td3', 'native'), ('sac', 'native'), ('ddpg', 'sb3'), ('td3', 'sb3'), ('sac', 'sb3')])
def test_checkpoints_reject_other_mechanism_and_preserve_old_missing_fields(tmp_path, algo, backend):
    algorithm = (f'tests/fixtures/native_algorithms/{algo}.yaml'
                 if backend == 'native' and algo != 'dqn' else f'configs/algo/{algo}.yaml')
    cfg = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml', algorithm)
    revised = replace(cfg, experiment=replace(cfg.experiment, artifact_schema_version=16),
                      auction_flow=replace(cfg.auction_flow, clearing_mechanism=VOLUME_MAX_CLEARING),
                      algo1=replace(cfg.algo1, clob_forecast_weights=()))
    legacy = make_agent(cfg, seed_everything(91, SEED_COMPONENTS))
    current = make_agent(revised, seed_everything(91, SEED_COMPONENTS))
    normalizer = FeatureNormalizer(cfg.grid.tau_cl, relative_prices=cfg.rl.relative_price_features,
        auction_exposure_features=cfg.rl.auction_exposure_features,
        phase_normalization=cfg.rl.phase_normalization, tau_op=cfg.grid.tau_op,
        auction_inventory_asinh=cfg.rl.auction_inventory_asinh,
        inventory_scale=cfg.grid.alpha/cfg.reward.lambda_inv).fit(np.zeros((2, 18)))
    legacy.set_feature_normalizer(normalizer)
    current.set_feature_normalizer(normalizer)
    old_path, new_path = tmp_path/'old.pt', tmp_path/'new.pt'
    legacy.save(old_path)
    # Simulate an actual pre-change checkpoint: neither new field existed.
    state = torch.load(old_path, weights_only=False)
    contract = state.get('dqn_contract', state.get('contract'))
    if contract:
        contract['auction_flow'].pop('clearing_mechanism')
        contract['algo1'].pop('clob_forecast_mechanism')
    torch.save(state, old_path)
    legacy.load(old_path)
    with pytest.raises(ValueError, match='mismatch'):
        current.load(old_path)
    current.save(new_path)
    current.load(new_path)
    with pytest.raises(ValueError, match='mismatch'):
        legacy.load(new_path)


def test_reporting_rejects_relabelled_legacy_results(fixture_run_dir):
    cfg = load_config(fixture_run_dir/'config_resolved.yaml')
    assert cfg.auction_flow.clearing_mechanism == LEGACY_CLEARING
    # Change only the requested config; retained legacy metadata cannot pass.
    revised = replace(cfg, experiment=replace(cfg.experiment, artifact_schema_version=16),
                      auction_flow=replace(cfg.auction_flow, clearing_mechanism=VOLUME_MAX_CLEARING),
                      algo1=replace(cfg.algo1, clob_forecast_weights=()))
    save_resolved(revised, fixture_run_dir/'config_resolved.yaml')
    with pytest.raises(ValueError, match='does not match'):
        collect_runs([fixture_run_dir])


def test_revised_training_evaluation_and_mixed_report_gate(tmp_path, fixture_run_dir):
    from lmm.experiments import train, evaluate
    overrides = dict(
        **{'experiment.results_root': str(tmp_path/'revised'), 'experiment.episodes': 1,
           'grid.tau_op': 3, 'grid.tau_cl': 6, 'grid.h': 3, 'grid.T_physical': 6.,
           'rl.normalizer_fit_episodes': 1, 'rl.validation_frequency_episodes': 1,
           'rl.validation_size': 1, 'rl.checkpoint_min_clob_updates': 0,
           'rl.checkpoint_min_auction_updates': 0, 'rl.test_size': 2,
           'benchmark.as_n_samples': 500})
    args = ['--config', 'configs/base.yaml', '--config', 'configs/synthetic_rough_heston.yaml',
            '--config', 'configs/algo/dqn.yaml', '--config', 'configs/clearing/max_volume_v2.yaml',
            '--run-name', 'smoke', '--seed', '9007']
    for key, value in overrides.items():
        args += ['-o', f'{key}={value}']
    assert train.main(args) == 0
    run = tmp_path/'revised/synthetic_rough_heston/smoke'
    assert evaluate.main(['--run-dir', str(run), '--n-episodes', '2', '--trace-episodes', '1']) == 0
    metadata = yaml.safe_load((run/'eval/metadata.yaml').read_text())
    assert metadata['clearing_mechanism'] == VOLUME_MAX_CLEARING
    assert metadata['artifact_schema_version'] == 16
    assert collect_runs([run])[0].cfg.auction_flow.clearing_mechanism == VOLUME_MAX_CLEARING
    with pytest.raises(ValueError, match='cannot pool'):
        collect_runs([fixture_run_dir, run])
    import pandas as pd
    trace = pd.read_csv(run/'eval/traces/dqn_ep0.csv')
    auction = trace[trace.phase == 'auction']
    assert (auction.clearing_mechanism == VOLUME_MAX_CLEARING).all()
    assert (auction.matched_volume >= 0).all()
    assert np.allclose(auction.h_next, .01 * auction.clearing_tick_index)
