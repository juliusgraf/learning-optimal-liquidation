"""Revised forecast staging, provenance and actual shell config forwarding."""
import json
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from lmm.config import LEGACY_CLEARING, VOLUME_MAX_CLEARING, load_config, save_resolved
from lmm.experiments import clearing_campaign as C
from lmm.experiments.run_matrix import ALGOS, ARMS

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = [.1, .2, .3, .4]


def _fake_fitter(monkeypatch, root):
    """Only generate test artifacts; coefficient estimation is tested separately."""
    calls = []
    original_run = subprocess.run
    def run(command, **kwargs):
        if '--setting' not in command:
            return original_run(command, **kwargs)
        calls.append(command)
        setting = command[command.index('--setting')+1]
        directory = C.forecast_directory(root, setting)
        directory.mkdir(parents=True)
        for name in C.FIT_FILES:
            (directory/name).write_text('test fixture\n')
        save_resolved(C.fit_config(REPO, root, setting), directory/'config_resolved.yaml')
        (directory/'protocol.json').write_text(json.dumps(dict(
            setting=setting, symbol=None, episodes_per_split=2,
            train_seed=194713, validation_seed=194719,
            clearing_mechanism=VOLUME_MAX_CLEARING, weights=WEIGHTS)))
        (directory/'forecast_overlay.yaml').write_text(yaml.safe_dump({'algo1': {
            'clob_forecast_weights': WEIGHTS, 'clob_forecast_mechanism': VOLUME_MAX_CLEARING}}))
    monkeypatch.setattr(C.subprocess, 'run', run)
    return calls


def test_complete_fits_reuse_only_exact_artifacts_and_protocol(tmp_path, monkeypatch):
    calls = _fake_fitter(monkeypatch, tmp_path)
    setting = C.SETTINGS[0]
    C.prepare_forecast(REPO, tmp_path, setting, smoke=True)
    C.prepare_forecast(REPO, tmp_path, setting, smoke=True)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='contract mismatch'):
        C.validate_forecast_fit(REPO, tmp_path, setting, smoke=False)
    directory = C.forecast_directory(tmp_path, setting)
    (directory/'forecasts.csv').write_text('modified')
    with pytest.raises(ValueError, match='artifacts changed'):
        C.prepare_forecast(REPO, tmp_path, setting, smoke=True)
    assert len(calls) == 1


def test_partial_fit_is_not_overwritten(tmp_path, monkeypatch):
    calls = _fake_fitter(monkeypatch, tmp_path)
    C.forecast_directory(tmp_path, C.SETTINGS[0]).mkdir(parents=True)
    with pytest.raises(ValueError, match='missing completed'):
        C.prepare_forecast(REPO, tmp_path, C.SETTINGS[0], smoke=True)
    assert not calls


def test_forecast_failure_does_not_write_completion(tmp_path, monkeypatch):
    original_run = subprocess.run
    def fail(command, **kwargs):
        if '--setting' not in command:
            return original_run(command, **kwargs)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(C.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        C.prepare_forecast(REPO, tmp_path, C.SETTINGS[0], smoke=True)
    assert not list(tmp_path.rglob('completion.json'))


def test_campaign_root_cannot_mix_mechanisms(tmp_path):
    run = tmp_path/'synthetic_rough_heston/dqn_seed42'
    run.mkdir(parents=True)
    # Missing historical mechanism field must continue to mean v1.
    (run/'config_resolved.yaml').write_text('auction_flow: {}\n')
    C.validate_campaign_root(tmp_path, LEGACY_CLEARING)
    with pytest.raises(ValueError, match='separate max_volume_v2 output root'):
        C.validate_campaign_root(tmp_path, VOLUME_MAX_CLEARING)


@pytest.mark.parametrize('setting', ['synthetic_rough_heston', 'historical_sp500_midquotes',
    'synthetic_rough_heston__mechanism_economic_dense'])
def test_publication_checks_run_weights_against_independent_fit(tmp_path, monkeypatch, setting):
    from lmm.experiments import publication as P
    base_setting = C.SETTINGS[setting == C.SETTINGS[1]]
    overlay = tmp_path/'forecast.yaml'
    overlay.write_text(yaml.safe_dump({'algo1': {
        'clob_forecast_weights': WEIGHTS, 'clob_forecast_mechanism': VOLUME_MAX_CLEARING}}))
    checked = []
    def validate(repo, root, requested_setting):
        checked.append((root, requested_setting))
        return overlay
    monkeypatch.setattr(C, 'validate_forecast_fit', validate)
    configs = ['configs/base.yaml', f'configs/{base_setting}.yaml', 'configs/algo/dqn.yaml']
    if '__' in setting:
        configs += [f'configs/treatment/{setting.split("__")[1]}.yaml']
    cfg = load_config(*configs, 'configs/clearing/max_volume_v2.yaml', overlay,
        overrides=[f'experiment.results_root={tmp_path}', f'experiment.name={setting}',
                   'experiment.master_seed=42', 'experiment.seeds=[42]'])
    run = SimpleNamespace(cfg=cfg, algo='dqn', setting=setting, seed=42,
                          run_dir=tmp_path/setting/'dqn_seed42')
    P._validate_canonical_publication_config(run)
    assert checked == [(tmp_path, base_setting)]
    run.cfg = replace(cfg, algo1=replace(cfg.algo1, clob_forecast_weights=(.9,.9,.9,.9)))
    with pytest.raises(ValueError, match='clob_forecast_weights'):
        P._validate_canonical_publication_config(run)


@pytest.mark.parametrize('block', ['synthetic', 'MSFT', *ARMS])
@pytest.mark.parametrize('algo', ALGOS)
def test_every_worker_configuration_uses_matching_revised_fit(tmp_path, block, algo):
    if block == 'MSFT' and not (REPO/'data/historical_sp500_midquotes_1m.csv').exists():
        pytest.skip('private historical artifact not installed')
    for setting in C.SETTINGS:
        directory = C.forecast_directory(tmp_path, setting)
        directory.mkdir(parents=True)
        (directory/'forecast_overlay.yaml').write_text(yaml.safe_dump({'algo1': {
            'clob_forecast_weights': WEIGHTS, 'clob_forecast_mechanism': VOLUME_MAX_CLEARING}}))
    fake = tmp_path/'record-python'
    fake.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                    'with open(os.environ["CALL_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    log = tmp_path/'calls.jsonl'
    env = dict(os.environ, LMM_PYTHON=str(fake), LMM_RESULTS_ROOT=str(tmp_path),
               LMM_CLEARING_MECHANISM=VOLUME_MAX_CLEARING,
               LMM_FORECAST_ROOT=str(tmp_path/'_forecasts'), CALL_LOG=str(log))
    result = subprocess.run(['/bin/bash', 'scripts/_run_matrix_job.sh', '42', algo, block, '0'],
                            cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    command = commands[0]
    configs = [command[i+1] for i, value in enumerate(command) if value == '--config']
    overrides = [command[i+1] for i, value in enumerate(command) if value == '-o']
    cfg = load_config(*configs, overrides=overrides)
    assert cfg.auction_flow.clearing_mechanism == cfg.algo1.clob_forecast_mechanism == VOLUME_MAX_CLEARING
    assert list(cfg.algo1.clob_forecast_weights) == WEIGHTS
    assert cfg.experiment.artifact_schema_version == 16
    assert cfg.experiment.results_root == tmp_path
    assert cfg.experiment.episodes == 800 and cfg.rl.validation_size == 128 and cfg.rl.test_size == 100
    assert configs[-1] == str(C.forecast_directory(tmp_path, C.SETTINGS[block == 'MSFT'])/'forecast_overlay.yaml')
    assert [c[1] for c in commands] == ['lmm.experiments.train', 'lmm.experiments.evaluate',
        'lmm.experiments.policy_differences', 'lmm.experiments.policy_differences', 'lmm.experiments.publication']
