"""Training-only forecast prerequisites for the revised multiseed campaign.

Fitted files live under their campaign root and are bound to config, code,
runtime and historical inputs. Never import/retag the retained v19 weights.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import yaml

from lmm.config import (
    LEGACY_CLEARING, VOLUME_MAX_CLEARING, economic_evaluation_config,
    load_config, to_dict,
)

SETTINGS = ('synthetic_rough_heston', 'historical_sp500_midquotes')
FIT_FILES = (
    'config_resolved.yaml', 'protocol.json', 'forecast_overlay.yaml',
    'forecasts.csv', 'error_by_episode.csv', 'summary.csv',
    'decision_weighted_summary.csv', 'aggregation.json',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def forecast_directory(root: Path, setting: str) -> Path:
    if setting not in SETTINGS:
        raise ValueError(f'unknown forecast setting {setting!r}')
    return root / '_forecasts' / setting


def fit_config(repo: Path, root: Path, setting: str):
    return economic_evaluation_config(load_config(
        repo/'configs/base.yaml', repo/f'configs/{setting}.yaml',
        repo/'configs/algo/dqn.yaml', repo/'configs/clearing/max_volume_v2.yaml',
        overrides=[f'experiment.results_root={root.resolve()}', 'algo1.clob_forecast_weights=[]'],
    ))


def fit_contract(repo: Path, root: Path, setting: str, smoke: bool) -> dict:
    cfg = fit_config(repo, root, setting)
    inputs = {}
    if cfg.midprice.historical is not None:
        csv = Path(cfg.midprice.historical.csv_path)
        if not csv.is_absolute():
            csv = repo / csv
        for path in (csv, Path(str(csv) + '.meta.json')):
            inputs[str(path.resolve())] = sha256(path)
    return dict(
        schema=1, mechanism=VOLUME_MAX_CLEARING, config=to_dict(cfg),
        git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        fitter_sha256=sha256(repo/'scripts/diagnose_forecast_credit.py'),
        python=sys.version, packages={name: version(name) for name in ('numpy', 'pandas', 'PyYAML', 'torch')},
        historical_inputs=inputs, setting=setting,
        symbol='MSFT' if setting == SETTINGS[1] else None,
        episodes_per_split=2 if smoke else 256,
        train_seed=194713, validation_seed=194719, fit_split='train',
        validation_role='diagnostic only; never used to fit or select weights',
    )


def validate_forecast_fit(repo: Path, root: Path, setting: str, *, smoke=False) -> Path:
    directory = forecast_directory(root, setting)
    manifest = directory/'completion.json'
    if not manifest.is_file():
        raise ValueError(f'{directory}: missing completed revised forecast fit; incomplete outputs are not reused')
    saved = json.loads(manifest.read_text())
    expected = fit_contract(repo, root, setting, smoke)
    if saved.get('contract') != expected:
        raise ValueError(f'{directory}: forecast config/code/runtime/input contract mismatch; use a fresh campaign root')
    hashes = {name: sha256(directory/name) for name in FIT_FILES}
    if saved.get('files') != hashes:
        raise ValueError(f'{directory}: forecast artifacts changed after completion')
    protocol = json.loads((directory/'protocol.json').read_text())
    for key in ('setting', 'symbol', 'episodes_per_split', 'train_seed', 'validation_seed'):
        if protocol.get(key) != expected[key]:
            raise ValueError(f'{directory}: forecast protocol mismatch at {key}')
    if protocol.get('clearing_mechanism') != VOLUME_MAX_CLEARING:
        raise ValueError(f'{directory}: legacy forecast cannot be used in revised clearing')
    if load_config(directory/'config_resolved.yaml') != fit_config(repo, root, setting):
        raise ValueError(f'{directory}: forecast resolved configuration mismatch')
    overlay = directory/'forecast_overlay.yaml'
    fitted = load_config(repo/'configs/base.yaml', repo/f'configs/{setting}.yaml',
                         repo/'configs/clearing/max_volume_v2.yaml', overlay)
    if len(fitted.algo1.clob_forecast_weights) != 4 or list(fitted.algo1.clob_forecast_weights) != protocol.get('weights'):
        raise ValueError(f'{directory}: fitted forecast coefficients disagree with protocol')
    return overlay


@dataclass(frozen=True)
class ForecastJob:
    setting: str

    @property
    def name(self):
        return 'forecast_' + self.setting

    def command(self, repo: Path, root: Path, smoke: bool):
        return [sys.executable, '-m', 'lmm.experiments.clearing_campaign',
                '--root', str(root), '--setting', self.setting, *(['--smoke'] if smoke else [])]


def prepare_forecast(repo: Path, root: Path, setting: str, *, smoke=False):
    directory = forecast_directory(root, setting)
    if directory.exists():
        validate_forecast_fit(repo, root, setting, smoke=smoke)
        print(f'Reusing validated revised forecast: {directory}', flush=True)
        return
    contract = fit_contract(repo, root, setting, smoke)
    command = [sys.executable, str(repo/'scripts/diagnose_forecast_credit.py'),
               '--setting', setting, '--config', str(repo/'configs/clearing/max_volume_v2.yaml'),
               '--episodes', str(contract['episodes_per_split']), '--output', str(directory),
               '--override', f'experiment.results_root={root.resolve()}']
    if contract['symbol']:
        command += ['--symbol', contract['symbol']]
    subprocess.run(command, cwd=repo, check=True)
    if fit_contract(repo, root, setting, smoke) != contract:
        raise ValueError('forecast inputs changed during fitting; no completion was written')
    # Written last: interrupted or partial fits can never qualify as complete.
    manifest = dict(contract=contract, files={name: sha256(directory/name) for name in FIT_FILES})
    temporary = directory/'.completion.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    temporary.replace(directory/'completion.json')
    validate_forecast_fit(repo, root, setting, smoke=smoke)


def validate_campaign_root(root: Path, mechanism: str):
    """Reject using an existing opposite-mechanism root before doing any work."""
    for path in root.glob('*/*/config_resolved.yaml'):
        cfg = yaml.safe_load(path.read_text()) or {}
        saved = cfg.get('auction_flow', {}).get('clearing_mechanism', LEGACY_CLEARING)
        if saved != mechanism:
            raise ValueError(f'{path}: campaign contains {saved}; choose a separate {mechanism} output root')


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--setting', required=True, choices=SETTINGS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    root = args.root.resolve()
    try:
        validate_campaign_root(root, VOLUME_MAX_CLEARING)
        prepare_forecast(repo, root, args.setting, smoke=args.smoke)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
