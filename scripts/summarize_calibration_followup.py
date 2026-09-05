"""Rebuild follow-up evidence from saved trials and training-date quote metadata.

No training, environment stepping, or held-out performance evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmm.config import load_config
from lmm.experiments.calibration import calibration_scales


def main():
    root = Path('results/calibration_followup')
    output = Path('docs/verification_followup')
    output.mkdir(parents=True, exist_ok=True)
    runs = [p.parent for p in sorted(root.glob('*/confirmation.csv'))]
    runs += [Path('results/pathology_diagnostic_20260905')/name
             for name in ('ddpg_clip_final', 'ddpg_historical_clip')]
    summary, validation, manifest = [], [], []
    for run in runs:
        data = pd.read_csv(run/'confirmation.csv')
        algo = next(x for x in data.policy.unique() if x not in ('as', 'twap'))
        protocol = json.loads((run/'protocol.json').read_text())
        selection = json.loads((run/'selection.json').read_text())
        learner = data[data.policy.eq(algo)].set_index('env_seed')
        noop = pd.read_csv(run/'auction_noop.csv').set_index('env_seed').loc[learner.index]
        np.testing.assert_allclose(learner.clob_qty, noop.clob_qty, atol=1e-10, rtol=0)
        cfg = load_config(run/'config.yaml')
        for policy, group in data.groupby('policy'):
            row = dict(run=run.name, seed=protocol['seed'], policy=policy, episodes=len(group),
                       selected_episode=selection['episode'] if policy == algo else 0,
                       **group[['pnl', 'objective', 'inventory', 'auction_qty', 'clob_qty', 'cancels']].mean().to_dict())
            row['opening_inventory'] = cfg.grid.I0-group.clob_qty.mean()
            if policy == algo:
                row['auction_objective_gain'] = (learner.objective-noop.objective).mean()
                row['auction_pnl_gain'] = (learner.pnl-noop.pnl).mean()
            summary.append(row)
        validation.append(pd.DataFrame(json.loads((run/'validation.json').read_text()))
                          .drop(columns='updates').assign(run=run.name))
        files = ('config.yaml', 'protocol.json', 'selection.json', 'confirmation.csv',
                 'validation.json', 'training.csv', 'auction_noop.csv', 'best.pt')
        manifest.append(dict(source=str(run), protocol=protocol,
                             scales=calibration_scales(cfg),
                             sha256={f: hashlib.sha256((run/f).read_bytes()).hexdigest() for f in files}))
    pd.DataFrame(summary).to_csv(output/'summary.csv', index=False)
    curves = pd.concat(validation)
    curves.to_csv(output/'validation.csv', index=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')

    metadata_path = Path('data/historical_sp500_midquotes_1m.csv.meta.json')
    metadata = json.loads(metadata_path.read_text())
    train = set(metadata['split_session_dates']['train'])
    quotes = [dict(date=s['date'], ticker=ticker, spread_bps=quality['median_spread_bps'])
              for s in metadata['sessions'] if s['date'] in train
              for ticker, quality in s['quote_quality'].items()]
    pd.DataFrame(quotes).to_csv(output/'training_quote_spreads.csv', index=False)
    (output/'quote_provenance.json').write_text(json.dumps(dict(
        source=str(metadata_path), sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        dates=sorted(train), statistic='median across training-session median selected-quote spreads'), indent=2)+'\n')

    # audit2's original v15 label used k*=2000 in error. Use its four penny
    # configurations only, and the corrected rerun for the actual v15 case.
    audit = pd.read_csv(root/'audit2/records.csv')
    audit = audit.loc[~audit.calibration.eq('v15')].copy()
    audit['k_star'] = 10000
    audit['source'] = str(root/'audit2/records.csv')
    corrected = pd.read_csv(root/'audit_v15_corrected/records.csv')
    corrected['source'] = str(root/'audit_v15_corrected/records.csv')
    probes = pd.concat([audit, corrected])
    probes.to_csv(output/'feasible_policy_records.csv', index=False)
    probes.groupby(['calibration', 'reserve', 'schedule']).mean(numeric_only=True).to_csv(output/'feasible_policy_summary.csv')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, names, title in zip(axes,
        [('ddpg_clip_final', 'ddpg_decay_v15', 'ddpg_deadline_synthetic'),
         ('ddpg_historical_clip', 'ddpg_decay_history', 'ddpg_deadline_history_msft')],
        ['Synthetic · seed 615', 'MSFT validation-date replay · seed 616']):
        for name, label in zip(names, ['Original potential, constant rate',
                                      'Fixed 60-episode half-life', 'Deadline-risk potential']):
            frame = curves[curves.run.eq(name)]
            ax.plot(frame.episode, frame.objective, 'o-', markersize=4, label=label)
        ax.axhline(0, color='gray', lw=.8)
        ax.set_xlabel('Training episodes')
        ax.set_ylabel('Economic validation objective')
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Neither DDPG modification repairs both settings; every checkpoint is shown')
    fig.savefig(output/'ddpg_decay.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
