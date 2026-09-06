"""Rebuild the complete bounded v19 evidence ledger without training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path('docs/verification_v19')


def main():
    rows, runs = [], {}
    for path in sorted(ROOT.glob('*/selection.json')):
        directory = path.parent
        cfg = yaml.safe_load((directory/'config.yaml').read_text())
        protocol = json.loads((directory/'protocol.json').read_text())
        validation = json.loads((directory/'validation.json').read_text())
        selected = json.loads(path.read_text())
        train = pd.read_csv(directory/'training.csv')
        all_confirmation = pd.read_csv(directory/'confirmation.csv')
        algo = cfg['algo']['name']
        confirmation = all_confirmation[all_confirmation.policy == algo].set_index('env_seed')
        noop = pd.read_csv(directory/'auction_noop.csv').set_index('env_seed')
        assert confirmation.index.is_unique and noop.index.is_unique
        assert set(confirmation.index) == set(noop.index)
        assert set(protocol['validation_seeds']).isdisjoint(confirmation.index)
        finite = all(np.isfinite(df.select_dtypes('number').to_numpy()).all()
                     for df in (confirmation, noop, pd.DataFrame(validation).drop(columns='updates')))
        losses = train.filter(regex=r'^(loss_|actor_loss_)').iloc[32:]
        finite_losses = np.isfinite(losses.to_numpy()).all()
        assert finite and finite_losses, directory
        factor = 10000/(cfg['grid']['S0']*cfg['grid']['I0'])
        row = dict(run=directory.name, algorithm=algo, seed=protocol['seed'],
                   setting=protocol['setting'], symbol=protocol.get('symbol'),
                   episodes=len(train), validation_paths=len(protocol['validation_seeds']),
                   confirmation_paths=len(confirmation),
                   initial_bps=validation[0]['objective']*factor,
                   last_bps=validation[-1]['objective']*factor,
                   selected_bps=selected['objective']*factor,
                   selected_episode=selected['episode'],
                   best_to_last_bps=(max(x['objective'] for x in validation)-validation[-1]['objective'])*factor,
                   confirmation_bps=confirmation.objective.mean()*factor,
                   confirmation_pnl_bps=confirmation.pnl.mean()*factor,
                   auction_bps=(confirmation.objective-noop.objective).mean()*factor,
                   opening_inventory=confirmation.inventory_at_auction_open.mean(),
                   final_inventory=confirmation.inventory.mean(),
                   absolute_final_inventory=confirmation.inventory.abs().mean(),
                   auction_qty=confirmation.auction_qty.mean(),
                   finite_evaluations=finite, finite_post_warmup_losses=finite_losses)
        for ref in ('as', 'twap'):
            reference = all_confirmation[all_confirmation.policy == ref].set_index('env_seed')
            assert set(reference.index) == set(confirmation.index)
            row['gap_'+ref+'_bps'] = (confirmation.objective-reference.objective).mean()*factor
        rows.append(row)
        runs[directory.name] = (confirmation.objective*factor, protocol)
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT/'learning_summary.csv', index=False)

    def difference(positive, negative, contrast):
        p, pm = runs[positive]; n, nm = runs[negative]
        assert set(p.index) == set(n.index)
        assert pm['validation_seeds'] == nm['validation_seeds']
        return dict(contrast=contrast, positive=positive, negative=negative,
                    mean_difference_bps=(p-n).mean(), n_training_seed_pairs=1,
                    n_matched_confirmation_paths=len(p))

    paired = []
    for algo in ('dqn', 'ddpg', 'td3', 'sac'):
        for seed in (1951, 1973):
            prefix = f'confirm_{algo}{seed}_'
            paired.append(difference(prefix+'candidate', prefix+'v18', 'final_spec_vs_v18'))
    for prefix in ('cat811_', 'synthetic1907_', 'confirm_cat1979_'):
        paired.append(difference(prefix+'ln', prefix+'v18', 'critic_layer_norm_only'))
    pd.DataFrame(paired).to_csv(ROOT/'paired_confirmation.csv', index=False)
    effects = []
    for algo in ('dqn', 'ddpg'):
        prefix = algo+'1931_'
        for contrast, positive, negative in (
            ('auction_credit', 'economic_dense', 'economic_sparse'),
            ('h_feature', 'economic_dense', 'feature_off'),
            ('h_anchor', 'indicative', 'full'),
            ('clob_preference', 'clob_only', 'economic_dense'),
            ('auction_preference', 'auction_only', 'economic_dense'),
            ('combined_preferences', 'full', 'economic_dense'),
            ('forecast_calibration', 'full', 'raw_forecast')):
            effects.append(difference(prefix+positive, prefix+negative, contrast))
    pd.DataFrame(effects).to_csv(ROOT/'mechanism_effects.csv', index=False)
    verification = dict(training_runs=len(summary), training_episodes=int(summary.episodes.sum()),
                        maximum_run_episodes=int(summary.episodes.max()),
                        finite_evaluations=bool(summary.finite_evaluations.all()),
                        finite_post_warmup_losses=bool(summary.finite_post_warmup_losses.all()),
                        final_test_used=False,
                        interpretation='Adaptive bounded development, not publication inference; all attempts retained.')
    (ROOT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
