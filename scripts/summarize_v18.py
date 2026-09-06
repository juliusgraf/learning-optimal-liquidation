"""Summarize every bounded v18 attempt, including rejected/immature candidates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/_development_v18')
    output = Path('docs/verification_v18')
    output.mkdir(parents=True, exist_ok=True)
    rows, validations, manifest = [], [], []
    for p in sorted(root.iterdir()):
        if not (p/'config.yaml').exists() or not (p/'validation.json').exists():
            continue
        cfg = yaml.safe_load((p/'config.yaml').read_text())
        hp, rl = cfg['algo']['hyperparams'], cfg['rl']
        protocol = json.loads((p/'protocol.json').read_text())
        v = json.loads((p/'validation.json').read_text())
        for point in v:
            validations.append(dict(run=p.name, **point))
        row = dict(run=p.name, algorithm=cfg['algo']['name'], seed=protocol['seed'],
                   setting=protocol['setting'], symbol=protocol.get('symbol'),
                   episodes=protocol['episodes'], flow=cfg['clob_flow']['lambda0'],
                   n_step=rl['n_step'], layer_norm=hp.get('layer_norm',False),
                   safe_initialization=hp.get('safe_auction_initialization',False),
                   grad_clip=hp.get('grad_clip_norm'),
                   initial_validation=v[0]['objective'], final_validation=v[-1]['objective'],
                   status='no_mature_checkpoint')
        if (p/'training.csv').exists():
            training = pd.read_csv(p/'training.csv')
            for phase in ['clob','auction']:
                column = f'loss_{phase}'
                row[f'max_{column}'] = training[column].max() if column in training else np.nan
        if (p/'confirmation.csv').exists() and (p/'auction_noop.csv').exists():
            d = pd.read_csv(p/'confirmation.csv')
            a = d[d.policy == cfg['algo']['name']].set_index('env_seed').sort_index()
            b = pd.read_csv(p/'auction_noop.csv').set_index('env_seed').sort_index()
            assert a.index.equals(b.index)
            np.testing.assert_allclose(a.clob_qty, b.clob_qty, atol=1e-10)
            selection = json.loads((p/'selection.json').read_text())
            row.update(status='completed', selected_episode=selection['episode'],
                       selected_validation=selection['objective'],
                       pnl=a.pnl.mean(), objective=a.objective.mean(),
                       auction_value=(a.objective-b.objective).mean(),
                       auction_value_se=(a.objective-b.objective).std()/np.sqrt(len(a)),
                       open_abs_inventory=a.inventory_at_auction_open.abs().mean(),
                       final_abs_inventory=a.inventory.abs().mean(),
                       risk_relief=.01*(a.inventory_at_auction_open**2-a.inventory**2).mean())
            for ref in ['as','twap']:
                reference = d[d.policy == ref].set_index('env_seed').sort_index()
                assert a.index.equals(reference.index)
                row[ref+'_objective'] = reference.objective.mean()
                row['gap_'+ref] = (a.objective-reference.objective).mean()
        rows.append(row)
        manifest.append(dict(run=str(p), artifacts={str(q.relative_to(p)):digest(q) for q in p.iterdir()
            if q.is_file() and q.suffix in ('.csv','.json','.yaml')}))
    pd.DataFrame(rows).to_csv(output/'all_attempts.csv', index=False)
    pd.DataFrame(validations).to_csv(output/'validation.csv', index=False)
    stress = []
    for p in sorted(root.glob('stress_*')):
        if not (p/'protocol.json').exists():
            continue
        d = pd.read_csv(p/'diagnostics.csv')
        v = pd.read_csv(p/'validation.csv')
        protocol = json.loads((p/'protocol.json').read_text())
        for phase, group in d.groupby('phase'):
            stress.append(dict(run=p.name, phase=phase, max_loss=group.loss.max(),
                               max_absolute_q=group.q_abs_max.max(),
                               initial_objective=v.objective_mean.iloc[0],
                               final_objective=v.objective_mean.iloc[-1],
                               final_objective_se=v.objective_se.iloc[-1],
                               learning_rate_episode=protocol['learning_rate_episode'],
                               extra_auction_updates=v.auction_updates.iloc[-1],
                               extra_clob_updates=v.clob_updates.iloc[-1]))
    pd.DataFrame(stress).to_csv(output/'replay_stress.csv', index=False)
    probes = []
    for name in ['auction_inventory', 'auction_inventory_clip10', 'transfer_validation']:
        path = root/name
        if (path/'protocol.json').exists():
            probes.append(dict(run=str(path), artifacts={str(q.relative_to(path)):digest(q)
                          for q in path.iterdir() if q.suffix in ['.csv','.json']}))
            for filename in ['summary.csv','protocol.json']:
                (output/f'{name}_{filename}').write_bytes((path/filename).read_bytes())
    (output/'manifest.json').write_text(json.dumps(dict(
        purpose=__doc__, no_publication_test=True, runs=manifest,
        total_training_episodes=sum(r['episodes'] for r in rows),
        conditional_probes=probes,
        replay_stress_artifacts={str(p):digest(p) for root_path in sorted(root.glob('stress_*'))
                                for p in root_path.iterdir() if p.suffix in ['.csv','.json']},
        protected_sha256={str(p):digest(p) for p in [Path('paper/main.tex'),Path('paper/results/tables_params/params_generative.tex')]},
        source_sha256={str(p):digest(p) for folder in ['src/lmm','configs','scripts'] for p in sorted(Path(folder).rglob('*'))
                       if p.suffix in ['.py','.yaml','.sh']},
    ), indent=2))
    print(pd.DataFrame(rows)[['run','status','objective','auction_value','final_validation']].to_string(index=False))


if __name__ == '__main__':
    main()
