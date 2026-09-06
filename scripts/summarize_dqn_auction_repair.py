"""Keep every bounded auction-repair attempt and diagnostic, including failures."""
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
    root = Path('results/_auction_repair_v18')
    output = Path('docs/verification_v18/auction_repair')
    output.mkdir(parents=True, exist_ok=True)
    attempts, paths, values, stress, transfers, manifest = [], [], [], [], [], []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        files = [q for q in p.iterdir() if q.suffix in ('.csv','.json','.yaml')]
        manifest.append(dict(run=str(p), artifacts={q.name:digest(q) for q in files}))
        if (p/'training.csv').exists():
            cfg = yaml.safe_load((p/'config.yaml').read_text())
            training = pd.read_csv(p/'training.csv')
            row = dict(run=p.name, algorithm=cfg['algo']['name'],
                       seed=cfg['experiment']['master_seed'], episodes=len(training),
                       status='incomplete',
                       control_exploration=cfg['algo']['hyperparams'].get('auction_control_exploration_probability',0.),
                       centered_phases=str(cfg['algo']['hyperparams'].get('reference_centered_phases',
                           ['clob','auction'] if cfg['algo']['hyperparams'].get('reference_centered_advantage') else [])),
                       known_fee=cfg['algo']['hyperparams'].get('known_auction_fee',False))
            if (p/'confirmation.csv').exists() and (p/'auction_noop.csv').exists():
                all_policies = pd.read_csv(p/'confirmation.csv')
                a = all_policies[all_policies.policy==row['algorithm']].set_index('env_seed').sort_index()
                b = pd.read_csv(p/'auction_noop.csv').set_index('env_seed').sort_index()
                assert a.index.equals(b.index)
                np.testing.assert_allclose(a.clob_qty,b.clob_qty,atol=1e-10)
                x = a.objective-b.objective
                selection = json.loads((p/'selection.json').read_text())
                row.update(status='completed', selected_episode=selection['episode'],
                    selected_validation=selection['objective'], paths=len(a), pnl=a.pnl.mean(),
                    objective=a.objective.mean(), auction_value=x.mean(), auction_value_se=x.sem(),
                    cancels=a.cancels.mean(), opening_abs_inventory=a.inventory_at_auction_open.abs().mean(),
                    final_abs_inventory=a.inventory.abs().mean())
                for ref in ['as','twap']:
                    r = all_policies[all_policies.policy==ref].set_index('env_seed').sort_index()
                    assert a.index.equals(r.index)
                    row[f'{ref}_objective'] = r.objective.mean()
                # Preserve pathwise confirmation, not just rounded positive summaries.
                for name in ['confirmation.csv','auction_noop.csv','validation.json','selection.json']:
                    dest=output/'runs'/p.name
                    dest.mkdir(parents=True,exist_ok=True)
                    (dest/name).write_bytes((p/name).read_bytes())
            attempts.append(row)
        if (p/'outcomes.csv').exists() and (p/'actions.csv').exists():
            d = pd.read_csv(p/'outcomes.csv')
            actions = pd.read_csv(p/'actions.csv')
            t = pd.read_csv(p/'terminal_advantages.csv')
            chosen = t[t.chosen]
            x = d.auction_value
            paths.append(dict(run=p.name, paths=len(d), objective=d.objective.mean(),
                pnl=d.pnl.mean(), auction_value=x.mean(), auction_value_se=x.sem(),
                ci_low=x.mean()-1.96*x.sem(),ci_high=x.mean()+1.96*x.sem(),
                cancels=actions.groupby('env_seed').cancel.sum().mean(),
                opening_abs_inventory=d.opening_inventory.abs().mean(),
                final_abs_inventory=d.final_inventory.abs().mean(),
                price_edge=d.price_edge.mean(), risk_relief=d.risk_relief.mean(), fees=d.fees.mean()))
            values.append(dict(run=p.name, paths=len(chosen),
                predicted_advantage=chosen.predicted_advantage.mean(),
                actual_shaped_advantage=chosen.shaped_advantage.mean(),
                actual_economic_advantage=chosen.economic_advantage.mean()))
            dest=output/'paths'/p.name
            dest.mkdir(parents=True,exist_ok=True)
            for name in ['outcomes.csv','terminal_advantages.csv','protocol.json']:
                (dest/name).write_bytes((p/name).read_bytes())
        if (p/'summary.json').exists() and (p/'protocol.json').exists():
            protocol=json.loads((p/'protocol.json').read_text())
            if 'clob_run' in protocol:
                transfers.append(dict(run=p.name,**json.loads((p/'summary.json').read_text())))
                dest=output/'transfers'/p.name
                dest.mkdir(parents=True,exist_ok=True)
                for name in ['outcomes.csv','summary.json','protocol.json']:
                    (dest/name).write_bytes((p/name).read_bytes())
        if (p/'diagnostics.csv').exists() and (p/'protocol.json').exists():
            d = pd.read_csv(p/'diagnostics.csv')
            v = pd.read_csv(p/'validation.csv')
            for phase,g in d.groupby('phase'):
                stress.append(dict(run=p.name,phase=phase,max_absolute_q=g.q_abs_max.max(),
                    max_loss=g.loss.max(),all_finite=bool(np.isfinite(g.select_dtypes('number')).all().all()),
                    final_objective=v.objective_mean.iloc[-1],final_objective_se=v.objective_se.iloc[-1],
                    extra_auction_updates=v.auction_updates.iloc[-1],extra_clob_updates=v.clob_updates.iloc[-1]))
            dest=output/'stress'/p.name
            dest.mkdir(parents=True,exist_ok=True)
            for name in ['diagnostics.csv','validation.csv','protocol.json']:
                (dest/name).write_bytes((p/name).read_bytes())
    for name,rows in [('all_attempts',attempts),('path_checks',paths),('terminal_values',values),('stress',stress),('auction_transfers',transfers)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    (output/'artifact_manifest.json').write_text(json.dumps(dict(
        purpose=__doc__,attempts=len(attempts),training_episodes=sum(a['episodes'] for a in attempts),
        all_attempts_included=True,no_final_test=True,artifacts=manifest,
        protected_sha256={str(p):digest(p) for p in [Path('paper/main.tex'),Path('paper/results/tables_params/params_generative.tex')]}
    ),indent=2))
    print(pd.DataFrame(attempts).to_string(index=False))
    print(pd.DataFrame(paths).to_string(index=False))


if __name__ == '__main__':
    main()
