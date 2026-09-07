"""Audit completed H-anchored conditioned-economic results without rerunning policies."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from lmm.experiments.cashflow_comparison import read_run, validate_pair, run_path
from lmm.experiments.make_report import economic_rows, require_paired, mean_interval
from lmm.experiments.protocol import PUBLICATION_SEEDS

ROOT = Path('results/revision_v19_economic_dense_h')
HEAD = Path('results/revision_v19')
OUT = Path(__file__).parent
status = json.loads((ROOT/'_orchestration/status.json').read_text())
assert status['state'] == 'complete' and len(status['completed']) == 40 and not status['failed']
rows = []
max_economic_error = 0.
for algo in ('dqn', 'ddpg', 'td3', 'sac'):
    for seed in PUBLICATION_SEEDS:
        h = read_run(run_path(HEAD, algo, seed, cashflow=False))
        c = read_run(run_path(ROOT, algo, seed, cashflow=True, conditioned=True))
        validate_pair(h, c, conditioned=True)
        he, ce = economic_rows(h, algo), economic_rows(c, algo)
        require_paired(he, ce)
        m = pd.read_csv(c.run_dir/'metrics.csv')
        sel = yaml.safe_load((c.run_dir/'checkpoints/best_selection.yaml').read_text())
        init = sel['economic_safety']['initial_validation_score']
        v = m[m.eval_risk_adjusted_pnl_mean.notna()]
        assert np.isfinite(m[['training_return', 'replay_return_unscaled', 'economic_objective']]).all().all()
        for col in ('loss_clob', 'loss_auction'):
            assert len(m[col].dropna()) and np.isfinite(m[col].dropna()).all()
        assert set(m.env_seed).isdisjoint(set(c.records.env_seed))
        assert set(sel['validation_seeds']).isdisjoint(set(c.records.env_seed))
        max_economic_error = max(max_economic_error, abs(m.training_return-m.economic_objective).max())
        rows.append(dict(algo=algo, seed=seed, headline_bps=he.objective_bps.mean(),
                         economic_dense_bps=ce.objective_bps.mean(),
                         difference_bps=(he.objective_bps-ce.objective_bps).mean(),
                         pnl_bps=ce.pnl_bps.mean(), penalty_bps=(ce.pnl_bps-ce.objective_bps).mean(),
                         auction_bps=ce.auction_value_bps.mean(),
                         open_inventory=ce.open_abs_inventory_pct.mean(),
                         close_inventory=ce.close_abs_inventory_pct.mean(),
                         episodes=len(m), selected_episode=sel['episode'], initial_val=init,
                         selected_val=sel['value'], final_val=v.eval_risk_adjusted_pnl_mean.iloc[-1],
                         late_clob_loss=m.loss_clob.tail(50).median(),
                         late_auction_loss=m.loss_auction.tail(50).median()))
data = pd.DataFrame(rows).sort_values(['algo','seed'])
saved = pd.read_csv(ROOT/'_comparison/by_seed.csv').sort_values(['algo','seed'])
assert data[['algo','seed']].reset_index(drop=True).equals(saved[['algo','seed']].reset_index(drop=True))
for col in ('headline_bps','economic_dense_bps','difference_bps'):
    np.testing.assert_allclose(data[col], saved[col], rtol=0, atol=1e-10)
raw = pd.read_csv('results/revision_v19_cashflow/_comparison/by_seed.csv').set_index(['algo','seed'])
summary = []
for algo, g in data.groupby('algo', sort=False):
    mean, lo, hi = mean_interval(g.difference_bps)
    conditioning = g.set_index('seed').economic_dense_bps - raw.loc[algo].cashflow_bps
    cm, cl, ch = mean_interval(conditioning)
    summary.append(dict(algo=algo, headline_bps=g.headline_bps.mean(), economic_dense_bps=g.economic_dense_bps.mean(),
                        preference_gap=mean, ci_low=lo, ci_high=hi, headline_wins=int((g.difference_bps>0).sum()),
                        median_preference_gap=g.difference_bps.median(),
                        conditioning_gain=cm, conditioning_ci_low=cl, conditioning_ci_high=ch,
                        selected_improves_initial=int((g.selected_val>g.initial_val).sum()),
                        final_improves_initial=int((g.final_val>g.initial_val).sum()),
                        mean_auction_bps=g.auction_bps.mean(), mean_pnl_bps=g.pnl_bps.mean(),
                        mean_penalty_bps=g.penalty_bps.mean(), mean_episodes=g.episodes.mean(),
                        mean_late_clob_loss=g.late_clob_loss.mean(), mean_late_auction_loss=g.late_auction_loss.mean()))
table = pd.DataFrame(summary)
data.to_csv(OUT/'by_seed.csv', index=False)
table.to_csv(OUT/'summary.csv', index=False)
verification = dict(validated_pairs=40, reproduced_effects=40, configuration_and_completion_hash_checks='passed',
                    training_episodes=int(data.episodes.sum()), seed_separation='passed',
                    nonfinite_required_values=0, economic_training_identity_max_error=float(max_economic_error))
(OUT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
print(table.to_string(index=False))
print(json.dumps(verification, indent=2))
