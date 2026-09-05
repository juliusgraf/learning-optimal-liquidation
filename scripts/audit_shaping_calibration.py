"""Frozen-policy reward sensitivity and economic scales; no parameter selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lmm.config import load_config, economic_evaluation_config
from lmm.env.mdp import make_env
from lmm.experiments.calibration import calibration_scales
from lmm.experiments.train import make_agent, wrap_env_for_agent
from lmm.rl.loops import run_episode, SEED_COMPONENTS
from lmm.utils.seeding import seed_everything


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=64)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = np.random.default_rng(518206).integers(0, 2**31-1, args.episodes).tolist()
    steps, episodes, configs = [], [], []
    for root in args.run:
        cfg = load_config(root/'config.yaml')
        protocol = json.loads((root/'protocol.json').read_text())
        agent = make_agent(cfg, seed_everything(protocol['seed'], SEED_COMPONENTS))
        agent.load(root/'best.pt')
        env = wrap_env_for_agent(make_env(economic_evaluation_config(cfg), symbol=protocol.get('symbol'), data_split='train'), cfg)
        def collect(i, phase, time, reward, cumulative, info, environment):
            if phase != 'clob' or info['E_t'] <= 0: return
            gap = max(info['H_used']-info['S_bullet'], 0.)
            steps.append(dict(run=root.name, seed=seed, time=time, execution=info['E_t'],
                              price=info['S_bullet'], gap=gap, gap_ticks=gap/cfg.grid.alpha))
        for seed in seeds:
            r = run_episode(env, agent, seed, chi=1., train=False, on_step=collect)
            episodes.append(dict(run=root.name, seed=seed, pnl=r.pnl, objective=r.risk_adjusted_pnl,
                                 opening_inventory=cfg.grid.I0-r.clob_exec_qty,
                                 clearing_gap=r.s_cl-r.residual_liquidation_price,
                                 strategic_price_displacement=r.agent_price_displacement,
                                 final_inventory=r.i_final, auction_quantity=r.auction_exec_qty))
        configs.append(dict(run=root.name, scales=calibration_scales(cfg),
                            checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest()))
        print(root.name, 'complete', flush=True)
    f = pd.DataFrame(steps)
    f.to_csv(args.output/'executions.csv', index=False)
    pd.DataFrame(episodes).to_csv(args.output/'episodes.csv', index=False)
    sensitivities = []
    for run, rows in f.groupby('run'):
        alpha = next(x['scales']['clob_zero_reward_gap']/x['scales']['clob_clipping_distance_ticks'] for x in configs if x['run']==run)
        for k in [100, 1000, 10000]:
            deduction = rows.price*rows.execution*np.minimum(rows.gap/(k*alpha), 1)
            sensitivities.append(dict(run=run, k_star=k,
                mean_episode_deduction=deduction.sum()/args.episodes,
                execution_weighted_clipped_fraction=float(np.sum(rows.execution*(rows.gap>=k*alpha))/rows.execution.sum()),
                positive_gap_execution_fraction=float(np.sum(rows.execution*(rows.gap>0))/rows.execution.sum()),
                gap_ticks_p95=rows.gap_ticks.quantile(.95), gap_ticks_max=rows.gap_ticks.max()))
    pd.DataFrame(sensitivities).to_csv(args.output/'k_star_sensitivity.csv', index=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(purpose=__doc__, seeds=seeds,
        evaluation_data_split='train', interpretation='Same fixed policies; changed reward accounting only, not retrained policy outcomes.',
        configurations=configs), indent=2))
    print(pd.DataFrame(sensitivities).round(4).to_string(index=False))


if __name__ == '__main__': main()
