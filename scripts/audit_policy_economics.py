"""Decompose frozen policies on reused development paths; no learning/selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lmm.config import load_config, economic_evaluation_config
from lmm.env.mdp import make_env
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
    paths = np.random.default_rng(791483).integers(0, 2**31-1, args.episodes).tolist()
    episodes, steps, manifest = [], [], []
    for root in args.run:
        cfg = load_config(root/'config.yaml')
        protocol = json.loads((root/'protocol.json').read_text())
        agent = make_agent(cfg, seed_everything(protocol['seed'], SEED_COMPONENTS))
        agent.load(root/'best.pt')
        base = make_env(economic_evaluation_config(cfg), symbol=protocol.get('symbol'), data_split='validation')
        env = wrap_env_for_agent(base, cfg)
        for variant in ['learned', 'no_cancel', 'no_auction']:
            class Policy:
                def __getattr__(self, name): return getattr(agent, name)
                def act(self, obs, mask, phase, **kwargs):
                    a = agent.act(obs, mask, phase, **kwargs)
                    if phase == 'auction' and variant != 'learned':
                        if cfg.algo.name == 'dqn':
                            order = base.auction_grid.actions[a]
                            if variant == 'no_auction': a = 0
                            else:
                                a = next(i for i, x in enumerate(base.auction_grid.actions)
                                         if x.K_a == order.K_a and x.ell == order.ell and x.cancel == 0)
                        else:
                            a = np.asarray(a).copy()
                            if len(a) == 3: a[2] = -1.
                            if variant == 'no_auction': a[:2] = [-1., 0.]
                    return a
            for seed in paths:
                result = run_episode(env, Policy(), seed, chi=1., train=False)
                auction_pnl = (base._S_cl-base.s_mid)*result.auction_exec_qty-result.cancellation_fees
                risk_gain = cfg.reward.lambda_inv*((cfg.grid.I0-result.clob_exec_qty)**2-result.i_final**2)
                episodes.append(dict(run=root.name, algorithm=cfg.algo.name, setting=cfg.experiment.setting,
                    variant=variant, seed=seed, pnl=result.pnl, objective=result.risk_adjusted_pnl,
                    opening_inventory=cfg.grid.I0-result.clob_exec_qty, final_inventory=result.i_final,
                    auction_qty=result.auction_exec_qty, clearing_gap=base._S_cl-base.s_mid,
                    fees=result.cancellation_fees, cancels=result.cancel_count,
                    auction_pnl=auction_pnl, auction_risk_gain=risk_gain,
                    auction_gain=auction_pnl+risk_gain))
                for row in result.action_records:
                    steps.append(dict(run=root.name, variant=variant, seed=seed, **row))
        manifest.append(dict(run=str(root), checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest()))
        print(root.name, 'complete', flush=True)
    frame = pd.DataFrame(episodes)
    frame.to_csv(args.output/'episodes.csv', index=False)
    pd.DataFrame(steps).to_csv(args.output/'steps.csv', index=False)
    means = frame.groupby(['run', 'variant']).mean(numeric_only=True)
    means.to_csv(args.output/'summary.csv')
    (args.output/'protocol.json').write_text(json.dumps(dict(purpose=__doc__,
        interpretation='No-cancel is a diagnostic policy ablation, not a replacement headline policy.',
        seeds=paths, runs=manifest), indent=2))
    print(means[['objective','opening_inventory','auction_qty','fees','auction_pnl','auction_risk_gain','auction_gain']].round(4).to_string())


if __name__ == '__main__': main()
