"""Policy-free dimensional and auction-capacity audit; no learner training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from lmm.config import load_config, save_resolved
from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.env.mdp import make_env


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--calibration', action='append', choices=['v15', 'penny_small', 'penny_b05', 'penny_b10', 'penny_b20'])
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    base = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml',
                       'configs/diagnostic/v15_reference.yaml')
    seeds = np.random.default_rng(98116).integers(0, 2**31-1, args.episodes).tolist()
    rows = []
    for name, alpha, beta, depth in [('v15', .05, .002, 1), ('penny_small', .01, .002, 1),
                                     ('penny_b05', .01, .05, 10), ('penny_b10', .01, .1, 10),
                                     ('penny_b20', .01, .2, 10)]:
        if args.calibration and name not in args.calibration:
            continue
        cfg = replace(base, grid=replace(base.grid, alpha=alpha),
                      actions=replace(base.actions, beta=beta),
                      auction_flow=replace(base.auction_flow, U1=.1*depth, U2=2*depth),
                      reward=replace(base.reward, k_star=base.reward.k_star if name == 'v15' else int(base.grid.S0/alpha)))
        save_resolved(cfg, args.output/f'{name}.yaml')
        env = make_env(cfg)
        for reserve in [0, 10, 20]:
            for schedule in ['noop', 'all', 'last10', 'last1']:
                for seed in seeds:
                    obs, _ = env.reset(seed=seed)
                    clob_cash = reward = shape = fees = 0.
                    inventory_open = None
                    while True:
                        if env.phase == 'clob':
                            volume = max(0, min(cfg.actions.V_max, int(env.inventory-reserve)))
                            action = ClobAction(volume, 1 if volume else 0)
                        else:
                            if inventory_open is None:
                                inventory_open = env.inventory
                            active = schedule == 'all' or (schedule == 'last10' and env.t >= 140) or (schedule == 'last1' and env.t >= 149)
                            action = AuctionAction(cfg.actions.auction_K_grid_max if active else 0., -10 if active else 0, 0)
                            if not env.action_mask()[env.auction_grid.actions.index(action)]:
                                action = AuctionAction(0., 0, 0)
                        obs, r, done, _, info = env.step(action)
                        reward += r
                        clob_cash += info.get('clob_economic_cash', 0)
                        shape += info.get('auction_interim_shaping', 0)
                        fees += info.get('cancellation_fee', 0)
                        if done:
                            pnl = clob_cash+info['S_cl']*info['Z']+env.s_mid*info['I_final']-cfg.grid.S0*cfg.grid.I0-fees
                            rows.append(dict(calibration=name, alpha=alpha, beta=beta, depth_scale=depth,
                                             k_star=cfg.reward.k_star, q=cfg.reward.q, lambda_inv=cfg.reward.lambda_inv,
                                             reserve=reserve, schedule=schedule, seed=seed,
                                             inventory_open=inventory_open, auction_fill=info['Z'],
                                             inventory_final=info['I_final'], pnl=pnl,
                                             objective=pnl-info['terminal_penalty'], shaped_return=reward,
                                             interim_credit=shape, clearing_displacement=info['S_cl']-env.s_mid))
                            break
        print(name, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'records.csv', index=False)
    frame.groupby(['calibration','reserve','schedule']).mean(numeric_only=True).to_csv(args.output/'summary.csv')
    (args.output/'protocol.json').write_text(json.dumps(dict(seeds=seeds,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='fixed feasible policy probes, not trained policy results'), indent=2))


if __name__ == '__main__':
    main()
