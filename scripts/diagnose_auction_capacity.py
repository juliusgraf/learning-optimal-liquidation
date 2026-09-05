"""Feasible myopic policy probes using current observed moments, never training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lmm.config import load_config, economic_evaluation_config, save_resolved
from lmm.env.action_spaces import ClobAction, AuctionAction
from lmm.env.mdp import make_env


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg=economic_evaluation_config(load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml', 'configs/algo/dqn.yaml', *args.config))
    save_resolved(cfg,args.output/'config.yaml')
    env=make_env(cfg)
    seeds=np.random.default_rng(73109).integers(0,2**31-1,args.episodes).tolist()
    grid=env.auction_grid.actions
    rows=[]
    for quote in [0,1,2,3]:
        for auction in ['noop','moment_greedy']:
            for seed in seeds:
                x,_=env.reset(seed=seed)
                cash=clob_qty=fees=0.
                while True:
                    if env.phase=='clob':
                        v=max(0,min(int(env.inventory),cfg.actions.V_max))
                        action=ClobAction(v,quote if v else 0)
                    elif auction=='noop':
                        action=AuctionAction(0,0,0)
                    else:
                        # Current state and admissibility only. The estimate
                        # is continuous; actual tick/pro-rata outcomes are
                        # subsequently recorded from the real simulator.
                        mask=env.action_mask()
                        candidates=[(i,a) for i,a in enumerate(grid) if mask[i]]
                        K=np.array([a.K_a for _,a in candidates])
                        ell=np.array([a.ell for _,a in candidates])
                        c=np.array([a.cancel for _,a in candidates])
                        mid=float(x[3]); anchor=float(x[2]) if cfg.rl.h_cl_feature_enabled else mid
                        price=mid+cfg.grid.alpha*(np.floor((anchor-mid)/cfg.grid.alpha+.5)+ell)
                        own_K=(1-c)*float(x[13])+K
                        own_W=(1-c)*(float(x[14])-mid*float(x[13]))+K*(price-mid)
                        displacement=(own_W+float(x[17])-mid*float(x[15])+float(x[16]))/(own_K+float(x[15]))
                        qty=own_K*displacement-own_W
                        fee=cfg.reward.d*max(float(x[0])-cfg.grid.tau_op,0)*c
                        score=displacement*qty-cfg.reward.lambda_inv*(float(x[1])-qty)**2-fee
                        action=candidates[int(np.argmax(score))][1]
                    x,_,done,_,info=env.step(action)
                    cash+=info.get('clob_economic_cash',0)
                    clob_qty+=info.get('E_t',0)
                    fees+=info.get('cancellation_fee',0)
                    if done:
                        pnl=cash+info['S_cl']*info['Z']+env.s_mid*info['I_final']-cfg.grid.I0*cfg.grid.S0-fees
                        rows.append(dict(quote=quote,auction=auction,seed=seed,pnl=pnl,
                                         objective=pnl-info['terminal_penalty'],
                                         inventory_open=cfg.grid.I0-clob_qty,
                                         inventory_final=info['I_final'],auction_fill=info['Z'],fees=fees))
                        break
    frame=pd.DataFrame(rows)
    frame.to_csv(args.output/'records.csv',index=False)
    summary=frame.groupby(['quote','auction']).mean(numeric_only=True)
    summary.to_csv(args.output/'summary.csv')
    (args.output/'protocol.json').write_text(json.dumps(dict(seeds=seeds,purpose=__doc__),indent=2))
    print(summary.round(4).to_string())


if __name__=='__main__':
    main()
