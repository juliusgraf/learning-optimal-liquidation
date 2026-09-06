"""Hold the CLOB controller fixed to attribute an auction-controller change.

Diagnostic only: no fitting, checkpoint selection or publication-policy splice.
Both checkpoints must use the identical frozen observation normalizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lmm.config import load_config, economic_evaluation_config
from lmm.env.mdp import make_env
from lmm.experiments.train import make_agent
from lmm.rl.loops import SEED_COMPONENTS, run_episode
from lmm.utils.seeding import seed_everything


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--clob-run',type=Path,required=True)
    p.add_argument('--auction-run',type=Path,required=True)
    p.add_argument('--auction-checkpoint',type=Path,default=Path('best.pt'),
                   help='Saved checkpoint relative to auction-run; diagnostics only, no reselection.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--path-seed',type=int,default=942287)
    p.add_argument('--episodes',type=int,default=512)
    args=p.parse_args()
    if not 1<=args.episodes<=512:
        p.error('bounded diagnostic requires 1..512 validation paths')
    args.output.mkdir(parents=True,exist_ok=False)
    learners=[]
    configs=[]
    checkpoints=[args.clob_run/'best.pt',args.auction_run/args.auction_checkpoint]
    for run,checkpoint in zip([args.clob_run,args.auction_run],checkpoints):
        cfg=load_config(run/'config.yaml')
        agent=make_agent(cfg,seed_everything(917,SEED_COMPONENTS))
        agent.load(checkpoint)
        agent.set_train(False)
        learners.append(agent)
        configs.append(cfg)
    clob,auction=learners
    assert clob._feature_normalizer.state_dict()==auction._feature_normalizer.state_dict()
    for field in ['grid','actions','reward','clob_flow','auction_flow','midprice','features']:
        assert getattr(configs[0],field)==getattr(configs[1],field),field
    protocol=json.loads((args.clob_run/'protocol.json').read_text())
    env=make_env(economic_evaluation_config(configs[0]),symbol=protocol.get('symbol'),data_split='validation')

    class Mixed:
        def __getattr__(self,name): return getattr(auction,name)
        def act(self,obs,mask,phase,**kwargs):
            return (clob if phase=='clob' else auction).act(obs,mask,phase,**kwargs)

    seeds=np.random.default_rng(args.path_seed).integers(0,2**31-1,args.episodes).tolist()
    rows=[]
    for seed in seeds:
        results=[]
        for label,policy in [('original',clob),('diagnostic_auction_transfer',Mixed())]:
            policy.bind(env)
            r=run_episode(env,policy,seed,chi=1.,train=False)
            opening=r.initial_inventory-r.clob_exec_qty
            contribution=(r.auction_exec_qty*(r.s_cl-r.residual_liquidation_price)
                          +configs[0].reward.lambda_inv*(opening**2-r.i_final**2)-r.cancellation_fees)
            rows.append(dict(policy=label,env_seed=seed,objective=r.risk_adjusted_pnl,
                             auction_value=contribution,opening_inventory=opening,
                             final_inventory=r.i_final,cancels=r.cancel_count))
            results.append(r)
        np.testing.assert_allclose(results[0].clob_exec_qty,results[1].clob_exec_qty,atol=1e-10)
        np.testing.assert_allclose(results[0].clob_cash,results[1].clob_cash,atol=1e-10)
    d=pd.DataFrame(rows)
    d.to_csv(args.output/'outcomes.csv',index=False)
    paired=d.pivot(index='env_seed',columns='policy',values='auction_value')
    delta=paired.diagnostic_auction_transfer-paired.original
    summary=dict(paths=len(seeds),original=paired.original.mean(),
                 transferred=paired.diagnostic_auction_transfer.mean(),
                 improvement=delta.mean(),improvement_se=delta.sem())
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'protocol.json').write_text(json.dumps(dict(
        purpose=__doc__,clob_run=str(args.clob_run),auction_run=str(args.auction_run),
        auction_checkpoint=str(checkpoints[1]),
        data_split='validation',path_seed=args.path_seed,seeds=seeds,
        checkpoint_sha256={str(checkpoint):hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                           for checkpoint in checkpoints},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
