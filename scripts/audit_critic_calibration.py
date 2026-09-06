"""Frozen deterministic critic versus realized conditioned on-policy returns."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lmm.config import load_config
from lmm.env.mdp import make_env
from lmm.experiments.train import make_agent,wrap_env_for_agent
from lmm.rl.conditioning import inventory_potential
from lmm.rl.loops import SEED_COMPONENTS,run_episode
from lmm.utils.seeding import seed_everything


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=64)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);out=[];economic=[]
    paths=np.random.default_rng(197527).integers(0,2**31-1,a.episodes)
    for root in a.run:
        cfg=load_config(root/'config.yaml');protocol=json.loads((root/'protocol.json').read_text())
        agent=make_agent(cfg,seed_everything(protocol['seed'],SEED_COMPONENTS));agent.load(root/'final.pt')
        env=make_env(cfg,symbol=protocol.get('symbol'),data_split='validation')
        wrapped=wrap_env_for_agent(env,cfg)
        for ep,seed in enumerate(paths):
            records=[]
            class Collector:
                def __getattr__(self,name):return getattr(agent,name)
                def act(self,obs,mask,phase,**kwargs):
                    action=agent.act(obs,mask,phase,**kwargs)
                    raw=getattr(env.features,phase+'_features')(env)
                    with torch.no_grad():
                        model=agent.models[phase]
                        q=model.critic(torch.as_tensor(obs[None],device=agent.device),
                                       torch.as_tensor(action[None],device=agent.device))[0].item()
                    records.append(dict(phase=phase,time=env.t,inventory=env.inventory,mid=env.s_mid,q=q,
                        phi=inventory_potential(raw,cfg) if cfg.reward.learning_potential else 0.))
                    return action
            def step(i,phase,t,reward,cum,info,environment):
                row=records[-1];row['reward']=reward;cv=0.
                if cfg.rl.market_return_control_variate and phase=='clob':
                    times=agent._feature_normalizer.inventory_reference_times
                    values=agent._feature_normalizer.inventory_reference_values
                    reference=np.interp(.5*(t+info['t_next']),times,values)
                    cv=-reference*(env.s_mid-row['mid'])
                row['cv']=cv
            res=run_episode(wrapped,Collector(),int(seed),chi=1.,train=False,on_step=step)
            future=0.
            for row in reversed(records):
                future+=row['reward']+row['cv'];row['mc']=future-row['phi'];row['q_error']=row['q']-row['mc']
                out.append(dict(run=root.name,episode=ep,env_seed=int(seed),**row))
            economic.append(dict(run=root.name,episode=ep,objective=res.risk_adjusted_pnl,
                training_return=res.training_return,inventory=res.i_final))
    d=pd.DataFrame(out);d.to_csv(a.output/'states.csv',index=False)
    e=pd.DataFrame(economic);e.to_csv(a.output/'episodes.csv',index=False)
    # Average within paths first; states are not independent replications.
    blocks=d.groupby(['run','phase','episode'])[['q','mc','q_error']].mean()
    summary=blocks.groupby(['run','phase']).mean();summary.to_csv(a.output/'summary.csv')
    (a.output/'protocol.json').write_text(json.dumps(dict(purpose=__doc__,data_split='validation',
        paths=paths.tolist(),runs=list(map(str,a.run)),checkpoint='final.pt',selection='none'),indent=2))
    print(summary.to_string());print(e.groupby('run').mean().to_string())


if __name__=='__main__':main()
