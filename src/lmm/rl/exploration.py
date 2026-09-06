"""Common finite warm-up for coverage of persistent signed auction exposure."""
from __future__ import annotations

import numpy as np

from lmm.env.action_spaces import ClobAction, ClobActionGrid, AuctionActionGrid, continuous_action_specs


class PersistentWarmup:
    """Exploration data only: fixed random proposals, no benchmark labels.

    Constant proposals expose accumulation that independent cancel coins
    almost never reach. All learning updates remain the agent's own updates.
    """
    def __init__(self, cfg, seed, episode):
        self.cfg=cfg
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),64107]))
        mode=int(episode)%4
        clob_mode=(int(episode)//4)%4
        self.volume=int(rng.integers(cfg.actions.V_max+1))
        self.delta=int(rng.integers(cfg.actions.L_max+1))
        if clob_mode==0: self.volume,self.delta=cfg.actions.V_max,min(1,cfg.actions.L_max)
        if clob_mode==1: self.volume,self.delta=0,0
        self.K=int(rng.integers(cfg.actions.K_max+1))
        self.ell=int(rng.integers(cfg.actions.B_max+1))*(1 if mode==2 else -1)
        self.cancel=int(mode==3)
        if mode==0: self.K=0
        # Zero slope has only the canonical zero offset in the action grid.
        # Leaving a random ell here makes the distance projection prefer a
        # positive slope at that ell over the intended no-order action.
        if self.K==0: self.ell=0
        self.clob=ClobActionGrid(cfg.actions)
        self.auction=AuctionActionGrid(cfg.actions)

    def act(self, observation, mask, phase):
        ap=self.cfg.actions
        if phase=='clob':
            v=max(0,min(self.volume,int(float(observation[1]))))
            action=ClobAction(v,self.delta if v else 0)
            index=self.clob.actions.index(action)
            raw=np.array([2*v/max(1,ap.V_max)-1,2*action.delta/max(1,ap.L_max)-1],dtype=np.float32)
        else:
            candidates=np.flatnonzero(mask)
            def distance(i):
                a=self.auction.actions[i]
                return abs(a.K_a/ap.beta-self.K)+abs(a.ell-self.ell)+100*abs(a.cancel-self.cancel)
            index=min(candidates,key=distance)
            action=self.auction.actions[index]
            raw=np.array([2*(action.K_a/ap.beta)/max(1,ap.K_max)-1,
                          action.ell/max(1,ap.B_max),2*action.cancel-1],dtype=np.float32)
            raw=raw[:continuous_action_specs(self.cfg)['auction'].dim]
        if self.cfg.algo.name=='dqn':
            return int(index)
        return raw
