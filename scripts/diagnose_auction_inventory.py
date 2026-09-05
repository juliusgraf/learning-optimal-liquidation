"""Evaluate learned auction controllers after feasible CLOB inventory probes.

These are conditional policy diagnostics, not headline policy performance and
not extra benchmarks. No learner updates, forced auction trades or fill clamps.
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
from lmm.env.action_spaces import ClobAction, ClobActionGrid
from lmm.experiments.train import make_agent, wrap_env_for_agent
from lmm.rl.loops import run_episode, SEED_COMPONENTS
from lmm.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = np.random.default_rng(370192).integers(0, 2**31-1, args.episodes).tolist()
    rows, manifest = [], []
    for root in args.run:
        cfg = load_config(root/'config.yaml')
        protocol = json.loads((root/'protocol.json').read_text())
        agent = make_agent(cfg, seed_everything(792, SEED_COMPONENTS))
        agent.load(root/'best.pt')
        ecfg = economic_evaluation_config(cfg)
        env = wrap_env_for_agent(make_env(ecfg, symbol=protocol.get('symbol'), data_split='validation'), cfg)
        grid = ClobActionGrid(cfg.actions)
        class Probe:
            def __init__(self, reserve, enabled): self.reserve, self.enabled = reserve, enabled
            def __getattr__(self, name): return getattr(agent, name)
            def act(self, obs, mask, phase, **kwargs):
                if phase == 'clob':
                    volume = max(0, min(cfg.actions.V_max, int(env.inventory-self.reserve)))
                    quote = 1 if volume else 0
                    if cfg.algo.name == 'dqn':
                        return grid.actions.index(ClobAction(volume, quote))
                    return np.array([2*volume/cfg.actions.V_max-1,
                                     2*quote/cfg.actions.L_max-1], dtype=np.float32)
                if self.enabled: return agent.act(obs, mask, phase, **kwargs)
                if cfg.algo.name == 'dqn': return 0
                dim = agent.models['auction'].action_space.shape[0]
                return np.array([-1., 0., -1.], dtype=np.float32)[:dim]
        for reserve in [0, 10, 20]:
            for enabled in [False, True]:
                for seed in seeds:
                    r = run_episode(env, Probe(reserve, enabled), seed, chi=1., train=False)
                    rows.append(dict(run=root.name, algorithm=cfg.algo.name, setting=cfg.experiment.setting,
                                     reserve=reserve, enabled=enabled, env_seed=seed, pnl=r.pnl,
                                     objective=r.risk_adjusted_pnl, opening_inventory=r.initial_inventory-r.clob_exec_qty,
                                     final_inventory=r.i_final, auction_quantity=r.auction_exec_qty,
                                     cancellation_fees=r.cancellation_fees, cancels=r.cancel_count))
        manifest.append(dict(run=str(root), source_protocol=protocol,
                             checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest()))
        print(root.name, 'complete', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'records.csv', index=False)
    summary = frame.groupby(['run', 'reserve', 'enabled']).mean(numeric_only=True)
    summary.to_csv(args.output/'summary.csv')
    (args.output/'protocol.json').write_text(json.dumps(dict(purpose=__doc__, seeds=seeds, runs=manifest), indent=2))
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
