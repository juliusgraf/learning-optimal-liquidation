"""Compare terminal DQN advantages with exact shaped/economic action outcomes.

No training or final-test paths. The terminal counterfactual uses an independent
copy of the same pre-action environment, including the live-order ledger.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lmm.config import load_config
from lmm.env.mdp import make_env
from lmm.experiments.train import make_agent
from lmm.rl.loops import SEED_COMPONENTS, run_episode
from lmm.utils.seeding import seed_everything


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=64)
    p.add_argument('--path-seed', type=int, help='Fresh validation flow paths; no checkpoint reselection.')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.run/'config.yaml')
    learner = make_agent(cfg, seed_everything(941, SEED_COMPONENTS))
    learner.load(args.run/'best.pt')
    protocol = json.loads((args.run/'protocol.json').read_text())
    env = make_env(cfg, symbol=protocol.get('symbol'), data_split='validation')
    rows, terminal, outcomes = [], [], []
    episode_seed = None

    class Inspect:
        def __getattr__(self, name): return getattr(learner, name)

        def act(self, obs, mask, phase, **kwargs):
            chosen = learner.act(obs, mask, phase, **kwargs)
            if phase != 'auction': return chosen
            with torch.no_grad():
                q = learner.q[phase](torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
            action = learner.auction_grid.actions[chosen]
            rows.append(dict(env_seed=episode_seed, time=env.t, action=chosen,
                             slope=action.K_a, offset=action.ell, cancel=action.cancel,
                             q_advantage=q[chosen]-q[0], q_noop=q[0]))
            if env.t == cfg.grid.tau_cl-1:
                trials = {}
                for a in sorted({0, chosen, *([1] if mask[1] else [])}):
                    clone = copy.deepcopy(env)
                    _, reward, done, _, info = clone.step(a)
                    assert done
                    economic = (info['auction_economic_cash']+info['residual_mark']
                                -info['terminal_penalty']-info['cancellation_fee'])
                    trials[a] = reward, economic
                for a, (reward, economic) in trials.items():
                    terminal.append(dict(env_seed=episode_seed, action=a, chosen=a==chosen,
                        predicted_advantage=q[a]-q[0], shaped_advantage=reward-trials[0][0],
                        economic_advantage=economic-trials[0][1]))
            return chosen

    agent = Inspect()
    seeds = (protocol['confirmation_seeds'][:args.episodes] if args.path_seed is None else
             np.random.default_rng(args.path_seed).integers(0,2**31-1,args.episodes).tolist())
    for episode_seed in seeds:
        result = run_episode(env, agent, episode_seed, chi=1., train=False)
        opened = result.initial_inventory-result.clob_exec_qty
        price_edge = result.auction_exec_qty*(result.s_cl-result.residual_liquidation_price)
        relief = cfg.reward.lambda_inv*(opened**2-result.i_final**2)
        outcomes.append(dict(env_seed=episode_seed, pnl=result.pnl, objective=result.risk_adjusted_pnl,
                             opening_inventory=opened, final_inventory=result.i_final,
                             price_edge=price_edge, risk_relief=relief, fees=result.cancellation_fees,
                             auction_value=price_edge+relief-result.cancellation_fees))
    pd.DataFrame(rows).to_csv(args.output/'actions.csv', index=False)
    pd.DataFrame(terminal).to_csv(args.output/'terminal_advantages.csv', index=False)
    pd.DataFrame(outcomes).to_csv(args.output/'outcomes.csv', index=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(run=str(args.run), seeds=seeds,
                   data_split='validation', path_seed=args.path_seed, purpose=__doc__), indent=2))
    print(pd.DataFrame(outcomes).mean().to_string())
    t = pd.DataFrame(terminal)
    print(t[t.chosen][['predicted_advantage','shaped_advantage','economic_advantage']].describe().to_string())


if __name__ == '__main__': main()
