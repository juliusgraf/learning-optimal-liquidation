"""Two synthetic CPU episodes. No training, file output, data or checkpoints."""
from pathlib import Path
import numpy as np
from lmm.config import load_config
from lmm.env.mdp import make_env

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = load_config(ROOT / 'configs/base.yaml', ROOT / 'configs/synthetic_rough_heston.yaml')
    if cfg.midprice.model != 'rough_heston':
        raise ValueError('smoke requires the synthetic generator')
    def episode(seed):
        env = make_env(cfg)
        obs, _ = env.reset(seed=seed)
        rewards = []
        done = False
        while not done:
            if not np.isfinite(obs).all() or len(rewards) > 1000:
                raise ValueError('invalid observation or terminal boundary')
            obs, reward, terminated, truncated, _ = env.step(0)
            if truncated:
                raise ValueError('unexpected truncation')
            rewards.append(reward); done = terminated
        if not np.isfinite(rewards).all() or env.t != cfg.grid.tau_cl:
            raise ValueError('nonfinite accounting or incorrect terminal time')
        return rewards, float(env.inventory)
    if episode(8675309) != episode(8675309):
        raise ValueError('same-seed smoke replay differs')
    print('PASS: two finite, identical synthetic CPU episodes reach terminal settlement; smoke only, no paper result.')


if __name__ == '__main__':
    main()
