"""Determinism tests (ruling D10): same config + seed => bit-identical
episode trajectories across two FRESH PROCESSES (states, rewards and the
H_cl trajectory are hashed inside each subprocess and compared here).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EPISODE_SCRIPT = """
import hashlib
import math
import sys

import numpy as np

from lmm.config import load_config
from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.env.mdp import make_env

setting, seed = sys.argv[1], int(sys.argv[2])
cfg = load_config("configs/base.yaml", f"configs/{setting}.yaml")
env = make_env(cfg, repo_root=".")

obs, _ = env.reset(seed=seed)
h = hashlib.sha256()
h.update(obs.tobytes())
while True:
    if env.phase == "clob":
        volume = float(min(5, max(0, math.floor(env.inventory))))
        a = ClobAction(volume, 2 if volume else 0)
    else:
        a = AuctionAction(2.0 * env.cfg.actions.beta, 2, int(env.cancel_admissible))
    obs, r, term, _, info = env.step(a)
    h.update(obs.tobytes())
    h.update(np.float64(r).tobytes())
    h.update(np.float64(info["H_used"]).tobytes())
    if term:
        break
h.update(np.float64(info["S_cl"]).tobytes())
h.update(np.float64(info["I_final"]).tobytes())
print(h.hexdigest())
"""


def episode_hash(setting: str, seed: int) -> str:
    out = subprocess.run(
        [sys.executable, "-c", EPISODE_SCRIPT, setting, str(seed)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize("setting", ["synthetic_rough_heston", pytest.param("historical_sp500_midquotes", marks=pytest.mark.market_data)])
def test_same_seed_identical_across_fresh_processes(setting):
    assert episode_hash(setting, 42) == episode_hash(setting, 42)


def test_different_seeds_differ():
    assert episode_hash("synthetic_rough_heston", 1) != episode_hash(
        "synthetic_rough_heston", 2
    )


def test_reset_seed_reproducible_in_process(synthetic_cfg):
    """reset(seed) alone determines the episode (no global RNG involved)."""
    import math

    from helpers import new_env
    from lmm.env.action_spaces import ClobAction, AuctionAction

    def run(seed):
        env = new_env(synthetic_cfg)
        env.reset(seed=seed)
        rewards = []
        while True:
            if env.phase == "clob":
                volume = float(min(5, max(0, math.floor(env.inventory))))
                a = ClobAction(volume, 2 if volume else 0)
            else:
                a = AuctionAction(2.0 * env.cfg.actions.beta, 2, int(env.cancel_admissible))
            _, r, term, _, _ = env.step(a)
            rewards.append(r)
            if term:
                return rewards

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_env_never_touches_global_rng(synthetic_cfg):
    """Ruling D10: NEVER global np.random.* or random.* — episode draws come
    from the env's private generator only."""
    import random

    import numpy as np

    from helpers import new_env
    from lmm.env.action_spaces import ClobAction

    np_state = np.random.get_state()[1].copy()
    py_state = random.getstate()

    env = new_env(synthetic_cfg)
    env.reset(seed=7)
    for _ in range(5):
        env.step(ClobAction(1.0, 2))

    np.testing.assert_array_equal(np.random.get_state()[1], np_state)
    assert random.getstate() == py_state
