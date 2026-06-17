"""Shared helpers for the Phase 3 market/env tests (imported as ``helpers``).

CRN note (ruling D10): the env's random draws are policy-independent, so two
envs reset with the same seed and driven with different actions share the
exogenous event stream EXCEPT for the auction's conditional index draws
(MM-cancel and taker-cancel fire only when the respective ledger is
nonempty). Tests that INJECT exogenous orders into one env of a pair use
``load_synthetic_cfg("auction_flow.p2=0.0", "auction_flow.p4=0.0")``, which
removes those conditional draws and makes the streams bit-identical by
construction.
"""

from __future__ import annotations

from pathlib import Path

from lmm.config import load_config
from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.env.mdp import make_env

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"

NOOP_CLOB = ClobAction(0.0, 0)
NOOP_AUCTION = AuctionAction(0.0, 0, 0)


def load_synthetic_cfg(*overrides: str):
    return load_config(
        CONFIGS / "base.yaml",
        CONFIGS / "synthetic_rough_heston.yaml",
        overrides=list(overrides),
    )


def load_dqn_cfg(*overrides: str):
    """Synthetic config + the DQN algo overlay (Phase 4 agent tests)."""
    return load_config(
        CONFIGS / "base.yaml",
        CONFIGS / "synthetic_rough_heston.yaml",
        CONFIGS / "algo" / "dqn.yaml",
        overrides=list(overrides),
    )


def load_algo_cfg(algo: str, *overrides: str):
    """Synthetic config + a continuous algo overlay (Phase 5 agent tests)."""
    return load_config(
        CONFIGS / "base.yaml",
        CONFIGS / "synthetic_rough_heston.yaml",
        CONFIGS / "algo" / f"{algo}.yaml",
        overrides=list(overrides),
    )


def load_ddpg_cfg(*overrides: str):
    return load_algo_cfg("ddpg", *overrides)


def load_td3_cfg(*overrides: str):
    return load_algo_cfg("td3", *overrides)


def load_sac_cfg(*overrides: str):
    return load_algo_cfg("sac", *overrides)


def load_historical_cfg(*overrides: str):
    return load_config(
        CONFIGS / "base.yaml",
        CONFIGS / "historical_sp500.yaml",
        overrides=list(overrides),
    )


def new_env(cfg, symbol=None):
    return make_env(cfg, symbol=symbol, repo_root=REPO_ROOT)


def drive_to_auction(env, seed: int, clob_action: ClobAction = NOOP_CLOB):
    """Reset and play no-op (or fixed) CLOB actions until the auction opens.

    Returns the list of (reward, info) of the CLOB steps; afterwards
    ``env.t == tau_op`` and the agent is about to take the first auction
    decision.
    """
    env.reset(seed=seed)
    out = []
    while env.phase == "clob":
        _, r, _, _, info = env.step(clob_action)
        out.append((r, info))
    return out


def step_pair(env_a, env_b, action_a, action_b=None):
    """Step a CRN pair; returns ((r_a, info_a), (r_b, info_b))."""
    if action_b is None:
        action_b = action_a
    _, r_a, _, _, info_a = env_a.step(action_a)
    _, r_b, _, _, info_b = env_b.step(action_b)
    return (r_a, info_a), (r_b, info_b)
