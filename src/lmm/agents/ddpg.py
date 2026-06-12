"""DDPG on the continuous-action relaxation (Phase 5; ruling D9).

Lives in the SEPARATE, clearly labeled relaxation: the discrete DQN setting
stays untouched; every mathematical change (action-space relaxation,
admissibility handling for continuous actions) is documented in
docs/continuous_action_extension.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lmm.agents.base import Agent, Transition
from lmm.config import ExperimentConfig
from lmm.utils.seeding import SeedBundle

__all__ = ["DDPGAgent"]


class DDPGAgent(Agent):
    """Deterministic policy gradient agent (Lillicrap et al. 2016 conventions)."""

    def __init__(self, cfg: ExperimentConfig, seeds: SeedBundle) -> None:
        self.cfg = cfg
        self.seeds = seeds

    def act(self, obs: np.ndarray, *, eval_mode: bool = False) -> np.ndarray:
        raise NotImplementedError("Phase 5")

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError("Phase 5")

    def update(self) -> dict[str, float]:
        raise NotImplementedError("Phase 5")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Phase 5")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("Phase 5")
