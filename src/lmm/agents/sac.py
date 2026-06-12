"""SAC on the continuous-action relaxation (Phase 5; ruling D9).

See docs/continuous_action_extension.md for the relaxation spec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lmm.agents.base import Agent, Transition
from lmm.config import ExperimentConfig
from lmm.utils.seeding import SeedBundle

__all__ = ["SACAgent"]


class SACAgent(Agent):
    """Soft actor-critic (Haarnoja et al. 2018 conventions, auto-alpha)."""

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
