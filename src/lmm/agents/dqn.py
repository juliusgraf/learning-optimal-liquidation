"""Textbook-standard DQN (Mnih et al. 2015 conventions; ruling D9).

Design (Phase 4 implements; docs/rl_design.md will carry the full spec):
- TWO phase networks Q_phi (CLOB, 8-dim features) and Q_psi (auction, 7-dim
  features) on time-augmented features — a documented DESIGN CHOICE (the
  phases have structurally different state/action spaces), not paper
  fidelity.
- Uniform replay, per-environment-step minibatch updates, target networks
  (hard update every C steps; soft-tau optional), epsilon-greedy with a
  configured schedule, Huber loss, Adam, gradient clipping, eval mode
  (epsilon = 0), checkpointing, full seeding (D10).
- Cross-phase junction: CLOB transitions with next_phase == "auction"
  bootstrap from the AUCTION target network; terminal handled by done=True.
- Admissibility masking (Adm(x)) in BOTH the argmax and the random draw.
- Bellman targets use chi = 0.99 (rl.chi).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lmm.agents.base import Agent, Transition
from lmm.config import ExperimentConfig
from lmm.utils.seeding import SeedBundle

__all__ = ["DQNAgent"]


class DQNAgent(Agent):
    """Two-phase DQN over the discrete action grids (Phase 4)."""

    def __init__(self, cfg: ExperimentConfig, seeds: SeedBundle) -> None:
        self.cfg = cfg
        self.seeds = seeds

    def act(self, obs: np.ndarray, *, eval_mode: bool = False) -> int:
        raise NotImplementedError("Phase 4")

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError("Phase 4")

    def update(self) -> dict[str, float]:
        raise NotImplementedError("Phase 4")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Phase 4")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("Phase 4")
