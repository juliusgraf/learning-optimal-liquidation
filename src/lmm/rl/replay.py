"""Uniform replay buffer with seeded sampling (ruling D9; Phase 4).

Sampling uses the buffer's PRIVATE ``np.random.Generator`` (ruling D10);
stored actions are the EXECUTED actions (AUDIT N12).
"""

from __future__ import annotations

import numpy as np

from lmm.agents.base import Transition

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """Fixed-capacity FIFO uniform replay (one instance per phase network)."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng

    def add(self, transition: Transition) -> None:
        raise NotImplementedError("Phase 4")

    def sample(self, batch_size: int) -> list[Transition]:
        """Uniform minibatch (with replacement matching standard DQN)."""
        raise NotImplementedError("Phase 4")

    def __len__(self) -> int:
        raise NotImplementedError("Phase 4")
