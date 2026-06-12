"""Agent interface shared by RL agents and benchmarks (ruling D9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

__all__ = ["Transition", "Agent"]

Action = Union[int, np.ndarray]  # discrete index (DQN) or continuous vector


@dataclass(frozen=True)
class Transition:
    """One environment transition as stored in replay.

    ``next_phase`` marks the cross-phase junction (CLAUDE.md, D9): a CLOB
    transition whose next state is in the auction bootstraps from the AUCTION
    target network. The terminal tau_cl reward is folded into the final
    auction transition with ``done=True`` and zero bootstrap.

    ``action`` is the EXECUTED action (admissibility is enforced by masking
    at selection, never by post-hoc projection — fixes AUDIT N12).
    """

    obs: np.ndarray
    action: Action
    reward: float
    next_obs: np.ndarray
    done: bool
    phase: str  # "clob" | "auction"
    next_phase: str  # phase of next_obs ("auction" for the junction)


class Agent(ABC):
    """Minimal agent interface: act / observe / update / save / load."""

    @abstractmethod
    def act(self, obs: np.ndarray, *, eval_mode: bool = False) -> Action:
        """Select an action for ``obs``.

        ``eval_mode=True`` disables exploration (epsilon = 0 / no noise).
        Implementations must restrict BOTH the greedy argmax and any random
        draw to the admissible set Adm(x) via the env's action mask.
        """

    @abstractmethod
    def observe(self, transition: Transition) -> None:
        """Record a transition (e.g. into replay). Benchmarks may no-op."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """One learning update; returns a dict of losses/diagnostics
        (empty dict if no update was performed)."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Checkpoint parameters and optimizer/schedule state."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore a checkpoint written by :meth:`save`."""
