"""Agent interface shared by RL agents and benchmarks (ruling D9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

__all__ = ["Transition", "Agent"]

Action = Union[int, np.ndarray]  # discrete index (DQN) or continuous vector


@dataclass(frozen=True)
class Transition:
    """One environment transition as seen by ``Agent.observe``.

    ``next_phase`` marks the cross-phase junction (CLAUDE.md, D9): a CLOB
    transition whose next state is in the auction bootstraps from the AUCTION
    target network. On the terminal transition (``done=True``) the env returns
    the combined reward r_step + r_tau_cl and exposes r_tau_cl in
    ``info["terminal_reward"]``; the RL agents un-fold it and bootstrap the
    terminal clearing reward as the known absorbing-state value, y = r_step +
    chi*r_tau_cl (item 1) — it is NOT folded into the stored step reward.

    ``action`` is the EXECUTED action (admissibility is enforced by masking
    at selection, never by post-hoc projection — fixes AUDIT N12).

    ``next_mask`` is Adm(x') over the NEXT phase's action grid (None iff
    ``done``): the Bellman masked max needs it at update time because the
    auction cancel-admissibility C(x') is not recoverable from the pruned
    feature vector. ``info`` carries per-step env diagnostics (E_t, S_bullet,
    ...) for non-learning consumers (benchmark execution-price tracking); it
    is NEVER stored in replay.
    """

    obs: np.ndarray
    action: Action
    reward: float
    next_obs: np.ndarray
    done: bool
    phase: str  # "clob" | "auction"
    next_phase: str  # phase of next_obs ("auction" for the junction)
    next_mask: np.ndarray | None = None  # Adm(x') on the next grid; None iff done
    info: dict[str, Any] | None = field(default=None, compare=False)


class Agent(ABC):
    """Minimal agent interface: act / observe / update / save / load,
    plus no-op lifecycle hooks (bind / start_episode / set_train)."""

    @abstractmethod
    def act(self, obs: np.ndarray, mask: np.ndarray, phase: str, *, eval_mode: bool = False) -> Action:
        """Select an action for ``obs`` in the given phase.

        ``mask`` is the env's admissibility mask Adm(x) over the phase grid.
        ``eval_mode=True`` disables exploration (epsilon = 0 / no noise).
        Implementations must restrict BOTH the greedy argmax and any random
        draw to the admissible set (masking, never projection — AUDIT N12).
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

    # -- lifecycle hooks (no-op by default) -----------------------------------

    def bind(self, env) -> None:
        """Give the agent a handle on its env (EVALUATION-time state access
        for the benchmarks: inventory, t, phase). RL agents ignore it."""

    def start_episode(self, episode: int) -> None:
        """Called before each episode (epsilon schedule, per-episode state)."""

    def set_train(self, training: bool) -> None:
        """Toggle train/eval mode (network .train()/.eval(), exploration)."""
