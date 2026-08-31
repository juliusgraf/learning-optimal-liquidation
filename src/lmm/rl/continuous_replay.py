"""Uniform replay buffer for the continuous-action relaxation (Phase 5; D9/D10).

Mirrors ``rl/replay.py::ReplayBuffer`` but stores the action as a FLOAT vector
(the raw normalized proposal in ``[-1, 1]``) and a
per-row ``next_cancel_admissible`` bool. The latter records the auction
admissibility state for diagnostics/checkpoint compatibility; all actor-critic
targets now remain in proposal space and leave the environment map to apply the
cancel mask. The discrete ``ReplayBuffer`` is left untouched.

One buffer per phase: the CLOB buffer's junction rows (next state at the auction
open) hold the (shorter) auction next-obs zero-padded to this buffer's
``next_obs_dim``; the agent splits the batch on ``junction`` and un-pads before
the auction target pass. Sampling uses the buffer's PRIVATE
``np.random.Generator`` (ruling D10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ContinuousReplayBatch", "ContinuousReplayBuffer"]


@dataclass(frozen=True)
class ContinuousReplayBatch:
    """One uniform minibatch as struct-of-arrays."""

    obs: np.ndarray  # (B, obs_dim) float32
    action: np.ndarray  # (B, action_dim) float32 raw normalized proposal [-1, 1]
    reward: np.ndarray  # (B,) float32
    next_obs: np.ndarray  # (B, next_obs_dim) float32, zero-padded on junction rows
    done: np.ndarray  # (B,) bool; True => add terminal_value, never a network value
    junction: np.ndarray  # (B,) bool; True => bootstrap from the auction networks
    next_cancel_admissible: np.ndarray  # (B,) bool; C(x') > 0 (auction next states)
    # (B,) known absorbing-state value g = r_tau_cl (reward_scale'd), nonzero
    # only on done rows; target adds it exactly once as y = r_step + g.
    terminal_value: np.ndarray | None = None
    # Deprecated compatibility field. Learners ignore it and replay emits ones.
    discount: np.ndarray | None = None


class ContinuousReplayBuffer:
    """Fixed-capacity FIFO ring buffer with uniform with-replacement sampling.

    ``next_obs_dim`` is the MAXIMUM over this phase's own next state and the
    junction next state; shorter rows are zero-padded on ``add``. Done rows
    store zeros / False (never read: zero bootstrap).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        next_obs_dim: int,
        rng: np.random.Generator,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.next_obs_dim = int(next_obs_dim)
        self.rng = rng
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, next_obs_dim), dtype=np.float32)
        self._done = np.zeros(capacity, dtype=bool)
        self._junction = np.zeros(capacity, dtype=bool)
        self._next_cancel_adm = np.zeros(capacity, dtype=bool)
        self._terminal_value = np.zeros(capacity, dtype=np.float32)
        self._discount = np.ones(capacity, dtype=np.float32)  # checkpoint compatibility only
        self._pos = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray | None,
        done: bool,
        junction: bool,
        next_cancel_admissible: bool,
        terminal_value: float = 0.0,
        discount: float = 1.0,
    ) -> None:
        """Append one transition, evicting FIFO at capacity.

        ``next_obs`` may be None only when ``done`` (terminal: the bootstrap is
        ``terminal_value`` = g = r_tau_cl, not the next state; item 1)."""
        if next_obs is None and not done:
            raise ValueError("next_obs may be None only on terminal transitions")
        i = self._pos
        self._obs[i] = obs
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action shape must be {(self.action_dim,)}, got {action.shape}")
        if not np.all(np.isfinite(action)) or np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("continuous replay actions must be finite raw proposals in [-1, 1]")
        self._action[i] = action
        self._reward[i] = float(reward)
        self._terminal_value[i] = float(terminal_value)
        # Retain the keyword for compatibility but remove all transition-
        # specific discount behavior from stored data.
        self._discount[i] = 1.0
        self._next_obs[i] = 0.0
        if next_obs is not None:
            self._next_obs[i, : len(next_obs)] = next_obs
        self._done[i] = bool(done)
        self._junction[i] = bool(junction)
        self._next_cancel_adm[i] = bool(next_cancel_admissible)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ContinuousReplayBatch:
        """Uniform minibatch WITH replacement from the buffer's private generator."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self.rng.integers(0, self._size, size=batch_size)
        return ContinuousReplayBatch(
            obs=self._obs[idx].copy(),
            action=self._action[idx].copy(),
            reward=self._reward[idx].copy(),
            next_obs=self._next_obs[idx].copy(),
            done=self._done[idx].copy(),
            junction=self._junction[idx].copy(),
            next_cancel_admissible=self._next_cancel_adm[idx].copy(),
            terminal_value=self._terminal_value[idx].copy(),
            discount=self._discount[idx].copy(),
        )

    def __len__(self) -> int:
        return self._size

    # -- resumable checkpoints (D10: bit-identical restarts) ------------------

    def state_dict(self) -> dict:
        return {
            "obs": self._obs.copy(),
            "action": self._action.copy(),
            "reward": self._reward.copy(),
            "next_obs": self._next_obs.copy(),
            "done": self._done.copy(),
            "junction": self._junction.copy(),
            "next_cancel_adm": self._next_cancel_adm.copy(),
            "terminal_value": self._terminal_value.copy(),
            "discount": np.ones_like(self._discount),
            "pos": self._pos,
            "size": self._size,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["obs"].shape != self._obs.shape:
            raise ValueError(
                f"checkpoint shape {state['obs'].shape} != buffer shape {self._obs.shape}"
            )
        self._obs[:] = state["obs"]
        self._action[:] = state["action"]
        self._reward[:] = state["reward"]
        self._next_obs[:] = state["next_obs"]
        self._done[:] = state["done"]
        self._junction[:] = state["junction"]
        self._next_cancel_adm[:] = state["next_cancel_adm"]
        if "terminal_value" in state:  # robust to pre-item-1 checkpoints
            self._terminal_value[:] = state["terminal_value"]
        # Old checkpoint factors are intentionally discarded: Bellman factor 1.
        self._discount.fill(1.0)
        self._pos = int(state["pos"])
        self._size = int(state["size"])
        self.rng.bit_generator.state = state["rng_state"]
