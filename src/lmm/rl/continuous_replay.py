"""Uniform replay buffer for the continuous-action relaxation (Phase 5; D9/D10).

Mirrors ``rl/replay.py::ReplayBuffer`` but stores the action as a FLOAT vector
(the committed continuous action of docs/continuous_action_extension.md §3) and
a per-row ``next_cancel_admissible`` bool (the auction cancel-admissibility
C(x') > 0 of the next state — the analogue of the DQN ``next_mask``, used by the
bootstrap target to clamp the target action's cancel logit). The discrete
``ReplayBuffer`` is left untouched.

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
    action: np.ndarray  # (B, action_dim) float32 (committed continuous action)
    reward: np.ndarray  # (B,) float32
    next_obs: np.ndarray  # (B, next_obs_dim) float32, zero-padded on junction rows
    done: np.ndarray  # (B,) bool; True => bootstrap from terminal_value (item 1)
    junction: np.ndarray  # (B,) bool; True => bootstrap from the auction networks
    next_cancel_admissible: np.ndarray  # (B,) bool; C(x') > 0 (auction next states)
    # (B,) known absorbing-state value g = r_tau_cl (reward_scale'd, clipped),
    # nonzero only on done rows; target bootstraps y = r_step + chi * g (item 1).
    terminal_value: np.ndarray | None = None


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
    ) -> None:
        """Append one transition, evicting FIFO at capacity.

        ``next_obs`` may be None only when ``done`` (terminal: the bootstrap is
        ``terminal_value`` = g = r_tau_cl, not the next state; item 1)."""
        if next_obs is None and not done:
            raise ValueError("next_obs may be None only on terminal transitions")
        i = self._pos
        self._obs[i] = obs
        self._action[i] = action
        self._reward[i] = float(reward)
        self._terminal_value[i] = float(terminal_value)
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
        self._pos = int(state["pos"])
        self._size = int(state["size"])
        self.rng.bit_generator.state = state["rng_state"]
