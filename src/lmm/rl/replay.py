"""Uniform replay buffer with seeded sampling (ruling D9; Phase 4).

Sampling uses the buffer's PRIVATE ``np.random.Generator`` (ruling D10);
stored actions are the EXECUTED actions (AUDIT N12). One buffer instance per
phase network; the CLOB buffer's junction rows (next state in the auction,
CLAUDE.md cross-phase rule) store the auction next-obs and the auction-grid
admissibility mask zero-padded to this buffer's widths — the agent splits the
batch on ``junction`` and un-pads before the target-network pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ReplayBatch", "ReplayBuffer"]


@dataclass(frozen=True)
class ReplayBatch:
    """One uniform minibatch as struct-of-arrays (np.float32/int64/bool)."""

    obs: np.ndarray  # (B, obs_dim)
    action: np.ndarray  # (B,) int64 grid indices
    reward: np.ndarray  # (B,)
    next_obs: np.ndarray  # (B, next_obs_dim), zero-padded on junction rows
    done: np.ndarray  # (B,) bool; True => zero bootstrap
    junction: np.ndarray  # (B,) bool; True => bootstrap from the OTHER phase
    next_mask: np.ndarray  # (B, mask_dim) bool admissibility of x' (Adm(x'))


class ReplayBuffer:
    """Fixed-capacity FIFO ring buffer with uniform with-replacement sampling.

    ``next_obs_dim``/``mask_dim`` are the MAXIMA over this phase's own next
    state and the junction next state; shorter rows are zero-/False-padded on
    ``add``. Done rows store an all-False mask (never read: zero bootstrap).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        next_obs_dim: int,
        mask_dim: int,
        rng: np.random.Generator,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.next_obs_dim = int(next_obs_dim)
        self.mask_dim = int(mask_dim)
        self.rng = rng
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros(capacity, dtype=np.int64)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, next_obs_dim), dtype=np.float32)
        self._done = np.zeros(capacity, dtype=bool)
        self._junction = np.zeros(capacity, dtype=bool)
        self._next_mask = np.zeros((capacity, mask_dim), dtype=bool)
        self._pos = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray | None,
        done: bool,
        junction: bool,
        next_mask: np.ndarray | None,
    ) -> None:
        """Append one transition, evicting FIFO at capacity.

        ``next_obs``/``next_mask`` may be None only when ``done`` (terminal:
        zero bootstrap; stored as zeros/all-False)."""
        if (next_obs is None or next_mask is None) and not done:
            raise ValueError("next_obs/next_mask may be None only on terminal transitions")
        i = self._pos
        self._obs[i] = obs
        self._action[i] = int(action)
        self._reward[i] = float(reward)
        self._next_obs[i] = 0.0
        self._next_mask[i] = False
        if next_obs is not None:
            self._next_obs[i, : len(next_obs)] = next_obs
        if next_mask is not None:
            self._next_mask[i, : len(next_mask)] = next_mask
        self._done[i] = bool(done)
        self._junction[i] = bool(junction)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Uniform minibatch WITH replacement (standard DQN), drawn from the
        buffer's private generator (ruling D10)."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self.rng.integers(0, self._size, size=batch_size)
        return ReplayBatch(
            obs=self._obs[idx].copy(),
            action=self._action[idx].copy(),
            reward=self._reward[idx].copy(),
            next_obs=self._next_obs[idx].copy(),
            done=self._done[idx].copy(),
            junction=self._junction[idx].copy(),
            next_mask=self._next_mask[idx].copy(),
        )

    def __len__(self) -> int:
        return self._size

    # -- resumable checkpoints (D10: bit-identical restarts) ------------------

    def state_dict(self) -> dict:
        """Full contents + cursor + sampling-RNG state (for --resume)."""
        return {
            "obs": self._obs.copy(),
            "action": self._action.copy(),
            "reward": self._reward.copy(),
            "next_obs": self._next_obs.copy(),
            "done": self._done.copy(),
            "junction": self._junction.copy(),
            "next_mask": self._next_mask.copy(),
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
        self._next_mask[:] = state["next_mask"]
        self._pos = int(state["pos"])
        self._size = int(state["size"])
        self.rng.bit_generator.state = state["rng_state"]
