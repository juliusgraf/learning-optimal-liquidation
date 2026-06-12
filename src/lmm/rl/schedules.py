"""Exploration schedules (ruling D9; Phase 4).

Default: exponential epsilon decay 1.0 -> 0.01 with a warmup, parameters from
configs/algo/dqn.yaml (legacy schedule recorded in AUDIT C.1).
"""

from __future__ import annotations

__all__ = ["ExponentialEpsilonSchedule"]


class ExponentialEpsilonSchedule:
    """epsilon(e) = end + (start - end) * exp(-(e - warmup)/decay) for
    e >= warmup, epsilon = start during warmup; clipped to [end, start]."""

    def __init__(self, start: float, end: float, decay_episodes: float, warmup_episodes: int) -> None:
        self.start = start
        self.end = end
        self.decay_episodes = decay_episodes
        self.warmup_episodes = warmup_episodes

    def value(self, episode: int) -> float:
        raise NotImplementedError("Phase 4")
