"""Exploration schedules (ruling D9; Phase 4).

Default: exponential epsilon decay 1.0 -> 0.01 with a warmup, parameters from
configs/algo/dqn.yaml (legacy schedule recorded in AUDIT C.1).
"""

from __future__ import annotations

import math

__all__ = ["ExponentialEpsilonSchedule"]


class ExponentialEpsilonSchedule:
    """epsilon(e) = start * exp(-rate * (e - warmup)) for e >= warmup with
    rate = -ln(end / start) / decay_episodes, epsilon = start during warmup;
    clipped to [end, start] (epsilon reaches ``end`` exactly at
    e = warmup + decay_episodes and stays there)."""

    def __init__(self, start: float, end: float, decay_episodes: float, warmup_episodes: int) -> None:
        if not 0.0 < end <= start:
            raise ValueError(f"need 0 < end <= start, got start={start}, end={end}")
        if decay_episodes <= 0.0:
            raise ValueError(f"decay_episodes must be > 0, got {decay_episodes}")
        self.start = float(start)
        self.end = float(end)
        self.decay_episodes = float(decay_episodes)
        self.warmup_episodes = int(warmup_episodes)
        self._rate = -math.log(self.end / self.start) / self.decay_episodes

    def value(self, episode: int) -> float:
        if episode < self.warmup_episodes:
            return self.start
        eps = self.start * math.exp(-self._rate * (episode - self.warmup_episodes))
        return min(self.start, max(self.end, eps))
