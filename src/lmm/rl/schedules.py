"""Exploration schedules for the revised DQN specification."""

from __future__ import annotations

__all__ = ["LinearEpsilonSchedule", "ExponentialEpsilonSchedule"]


class LinearEpsilonSchedule:
    """Warm-up followed by an exact linear decay.

    The endpoints are entirely configuration-driven.  With the active DQN
    values, episodes 0--49 use epsilon 1, episode 50 starts the linear decay,
    and episode 650 reaches 0.01 exactly.  Values thereafter remain at the
    endpoint.  The active DQN applies this same schedule in both phases;
    phase multipliers and delayed unlocks are explicit ablations only.
    """

    def __init__(self, start: float, end: float, decay_episodes: float, warmup_episodes: int) -> None:
        if not 0.0 < end <= start:
            raise ValueError(f"need 0 < end <= start, got start={start}, end={end}")
        if decay_episodes <= 0.0:
            raise ValueError(f"decay_episodes must be > 0, got {decay_episodes}")
        self.start = float(start)
        self.end = float(end)
        self.decay_episodes = float(decay_episodes)
        self.warmup_episodes = int(warmup_episodes)

    def value(self, episode: int) -> float:
        if episode < self.warmup_episodes:
            return self.start
        progress = (float(episode) - self.warmup_episodes) / self.decay_episodes
        eps = self.start + progress * (self.end - self.start)
        return min(self.start, max(self.end, eps))


# Import compatibility for downstream code written against the former class
# name.  The behavior intentionally follows the revised linear specification.
ExponentialEpsilonSchedule = LinearEpsilonSchedule
