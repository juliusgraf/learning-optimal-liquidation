"""Exploration schedules for the revised DQN specification."""

from __future__ import annotations


def learning_rate_factor(rl, episode: int) -> float:
    """Fixed, outcome-independent decay; zero half-life preserves old runs."""
    half_life = rl.learning_rate_half_life_episodes
    if half_life == 0:
        return 1.0
    return max(rl.learning_rate_min_fraction, 2.0 ** (-max(0, episode) / half_life))


def learning_conditioning_contract(rl):
    """Only enabled extensions enter the contract of compatible old runs."""
    extra = {}
    if rl.phase_normalization:
        extra['phase_normalization'] = True
    if rl.auction_inventory_asinh:
        extra['auction_inventory_asinh'] = True
    if rl.learning_credit_baseline:
        extra['learning_credit_baseline'] = True
    if rl.learning_clob_inventory_potential:
        extra['learning_clob_inventory_potential'] = True
    if rl.auction_exposure_features:
        extra['auction_exposure_features'] = True
    if rl.market_return_control_variate:
        extra['market_return_control_variate'] = True
        if rl.market_control_reference != 'linear':
            extra['market_control_reference'] = rl.market_control_reference
    if rl.structured_warmup_episodes:
        extra['structured_warmup_episodes'] = rl.structured_warmup_episodes
    if rl.learning_starts_after_warmup:
        extra['learning_starts_after_warmup'] = True
    if rl.learning_rate_half_life_episodes:
        extra['learning_rate_schedule'] = [rl.learning_rate_half_life_episodes, rl.learning_rate_min_fraction]
    return extra

__all__ = ["LinearEpsilonSchedule", "ExponentialEpsilonSchedule"]


class LinearEpsilonSchedule:
    """Warm-up followed by an exact linear decay.

    The endpoints are entirely configuration-driven. With the active DQN
    values, episodes 0--4 use epsilon 1, episode 5 starts the linear decay,
    and episode 95 reaches 0.05 exactly. Values thereafter remain at the
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
