"""TD3 on the continuous-action relaxation (Phase 5; ruling D9).

Twin-delayed DDPG (Fujimoto et al. 2018 conventions): twin critics with a
clipped-double-Q (min) target, target-policy smoothing (clipped Gaussian on the
target action), and delayed actor + target updates. Full spec:
docs/continuous_action_extension.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from lmm.agents.continuous_base import ContinuousActorCriticAgent, ContinuousHyperparams
from lmm.config import ExperimentConfig, build_hyperparams
from lmm.utils.seeding import SeedBundle

__all__ = ["TD3Agent", "TD3Hyperparams"]


@dataclass(frozen=True)
class TD3Hyperparams(ContinuousHyperparams):
    """TD3-specific knobs (configs/algo/td3.yaml)."""

    exploration_noise: str  # "gaussian" | "ou"
    exploration_noise_std: float
    target_noise_std: float  # target-policy smoothing sigma in normalized space
    target_noise_clip: float  # smoothing clip in normalized space
    policy_delay: int  # delayed actor / target-sync cadence
    min_buffer_clob: Optional[int] = None
    min_buffer_auction: Optional[int] = None
    safe_auction_initialization: bool = False


class TD3Agent(ContinuousActorCriticAgent):
    """Twin-delayed DDPG: twin critics, min-target, smoothing, delayed updates."""

    n_critics = 2
    uses_target_actor = True

    def __init__(self, cfg: ExperimentConfig, seeds: SeedBundle) -> None:
        if cfg.algo is None or cfg.algo.name != "td3":
            raise ValueError("TD3Agent requires algo.name == 'td3' in the config")
        hp = build_hyperparams(TD3Hyperparams, cfg.algo.hyperparams)
        super().__init__(cfg, seeds, hp)
        self._exploration_scale = hp.exploration_noise_std

    def _should_update_actor(self, phase: str) -> bool:
        """Delayed policy + target updates (Fujimoto et al. 2018)."""
        return self._update_count[phase] % self.hp.policy_delay == 0

    def _target_next_action(self, phase: str, next_obs: torch.Tensor, cadm: np.ndarray):
        """Apply clipped-Gaussian smoothing directly in normalized space."""
        low, high = self._low[phase], self._high[phase]
        a = self.actor_target[phase](next_obs)
        noise = torch.randn_like(a) * self.hp.target_noise_std
        noise = torch.clamp(noise, -self.hp.target_noise_clip, self.hp.target_noise_clip)
        a = torch.clamp(a + noise, low, high)
        return a, None
