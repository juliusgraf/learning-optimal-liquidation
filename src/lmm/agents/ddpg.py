"""DDPG on the continuous-action relaxation (Phase 5; ruling D9).

Deterministic policy gradient (Lillicrap et al. 2016 conventions): a single
critic per phase, a deterministic actor, Gaussian (default) or OU exploration
noise, and Polyak target updates. Lives in the SEPARATE, clearly labeled
relaxation; the discrete DQN setting stays untouched. Full spec:
docs/continuous_action_extension.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from lmm.agents.continuous_base import ContinuousActorCriticAgent, ContinuousHyperparams
from lmm.config import ExperimentConfig, build_hyperparams
from lmm.utils.seeding import SeedBundle

__all__ = ["DDPGAgent", "DDPGHyperparams"]


@dataclass(frozen=True)
class DDPGHyperparams(ContinuousHyperparams):
    """DDPG-specific knobs (configs/algo/ddpg.yaml)."""

    exploration_noise: str  # "gaussian" | "ou"
    exploration_noise_std: float  # standard deviation in normalized [-1, 1] space


class DDPGAgent(ContinuousActorCriticAgent):
    """Deterministic policy gradient agent (single critic per phase)."""

    n_critics = 1
    uses_target_actor = True

    def __init__(self, cfg: ExperimentConfig, seeds: SeedBundle) -> None:
        if cfg.algo is None or cfg.algo.name != "ddpg":
            raise ValueError("DDPGAgent requires algo.name == 'ddpg' in the config")
        hp = build_hyperparams(DDPGHyperparams, cfg.algo.hyperparams)
        super().__init__(cfg, seeds, hp)
        self._exploration_scale = hp.exploration_noise_std
