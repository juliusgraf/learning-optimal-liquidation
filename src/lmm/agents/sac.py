"""SAC on the continuous-action relaxation (Phase 5; ruling D9).

Soft actor-critic (Haarnoja et al. 2018 conventions, automatic temperature):
a tanh-Gaussian actor with reparameterized sampling, twin critics with a
min-target plus the entropy bonus, and an auto-tuned temperature per phase
(target entropy -2 in the CLOB and -3 in the auction). No target actor
action). Full spec: docs/continuous_action_extension.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from lmm.agents.continuous_base import ContinuousActorCriticAgent, ContinuousHyperparams
from lmm.config import ExperimentConfig, build_hyperparams
from lmm.rl.networks import SquashedGaussianActor
from lmm.utils.seeding import SeedBundle

__all__ = ["SACAgent", "SACHyperparams"]


@dataclass(frozen=True)
class SACHyperparams(ContinuousHyperparams):
    """SAC-specific knobs (configs/algo/sac.yaml)."""

    alpha_lr: float  # temperature Adam lr
    auto_alpha: bool  # auto-tune the entropy temperature
    target_entropy: str  # "auto" => -dim(A) per phase


class SACAgent(ContinuousActorCriticAgent):
    """Soft actor-critic (twin critics, auto temperature, no target actor)."""

    n_critics = 2
    uses_target_actor = False

    def __init__(self, cfg: ExperimentConfig, seeds: SeedBundle) -> None:
        if cfg.algo is None or cfg.algo.name != "sac":
            raise ValueError("SACAgent requires algo.name == 'sac' in the config")
        hp = build_hyperparams(SACHyperparams, cfg.algo.hyperparams)
        super().__init__(cfg, seeds, hp)
        self._exploration_scale = 0.0  # intrinsic stochastic policy; no action noise

    # -- actor / temperature setup -------------------------------------------

    def _make_actor(self, phase: str, obs_dim: int, hidden, act_cls):
        dim = self._act_dim[phase]
        return SquashedGaussianActor(
            obs_dim,
            -np.ones(dim, dtype=np.float32),
            np.ones(dim, dtype=np.float32),
            hidden,
            act_cls,
        )

    def _post_setup(self, seeds: SeedBundle) -> None:
        if self.hp.target_entropy != "auto":
            raise ValueError(f"target_entropy must be 'auto', got {self.hp.target_entropy!r}")
        self.target_entropy = {"clob": -2.0, "auction": -3.0}
        self.log_alpha = {
            p: torch.zeros(1, requires_grad=True, device=self.device) for p in self.PHASES
        }
        self.alpha_optim = {
            p: torch.optim.Adam([self.log_alpha[p]], lr=self.hp.alpha_lr) for p in self.PHASES
        }

    def _alpha(self, phase: str) -> torch.Tensor:
        return self.log_alpha[phase].exp().squeeze()

    # -- acting / targets / actor update -------------------------------------

    def _select_action(self, phase: str, obs_t: torch.Tensor, *, greedy: bool):
        action, _ = self.actor[phase](obs_t, deterministic=greedy, with_logprob=False)
        return action.squeeze(0).cpu().numpy().astype("float32")

    def _target_next_action(self, phase: str, next_obs: torch.Tensor, cadm):
        """Sample from the CURRENT actor (SAC has no target actor); return the
        log-prob for the entropy term.  Like DDPG/TD3 after the comparability
        cleanup, this is a proposal action; Gamma_x applies the env mask."""
        return self.actor[phase](next_obs)

    def _min_online_q(self, phase: str, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        qs = [c(obs, action) for c in self.critics[phase]]
        q = qs[0]
        for qi in qs[1:]:
            q = torch.minimum(q, qi)
        return q

    def _update_actor(self, phase: str, obs: torch.Tensor) -> dict[str, float]:
        a, logp = self.actor[phase](obs)
        q = self._min_online_q(phase, obs, a)
        alpha = self._alpha(phase).detach()
        actor_loss = (alpha * logp - q).mean()
        self.actor_optim[phase].zero_grad()
        actor_loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.actor[phase].parameters(), self.hp.grad_clip_norm)
        self.actor_optim[phase].step()
        out = {"actor_loss": float(actor_loss.item()), "actor_grad_norm": float(gnorm.item())}
        if self.hp.auto_alpha:
            alpha_loss = -(self.log_alpha[phase] * (logp.detach() + self.target_entropy[phase])).mean()
            self.alpha_optim[phase].zero_grad()
            alpha_loss.backward()
            self.alpha_optim[phase].step()
            out["alpha"] = float(self._alpha(phase).item())
            out["alpha_loss"] = float(alpha_loss.item())
            out["entropy"] = float((-logp).mean().item())
        return out

    # -- checkpoint extras (temperature) -------------------------------------

    def _extra_state(self):
        return {
            "log_alpha": {p: self.log_alpha[p].detach().cpu().clone() for p in self.PHASES},
            "alpha_optim": {p: self.alpha_optim[p].state_dict() for p in self.PHASES},
        }

    def _load_extra_state(self, extra) -> None:
        if extra is None:
            return
        for p in self.PHASES:
            with torch.no_grad():
                self.log_alpha[p].copy_(extra["log_alpha"][p].to(self.device))
            self.alpha_optim[p].load_state_dict(extra["alpha_optim"][p])
