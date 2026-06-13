"""Network factory and actor/critic modules (ruling D9).

The DQN MLP factory (``mlp``, Phase 4) is recorded in docs/rl_design.md. The
continuous actor/critic modules (Phase 5; DDPG/TD3/SAC) are recorded in
docs/continuous_action_extension.md §4. Architecture is configured in
configs/algo/*.yaml.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "ACTIVATIONS",
    "mlp",
    "DeterministicActor",
    "Critic",
    "SquashedGaussianActor",
    "LOG_STD_MIN",
    "LOG_STD_MAX",
]

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}

# SAC log-std clamp (Haarnoja et al. 2018 conventions).
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Standard MLP: Linear-activation blocks, linear output head."""
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError(f"in_dim and out_dim must be positive, got {in_dim}, {out_dim}")
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, int(width)))
        layers.append(activation())
        prev = int(width)
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _trunk(in_dim: int, hidden: Sequence[int], activation: type[nn.Module]) -> tuple[nn.Sequential, int]:
    """A Linear-activation body (no output head); returns (module, out_width)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, int(width)))
        layers.append(activation())
        prev = int(width)
    return nn.Sequential(*layers), prev


class DeterministicActor(nn.Module):
    """Deterministic policy mu(x) in [low, high] (DDPG/TD3;
    docs/continuous_action_extension.md §4): MLP -> tanh -> affine scale."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        action_dim = int(np.asarray(action_low).shape[0])
        self.net = mlp(obs_dim, hidden, action_dim, activation)
        self.register_buffer("_low", torch.as_tensor(np.asarray(action_low), dtype=torch.float32))
        self.register_buffer("_high", torch.as_tensor(np.asarray(action_high), dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.net(obs))  # (-1, 1)
        return self._low + (self._high - self._low) * (u + 1.0) / 2.0


class Critic(nn.Module):
    """State-action value Q(x, a) -> scalar (concat input MLP)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden, 1, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy (SAC; docs/continuous_action_extension.md
    §4). Reparameterized sampling with the tanh + affine-scaling log-prob
    correction so log pi(a|x) is the density of the SCALED action."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        action_dim = int(np.asarray(action_low).shape[0])
        self.trunk, width = _trunk(obs_dim, hidden, activation)
        self.mean_head = nn.Linear(width, action_dim)
        self.log_std_head = nn.Linear(width, action_dim)
        self.register_buffer("_low", torch.as_tensor(np.asarray(action_low), dtype=torch.float32))
        self.register_buffer("_high", torch.as_tensor(np.asarray(action_high), dtype=torch.float32))

    def _scale(self, tanh_u: torch.Tensor) -> torch.Tensor:
        return self._low + (self._high - self._low) * (tanh_u + 1.0) / 2.0

    def forward(
        self, obs: torch.Tensor, *, deterministic: bool = False, with_logprob: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        if deterministic:
            u = mean
        else:
            u = torch.distributions.Normal(mean, std).rsample()
        tanh_u = torch.tanh(u)
        action = self._scale(tanh_u)
        if not with_logprob:
            return action, None
        logp = torch.distributions.Normal(mean, std).log_prob(u).sum(-1)
        # tanh change-of-variables and the affine box scaling (constant) Jacobian.
        logp = logp - torch.log(1.0 - tanh_u.pow(2) + 1e-6).sum(-1)
        logp = logp - torch.log((self._high - self._low) / 2.0).sum()
        return action, logp
