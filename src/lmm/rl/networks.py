"""Discrete-Q and continuous actor/critic network modules (ruling D9).

The coordinate-conditioned DQN network is recorded in docs/rl_design.md. The
continuous actor/critic modules (DDPG/TD3/SAC) are recorded in
docs/continuous_action_extension.md §4. Architecture is configured in
configs/algo/*.yaml; ``mlp`` remains their shared feed-forward factory.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "ACTIVATIONS",
    "mlp",
    "StructuredDiscreteQ",
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


class StructuredDiscreteQ(nn.Module):
    """Action-coordinate-conditioned Q network for an exact discrete grid.

    A conventional DQN output layer assigns an independent parameter vector to
    every discrete action.  That is a poor statistical fit for the auction
    grid, whose 1,346 entries are structured combinations of only three
    coordinates.  This module instead embeds the normalized coordinates with a
    shared encoder and scores every exact grid entry with a state-dependent
    query::

        Q(s, a) = V(s) + <q(s), e(a)> / sqrt(r) + w_prior 1{a != a_noop}.

    ``forward(obs)`` retains the ordinary vector-valued DQN interface.  Passing
    one action index per observation evaluates only the sampled entries, which
    avoids materializing a batch-by-grid tensor during the gradient step.  The
    action coordinates are a registered buffer (checkpointed but not learned),
    while all action dependence is learned through shared encoder parameters;
    there is no per-action parameter table.

    When ``initial_noop_margin`` is supplied, the value/query heads start at
    zero and the final scalar coefficient starts at ``-margin``.  Thus the
    canonical no-op is the unique initial greedy action, while the coefficient
    remains trainable and can be overcome by evidence.
    """

    ARCHITECTURE = "coordinate_conditioned_bilinear_v1"

    def __init__(
        self,
        obs_dim: int,
        hidden: Sequence[int],
        action_coordinates: np.ndarray,
        action_embedding_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        *,
        noop_index: int = 0,
        initial_noop_margin: float | None = None,
    ) -> None:
        super().__init__()
        coordinates = np.asarray(action_coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[0] <= 0 or coordinates.shape[1] <= 0:
            raise ValueError(
                "action_coordinates must have shape (n_actions, action_dim) "
                f"with positive dimensions, got {coordinates.shape}"
            )
        if not np.isfinite(coordinates).all():
            raise ValueError("action_coordinates must be finite")
        widths = tuple(int(width) for width in hidden)
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("hidden must contain positive layer widths")
        rank = int(action_embedding_dim)
        if rank <= 0:
            raise ValueError("action_embedding_dim must be positive")
        if not 0 <= int(noop_index) < coordinates.shape[0]:
            raise ValueError("noop_index is outside the action grid")
        if initial_noop_margin is not None and float(initial_noop_margin) < 0.0:
            raise ValueError("initial_noop_margin must be nonnegative")

        self.n_actions = int(coordinates.shape[0])
        self.action_dim = int(coordinates.shape[1])
        self.action_embedding_dim = rank
        self.noop_index = int(noop_index)
        self.register_buffer(
            "action_coordinates", torch.as_tensor(coordinates, dtype=torch.float32)
        )
        nonnoop = np.ones(self.n_actions, dtype=np.float32)
        nonnoop[self.noop_index] = 0.0
        self.register_buffer("_nonnoop", torch.as_tensor(nonnoop))

        trunk: list[nn.Module] = []
        previous = int(obs_dim)
        for width in widths:
            trunk.extend((nn.Linear(previous, width), activation()))
            previous = width
        self.state_trunk = nn.Sequential(*trunk)
        self.value_head = nn.Linear(previous, 1)
        self.query_head = nn.Linear(previous, rank)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, rank),
            activation(),
            nn.Linear(rank, rank),
        )
        prior = 0.0 if initial_noop_margin is None else -float(initial_noop_margin)
        self.nonnoop_prior = nn.Parameter(torch.tensor(prior, dtype=torch.float32))

        if initial_noop_margin is not None:
            # Preserve the exact safe auction policy at initialization without
            # introducing a 1,346-row action-specific output parameter.
            with torch.no_grad():
                self.value_head.weight.zero_()
                self.value_head.bias.zero_()
                self.query_head.weight.zero_()
                self.query_head.bias.zero_()

    def forward(
        self,
        obs: torch.Tensor,
        action_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return all Q values, or one indexed Q value per observation."""
        if obs.ndim != 2:
            raise ValueError(f"obs must have shape (batch, obs_dim), got {tuple(obs.shape)}")
        state = self.state_trunk(obs)
        value = self.value_head(state).squeeze(-1)
        query = self.query_head(state)
        scale = math.sqrt(float(self.action_embedding_dim))

        if action_indices is None:
            embedding = self.action_encoder(self.action_coordinates)
            advantage = query @ embedding.transpose(0, 1) / scale
            return value.unsqueeze(1) + advantage + self.nonnoop_prior * self._nonnoop

        indices = action_indices.to(device=obs.device, dtype=torch.int64)
        if indices.ndim != 1 or indices.shape[0] != obs.shape[0]:
            raise ValueError(
                "action_indices must have shape (batch,), got "
                f"{tuple(indices.shape)} for batch {obs.shape[0]}"
            )
        if bool(((indices < 0) | (indices >= self.n_actions)).any()):
            raise ValueError("action_indices contains an index outside the action grid")
        embedding = self.action_encoder(self.action_coordinates[indices])
        advantage = (query * embedding).sum(dim=1) / scale
        return value + advantage + self.nonnoop_prior * self._nonnoop[indices]


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
