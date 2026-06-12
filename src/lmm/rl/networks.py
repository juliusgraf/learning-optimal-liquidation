"""Network factory (ruling D9; Phase 4).

Architecture is configured in configs/algo/*.yaml and recorded in
docs/rl_design.md (the author rewrites paper Section 4 from that document).
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn

__all__ = ["ACTIVATIONS", "mlp"]

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


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
