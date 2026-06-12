"""Network factory (ruling D9; Phase 4).

Architecture is configured in configs/algo/*.yaml and recorded in
docs/rl_design.md (the author rewrites paper Section 4 from that document).
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn

__all__ = ["mlp"]


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Standard MLP: Linear-activation blocks, linear output head."""
    raise NotImplementedError("Phase 4")
