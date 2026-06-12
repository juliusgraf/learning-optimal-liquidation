"""Action grids and admissibility masks (paper `sec:MDP`, Adm(x); AUDIT A.8).

Discrete grids (legacy values as config defaults):
- CLOB: {(0,0)} u {1..30} x {1..12} = 361 actions (v, delta). The delta grid
  is explicit — no silent clamping (fixes AUDIT N6).
- Auction: K in {0} u linspace(1, 33.33, 10), offset in {-12..12},
  c in {0,1} = 550 actions. K^a = 0 == abstain; S^a = S_mid + offset*alpha
  snapped to the tick grid (AUDIT N4).

Admissibility (CLAUDE.md, time-free): a^1 <= x^1, a^2 >= x^10/alpha
(structural on this grid), a^5 <= C(x). Exploration must SAMPLE from Adm(x):
agents mask both the greedy argmax and the random draw (masking preferred
over projection — fixes AUDIT N12; stored action == executed action).

Phase 3 fills in the bodies; the continuous relaxation is Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import ActionGridParams

__all__ = ["ClobAction", "AuctionAction", "ClobActionGrid", "AuctionActionGrid", "ContinuousActionSpec"]


@dataclass(frozen=True)
class ClobAction:
    """CLOB action (A^1, A^2): volume v_t and book-level offset delta_t."""

    volume: int
    delta: int


@dataclass(frozen=True)
class AuctionAction:
    """Auction action (A^3, A^4, A^5): slope K^a, quote offset, scalar cancel."""

    K_a: float
    offset: int
    cancel: int  # c_t in {0, 1}


class ClobActionGrid:
    """Discrete CLOB grid with admissibility masking (a^1 <= inventory)."""

    def __init__(self, params: ActionGridParams) -> None:
        self.params = params

    def __len__(self) -> int:
        raise NotImplementedError("Phase 3")

    def decode(self, index: int) -> ClobAction:
        raise NotImplementedError("Phase 3")

    def mask(self, inventory: float) -> np.ndarray:
        """Boolean mask: True where (v, delta) is admissible (v <= inventory)."""
        raise NotImplementedError("Phase 3")


class AuctionActionGrid:
    """Discrete auction grid with admissibility masking (a^5 <= C(x))."""

    def __init__(self, params: ActionGridParams) -> None:
        self.params = params

    def __len__(self) -> int:
        raise NotImplementedError("Phase 3")

    def decode(self, index: int) -> AuctionAction:
        raise NotImplementedError("Phase 3")

    def mask(self, cancel_admissible: bool) -> np.ndarray:
        """Boolean mask: c = 1 actions admissible iff ``cancel_admissible``."""
        raise NotImplementedError("Phase 3")


@dataclass(frozen=True)
class ContinuousActionSpec:
    """Bounds of the continuous-action relaxation (Phase 5; DDPG/TD3/SAC).

    A SEPARATE, clearly labeled relaxation of the discrete grids; every
    mathematical change is documented in docs/continuous_action_extension.md.
    """

    low: np.ndarray
    high: np.ndarray
