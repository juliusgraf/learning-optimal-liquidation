"""Action grids and admissibility masks (paper `sec:MDP`, Adm(x); AUDIT A.8).

Discrete grids (legacy values as config defaults, main.py:1424-1435):
- CLOB: {(0,0)} u {1..30} x {1..12} = 361 actions (v, delta), v-major order
  (index-compatible with legacy). The delta grid is explicit and applied
  LITERALLY — no silent clamping (fixes AUDIT N6): delta = 12 = Lc quotes one
  tick past the deepest refreshed exogenous level (see market/clob.py).
- Auction: K in {0} u linspace(1, K_max, 10), offset in {-12..12},
  c in {0,1} = 550 actions, (K, offset, c)-major order. K^a = 0 == abstain;
  S^a = alpha*(floor(S_mid_frozen/alpha) + offset) is tick-snapped by the
  env (AUDIT N4), so S^a in alpha*N holds structurally.

Admissibility (CLAUDE.md, time-free): a^1 <= x^1 (volume <= inventory),
a^2 >= x^10/alpha (structural on these grids: delta >= 0 => price >= mid
tick), a^5 <= C(x) (cancel-all only if a live prior K^a > 0 order exists).
Exploration must SAMPLE from Adm(x): agents mask both the greedy argmax and
the random draw (masking preferred over projection — fixes AUDIT N12; the
stored action always equals the executed action). The env additionally
REJECTS inadmissible submissions with ValueError (env/mdp.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.config import ActionGridParams

__all__ = [
    "ClobAction",
    "AuctionAction",
    "ClobActionGrid",
    "AuctionActionGrid",
    "ContinuousActionSpec",
]


@dataclass(frozen=True)
class ClobAction:
    """CLOB action (A^1, A^2): volume v_t and book-level offset delta_t.

    delta is the tick offset above the mid tick (the agent sells on the ask
    side; delta >= 0 so the quote alpha*(k_mid + delta) >= S^mid tick)."""

    volume: float
    delta: int


@dataclass(frozen=True)
class AuctionAction:
    """Auction action (A^3, A^4, A^5): slope K^a, quote tick offset, scalar
    cancel-all c_t in {0, 1} (ruling D4)."""

    K_a: float
    offset: int
    cancel: int


class ClobActionGrid:
    """Discrete CLOB grid with admissibility masking (a^1 <= inventory)."""

    def __init__(self, params: ActionGridParams) -> None:
        self.params = params
        deltas = range(params.clob_delta_min, params.clob_delta_max + 1)
        self.actions: tuple[ClobAction, ...] = (ClobAction(0.0, 0),) + tuple(
            ClobAction(float(v), d)
            for v in range(1, params.clob_volume_max + 1)
            for d in deltas
        )

    def __len__(self) -> int:
        return len(self.actions)

    def decode(self, index: int) -> ClobAction:
        return self.actions[index]

    def mask(self, inventory: float) -> np.ndarray:
        """Boolean mask: True where (v, delta) is admissible (v <= x^1).

        The quote constraint a^2 >= x^10/alpha is structural (delta >= 0)."""
        return np.fromiter(
            (a.volume <= inventory for a in self.actions), dtype=bool, count=len(self.actions)
        )


class AuctionActionGrid:
    """Discrete auction grid with admissibility masking (a^5 <= C(x))."""

    def __init__(self, params: ActionGridParams) -> None:
        self.params = params
        K_choices = [0.0] + list(
            np.linspace(params.auction_K_grid_min, params.auction_K_grid_max, params.auction_K_grid_n)
        )
        offsets = range(-params.auction_offset_max, params.auction_offset_max + 1)
        self.actions: tuple[AuctionAction, ...] = tuple(
            AuctionAction(float(K), off, c)
            for K in K_choices
            for off in offsets
            for c in (0, 1)
        )
        self._cancel_flags = np.fromiter(
            (a.cancel == 1 for a in self.actions), dtype=bool, count=len(self.actions)
        )

    def __len__(self) -> int:
        return len(self.actions)

    def decode(self, index: int) -> AuctionAction:
        return self.actions[index]

    def mask(self, cancel_admissible: bool) -> np.ndarray:
        """Boolean mask: c = 1 actions admissible iff ``cancel_admissible``
        (= C(x) > 0, ruling D4). K >= 0 and the tick-grid quote are
        structural on this grid."""
        if cancel_admissible:
            return np.ones(len(self.actions), dtype=bool)
        return ~self._cancel_flags


@dataclass(frozen=True)
class ContinuousActionSpec:
    """Bounds of the continuous-action relaxation (Phase 5; DDPG/TD3/SAC).

    A SEPARATE, clearly labeled relaxation of the discrete grids; every
    mathematical change is documented in docs/continuous_action_extension.md.
    """

    low: np.ndarray
    high: np.ndarray

    @staticmethod
    def clob(params: ActionGridParams) -> "ContinuousActionSpec":
        return ContinuousActionSpec(
            low=np.array([0.0, float(params.clob_delta_min)]),
            high=np.array([float(params.clob_volume_max), float(params.clob_delta_max)]),
        )

    @staticmethod
    def auction(params: ActionGridParams) -> "ContinuousActionSpec":
        return ContinuousActionSpec(
            low=np.array([0.0, -float(params.auction_offset_max), 0.0]),
            high=np.array([params.auction_K_grid_max, float(params.auction_offset_max), 1.0]),
        )
