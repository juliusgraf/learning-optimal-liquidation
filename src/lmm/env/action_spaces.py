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
from typing import TYPE_CHECKING

import gymnasium
import numpy as np

from lmm.config import ActionGridParams, ExperimentConfig

if TYPE_CHECKING:  # avoid the env <-> action_spaces import cycle at runtime
    from lmm.env.mdp import MarketMakingEnv

__all__ = [
    "ClobAction",
    "AuctionAction",
    "ClobActionGrid",
    "AuctionActionGrid",
    "ContinuousActionSpec",
    "continuous_action_specs",
    "ContinuousActionAdapter",
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
    cancel-all c_t in {0, 1} (ruling D4).

    ``one_sided=True`` marks the BENCHMARK hockey-stick supply
    K^a (p - S^a)_+ (ruling D16; benchmarks only liquidate). It is never on
    the discrete grid — only raw actions submitted by the benchmark agents
    carry it."""

    K_a: float
    offset: int
    cancel: int
    one_sided: bool = False


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

    @property
    def dim(self) -> int:
        return int(np.asarray(self.low).shape[0])

    def box(self) -> "gymnasium.spaces.Box":
        """A ``gymnasium`` Box mirroring these bounds (float32)."""
        return gymnasium.spaces.Box(
            low=np.asarray(self.low, dtype=np.float32),
            high=np.asarray(self.high, dtype=np.float32),
            dtype=np.float32,
        )

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


def continuous_action_specs(
    cfg: ExperimentConfig, continuous_cancel: str = "threshold"
) -> dict[str, ContinuousActionSpec]:
    """Per-phase continuous action bounds (Phase 5; docs/continuous_action_extension.md §1.2).

    ``continuous_cancel`` selects the auction action dimension:
    ``"threshold"`` (default) keeps the 3-dim (K^a, s_off, c_logit) action;
    ``"never"`` drops the cancel coordinate (2-dim (K^a, s_off), c == 0).
    """
    clob = ContinuousActionSpec.clob(cfg.actions)
    auction = ContinuousActionSpec.auction(cfg.actions)
    if continuous_cancel == "never":
        auction = ContinuousActionSpec(low=auction.low[:2].copy(), high=auction.high[:2].copy())
    elif continuous_cancel != "threshold":
        raise ValueError(
            f"continuous_cancel must be 'threshold' or 'never', got {continuous_cancel!r}"
        )
    return {"clob": clob, "auction": auction}


class ContinuousActionAdapter:
    """Wrap :class:`~lmm.env.mdp.MarketMakingEnv` with per-phase Box action
    spaces for the continuous-control relaxation (Phase 5; ruling D9).

    The discrete DQN setting is untouched: this adapter is only used by the
    continuous agents (DDPG/TD3/SAC). It exposes ``gymnasium`` Box action
    spaces per phase and projects/snaps a continuous action vector to a
    :class:`ClobAction` / :class:`AuctionAction` before delegating to the inner
    env (docs/continuous_action_extension.md §1.3):

    - CLOB ``(v, delta)``: ``v`` projected to ``[0, min(V, I_t)]`` (continuous),
      ``delta`` snapped to the nearest tick (integer book level);
    - auction ``(K^a, s_off[, c_logit])``: ``K^a`` projected to ``[0, K_max]``
      (continuous, NOT snapped), ``s_off`` snapped to the nearest integer tick
      offset, ``c = 1{c_logit > 0.5}`` THEN masked by cancel-admissibility.

    Execution and rewards run on the snapped action, so the env DYNAMICS are
    identical to the discrete case (same transition function, possibly off-grid
    inputs). ``info["executed_action_vec"]`` carries the committed continuous
    action (after projection, before snapping) for the replay buffer.

    Attribute access (``phase``, ``t``, ``grid``, ``eq2``, ``inventory``,
    ``action_mask``, ``reset``) is delegated EXPLICITLY to the inner env, so
    ``rl/loops.py::run_episode`` drives it exactly like the raw env.
    """

    def __init__(self, env: "MarketMakingEnv", *, continuous_cancel: str = "threshold") -> None:
        self.env = env
        self.cfg = env.cfg
        self.continuous_cancel = continuous_cancel
        self.specs = continuous_action_specs(env.cfg, continuous_cancel)
        self._boxes = {phase: spec.box() for phase, spec in self.specs.items()}
        ap = env.cfg.actions
        self._delta_min = ap.clob_delta_min
        self._delta_max = ap.clob_delta_max
        self._volume_max = float(ap.clob_volume_max)
        self._K_max = float(ap.auction_K_grid_max)
        self._offset_max = ap.auction_offset_max
        # The env always starts in the CLOB phase after reset(); _phase is not
        # set until then, so default to the CLOB box at construction.
        self.action_space = self._boxes["clob"]
        self.observation_space = env.observation_space

    # -- explicit delegation (run_episode contract) --------------------------

    @property
    def phase(self) -> str:
        return self.env.phase

    @property
    def t(self) -> float:
        return self.env.t

    @property
    def grid(self):
        return self.env.grid

    @property
    def eq2(self):
        return self.env.eq2

    @property
    def inventory(self) -> float:
        return self.env.inventory

    @property
    def s_mid(self) -> float:
        return self.env.s_mid

    @property
    def generator(self):
        return self.env.generator

    def action_mask(self) -> np.ndarray:
        return self.env.action_mask()

    # -- gymnasium-style API -------------------------------------------------

    def reset(self, *, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        self.action_space = self._boxes[self.env.phase]
        self.observation_space = self.env.observation_space
        return out

    def step(self, action):
        """Project/snap ``action`` for the CURRENT phase, delegate to the inner
        env, and attach the committed continuous action vector to ``info``."""
        if self.env.phase == "clob":
            order, committed = self._project_clob(action)
        else:
            order, committed = self._project_auction(action)
        obs, reward, terminated, truncated, info = self.env.step(order)
        info = dict(info)
        info["executed_action_vec"] = committed
        self.action_space = self._boxes[self.env.phase]
        self.observation_space = self.env.observation_space
        return obs, reward, terminated, truncated, info

    # -- projection / snapping (docs/continuous_action_extension.md §1.3) ----

    def _project_clob(self, action) -> tuple[ClobAction, np.ndarray]:
        a = np.asarray(action, dtype=float).reshape(-1)
        inv = float(self.env.inventory)
        # Floor the upper bound at 0: inventory can drift to a tiny negative
        # float after near-full liquidation, and np.clip with lo > hi returns
        # hi (which would be a negative volume the env rejects).
        v = float(np.clip(a[0], 0.0, max(0.0, min(self._volume_max, inv))))
        delta_c = float(np.clip(a[1], self._delta_min, self._delta_max))
        committed = np.array([v, delta_c], dtype=np.float32)
        order = ClobAction(volume=v, delta=int(np.round(delta_c)))
        return order, committed

    def _project_auction(self, action) -> tuple[AuctionAction, np.ndarray]:
        a = np.asarray(action, dtype=float).reshape(-1)
        K = float(np.clip(a[0], 0.0, self._K_max))
        off_c = float(np.clip(a[1], -self._offset_max, self._offset_max))
        offset = int(np.clip(np.round(a[1]), -self._offset_max, self._offset_max))
        if self.continuous_cancel == "never":
            cancel = 0
            committed = np.array([K, off_c], dtype=np.float32)
        else:
            c_logit = float(np.clip(a[2], 0.0, 1.0))
            # threshold THEN admissibility mask: a^5 <= C(x). In the auction
            # phase action_mask().all() == ledger.cancel_admissible() (the grid
            # mask is all-True iff a cancel-all is admissible).
            cancel = int(c_logit > 0.5 and bool(self.env.action_mask().all()))
            committed = np.array([K, off_c, c_logit], dtype=np.float32)
        order = AuctionAction(K_a=K, offset=offset, cancel=cancel)
        return order, committed
