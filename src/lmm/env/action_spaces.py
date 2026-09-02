"""Discrete five-coordinate actions and projected continuous proposals.

The manuscript action is ``(v, delta, K, ell, c)``.  The local auction
coordinate ``ell`` is resolved inside the simulator to the absolute frozen-mid
coordinate ``b``; those two quantities deliberately use different types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import gymnasium
import numpy as np

from lmm.config import ActionGridParams, ExperimentConfig

if TYPE_CHECKING:  # avoid the env <-> action_spaces import cycle at runtime
    from lmm.env.mdp import MarketMakingEnv

__all__ = [
    "ClobAction",
    "AuctionAction",
    "FiveCoordinateAction",
    "to_five_coordinate",
    "from_five_coordinate",
    "round_half_up",
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

    def as_five_coordinate(self) -> "FiveCoordinateAction":
        return FiveCoordinateAction(self.volume, self.delta, 0.0, 0, 0)


@dataclass(frozen=True)
class AuctionAction:
    """Public manuscript auction action ``(K^a,ell,c)``.

    ``ell`` is the local integer displacement from the current indicative
    price. The environment privately derives and validates absolute ``b``.
    """

    K_a: float
    ell: int
    cancel: int

    def as_five_coordinate(self) -> "FiveCoordinateAction":
        return FiveCoordinateAction(0.0, 0, self.K_a, self.ell, self.cancel)


@dataclass(frozen=True)
class _BenchmarkAuctionAction:
    """Private trusted path for the capped positive-part AS/TWAP schedule."""

    K_a: float
    reference_price: float
    quantity_cap: float
    cancel: int = 0


@dataclass(frozen=True, order=True)
class FiveCoordinateAction:
    """Common manuscript action ``(v, delta, K, ell, c)``."""

    volume: float
    delta: int
    K_a: float
    ell: int
    cancel: int

    def as_tuple(self) -> tuple[float, int, float, int, int]:
        return (self.volume, self.delta, self.K_a, self.ell, self.cancel)


def to_five_coordinate(
    action: ClobAction | AuctionAction | FiveCoordinateAction,
) -> FiveCoordinateAction:
    """Convert a phase action to the common five-coordinate representation."""
    if isinstance(action, FiveCoordinateAction):
        return action
    if isinstance(action, (ClobAction, AuctionAction)):
        return action.as_five_coordinate()
    raise TypeError(f"unsupported action type {type(action).__name__}")


def from_five_coordinate(
    action: FiveCoordinateAction | Sequence[float], phase: str
) -> ClobAction | AuctionAction:
    """Convert a common action to the legacy phase-specific action object."""
    if not isinstance(action, FiveCoordinateAction):
        values = tuple(action)
        if len(values) != 5:
            raise ValueError(f"five-coordinate action must have length 5, got {len(values)}")
        action = FiveCoordinateAction(
            float(values[0]), int(values[1]), float(values[2]), int(values[3]), int(values[4])
        )
    if phase == "clob":
        if (action.K_a, action.ell, action.cancel) != (0.0, 0, 0):
            raise ValueError("CLOB action requires (K,ell,c)=(0,0,0)")
        return ClobAction(action.volume, action.delta)
    if phase == "auction":
        if (action.volume, action.delta) != (0.0, 0):
            raise ValueError("auction action requires (v,delta)=(0,0)")
        return AuctionAction(action.K_a, action.ell, action.cancel)
    raise ValueError(f"unknown phase {phase!r}")


def round_half_up(value: float) -> int:
    """Deterministic manuscript rounding: ``floor(value + 1/2)``."""
    return math.floor(float(value) + 0.5)


class ClobActionGrid:
    """Lexicographic CLOB list: one wait plus ``v=1..V_max, delta=0..L_max``."""

    def __init__(self, params: ActionGridParams) -> None:
        self.params = params
        deltas = range(0, params.L_max + 1)
        self.actions: tuple[ClobAction, ...] = (ClobAction(0.0, 0),) + tuple(
            ClobAction(float(v), d)
            for v in range(1, params.V_max + 1)
            for d in deltas
        )
        keys = [to_five_coordinate(a).as_tuple() for a in self.actions]
        assert keys == sorted(keys), "CLOB action list must be lexicographic"

    def __len__(self) -> int:
        return len(self.actions)

    def decode(self, index: int) -> ClobAction:
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("CLOB action index must be an integer")
        if not 0 <= int(index) < len(self.actions):
            raise ValueError(
                f"CLOB action index {index} is outside [0,{len(self.actions) - 1}]"
            )
        return self.actions[index]

    def mask(self, inventory: float) -> np.ndarray:
        """Mask integer volume at ``min(V_max, floor(inventory))``."""
        max_volume = min(self.params.V_max, max(0, math.floor(float(inventory))))
        return np.fromiter(
            (a.volume <= max_volume for a in self.actions), dtype=bool, count=len(self.actions)
        )


class AuctionActionGrid:
    """Lexicographic full manuscript lattice in ``(K^a,ell,c)``."""

    def __init__(self, params: ActionGridParams) -> None:
        self.params = params
        K_choices = tuple(params.beta * k for k in params.auction_K_multipliers)
        offsets = range(-params.B_max, params.B_max + 1)
        if params.auction_cancel_mode == "enabled":
            self.actions = (
                AuctionAction(0.0, 0, 0),
                AuctionAction(0.0, 0, 1),
            ) + tuple(
                AuctionAction(float(K), ell, c)
                for K in K_choices
                for ell in offsets
                for c in (0, 1)
            )
        elif params.auction_cancel_mode == "never":
            self.actions = (AuctionAction(0.0, 0, 0),) + tuple(
                AuctionAction(float(K), ell, 0)
                for K in K_choices
                for ell in offsets
            )
        else:
            raise ValueError(
                "actions.auction_cancel_mode must be 'enabled' or 'never', "
                f"got {params.auction_cancel_mode!r}"
            )
        keys = [to_five_coordinate(a).as_tuple() for a in self.actions]
        assert keys == sorted(keys), "auction action list must be lexicographic"
        self._cancel_flags = np.fromiter(
            (a.cancel == 1 for a in self.actions), dtype=bool, count=len(self.actions)
        )

    def __len__(self) -> int:
        return len(self.actions)

    def decode(self, index: int) -> AuctionAction:
        """Return the public local-coordinate action at ``index``."""
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("auction action index must be an integer")
        if not 0 <= int(index) < len(self.actions):
            raise ValueError(
                f"auction action index {index} is outside [0,{len(self.actions) - 1}]"
            )
        return self.actions[index]

    def mask(
        self,
        cancel_admissible: bool,
        *,
        frozen_mid: float | None = None,
        alpha: float | None = None,
        indicative_b_ticks: int = 0,
    ) -> np.ndarray:
        """Mask cancellation and resolved absolute-``b`` admissibility.

        ``indicative_b_ticks`` is the frozen-mid coordinate of ``H_t^cl``;
        execution resolves ``b=indicative_b_ticks+ell``. Supplying exactly one
        of ``frozen_mid`` and ``alpha`` is an error.
        """
        if cancel_admissible:
            mask = np.ones(len(self.actions), dtype=bool)
        else:
            mask = ~self._cancel_flags
        if (frozen_mid is None) != (alpha is None):
            raise ValueError("frozen_mid and alpha must be supplied together")
        if frozen_mid is not None:
            assert alpha is not None
            offset_admissible = np.fromiter(
                (
                    a.K_a == 0.0
                    or (
                        abs(int(indicative_b_ticks) + a.ell)
                        <= self.params.B_inf
                        and float(frozen_mid)
                        + float(alpha)
                        * (int(indicative_b_ticks) + a.ell)
                        >= 0.0
                    )
                    for a in self.actions
                ),
                dtype=bool,
                count=len(self.actions),
            )
            mask &= offset_admissible
        return mask


@dataclass(frozen=True)
class ContinuousActionSpec:
    """Normalized raw actor box; every coordinate lies in ``[-1, 1]``."""

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
            low=-np.ones(2, dtype=float),
            high=np.ones(2, dtype=float),
        )

    @staticmethod
    def auction(params: ActionGridParams) -> "ContinuousActionSpec":
        return ContinuousActionSpec(
            low=-np.ones(3, dtype=float),
            high=np.ones(3, dtype=float),
        )


def _continuous_cancel_from_actions(params: ActionGridParams) -> str:
    if params.auction_cancel_mode == "enabled":
        return "threshold"
    if params.auction_cancel_mode == "never":
        return "never"
    raise ValueError(
        "actions.auction_cancel_mode must be 'enabled' or 'never', "
        f"got {params.auction_cancel_mode!r}"
    )


def continuous_action_specs(cfg: ExperimentConfig) -> dict[str, ContinuousActionSpec]:
    """Per-phase normalized raw boxes for projected actor-critic policies."""
    continuous_cancel = _continuous_cancel_from_actions(cfg.actions)
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
    """Project normalized raw actor actions in ``[-1,1]`` to market actions.

    The clipped *raw* proposal is retained in ``proposal_action_vec`` for
    replay.  Integer coordinates use :func:`round_half_up`, never NumPy's
    bankers' rounding.
    """

    def __init__(self, env: "MarketMakingEnv") -> None:
        self.env = env
        self.cfg = env.cfg
        continuous_cancel = _continuous_cancel_from_actions(env.cfg.actions)
        self.continuous_cancel = continuous_cancel
        self.specs = continuous_action_specs(env.cfg)
        self._boxes = {phase: spec.box() for phase, spec in self.specs.items()}
        ap = env.cfg.actions
        self._volume_max = int(ap.V_max)
        self._delta_max = int(ap.L_max)
        self._beta = float(ap.beta)
        self._K_index_max = int(ap.K_max)
        self._absolute_offset_max = int(ap.B_inf)
        self._local_offset_max = int(ap.B_max)
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
    def h_cl(self) -> float:
        return self.env.h_cl

    @property
    def decision_index(self) -> int:
        return self.env.decision_index

    @property
    def completed_episode_grid(self):
        """The realized grid, available only after the episode has ended."""
        return self.env.completed_episode_grid

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
            order, committed, projection = self._project_clob(action)
        else:
            order, committed, projection = self._project_auction(action)
        obs, reward, terminated, truncated, info = self.env.step(order)
        info = dict(info)
        info["proposal_action_vec"] = committed
        info["raw_action_vec"] = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        info["projected_action_five"] = to_five_coordinate(order).as_tuple()
        info["projection_diagnostics"] = projection
        next_phase = self.env.phase
        if next_phase in self._boxes:
            self.action_space = self._boxes[next_phase]
        elif not (terminated or truncated):
            raise RuntimeError(f"no continuous action space for active phase {next_phase!r}")
        self.observation_space = self.env.observation_space
        return obs, reward, terminated, truncated, info

    # -- projection / snapping (docs/continuous_action_extension.md §1.3) ----

    def _project_clob(
        self, action
    ) -> tuple[ClobAction, np.ndarray, dict[str, float | int | bool]]:
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != 2:
            raise ValueError(f"CLOB raw action must have length 2, got {a.size}")
        raw = np.clip(a, -1.0, 1.0)
        v_raw = (raw[0] + 1.0) * self._volume_max / 2.0
        delta_raw = (raw[1] + 1.0) * self._delta_max / 2.0
        inventory_cap = max(0, math.floor(float(self.env.inventory)))
        v_rounded = round_half_up(v_raw)
        delta_rounded = round_half_up(delta_raw)
        v = min(inventory_cap, self._volume_max, v_rounded)
        delta = 0 if v == 0 else int(np.clip(delta_rounded, 0, self._delta_max))
        committed = raw.astype(np.float32, copy=True)
        order = ClobAction(volume=float(v), delta=delta)
        diagnostics: dict[str, float | int | bool] = {
            "input_clipped": bool(np.any(a != raw)),
            "bound_saturation_count": int(np.count_nonzero(np.isclose(np.abs(raw), 1.0))),
            "rounded_coordinate_count": int(not np.isclose(v_raw, v_rounded))
            + int(v > 0 and not np.isclose(delta_raw, delta_rounded)),
            "inventory_projection": bool(v_rounded > inventory_cap),
            "ell_admissibility_projection": False,
            "cancel_threshold_positive": False,
            "cancel_executed": False,
        }
        return order, committed, diagnostics

    def _project_auction(
        self, action
    ) -> tuple[AuctionAction, np.ndarray, dict[str, float | int | bool]]:
        a = np.asarray(action, dtype=float).reshape(-1)
        expected = 2 if self.continuous_cancel == "never" else 3
        if a.size != expected:
            raise ValueError(f"auction raw action must have length {expected}, got {a.size}")
        raw = np.clip(a, -1.0, 1.0)
        K_raw = (raw[0] + 1.0) * self._beta * self._K_index_max / 2.0
        k_unclipped = round_half_up(K_raw / self._beta)
        k = int(np.clip(k_unclipped, 0, self._K_index_max))
        K = self._beta * k
        ell_raw = self._local_offset_max * raw[1]
        ell_unclipped = round_half_up(ell_raw)
        ell = int(
            np.clip(
                ell_unclipped,
                -self._local_offset_max,
                self._local_offset_max,
            )
        )
        ell_before_admissibility = ell
        if k > 0:
            alpha = float(self.cfg.grid.alpha)
            center_b = self.env._indicative_b_ticks()
            minimum_b = math.ceil(-float(self.env.s_mid) / alpha)
            lower = max(
                -self._local_offset_max,
                -self._absolute_offset_max - center_b,
                minimum_b - center_b,
            )
            upper = min(
                self._local_offset_max,
                self._absolute_offset_max - center_b,
            )
            if lower > upper:
                raise ValueError(
                    "no local auction ell resolves to an admissible absolute b"
                )
            ell = int(np.clip(ell, lower, upper))
        else:
            ell = 0
        if self.continuous_cancel == "never":
            cancel = 0
            committed = raw.astype(np.float32, copy=True)
        else:
            cancel = int(raw[2] >= 0.0 and bool(self.env.cancel_admissible))
            committed = raw.astype(np.float32, copy=True)
        order = AuctionAction(K_a=K, ell=ell, cancel=cancel)
        diagnostics: dict[str, float | int | bool] = {
            "input_clipped": bool(np.any(a != raw)),
            "bound_saturation_count": int(np.count_nonzero(np.isclose(np.abs(raw), 1.0))),
            "rounded_coordinate_count": int(
                not np.isclose(K_raw / self._beta, k_unclipped)
            )
            + int(k > 0 and not np.isclose(ell_raw, ell_unclipped)),
            "inventory_projection": False,
            "ell_admissibility_projection": bool(
                ell != ell_before_admissibility
            ),
            "cancel_threshold_positive": bool(
                self.continuous_cancel == "threshold" and raw[2] >= 0.0
            ),
            "cancel_executed": bool(cancel),
        }
        return order, committed, diagnostics
