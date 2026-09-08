"""Auction state, proposal validity, and strategic-order liveness.

The revised auction is deliberately transactional.  The six exogenous event
indicators are sampled together and then processed in manuscript order.  A
tentative mutation is committed only when the resulting *exogenous* book
satisfies

``G = sum(K) >= D_mu`` and ``J + I >= 0``.

Accepted market-order arrivals have cumulative, side-specific indices.
Cancellation sets an indexed volume to zero without decrementing the index.
The carry-over book always contains one non-cancellable schedule: either the
explicit depth-floor fallback or the selected persistent carry-over schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from lmm.config import AuctionFlowParams, ClobFlowParams, GridParams
from lmm.market.clearing import CappedPositivePartSchedule, CarryoverCalibration
from lmm.market.clob import sample_mo_volume

__all__ = [
    "AgentOrderLedger",
    "AuctionEvents",
    "ExogenousAuctionFlow",
    "ProposalCounts",
    "ScheduleRecord",
]

_EPS = 1e-12
_EVENT_NAMES = (
    "schedule_arrival",
    "schedule_cancel",
    "buy_arrival",
    "buy_cancel",
    "sell_arrival",
    "sell_cancel",
)


@dataclass
class ScheduleRecord:
    """One exogenous linear schedule ``K * (p-S)``."""

    schedule_id: int
    slope: float
    reference: float
    provenance: str
    persistent: bool = False
    active: bool = True


@dataclass
class ProposalCounts:
    """Cumulative disposition counts for one proposal type."""

    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    ineligible: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "proposed": int(self.proposed),
            "accepted": int(self.accepted),
            "rejected": int(self.rejected),
            "ineligible": int(self.ineligible),
        }


@dataclass(frozen=True)
class AuctionEvents:
    """Disposition of all six proposals at one auction decision.

    A status is one of ``not_proposed``, ``accepted``, ``rejected`` or
    ``ineligible``.  The boolean compatibility properties intentionally mean
    *accepted*, not merely proposed.
    """

    schedule_arrival: str = "not_proposed"
    schedule_cancel: str = "not_proposed"
    buy_arrival: str = "not_proposed"
    buy_cancel: str = "not_proposed"
    sell_arrival: str = "not_proposed"
    sell_cancel: str = "not_proposed"

    def status(self, event_name: str) -> str:
        if event_name not in _EVENT_NAMES:
            raise KeyError(event_name)
        return str(getattr(self, event_name))

    def as_dict(self) -> dict[str, str]:
        return {name: self.status(name) for name in _EVENT_NAMES}

    @property
    def new_mm(self) -> bool:
        return self.schedule_arrival == "accepted"

    @property
    def mm_cancelled(self) -> bool:
        return self.schedule_cancel == "accepted"

    @property
    def new_buy_taker(self) -> bool:
        return self.buy_arrival == "accepted"

    @property
    def buy_taker_cancelled(self) -> bool:
        return self.buy_cancel == "accepted"

    @property
    def new_sell_taker(self) -> bool:
        return self.sell_arrival == "accepted"

    @property
    def sell_taker_cancelled(self) -> bool:
        return self.sell_cancel == "accepted"


class ExogenousAuctionFlow:
    """Valid exogenous auction book with transactional proposal processing."""

    def __init__(
        self,
        params: AuctionFlowParams,
        grid: GridParams,
        clob_params: ClobFlowParams,
    ) -> None:
        self.params = params
        self.grid = grid
        self.clob_params = clob_params
        self.frozen_mid = float(grid.S0)
        self._schedules: list[ScheduleRecord] = []
        self._buy_volumes: list[float] = []
        self._sell_volumes: list[float] = []
        self._next_schedule_id = 0
        self._counts = {name: ProposalCounts() for name in _EVENT_NAMES}

    def reset(
        self,
        frozen_mid: float,
        carryover: CarryoverCalibration | None = None,
    ) -> None:
        """Install carry-over schedules and establish the persistent floor.

        If carry-over depth is below ``D_mu``, the shortfall is supplied by
        ``(D_mu-D*) * (p-S_mid_open)``.  Otherwise exactly one carry-over
        schedule is persistent: maximum slope, then lower reference, then
        stable schedule id.
        """
        self.frozen_mid = float(frozen_mid)
        self._schedules = []
        self._buy_volumes = []
        self._sell_volumes = []
        self._next_schedule_id = 0
        self._counts = {name: ProposalCounts() for name in _EVENT_NAMES}

        if carryover is not None:
            for slope, reference in zip(carryover.slopes, carryover.references, strict=True):
                if float(slope) > 0.0:
                    self._append_schedule(
                        float(slope),
                        float(reference),
                        provenance="carryover",
                    )

        total = sum(rec.slope for rec in self._schedules if rec.active)
        shortfall = float(self.params.D_mu) - float(total)
        if shortfall > _EPS:
            self._append_schedule(
                shortfall,
                self.frozen_mid,
                provenance="fallback",
                persistent=True,
            )
        else:
            active = [rec for rec in self._schedules if rec.active]
            if not active:
                # This can only occur for a non-positive configured floor, but
                # retaining a persistent zero-free book would break validity.
                raise ValueError("D_mu must be positive when no carry-over schedule exists")
            selected = min(
                active,
                key=lambda rec: (-rec.slope, rec.reference, rec.schedule_id),
            )
            selected.persistent = True
        self.assert_valid()

    def _append_schedule(
        self,
        slope: float,
        reference: float,
        provenance: str,
        persistent: bool = False,
    ) -> ScheduleRecord:
        if slope <= 0.0:
            raise ValueError(f"schedule slope must be positive, got {slope}")
        rec = ScheduleRecord(
            schedule_id=self._next_schedule_id,
            slope=float(slope),
            reference=float(reference),
            provenance=str(provenance),
            persistent=bool(persistent),
        )
        self._next_schedule_id += 1
        self._schedules.append(rec)
        return rec

    @property
    def active_schedules(self) -> tuple[ScheduleRecord, ...]:
        return tuple(rec for rec in self._schedules if rec.active)

    @property
    def schedules(self) -> tuple[ScheduleRecord, ...]:
        """All schedules, including cancelled records, in stable-id order."""
        return tuple(self._schedules)

    def supply_curves(self) -> tuple[np.ndarray, np.ndarray]:
        active = self.active_schedules
        return (
            np.asarray([rec.slope for rec in active], dtype=float),
            np.asarray([rec.reference for rec in active], dtype=float),
        )

    @property
    def supply_K(self) -> list[float]:
        """Compatibility view of active slopes."""
        return [rec.slope for rec in self.active_schedules]

    @property
    def supply_S(self) -> list[float]:
        """Compatibility view of active references."""
        return [rec.reference for rec in self.active_schedules]

    @property
    def buy_volumes(self) -> np.ndarray:
        return np.asarray(self._buy_volumes, dtype=float)

    @property
    def sell_volumes(self) -> np.ndarray:
        return np.asarray(self._sell_volumes, dtype=float)

    @property
    def n_mm(self) -> int:
        return len(self.active_schedules)

    @property
    def n_buy(self) -> int:
        """Cumulative accepted buy-arrival count; cancellation does not decrement."""
        return len(self._buy_volumes)

    @property
    def n_sell(self) -> int:
        """Cumulative accepted sell-arrival count; cancellation does not decrement."""
        return len(self._sell_volumes)

    def buy_market_volume(self) -> float:
        return float(sum(self._buy_volumes))

    def sell_market_volume(self) -> float:
        return float(sum(self._sell_volumes))

    def net_market_volume(self) -> float:
        return self.buy_market_volume() - self.sell_market_volume()

    def aggregates(self) -> tuple[float, float, float, float]:
        """Return ``(G, J, I, R)`` for the active exogenous book."""
        K, S = self.supply_curves()
        G = float(np.sum(K))
        J = float(K @ S) if len(K) else 0.0
        I = self.net_market_volume()
        return G, J, I, J + I

    def is_valid(self) -> bool:
        G, _, _, R = self.aggregates()
        return G + _EPS >= float(self.params.D_mu) and R >= -_EPS

    def assert_valid(self) -> None:
        G, J, I, R = self.aggregates()
        if G + _EPS < float(self.params.D_mu) or R < -_EPS:
            raise AssertionError(
                "invalid exogenous auction book: "
                f"G={G:.12g}, J={J:.12g}, I={I:.12g}, R={R:.12g}, "
                f"D_mu={self.params.D_mu:.12g}"
            )

    @property
    def proposal_counts(self) -> Mapping[str, Mapping[str, int]]:
        return {name: counts.as_dict() for name, counts in self._counts.items()}

    def _record(self, name: str, status: str) -> str:
        counts = self._counts[name]
        counts.proposed += 1
        if status == "accepted":
            counts.accepted += 1
        elif status == "rejected":
            counts.rejected += 1
        elif status == "ineligible":
            counts.ineligible += 1
        else:
            raise ValueError(f"unknown proposal status {status!r}")
        return status

    def step(self, rng: np.random.Generator) -> AuctionEvents:
        """Sample all indicators, then process ``B,D,J+,G+,J-,G-``.

        Here ``B``/``D`` are schedule arrival/cancellation, ``J`` market-order
        arrival, and ``G`` market-order cancellation.  Conditional marks are
        sampled only for proposed, eligible operations.
        """
        p = self.params
        proposed = rng.random(6) < np.asarray(
            [p.p1, p.p2, p.p3, p.p4, p.p3, p.p4], dtype=float
        )
        status = {name: "not_proposed" for name in _EVENT_NAMES}

        # B: exogenous linear-schedule arrival.
        if proposed[0]:
            slope = float(rng.uniform(p.U1, p.U2))
            offset = int(rng.integers(p.M1, p.M2 + 1))
            rec = self._append_schedule(
                slope,
                self.frozen_mid + self.grid.alpha * offset,
                provenance="auction",
            )
            if self.is_valid():
                status["schedule_arrival"] = self._record("schedule_arrival", "accepted")
            else:
                self._schedules.pop()
                self._next_schedule_id -= 1
                status["schedule_arrival"] = self._record("schedule_arrival", "rejected")

        # D: cancellation of one active, non-persistent exogenous schedule.
        if proposed[1]:
            eligible = [rec for rec in self._schedules if rec.active and not rec.persistent]
            if not eligible:
                status["schedule_cancel"] = self._record("schedule_cancel", "ineligible")
            else:
                rec = eligible[int(rng.integers(0, len(eligible)))]
                rec.active = False
                if self.is_valid():
                    status["schedule_cancel"] = self._record("schedule_cancel", "accepted")
                else:
                    rec.active = True
                    status["schedule_cancel"] = self._record("schedule_cancel", "rejected")

        # J+: buy market-order arrival.
        if proposed[2]:
            volume = sample_mo_volume(
                rng,
                self.clob_params.v_m,
                self.clob_params.gamma_m,
                self.clob_params.V_max,
            )
            self._buy_volumes.append(volume)
            if self.is_valid():
                status["buy_arrival"] = self._record("buy_arrival", "accepted")
            else:
                self._buy_volumes.pop()
                status["buy_arrival"] = self._record("buy_arrival", "rejected")

        # G+: cancellation of one active buy market order.
        if proposed[3]:
            eligible = [i for i, volume in enumerate(self._buy_volumes) if volume > 0.0]
            if not eligible:
                status["buy_cancel"] = self._record("buy_cancel", "ineligible")
            else:
                idx = eligible[int(rng.integers(0, len(eligible)))]
                old = self._buy_volumes[idx]
                self._buy_volumes[idx] = 0.0
                if self.is_valid():
                    status["buy_cancel"] = self._record("buy_cancel", "accepted")
                else:
                    self._buy_volumes[idx] = old
                    status["buy_cancel"] = self._record("buy_cancel", "rejected")

        # J-: sell market-order arrival.
        if proposed[4]:
            volume = sample_mo_volume(
                rng,
                self.clob_params.v_m,
                self.clob_params.gamma_m,
                self.clob_params.V_max,
            )
            self._sell_volumes.append(volume)
            if self.is_valid():
                status["sell_arrival"] = self._record("sell_arrival", "accepted")
            else:
                self._sell_volumes.pop()
                status["sell_arrival"] = self._record("sell_arrival", "rejected")

        # G-: cancellation of one active sell market order.
        if proposed[5]:
            eligible = [i for i, volume in enumerate(self._sell_volumes) if volume > 0.0]
            if not eligible:
                status["sell_cancel"] = self._record("sell_cancel", "ineligible")
            else:
                idx = eligible[int(rng.integers(0, len(eligible)))]
                old = self._sell_volumes[idx]
                self._sell_volumes[idx] = 0.0
                if self.is_valid():
                    status["sell_cancel"] = self._record("sell_cancel", "accepted")
                else:
                    self._sell_volumes[idx] = old
                    status["sell_cancel"] = self._record("sell_cancel", "rejected")

        self.assert_valid()
        return AuctionEvents(**status)

    # Test/diagnostic injection hooks use the same validity contract.
    def inject_market_maker(
        self,
        K: float,
        S: float,
        *,
        persistent: bool = False,
        provenance: str = "injected",
    ) -> None:
        rec = self._append_schedule(K, S, provenance, persistent=persistent)
        if not self.is_valid():
            self._schedules.pop()
            self._next_schedule_id -= 1
            raise ValueError("injected schedule would invalidate the exogenous auction book")
        assert rec.active

    def inject_taker(self, side: int, volume: float) -> None:
        if volume < 0.0:
            raise ValueError("market-order volume must be nonnegative")
        target = self._buy_volumes if side > 0 else self._sell_volumes
        target.append(float(volume))
        if not self.is_valid():
            target.pop()
            raise ValueError("injected market order would invalidate the exogenous auction book")


class AgentOrderLedger:
    """Strategic auction schedules and cancel-all liveness history."""

    def __init__(self, grid: GridParams) -> None:
        self.grid = grid
        self.n_slots = int(grid.tau_cl - grid.tau_op)
        self.K = np.zeros(self.n_slots, dtype=float)
        self.S = np.zeros(self.n_slots, dtype=float)
        self.submitted = np.zeros(self.n_slots, dtype=bool)
        self.live = np.zeros(self.n_slots, dtype=bool)
        self.one_sided = np.zeros(self.n_slots, dtype=bool)
        self.quantity_cap = np.full(self.n_slots, np.nan, dtype=float)
        self.reference_price = np.full(self.n_slots, np.nan, dtype=float)

    def reset(self) -> None:
        self.K.fill(0.0)
        self.S.fill(0.0)
        self.submitted.fill(False)
        self.live.fill(False)
        self.one_sided.fill(False)
        self.quantity_cap.fill(np.nan)
        self.reference_price.fill(np.nan)

    def _slot(self, t: int | float) -> int:
        raw = float(t) - float(self.grid.tau_op)
        i = int(round(raw))
        if not np.isclose(raw, i) or not 0 <= i < self.n_slots:
            raise ValueError(
                f"auction time {t} outside integer grid "
                f"[{self.grid.tau_op}, {self.grid.tau_cl - 1}]"
            )
        return i

    def submit(
        self,
        t: int | float,
        K_a: float,
        S_a: float,
        one_sided: bool = False,
        quantity_cap: float | None = None,
        reference_price: float | None = None,
    ) -> None:
        """Submit the current schedule; ``K_a=0`` is the canonical no-op."""
        if K_a < 0.0:
            raise ValueError(f"K^a must be nonnegative, got {K_a}")
        if K_a == 0.0:
            return
        external = bool(one_sided or quantity_cap is not None)
        if external and np.any(self.live & self.one_sided):
            raise ValueError("at most one external capped schedule may be live")
        if quantity_cap is not None and quantity_cap < 0.0:
            raise ValueError("external schedule cap must be nonnegative")
        i = self._slot(t)
        self.K[i] = float(K_a)
        self.S[i] = float(S_a)
        self.submitted[i] = True
        self.live[i] = True
        self.one_sided[i] = external
        if external:
            self.quantity_cap[i] = float("inf") if quantity_cap is None else float(quantity_cap)
            self.reference_price[i] = (
                float(S_a) if reference_price is None else float(reference_price)
            )

    def apply_cancel_all(self, t: int | float) -> None:
        """Deactivate every positive schedule submitted strictly before ``t``."""
        self.live[: self._slot(t)] = False

    def cancel_admissible(self) -> bool:
        return bool(np.any(self.live & (self.K > 0.0)))

    def theta(self) -> np.ndarray:
        return (self.submitted & ~self.live).astype(float)

    def history_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        cut = int(np.clip(np.floor(t) - self.grid.tau_op, 0, self.n_slots))
        S = np.zeros(self.n_slots, dtype=float)
        K = np.zeros(self.n_slots, dtype=float)
        S[:cut] = self.S[:cut] * self.submitted[:cut]
        K[:cut] = self.K[:cut] * self.submitted[:cut]
        return S, K

    def live_orders(self) -> tuple[np.ndarray, np.ndarray]:
        mask = self.live & ~self.one_sided
        return self.K[mask].copy(), self.S[mask].copy()

    def live_orders_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None]:
        K, S = self.live_orders()
        ext = np.flatnonzero(self.live & self.one_sided)
        hockey = None
        if len(ext):
            i = int(ext[0])
            hockey = (float(self.K[i]), float(self.reference_price[i]))
        return K, S, hockey

    def aggregates(self) -> tuple[float, float]:
        """Return live linear strategic ``(A_own, B_own)``."""
        K, S = self.live_orders()
        return float(np.sum(K)), float(K @ S) if len(K) else 0.0

    def external_schedule(self) -> CappedPositivePartSchedule | None:
        ext = np.flatnonzero(self.live & self.one_sided)
        if not len(ext):
            return None
        if len(ext) > 1:
            raise AssertionError("more than one external capped schedule is live")
        i = int(ext[0])
        return CappedPositivePartSchedule(
            slope=float(self.K[i]),
            reference=float(self.reference_price[i]),
            cap=float(self.quantity_cap[i]),
        )

    def aggregate_value(self, price: float) -> float:
        K, S = self.live_orders()
        value = float(np.sum(K * (float(price) - S)))
        external = self.external_schedule()
        if external is not None:
            value += external.value(float(price))
        return value
