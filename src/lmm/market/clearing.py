"""Projected-price calibration, clearing, tick projection, and allocation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Protocol

import numpy as np

from lmm.config import Algo1Params, GridParams, LEGACY_CLEARING, VOLUME_MAX_CLEARING
from lmm.market.clob import BookSnapshot

__all__ = [
    "Algo1Diagnostics",
    "CarryoverCalibration",
    "Algo1Estimator",
    "ClearingInputs",
    "ClearingResult",
    "TerminalAllocation",
    "CappedPositivePartSchedule",
    "Eq2Cache",
    "round_half_up_to_tick",
    "bracketing_tick_indices",
    "select_clearing_tick",
    "supply_demand",
    "clear_linear",
    "clear_with_external_schedule",
    "allocate_pro_rata",
    "allocate_terminal",
    "solve_clearing",
    "solve_linear_clearing",
    "solve_clearing_with_hockey_stick",
    "solve_monotone_clearing",
]

logger = logging.getLogger(__name__)

_EPS = 1e-12
_SLOPE_EPS = 1e-8  # compatibility path only; revised clearing uses D_mu.
# Quantities within atol + rtol*max(|a|,|b|) are tied. Grid membership and
# distance ties use max(1e-12, 8 ulps of p/alpha), in TICK units. These are
# numerical conventions of max_volume_v2, not configurable economic bands.
VOLUME_ATOL = 1e-12
VOLUME_RTOL = 1e-12
TICK_ATOL = 1e-12
TICK_ULPS = 8


def _plain_map(values: Mapping[int, float]) -> dict[int, float]:
    return {int(k): float(v) for k, v in sorted(values.items())}


@dataclass(frozen=True)
class Algo1Diagnostics:
    """All quantities required to audit one Algorithm-1 observation."""

    decision_index: int
    Q: Mapping[int, float]
    e_hat: Mapping[int, float]
    varsigma_hat: Mapping[int, float]
    K_hat: Mapping[int, float]
    s_tilde: float | None
    H: float


@dataclass(frozen=True)
class CarryoverCalibration:
    """Final moments after replacing ``O_n`` by the residual CLOB book."""

    n: int
    Q_star: Mapping[int, float]
    e_hat: Mapping[int, float]
    varsigma_hat: Mapping[int, float]
    K_hat: Mapping[int, float]
    ticks: np.ndarray
    slopes: np.ndarray
    references: np.ndarray

    def __post_init__(self) -> None:
        for name in ("ticks", "slopes", "references"):
            values = np.asarray(getattr(self, name)).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def total_slope(self) -> float:
        return float(np.sum(self.slopes))


class Algo1Estimator:
    """Decision-indexed implementation of manuscript Algorithm 1.

    ``H_{t_0}=H0`` is established by :meth:`reset`; there is no observation at
    index zero.  For each ``i>=1``, :meth:`observe` consumes the *pre-action
    exogenous* snapshot ``O_i``.  Missing price levels contribute zero to the
    moments.  The strategic CLOB remainder is structurally ignored because
    only :meth:`BookSnapshot.exogenous_volume_by_tick` is read.
    """

    def __init__(self, params: Algo1Params, grid: GridParams) -> None:
        self.params = params
        self.grid = grid
        self.n_khat_clamped = 0
        self.n_zero_slope_skips = 0
        self._h = float(grid.S0)
        self._last_index = 0
        self._history: dict[int, dict[int, float]] = {}
        self._diagnostics: dict[int, Algo1Diagnostics] = {}
        # Running first and second raw sums make the per-observation moment
        # update O(number of active levels), rather than repeatedly scanning
        # the complete O_1,...,O_i history.  Missing levels are zeros and hence
        # require no explicit update.  This is algebraically identical to the
        # manuscript formula and matters for empirically wide historical books.
        self._sum_q: dict[int, float] = {}
        self._sum_q2: dict[int, float] = {}

    @property
    def eta_H(self) -> float:
        # ``tau`` is the pre-revision field name; the compatibility fallback
        # lets config integration land independently.
        return float(getattr(self.params, "eta_H", getattr(self.params, "tau")))

    def reset(self, h0: float | None = None) -> None:
        """Start at ``i=0`` with explicit ``H0`` and no moment observation."""
        configured = getattr(self.params, "H0", None)
        if h0 is None:
            h0 = self.grid.S0 if configured is None else configured
        self._h = float(h0)
        self._last_index = 0
        self._history.clear()
        self._diagnostics.clear()
        self._sum_q.clear()
        self._sum_q2.clear()
        self.n_khat_clamped = 0
        self.n_zero_slope_skips = 0

    def observe(self, decision_index: int, snapshot: BookSnapshot) -> Algo1Diagnostics:
        """Observe ``O_i`` immediately before the action at ``t_i`` (``i>=1``)."""
        i = int(decision_index)
        if i < 1:
            raise ValueError("Algorithm 1 has H0 at i=0; observations start at i=1")
        if i != self._last_index + 1:
            raise ValueError(
                f"Algorithm-1 observations must be sequential; expected "
                f"{self._last_index + 1}, got {i}"
            )
        q = snapshot.exogenous_volume_by_tick(eps=_EPS)
        self._history[i] = q
        for tick, volume in q.items():
            self._sum_q[tick] = self._sum_q.get(tick, 0.0) + float(volume)
            self._sum_q2[tick] = self._sum_q2.get(tick, 0.0) + float(volume) ** 2
        e_hat, var_hat, k_hat = self._moments_from_sums(
            sum_q=self._sum_q,
            sum_q2=self._sum_q2,
            denominator=i,
            active_ticks=tuple(q),
        )

        den = float(sum(k_hat.values()))
        s_tilde: float | None = None
        if q and den > 0.0:
            s_tilde = float(
                sum(k_hat[k] * self.grid.alpha * k for k in k_hat) / den
            )
            self._h += self.eta_H * (s_tilde - self._h)
        else:
            self.n_zero_slope_skips += 1

        diag = Algo1Diagnostics(
            decision_index=i,
            Q=_plain_map(q),
            e_hat=_plain_map(e_hat),
            varsigma_hat=_plain_map(var_hat),
            K_hat=_plain_map(k_hat),
            s_tilde=s_tilde,
            H=float(self._h),
        )
        self._diagnostics[i] = diag
        self._last_index = i
        self._log_diagnostics(diag)
        return diag

    def update(self, snapshot: BookSnapshot) -> float:
        """Compatibility shim: observe the next index and return only ``H``."""
        return self.observe(self._last_index + 1, snapshot).H

    def final_replacement_calibration(
        self,
        residual: BookSnapshot,
        n: int | None = None,
    ) -> CarryoverCalibration:
        """Replace ``O_n`` by ``O*`` without changing the already observed H.

        For ``n>=1`` this uses ``O_1,...,O_{n-1},O*`` with denominator ``n``.
        The explicit ``n=0`` edge case uses the residual snapshot alone with
        denominator one.  Only levels still present in ``O*`` can carry over.
        """
        final_index = self._last_index if n is None else int(n)
        q_star = residual.exogenous_volume_by_tick(eps=_EPS)
        if final_index < 0:
            raise ValueError("n must be nonnegative")
        if final_index == 0:
            histories = {1: q_star}
            denominator = 1
        else:
            missing = [i for i in range(1, final_index) if i not in self._history]
            if missing:
                raise ValueError(f"missing Algorithm-1 snapshots before n: {missing}")
            histories = {i: self._history[i] for i in range(1, final_index)}
            histories[final_index] = q_star
            denominator = final_index

        if final_index == self._last_index and final_index > 0:
            # Common environment path: replace O_n in the running sufficient
            # statistics without rebuilding all prior per-tick histories.
            sum_q = dict(self._sum_q)
            sum_q2 = dict(self._sum_q2)
            for tick, volume in self._history[final_index].items():
                sum_q[tick] = sum_q.get(tick, 0.0) - float(volume)
                sum_q2[tick] = sum_q2.get(tick, 0.0) - float(volume) ** 2
            for tick, volume in q_star.items():
                sum_q[tick] = sum_q.get(tick, 0.0) + float(volume)
                sum_q2[tick] = sum_q2.get(tick, 0.0) + float(volume) ** 2
            e_hat, var_hat, k_hat = self._moments_from_sums(
                sum_q=sum_q,
                sum_q2=sum_q2,
                denominator=denominator,
                active_ticks=tuple(q_star),
                count_clamps=False,
            )
        else:
            e_hat, var_hat, k_hat = self._moments(
                histories=histories,
                denominator=denominator,
                active_ticks=tuple(q_star),
                count_clamps=False,
            )
        kept_ticks = np.asarray(
            sorted(k for k, q in q_star.items() if q > 0.0 and k_hat.get(k, 0.0) > 0.0),
            dtype=int,
        )
        slopes = np.asarray([k_hat[int(k)] for k in kept_ticks], dtype=float)
        references = self.grid.alpha * kept_ticks.astype(float)
        calibration = CarryoverCalibration(
            n=final_index,
            Q_star=_plain_map(q_star),
            e_hat=_plain_map(e_hat),
            varsigma_hat=_plain_map(var_hat),
            K_hat=_plain_map(k_hat),
            ticks=kept_ticks,
            slopes=slopes,
            references=references,
        )
        logger.debug(
            "Algorithm 1 residual replacement n=%d Q*=%s K*=%s D*=%.12g",
            final_index,
            calibration.Q_star,
            calibration.K_hat,
            calibration.total_slope,
        )
        return calibration

    # Alias chosen for readable integration call sites.
    calibrate_carryover = final_replacement_calibration

    def _moments(
        self,
        histories: Mapping[int, Mapping[int, float]],
        denominator: int,
        active_ticks: tuple[int, ...],
        count_clamps: bool = True,
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        if denominator <= 0:
            raise ValueError("moment denominator must be positive")
        sum_q: dict[int, float] = {}
        sum_q2: dict[int, float] = {}
        for values in histories.values():
            for tick, volume in values.items():
                sum_q[tick] = sum_q.get(tick, 0.0) + float(volume)
                sum_q2[tick] = sum_q2.get(tick, 0.0) + float(volume) ** 2
        return self._moments_from_sums(
            sum_q=sum_q,
            sum_q2=sum_q2,
            denominator=denominator,
            active_ticks=active_ticks,
            count_clamps=count_clamps,
        )

    def _moments_from_sums(
        self,
        sum_q: Mapping[int, float],
        sum_q2: Mapping[int, float],
        denominator: int,
        active_ticks: tuple[int, ...],
        count_clamps: bool = True,
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        """Evaluate Algorithm-1 moments from exact sufficient statistics."""
        if denominator <= 0:
            raise ValueError("moment denominator must be positive")
        e_hat: dict[int, float] = {}
        var_hat: dict[int, float] = {}
        k_hat: dict[int, float] = {}
        for tick in sorted(set(active_ticks)):
            total = float(sum_q.get(tick, 0.0))
            total_sq = float(sum_q2.get(tick, 0.0))
            e = total / denominator
            var = total_sq / denominator
            raw = 0.0 if e <= 0.0 else (2.0 * e - var / e) / self.grid.alpha
            if raw < 0.0 and count_clamps:
                self.n_khat_clamped += 1
            e_hat[tick] = e
            var_hat[tick] = var
            k_hat[tick] = max(0.0, raw)
        return e_hat, var_hat, k_hat

    @staticmethod
    def _log_diagnostics(diag: Algo1Diagnostics) -> None:
        logger.debug(
            "Algorithm 1 i=%d Q=%s e=%s varsigma=%s K=%s S_tilde=%s H=%.12g",
            diag.decision_index,
            diag.Q,
            diag.e_hat,
            diag.varsigma_hat,
            diag.K_hat,
            diag.s_tilde,
            diag.H,
        )

    @property
    def h(self) -> float:
        return float(self._h)

    @property
    def diagnostics(self) -> Mapping[int, Algo1Diagnostics]:
        return dict(self._diagnostics)


@dataclass(frozen=True)
class ClearingInputs:
    """Linear book components using the manuscript buy-positive convention."""

    K_exo: np.ndarray
    S_exo: np.ndarray
    K_agent: np.ndarray
    S_agent: np.ndarray
    net_market_volume: float
    fallback_mid: float
    hockey: tuple[float, float] | None = None  # compatibility only
    # Net-only legacy callers imply the minimal one-sided decomposition.
    # Explicit sides must both be supplied and agree with buy-minus-sell net.
    buy_market_volume: float | None = None
    sell_market_volume: float | None = None


@dataclass(frozen=True)
class ClearingResult:
    """Continuous and tick-projected clearing diagnostics."""

    D: float
    R: float
    continuous_price: float
    tick_price: float
    residual_at_tick: float
    nonlinear: bool = False
    tick_index: int | None = None
    matched_volume: float | None = None


@dataclass(frozen=True)
class TerminalAllocation:
    """Pro-rata terminal allocation after aggregating the strategic schedule."""

    requested_agent: float
    actual_agent: float
    Q_supply: float
    Q_demand: float
    rho_supply: float
    rho_demand: float
    executed_supply: float
    executed_demand: float
    self_trade_count: int = 0


class NetSupplySchedule(Protocol):
    def value(self, price: float) -> float: ...


@dataclass(frozen=True)
class CappedPositivePartSchedule:
    """External benchmark schedule ``min(cap, slope*(p-reference)_+)``."""

    slope: float
    reference: float
    cap: float

    def __post_init__(self) -> None:
        if self.slope < 0.0 or self.cap < 0.0:
            raise ValueError("benchmark slope and cap must be nonnegative")

    def value(self, price: float) -> float:
        return float(min(self.cap, self.slope * max(float(price) - self.reference, 0.0)))

    __call__ = value


def round_half_up_to_tick(price: float, alpha: float) -> float:
    """Deterministic ``alpha*floor(price/alpha + 1/2)`` projection."""
    if alpha <= 0.0:
        raise ValueError("tick size alpha must be positive")
    return float(alpha * math.floor(float(price) / alpha + 0.5))


def _linear_aggregates(inputs: ClearingInputs) -> tuple[float, float]:
    K_exo = np.asarray(inputs.K_exo, dtype=float)
    S_exo = np.asarray(inputs.S_exo, dtype=float)
    K_agent = np.asarray(inputs.K_agent, dtype=float)
    S_agent = np.asarray(inputs.S_agent, dtype=float)
    if (K_exo.ndim != 1 or K_agent.ndim != 1
            or K_exo.shape != S_exo.shape or K_agent.shape != S_agent.shape):
        raise ValueError("each slope array must match its reference-price array")
    if not all(np.all(np.isfinite(x)) for x in (K_exo, S_exo, K_agent, S_agent)):
        raise ValueError("linear schedules must be finite")
    if not math.isfinite(inputs.net_market_volume):
        raise ValueError("net market volume must be finite")
    if np.any(K_exo < 0.0) or np.any(K_agent < 0.0):
        raise ValueError("linear schedule slopes must be nonnegative")
    D = float(np.sum(K_exo) + np.sum(K_agent))
    R = float(K_exo @ S_exo + K_agent @ S_agent + inputs.net_market_volume)
    return D, R


def _tick_tolerance(coordinate: float) -> float:
    return max(TICK_ATOL, TICK_ULPS * math.ulp(coordinate))


def bracketing_tick_indices(p_star: float, alpha: float) -> tuple[int, ...]:
    """Unique floor/ceil indices, snapping only machine-close on-grid roots.

    Prices are nonnegative. Coordinates >= 2**48 are rejected because binary64
    no longer resolves our sub-tick tie convention reliably there.
    """
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(p_star) or p_star < 0:
        raise ValueError("finite nonnegative root and finite positive alpha required")
    x = p_star / alpha
    if not math.isfinite(x) or x >= 2**48:
        raise ValueError("root/tick size exceeds supported tick-index precision")
    nearest = round(x)
    if abs(x - nearest) <= _tick_tolerance(x):
        return (int(nearest),)
    return (math.floor(x), math.ceil(x))


def supply_demand(
    exogenous_schedule_values: np.ndarray,
    agent_net_quantity: float,
    buy_market_volume: float,
    sell_market_volume: float,
) -> tuple[float, float]:
    """Keep exogenous participants separate, but split the ONE net agent."""
    values = np.asarray(exogenous_schedule_values, dtype=float)
    agent, buy, sell = map(float, (agent_net_quantity, buy_market_volume, sell_market_volume))
    if (not np.all(np.isfinite(values)) or not all(map(math.isfinite, (agent, buy, sell)))
            or min(buy, sell) < 0):
        raise ValueError("schedule values must be finite and market-order volumes nonnegative")
    return (
        float(np.sum(np.maximum(values, 0.0))) + max(agent, 0.0) + sell,
        float(np.sum(np.maximum(-values, 0.0))) + max(-agent, 0.0) + buy,
    )


def _market_sides(inputs: ClearingInputs) -> tuple[float, float]:
    buy, sell = inputs.buy_market_volume, inputs.sell_market_volume
    if buy is None and sell is None:
        return max(inputs.net_market_volume, 0.0), max(-inputs.net_market_volume, 0.0)
    if buy is None or sell is None:
        raise ValueError("provide both buy and sell market volumes, or neither")
    if not all(map(math.isfinite, (buy, sell))) or min(buy, sell) < 0:
        raise ValueError("market-order volumes must be finite and nonnegative")
    if not math.isclose(buy - sell, inputs.net_market_volume, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("net market volume must equal buy minus sell")
    return float(buy), float(sell)


def _schedule_values(inputs, price, external_agent_schedule=None):
    exogenous = np.asarray(inputs.K_exo) * (price - np.asarray(inputs.S_exo))
    agent = float(np.sum(np.asarray(inputs.K_agent) * (price - np.asarray(inputs.S_agent))))
    if external_agent_schedule is not None:
        agent += float(external_agent_schedule.value(price))
    return exogenous, agent


def _book_supply_demand(inputs, price, external_agent_schedule=None):
    return supply_demand(*_schedule_values(inputs, price, external_agent_schedule), *_market_sides(inputs))


def select_clearing_tick(
    p_star: float,
    alpha: float,
    quantities: Callable[[float], tuple[float, float]],
    *,
    mechanism: str = VOLUME_MAX_CLEARING,
) -> int:
    """Maximize (matched volume, -distance to root, price) on adjacent ticks.

    ``quantities`` must supply nondecreasing supply and nonincreasing demand
    from the same book whose admissible continuous root is ``p_star``.
    Only this projection changes; raw Algorithm 1 and other rounding do not.
    """
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(p_star) or p_star < 0:
        raise ValueError("finite nonnegative root and finite positive alpha required")
    if mechanism == LEGACY_CLEARING:
        return math.floor(p_star / alpha + 0.5)
    if mechanism != VOLUME_MAX_CLEARING:
        raise ValueError(f"unknown clearing mechanism {mechanism!r}")
    candidates = bracketing_tick_indices(p_star, alpha)
    best = candidates[0]
    if len(candidates) == 1:
        return best
    volumes = [min(quantities(alpha * k)) for k in candidates]
    if not all(math.isfinite(v) and v >= 0 for v in volumes):
        raise ValueError("matched volumes must be finite and nonnegative")
    volume_tol = VOLUME_ATOL + VOLUME_RTOL * max(volumes)
    if abs(volumes[1] - volumes[0]) > volume_tol:
        return candidates[int(volumes[1] > volumes[0])]
    x = p_star / alpha
    distances = [abs(k - x) for k in candidates]
    if abs(distances[1] - distances[0]) <= _tick_tolerance(x):
        return candidates[1]
    return candidates[int(distances[1] < distances[0])]


def clear_linear(
    inputs: ClearingInputs, alpha: float, D_mu: float, *, mechanism: str = VOLUME_MAX_CLEARING,
) -> ClearingResult:
    """Project the linear root; revised residual bound is D*alpha (linear only)."""
    D, R = _linear_aggregates(inputs)
    if not math.isfinite(D_mu) or D_mu <= 0 or D <= 0:
        raise ValueError("D_mu and aggregate slope must be finite and positive")
    if D + _EPS < D_mu:
        raise AssertionError(f"invalid auction book: D={D} < D_mu={D_mu}")
    if R < -_EPS:
        raise AssertionError(f"invalid auction book: R={R} < 0")
    R = max(R, 0.0)
    continuous = R / D
    k = select_clearing_tick(continuous, alpha, lambda p: _book_supply_demand(inputs, p), mechanism=mechanism)
    tick = alpha * k
    residual = D * tick - R
    volume = min(_book_supply_demand(inputs, tick))
    return ClearingResult(D, R, continuous, tick, residual, tick_index=k, matched_volume=volume)


def clear_with_external_schedule(
    inputs: ClearingInputs,
    schedule: NetSupplySchedule,
    alpha: float,
    D_mu: float,
    *,
    mechanism: str = VOLUME_MAX_CLEARING,
) -> ClearingResult:
    """Clear a linear background plus one external monotone benchmark order."""
    D, R = _linear_aggregates(inputs)
    if not math.isfinite(D_mu) or D_mu <= 0 or D <= 0:
        raise ValueError("D_mu and aggregate slope must be finite and positive")
    if D + _EPS < D_mu:
        raise AssertionError(f"invalid auction book: D={D} < D_mu={D_mu}")
    if mechanism == LEGACY_CLEARING:
        if R < -_EPS:
            raise AssertionError(f"invalid auction book: R={R} < 0")
        R = max(R, 0.0)
    # With a signed external schedule, R alone need not be nonnegative:
    # admissibility is excess(0)<=0 for the COMPLETE book, checked below.

    def excess(price: float) -> float:
        return D * price - R + float(schedule.value(price))

    upper = max(float(inputs.fallback_mid) * 2.0, R / D + 1.0, 1.0)
    if excess(0.0) > 0.0:
        raise ValueError("external schedule has no admissible nonnegative root")
    # Preserve the legacy root solve exactly. Revised roots are refined to
    # sub-tick machine precision before classifying their adjacent ticks.
    tol = 1e-10 if mechanism == LEGACY_CLEARING else alpha * TICK_ATOL
    continuous = solve_monotone_clearing(excess, (0.0, upper), tol=tol)
    k = select_clearing_tick(continuous, alpha, lambda p: _book_supply_demand(inputs, p, schedule), mechanism=mechanism)
    tick = alpha * k
    return ClearingResult(D, R, continuous, tick, excess(tick), nonlinear=True,
                          tick_index=k, matched_volume=min(_book_supply_demand(inputs, tick, schedule)))


def allocate_pro_rata(
    exogenous_schedule_values: np.ndarray,
    agent_net_quantity: float,
    buy_market_volume: float,
    sell_market_volume: float,
) -> TerminalAllocation:
    """Allocate the rounded-price imbalance with the manuscript robust ratios."""
    agent = float(agent_net_quantity)  # aggregate before positive/negative split
    Q_supply, Q_demand = supply_demand(
        exogenous_schedule_values, agent, buy_market_volume, sell_market_volume,
    )

    rho_supply = 1.0 if Q_supply == 0.0 or Q_supply <= Q_demand else Q_demand / Q_supply
    rho_demand = 1.0 if Q_demand == 0.0 or Q_demand <= Q_supply else Q_supply / Q_demand
    actual = rho_supply * max(agent, 0.0) - rho_demand * max(-agent, 0.0)
    executed_supply = rho_supply * Q_supply
    executed_demand = rho_demand * Q_demand
    if not math.isclose(executed_supply, executed_demand, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(
            f"pro-rata allocation is unbalanced: {executed_supply} != {executed_demand}"
        )
    return TerminalAllocation(
        requested_agent=agent,
        actual_agent=float(actual),
        Q_supply=Q_supply,
        Q_demand=Q_demand,
        rho_supply=float(rho_supply),
        rho_demand=float(rho_demand),
        executed_supply=float(executed_supply),
        executed_demand=float(executed_demand),
        self_trade_count=0,
    )


def allocate_terminal(
    inputs: ClearingInputs,
    tick_price: float,
    external_agent_schedule: NetSupplySchedule | None = None,
) -> TerminalAllocation:
    """Evaluate all schedules at the rounded price and allocate actual ``Z``."""
    return allocate_pro_rata(
        *_schedule_values(inputs, float(tick_price), external_agent_schedule),
        *_market_sides(inputs),
    )


class Eq2Cache:
    """Lagged auction indicative-price cache with revised rounded-result API."""

    def __init__(self, grid: GridParams, mechanism: str = VOLUME_MAX_CLEARING) -> None:
        self.grid = grid
        self.mechanism = mechanism
        self.n_degenerate_fallbacks = 0  # compatibility metric; revised path stays zero
        self._h = float(grid.S0)
        self._result: ClearingResult | None = None

    def reset(self, h_from_algo1: float) -> None:
        self._h = float(h_from_algo1)
        self._result = None
        self.n_degenerate_fallbacks = 0

    def recompute_result(
        self,
        inputs: ClearingInputs,
        D_mu: float,
        external_schedule: NetSupplySchedule | None = None,
    ) -> ClearingResult:
        result = (
            clear_linear(inputs, self.grid.alpha, D_mu, mechanism=self.mechanism)
            if external_schedule is None
            else clear_with_external_schedule(inputs, external_schedule, self.grid.alpha, D_mu, mechanism=self.mechanism)
        )
        self._result = result
        self._h = result.tick_price
        return result

    def recompute(self, inputs: ClearingInputs) -> float:
        """Projected convenience API; only degenerate legacy books fall back."""
        root, degenerate = solve_clearing(inputs, alpha=self.grid.alpha, mechanism=self.mechanism)
        self.n_degenerate_fallbacks += int(degenerate)
        self._h = float(root)
        self._result = None
        return self._h

    def read(self) -> float:
        return float(self._h)

    @property
    def result(self) -> ClearingResult | None:
        return self._result


# ---------------------------------------------------------------------------
# Compatibility clearing API.  New code should call clear_linear /
# clear_with_external_schedule and consume ClearingResult.
# ---------------------------------------------------------------------------


def solve_linear_clearing(
    inputs: ClearingInputs, *, alpha: float | None = None, mechanism: str = VOLUME_MAX_CLEARING,
) -> tuple[float, bool]:
    """Continuous compatibility solver; pass alpha for the shared tick rule.

    Omitting alpha intentionally returns a raw root, never a settlement price.
    Degenerate compatibility books retain the historical fallback.
    """
    D, R = _linear_aggregates(inputs)
    if D <= _SLOPE_EPS:
        return float(inputs.fallback_mid), True
    if alpha is not None:
        return clear_linear(inputs, alpha, _SLOPE_EPS, mechanism=mechanism).tick_price, False
    return float(R / D), False


def solve_clearing(
    inputs: ClearingInputs, *, alpha: float | None = None, mechanism: str = VOLUME_MAX_CLEARING,
) -> tuple[float, bool]:
    """Dispatch continuous (alpha omitted) or projected clearing explicitly."""
    if inputs.hockey is None:
        return solve_linear_clearing(inputs, alpha=alpha, mechanism=mechanism)
    return solve_clearing_with_hockey_stick(inputs, *inputs.hockey, alpha=alpha, mechanism=mechanism)


def solve_clearing_with_hockey_stick(
    inputs: ClearingInputs,
    z_slope: float,
    s_tilde: float,
    *,
    alpha: float | None = None,
    mechanism: str = VOLUME_MAX_CLEARING,
) -> tuple[float, bool]:
    """Uncapped hockey-stick root; optional alpha uses aggregate-agent volume."""
    if z_slope < 0.0:
        raise ValueError("benchmark slope must be nonnegative")
    base = replace(inputs, hockey=None)
    if alpha is not None:
        root, degenerate = solve_clearing_with_hockey_stick(base, z_slope, s_tilde)
        if degenerate:
            return root, True
        schedule = CappedPositivePartSchedule(z_slope, s_tilde, math.inf)
        k = select_clearing_tick(root, alpha, lambda p: _book_supply_demand(base, p, schedule), mechanism=mechanism)
        return alpha * k, False
    root, degenerate = solve_linear_clearing(base)
    if not degenerate and root <= s_tilde:
        return root, False
    D, R = _linear_aggregates(base)
    D_with = D + z_slope
    if D_with <= _SLOPE_EPS:
        return float(inputs.fallback_mid), True
    root_with = (R + z_slope * s_tilde) / D_with
    if root_with >= s_tilde:
        return float(root_with), False
    return float(inputs.fallback_mid), True


def solve_monotone_clearing(
    excess_supply: Callable[[float], float],
    bracket: tuple[float, float],
    tol: float = 1e-10,
    max_widen: int = 200,
) -> float:
    """Bisection with geometric bracket widening for monotone net supply."""
    lo, hi = map(float, bracket)
    if not all(map(math.isfinite, (lo, hi, tol))) or lo >= hi or tol <= 0:
        raise ValueError(f"invalid bracket {bracket!r}")
    f_lo, f_hi = float(excess_supply(lo)), float(excess_supply(hi))
    width = hi - lo
    n = 0
    while f_lo > 0.0 and n < max_widen:
        lo -= width
        width *= 2.0
        f_lo = float(excess_supply(lo))
        n += 1
    while f_hi < 0.0 and n < max_widen:
        hi += width
        width *= 2.0
        f_hi = float(excess_supply(hi))
        n += 1
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError("no sign change found; excess supply has no root")
    if not all(map(math.isfinite, (f_lo, f_hi))):
        raise ValueError("nonfinite excess supply at bracket")
    if tol < 1e-10:
        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break  # machine precision; prevents stalled sub-ulp bisection
        value = float(excess_supply(mid))
        if not math.isfinite(value):
            raise ValueError("nonfinite excess supply during root solve")
        if value == 0.0 and tol < 1e-10:
            return mid
        if value < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
