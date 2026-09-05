"""Common manuscript feature vector and leak-free training normalization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import numpy as np

from lmm.config import ClobFlowParams, FeatureParams, GridParams
from lmm.config import AuctionFlowParams

__all__ = [
    "COMMON_FEATURES",
    "FeatureExtractor",
    "FeatureNormalizer",
    "CLOB_FEATURE_DIM",
    "AUCTION_FEATURE_DIM",
]

COMMON_FEATURES = (
    "time",
    "inventory",
    "h_cl",
    "s_mid",
    "decision_index",
    "depth_ask",
    "depth_bid",
    "top_ask",
    "top_bid",
    "n_mm",
    "n_buy",
    "n_sell",
    "cancel_admissible",
    "own_slope",
    "own_weighted_quote",
    "exogenous_slope",
    "auction_imbalance",
    "exogenous_weighted_quote",
)

CLOB_FEATURE_DIM = len(COMMON_FEATURES)
AUCTION_FEATURE_DIM = len(COMMON_FEATURES)


class FeatureNormalizer:
    """Fit normalization statistics on training observations, then freeze.

    Physical time (index 0) and decision index (index 4) always divide by
    ``tau_cl``.  The cancellation bit (index 12) passes through.  By default
    every other numeric coordinate is standardized with training-only
    population mean/std; a zero-variance coordinate receives scale one.
    ``transform`` is intentionally unavailable until :meth:`freeze` prevents
    validation/test observations from mutating or influencing the state.
    Phase normalization fits separate population statistics on either side
    of tau_op. The optional signed asinh transform is invertible and resolves
    small auction inventories without clipping large or negative exposures.
    """

    VERSION = 1
    TIME_INDEX = 0
    DECISION_INDEX = 4
    CANCEL_INDEX = 12
    H_CL_INDEX = 2

    def __init__(
        self,
        tau_cl: float,
        fit_indices: Iterable[int] | None = None,
        *,
        eps: float = 1e-12,
        zero_h_cl: bool = False,
        relative_prices: bool = False,
        auction_exposure_features: bool = False,
        phase_normalization: bool = False,
        tau_op: float | None = None,
        auction_inventory_asinh: bool = False,
        inventory_scale: float = 1.,
    ) -> None:
        if tau_cl <= 0:
            raise ValueError(f"tau_cl must be positive, got {tau_cl}")
        self.tau_cl = float(tau_cl)
        excluded = {self.TIME_INDEX, self.DECISION_INDEX, self.CANCEL_INDEX}
        indices = tuple(i for i in range(len(COMMON_FEATURES)) if i not in excluded)
        self.fit_indices = tuple(indices if fit_indices is None else fit_indices)
        if len(set(self.fit_indices)) != len(self.fit_indices):
            raise ValueError("fit_indices must be unique")
        if any(i < 0 or i >= len(COMMON_FEATURES) or i in excluded for i in self.fit_indices):
            raise ValueError("fit_indices cannot include time, decision index, cancel, or out-of-range")
        self.eps = float(eps)
        self.zero_h_cl = bool(zero_h_cl)
        self.relative_prices = bool(relative_prices)
        self.auction_exposure_features = bool(auction_exposure_features)
        if self.auction_exposure_features and not self.relative_prices:
            raise ValueError('auction exposure coordinates require relative prices')
        self.count = 0
        self.mean = np.zeros(len(COMMON_FEATURES), dtype=np.float64)
        self.m2 = np.zeros(len(COMMON_FEATURES), dtype=np.float64)
        self.scale = np.ones(len(COMMON_FEATURES), dtype=np.float64)
        self.frozen = False
        self.inventory_reference_times = None
        self.inventory_reference_values = None
        self.phase_normalization = bool(phase_normalization)
        self.auction_inventory_asinh = bool(auction_inventory_asinh)
        self.inventory_scale = float(inventory_scale)
        if self.auction_inventory_asinh and (
            not self.auction_exposure_features or not np.isfinite(self.inventory_scale) or self.inventory_scale <= 0
        ):
            raise ValueError('auction inventory asinh requires exposure coordinates and a positive finite inventory scale')
        self.tau_op = tau_op
        self.phase_count = np.zeros(2, dtype=np.int64)
        self.phase_mean = np.zeros((2, len(COMMON_FEATURES)), dtype=np.float64)
        self.phase_m2 = np.zeros_like(self.phase_mean)
        self.phase_scale = np.ones_like(self.phase_mean)
        if self.phase_normalization and (
            tau_op is None or not np.isfinite(tau_op) or not 0 < tau_op < tau_cl
        ):
            raise ValueError('phase normalization requires 0 < tau_op < tau_cl')

    @staticmethod
    def _rows(observations: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
        rows = np.asarray(observations, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != len(COMMON_FEATURES):
            raise ValueError(
                f"observations must have shape (n,{len(COMMON_FEATURES)}), got {rows.shape}"
            )
        if not np.isfinite(rows).all():
            raise ValueError("observations must be finite")
        return rows

    def update(self, observations: np.ndarray | Iterable[Iterable[float]]) -> None:
        """Accumulate training observations with vector Welford updates."""
        if self.frozen:
            raise RuntimeError("normalizer is frozen; future observations cannot update it")
        idx = np.asarray(self.fit_indices, dtype=int)
        for row in self._coordinates(observations):
            self.count += 1
            delta = row[idx] - self.mean[idx]
            self.mean[idx] += delta / self.count
            delta2 = row[idx] - self.mean[idx]
            self.m2[idx] += delta * delta2
            if self.phase_normalization:
                phase = int(self._phase_indices(row[None])[0])
                self.phase_count[phase] += 1
                delta = row[idx] - self.phase_mean[phase, idx]
                self.phase_mean[phase, idx] += delta / self.phase_count[phase]
                self.phase_m2[phase, idx] += delta * (row[idx]-self.phase_mean[phase, idx])

    def _phase_indices(self, rows):
        # CLOB-inactive auction slopes are zero. In the no-auction treatment
        # the terminal CLOB observation has t=tau_op but still no auction
        # book, and must not require statistics from an unvisited phase.
        return ((rows[:, self.TIME_INDEX] >= self.tau_op)
                & ((rows[:, 13]+rows[:, 15]) > 0)).astype(int)

    def _coordinates(self, observations):
        rows = self._rows(observations).copy()
        if self.relative_prices:
            mid = rows[:, 3].copy()
            rows[:, 2] -= mid
            rows[:, 14] -= mid * rows[:, 13]
            rows[:, 17] -= mid * rows[:, 15]
        if self.auction_exposure_features:
            slope = rows[:, 13]+rows[:, 15]
            active = slope > 0
            displacement = np.divide(rows[:, 14]+rows[:, 17]+rows[:, 16],
                                     slope, out=np.zeros(len(rows)), where=active)
            quantity = rows[:, 13]*displacement-rows[:, 14]
            # Invertible, observable-state coordinates: remaining inventory
            # under the continuous aggregate root and its price displacement.
            # No future price or actual pro-rata allocation is supplied.
            rows[active, 14] = rows[active, 1]-quantity[active]
            rows[active, 17] = displacement[active]
            if self.auction_inventory_asinh:
                rows[active, 1] = np.arcsinh(rows[active, 1]/self.inventory_scale)
                rows[active, 14] = np.arcsinh(rows[active, 14]/self.inventory_scale)
        return rows

    def fit(self, observations: np.ndarray | Iterable[Iterable[float]]) -> "FeatureNormalizer":
        """Fit once on training observations and freeze the resulting state."""
        if self.count or self.frozen:
            raise RuntimeError("fit requires a fresh normalizer")
        self.update(observations)
        return self.freeze()

    def freeze(self) -> "FeatureNormalizer":
        if self.count <= 0:
            raise RuntimeError("cannot freeze without training observations")
        idx = np.asarray(self.fit_indices, dtype=int)
        std = np.sqrt(self.m2[idx] / self.count)
        self.scale[idx] = np.where(std > self.eps, std, 1.0)
        if self.phase_normalization:
            for phase, count in enumerate(self.phase_count):
                if count:
                    std = np.sqrt(self.phase_m2[phase, idx] / count)
                    self.phase_scale[phase, idx] = np.where(std > self.eps, std, 1.0)
        self.frozen = True
        return self

    def transform(self, observations: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
        if not self.frozen:
            raise RuntimeError("freeze the normalizer before transforming observations")
        original = np.asarray(observations)
        rows = self._coordinates(observations)
        phases = self._phase_indices(rows) if self.phase_normalization else None
        rows[:, self.TIME_INDEX] /= self.tau_cl
        rows[:, self.DECISION_INDEX] /= self.tau_cl
        idx = np.asarray(self.fit_indices, dtype=int)
        if self.phase_normalization:
            if np.any(self.phase_count[phases] == 0):
                raise ValueError('phase normalization has no training observations for this phase')
            rows[:, idx] = (rows[:, idx] - self.phase_mean[phases][:, idx]) / self.phase_scale[phases][:, idx]
        else:
            rows[:, idx] = (rows[:, idx] - self.mean[idx]) / self.scale[idx]
        if self.zero_h_cl:
            rows[:, self.H_CL_INDEX] = 0.0
        out = rows.astype(np.float32)
        return out[0] if original.ndim == 1 else out

    def state_dict(self) -> dict[str, Any]:
        state = {
            "version": self.VERSION,
            "tau_cl": self.tau_cl,
            "fit_indices": list(self.fit_indices),
            "eps": self.eps,
            "zero_h_cl": self.zero_h_cl,
            "relative_prices": self.relative_prices,
            "auction_exposure_features": self.auction_exposure_features,
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "scale": self.scale.tolist(),
            "frozen": self.frozen,
            "inventory_reference_times": self.inventory_reference_times,
            "inventory_reference_values": self.inventory_reference_values,
        }
        if self.phase_normalization:
            state.update(phase_normalization=True, tau_op=self.tau_op,
                         phase_count=self.phase_count.tolist(), phase_mean=self.phase_mean.tolist(),
                         phase_m2=self.phase_m2.tolist(), phase_scale=self.phase_scale.tolist())
        if self.auction_inventory_asinh:
            state.update(auction_inventory_asinh=True, inventory_scale=self.inventory_scale)
        return state

    def set_inventory_reference(self, times, values) -> None:
        """Store a frozen, policy-independent training calibration curve."""
        if self.frozen:
            raise RuntimeError('normalizer is frozen; inventory reference cannot change')
        if (times is None) != (values is None):
            raise ValueError('inventory reference requires both times and values')
        if times is None:
            self.inventory_reference_times = self.inventory_reference_values = None
            return
        t, v = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
        if (t.ndim != 1 or v.shape != t.shape or len(t) < 2
                or not np.isfinite(t).all() or not np.isfinite(v).all()
                or t[0] != 0 or t[-1] > self.tau_cl or np.any(np.diff(t) <= 0)
                or np.any(v < 0)):
            raise ValueError('invalid frozen inventory reference curve')
        self.inventory_reference_times, self.inventory_reference_values = t.tolist(), v.tolist()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError(f"unsupported normalizer state version {state.get('version')!r}")
        if float(state["tau_cl"]) != self.tau_cl:
            raise ValueError("normalizer tau_cl mismatch")
        if tuple(state["fit_indices"]) != self.fit_indices:
            raise ValueError("normalizer fit_indices mismatch")
        self.eps = float(state["eps"])
        self.zero_h_cl = bool(state.get("zero_h_cl", False))
        self.relative_prices = bool(state.get("relative_prices", False))
        self.auction_exposure_features = bool(state.get("auction_exposure_features", False))
        self.auction_inventory_asinh = bool(state.get('auction_inventory_asinh', False))
        self.inventory_scale = float(state.get('inventory_scale', 1.))
        if self.auction_inventory_asinh and (
            not self.auction_exposure_features or not np.isfinite(self.inventory_scale) or self.inventory_scale <= 0
        ):
            raise ValueError('invalid auction inventory asinh state')
        self.phase_normalization = bool(state.get('phase_normalization', False))
        self.tau_op = state.get('tau_op')
        if self.phase_normalization:
            if self.tau_op is None or not np.isfinite(self.tau_op) or not 0 < self.tau_op < self.tau_cl:
                raise ValueError('invalid phase-normalization boundary')
            for name in ('phase_count', 'phase_mean', 'phase_m2', 'phase_scale'):
                value = np.asarray(state[name], dtype=float)
                shape = (2,) if name == 'phase_count' else (2, len(COMMON_FEATURES))
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'invalid normalizer {name}')
                if name in ('phase_count', 'phase_m2') and np.any(value < 0):
                    raise ValueError(f'negative normalizer {name}')
                if name == 'phase_count' and np.any(value != np.floor(value)):
                    raise ValueError('noninteger normalizer phase_count')
                if name == 'phase_scale' and np.any(value <= 0):
                    raise ValueError('nonpositive normalizer phase_scale')
                setattr(self, name, value.astype(np.int64) if name == 'phase_count' else value.copy())
        self.count = int(state["count"])
        for name in ("mean", "m2", "scale"):
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != (len(COMMON_FEATURES),):
                raise ValueError(f"normalizer {name} has shape {value.shape}")
            setattr(self, name, value.copy())
        self.frozen = False
        self.set_inventory_reference(state.get('inventory_reference_times'),
                                     state.get('inventory_reference_values'))
        self.frozen = bool(state["frozen"])


class FeatureExtractor:
    """Materialize the common 18-coordinate manuscript feature map.

    The environment accessors implement phase-inactive zeros.  No statistics
    are estimated here: :class:`FeatureNormalizer` is fitted once on the
    training split and applied outside the simulator so validation and test
    observations cannot influence it.
    """

    def __init__(
        self,
        params: FeatureParams,
        grid: GridParams,
        clob_flow: ClobFlowParams,
        auction_flow: AuctionFlowParams,
    ) -> None:
        self.params = params
        self.grid = grid

        # Keep the constructor arguments explicit even though only ``params``
        # drives extraction; this preserves the public factory contract and
        # makes configuration mismatches fail at environment construction.
        del clob_flow, auction_flow
        if tuple(params.clob) != COMMON_FEATURES:
            raise ValueError(
                "features.clob must use the exact common manuscript order: "
                f"{COMMON_FEATURES!r}"
            )
        if tuple(params.auction) != COMMON_FEATURES:
            raise ValueError(
                "features.auction must use the exact common manuscript order: "
                f"{COMMON_FEATURES!r}"
            )

        registry: dict[str, Callable] = {
            "time": lambda env: env.t,
            "inventory": lambda env: env.inventory,
            # The H-off treatments remove the signal from the environment's
            # raw observation itself, not merely in an agent-side normalizer.
            "h_cl": lambda env: env.h_cl if env.cfg.rl.h_cl_feature_enabled else 0.0,
            "s_mid": lambda env: env.s_mid,
            "decision_index": lambda env: env.decision_index,
            "depth_ask": lambda env: env.depth_ask,
            "depth_bid": lambda env: env.depth_bid,
            "top_ask": lambda env: env.top_ask,
            "top_bid": lambda env: env.top_bid,
            "n_mm": lambda env: env.n_mm,
            "n_buy": lambda env: env.n_buy,
            "n_sell": lambda env: env.n_sell,
            "cancel_admissible": lambda env: float(env.cancel_admissible),
            "own_slope": lambda env: env.own_slope,
            "own_weighted_quote": lambda env: env.own_weighted_quote,
            "exogenous_slope": lambda env: env.exogenous_slope,
            "auction_imbalance": lambda env: env.auction_imbalance,
            "exogenous_weighted_quote": lambda env: env.exogenous_weighted_quote,
        }
        self._clob_getters = [registry[name] for name in COMMON_FEATURES]
        self._auction_getters = [registry[name] for name in COMMON_FEATURES]

    def clob_features(self, env) -> np.ndarray:
        """Raw common feature vector at a CLOB decision."""
        return np.array([g(env) for g in self._clob_getters], dtype=np.float32)

    def auction_features(self, env) -> np.ndarray:
        """Raw common feature vector at an auction decision."""
        return np.array([g(env) for g in self._auction_getters], dtype=np.float32)
