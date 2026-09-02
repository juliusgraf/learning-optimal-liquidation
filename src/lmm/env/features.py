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
        self.count = 0
        self.mean = np.zeros(len(COMMON_FEATURES), dtype=np.float64)
        self.m2 = np.zeros(len(COMMON_FEATURES), dtype=np.float64)
        self.scale = np.ones(len(COMMON_FEATURES), dtype=np.float64)
        self.frozen = False

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
        for row in self._rows(observations):
            self.count += 1
            delta = row[idx] - self.mean[idx]
            self.mean[idx] += delta / self.count
            delta2 = row[idx] - self.mean[idx]
            self.m2[idx] += delta * delta2

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
        self.frozen = True
        return self

    def transform(self, observations: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
        if not self.frozen:
            raise RuntimeError("freeze the normalizer before transforming observations")
        original = np.asarray(observations)
        rows = self._rows(observations).copy()
        rows[:, self.TIME_INDEX] /= self.tau_cl
        rows[:, self.DECISION_INDEX] /= self.tau_cl
        idx = np.asarray(self.fit_indices, dtype=int)
        rows[:, idx] = (rows[:, idx] - self.mean[idx]) / self.scale[idx]
        if self.zero_h_cl:
            rows[:, self.H_CL_INDEX] = 0.0
        out = rows.astype(np.float32)
        return out[0] if original.ndim == 1 else out

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "tau_cl": self.tau_cl,
            "fit_indices": list(self.fit_indices),
            "eps": self.eps,
            "zero_h_cl": self.zero_h_cl,
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "scale": self.scale.tolist(),
            "frozen": self.frozen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError(f"unsupported normalizer state version {state.get('version')!r}")
        if float(state["tau_cl"]) != self.tau_cl:
            raise ValueError("normalizer tau_cl mismatch")
        if tuple(state["fit_indices"]) != self.fit_indices:
            raise ValueError("normalizer fit_indices mismatch")
        self.eps = float(state["eps"])
        self.zero_h_cl = bool(state.get("zero_h_cl", False))
        self.count = int(state["count"])
        for name in ("mean", "m2", "scale"):
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != (len(COMMON_FEATURES),):
                raise ValueError(f"normalizer {name} has shape {value.shape}")
            setattr(self, name, value.copy())
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
