"""Mid-price models (paper `sub:rough`, `sub:historical`; rulings D13, D14).

The env (not the model) freezes the mid at tau_op during the auction phase:
``advance_to`` is called only for CLOB decision times and once more for
tau_op itself; the env stores that last value as the frozen auction mid.

RNG discipline (ruling D10): each model consumes draws from the generator
passed to ``reset`` in a fixed per-step order, unconditionally on the agent's
policy, so seeded trajectories are policy-independent (common random
numbers). ``HistoricalMidPrice`` draws nothing.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from lmm.config import (
    ExperimentConfig,
    GridParams,
    HistoricalParams,
    RoughHestonParams,
)

__all__ = [
    "MidPriceModel",
    "RoughHestonMidPrice",
    "HistoricalMidPrice",
    "load_mid_paths",
    "build_midprice",
]


class MidPriceModel(ABC):
    """Common interface for the exogenous mid-price process S^mid."""

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> float:
        """Start a new episode path; returns S^mid_0."""

    @abstractmethod
    def advance_to(self, t: float) -> float:
        """Advance the path to (decision) time ``t`` and return S^mid_t.

        ``t`` is monotone within an episode. Gaussian draws are consumed in a
        path-stable order so seeded trajectories are policy-independent (CRN).
        """


class RoughHestonMidPrice(MidPriceModel):
    """Rough Heston Euler scheme of Richard et al. (`sub:rough`; ruling D14).

    Implements the revised manuscript scheme: kernel
    K(u) = u^(H-1/2)/Gamma(H+1/2), (V)_+ truncation in the variance
    recursion, correlated log-price update (rho dW_v + sqrt(1-rho^2) dW_perp,
    drift -V_+/2), and physical-time scaling ``bar(t)=t/s_star``. The active
    synthetic simulator uses minutes, so ``s_star`` is expressed in trading
    minutes per year; raw minutes are never used as Brownian variances.

    Per ruling D14, performance work only VECTORIZES the same recursion:
    the per-step variance update

        V_k = v0 + sum_{i<k} K(t_k - t_i) * drift_i,
        drift_i = (theta - kappa*(V_i)_+) * (t_{i+1} - t_i)
                  + xi * sqrt((V_i)_+) * dW_i,

    is computed with an incrementally built ``drift`` array and one numpy
    kernel evaluation over the history (``method="vectorized"``, default).
    ``method="naive"`` keeps the verbatim legacy per-step Python loop as the
    reference implementation; ``tests/test_rough_heston.py`` asserts allclose
    on seeded paths. Both methods draw (Z_v, Z_perp) on every update — even
    when (V)_+ = 0 skips the price move — so the stream is stable.
    """

    def __init__(
        self,
        params: RoughHestonParams,
        grid: GridParams,
        method: str = "vectorized",
    ) -> None:
        if method not in ("vectorized", "naive"):
            raise ValueError(f"unknown method {method!r}")
        self.params = params
        self.grid = grid
        self.method = method
        self._kernel_const = 1.0 / math.gamma(params.H + 0.5)
        # Calendar conversion: bar(t)=t/s_star, with t and s_star expressed in
        # the same configured clock unit (minutes for all active experiments).
        self._years_per_grid_unit = 1.0 / params.time_units_per_year
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> float:
        self._rng = rng
        self._grid_t = 0.0
        self.mid = float(self.grid.S0)
        self._Y = math.log(self.mid)
        # History in PHYSICAL YEARS (legacy rh_times/rh_V/rh_dW).
        self._times: list[float] = [0.0]
        self._V: list[float] = [float(self.params.v0)]
        self._dW: list[float] = []
        self._drift: list[float] = []  # drift_i, vectorized path only
        return self.mid

    def advance_to(self, t: float) -> float:
        if self._rng is None:
            raise RuntimeError("reset(rng) must be called before advance_to")
        dt_years = (float(t) - self._grid_t) * self._years_per_grid_unit
        self._grid_t = float(t)
        if dt_years <= 0.0:  # legacy guard; never hit on a strictly increasing grid
            return self.mid
        if self.method == "vectorized":
            self._update_vectorized(dt_years)
        else:
            self._update_naive(dt_years)
        return self.mid

    # -- shared per-update head: draws and the log-price move ----------------

    def _draw_and_move_price(self, dt: float) -> float:
        """Draw (Z_v, Z_perp), apply the correlated log-price update, return dW_v."""
        p = self.params
        sqrt_dt = math.sqrt(dt)
        z_v = float(self._rng.normal())
        z_perp = float(self._rng.normal())
        dw_v = sqrt_dt * z_v
        dw_perp = sqrt_dt * z_perp

        v_prev_pos = max(float(self._V[-1]), 0.0)
        if v_prev_pos > 0.0:
            dw_s = p.rho * dw_v + math.sqrt(max(1.0 - p.rho * p.rho, 0.0)) * dw_perp
            self._Y += -0.5 * v_prev_pos * dt + math.sqrt(v_prev_pos) * dw_s
        return dw_v

    # -- vectorized update (ruling D14: same recursion, numpy over history) --

    def _update_vectorized(self, dt: float) -> None:
        p = self.params
        dw_v = self._draw_and_move_price(dt)
        t_new = self._times[-1] + dt

        # drift_{k-1} becomes computable once t_k is known.
        v_last_pos = max(self._V[-1], 0.0)
        self._drift.append(
            (p.theta - p.kappa * v_last_pos) * dt + p.xi * math.sqrt(v_last_pos) * dw_v
        )
        self._times.append(t_new)
        self._dW.append(dw_v)

        times = np.asarray(self._times[:-1])
        kernel = self._kernel_const * (t_new - times) ** (p.H - 0.5)
        v_new = p.v0 + float(kernel @ np.asarray(self._drift))
        self._V.append(v_new)
        self.mid = float(math.exp(self._Y))

    # -- naive update (verbatim legacy loop; test reference only) ------------

    def _update_naive(self, dt: float) -> None:
        p = self.params
        dw_v = self._draw_and_move_price(dt)
        t_new = self._times[-1] + dt
        self._times.append(t_new)
        self._dW.append(dw_v)

        k = len(self._times) - 1
        v_new = p.v0
        for i in range(k):
            t_i = self._times[i]
            dt_i = self._times[i + 1] - self._times[i]
            u = t_new - t_i
            kernel = self._kernel_const * (u ** (p.H - 0.5)) if u > 0.0 else 0.0
            v_i_pos = max(self._V[i], 0.0)
            v_new += kernel * (p.theta - p.kappa * v_i_pos) * dt_i
            v_new += kernel * p.xi * math.sqrt(v_i_pos) * self._dW[i]
        self._V.append(v_new)
        self.mid = float(math.exp(self._Y))

    @property
    def variance_path(self) -> np.ndarray:
        """The V_t path so far (test/diagnostic accessor)."""
        return np.asarray(self._V)


class HistoricalMidPrice(MidPriceModel):
    """Fixed historical path replay (`sub:historical`; ruling D13).

    Paths are materialized on the one-minute physical grid, including
    ``tau_op``.  Each value is the most recent source observation at or before
    that minute, so a future bar is never used.  ``advance_to(t)`` therefore
    selects ``floor(t)`` and the auction-open call at ``tau_op`` uses the
    observation available at that physical time (not the previous row by
    convention).
    """

    def __init__(self, params: HistoricalParams, grid: GridParams, path: np.ndarray) -> None:
        paths = np.asarray(path, dtype=float)
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        if paths.ndim != 2 or paths.shape[1] == 0:
            raise ValueError("historical mid path is empty")
        if params.path_policy not in ("fixed", "split_pool"):
            raise NotImplementedError(
                f"unsupported historical path_policy={params.path_policy!r}"
            )
        self.params = params
        self.grid = grid
        self.paths = paths
        self.path = self.paths[0]
        self.path_index = 0
        self.mid = float(self.path[0])

    def reset(self, rng: np.random.Generator) -> float:
        if self.params.path_policy == "split_pool" and len(self.paths) > 1:
            self.path_index = int(rng.integers(0, len(self.paths)))
        else:
            self.path_index = 0
        self.path = self.paths[self.path_index]
        self.mid = float(self.path[0])
        return self.mid

    def advance_to(self, t: float) -> float:
        idx = int(math.floor(t))
        idx = max(0, min(idx, len(self.path) - 1))
        self.mid = float(self.path[idx])
        return self.mid


def load_mid_paths(
    params: HistoricalParams,
    repo_root: str | Path = ".",
    split: str | None = None,
) -> dict[str, np.ndarray]:
    """Load normalized per-session source paths for one chronological split.

    This public helper retains the row-array interface used by data-pipeline
    tooling.  Environment construction additionally regularizes each selected
    session onto minutes ``0..tau_op`` with a latest-at-or-before lookup; see
    :func:`_load_regularized_mid_paths`.
    """
    import pandas as pd

    if split is not None and not params.split_id.startswith("legacy_"):
        from lmm.data.load_yfinance_data import validate_historical_artifact

        validate_historical_artifact(
            params,
            repo_root,
            horizon=max(int(params.n_rows) - 1, 0),
        )

    csv_path = Path(repo_root) / params.csv_path
    df = pd.read_csv(csv_path)
    if "Datetime" not in df:
        raise ValueError(f"{csv_path}: missing Datetime column")
    timestamps = pd.to_datetime(df["Datetime"], errors="raise", utc=True)
    try:
        timestamps = timestamps.dt.tz_convert(params.timezone)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{csv_path}: invalid historical timezone {params.timezone!r}"
        ) from exc
    df = (
        df.assign(_timestamp=timestamps)
        .sort_values("_timestamp")
        .reset_index(drop=True)
    )
    if df["_timestamp"].duplicated().any():
        duplicate = df.loc[df["_timestamp"].duplicated(), "_timestamp"].iloc[0]
        raise ValueError(f"{csv_path}: duplicate historical timestamp {duplicate}")
    split_ranges = {
        "train": params.train_date_range,
        "validation": params.validation_date_range,
        "test": params.test_date_range,
    }
    if split is not None and split not in split_ranges:
        raise ValueError(f"historical split must be train|validation|test, got {split!r}")
    selected = df
    if split is not None:
        bounds = tuple(split_ranges[split])
        if bounds:
            if len(bounds) != 2:
                raise ValueError(f"{split}_date_range must contain [start,end]")
            dates = df["_timestamp"].dt.date
            start = pd.Timestamp(bounds[0]).date()
            end = pd.Timestamp(bounds[1]).date()
            if end < start:
                raise ValueError(f"{split} date range ends before it starts")
            selected = df[(dates >= start) & (dates <= end)]
        elif not params.split_id.startswith("legacy_"):
            raise ValueError(
                f"historical split {split!r} has no configured chronological date range"
            )
    if selected.empty:
        raise ValueError(f"historical split {split!r} contains no rows in {csv_path}")

    # Each calendar session is one candidate path.  No row from another split
    # can enter the pool, and each path is normalized independently.
    session_dates = selected["_timestamp"].dt.date
    groups = [group for _, group in selected.groupby(session_dates, sort=True)]
    paths: dict[str, np.ndarray] = {}
    for sym in params.symbols:
        candidates: list[np.ndarray] = []
        for group in groups:
            if sym not in group:
                raise ValueError(f"{csv_path}: missing symbol column {sym!r}")
            series = group[sym].astype(float)
            if params.missing_data_treatment == "ffill":
                series = series.ffill()
            elif params.missing_data_treatment != "error":
                raise ValueError(
                    "missing_data_treatment must be 'error' or 'ffill'"
                )
            raw = series.to_numpy(dtype=float)[: params.n_rows]
            if len(raw) < params.n_rows:
                continue
            if not np.isfinite(raw).all():
                raise ValueError(
                    f"{csv_path}: split={split!r}, symbol={sym!r} contains missing/nonfinite values"
                )
            if raw[0] == 0.0:
                raise ValueError(f"{csv_path}: cannot normalize {sym!r} from zero")
            candidates.append(raw * (params.normalize_first / raw[0]))
        if not candidates:
            raise ValueError(
                f"{csv_path}: split={split!r}, symbol={sym!r} has no session "
                f"with at least n_rows={params.n_rows}"
            )
        paths[sym] = candidates[0] if len(candidates) == 1 else np.stack(candidates)
    return paths


def _load_regularized_mid_paths(
    params: HistoricalParams,
    repo_root: str | Path,
    split: str,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Materialize one-minute paths by latest observation at or before time."""
    import pandas as pd

    if not params.split_id.startswith("legacy_"):
        from lmm.data.load_yfinance_data import validate_historical_artifact

        validate_historical_artifact(params, repo_root, horizon=horizon)

    csv_path = Path(repo_root) / params.csv_path
    df = pd.read_csv(csv_path)
    if "Datetime" not in df:
        raise ValueError(f"{csv_path}: missing Datetime column")
    timestamps = pd.to_datetime(df["Datetime"], errors="raise", utc=True)
    try:
        timestamps = timestamps.dt.tz_convert(params.timezone)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{csv_path}: invalid historical timezone {params.timezone!r}"
        ) from exc
    df = df.assign(_timestamp=timestamps).sort_values("_timestamp")
    if df["_timestamp"].duplicated().any():
        duplicate = df.loc[df["_timestamp"].duplicated(), "_timestamp"].iloc[0]
        raise ValueError(f"{csv_path}: duplicate historical timestamp {duplicate}")

    ranges = {
        "train": params.train_date_range,
        "validation": params.validation_date_range,
        "test": params.test_date_range,
    }
    if split not in ranges:
        raise ValueError(f"historical split must be train|validation|test, got {split!r}")
    bounds = tuple(ranges[split])
    if bounds:
        start, end = (pd.Timestamp(value).date() for value in bounds)
        local_dates = df["_timestamp"].dt.date
        df = df[(local_dates >= start) & (local_dates <= end)]
    elif not params.split_id.startswith("legacy_"):
        raise ValueError(f"historical split {split!r} has no configured date range")
    if df.empty:
        raise ValueError(f"historical split {split!r} contains no rows in {csv_path}")

    local_dates = df["_timestamp"].dt.date
    groups = [group for _, group in df.groupby(local_dates, sort=True)]
    query_minutes = np.arange(int(horizon) + 1, dtype=float)
    paths: dict[str, list[np.ndarray]] = {symbol: [] for symbol in params.symbols}
    for group in groups:
        observed_minutes = (
            (group["_timestamp"] - group["_timestamp"].iloc[0])
            .dt.total_seconds()
            .to_numpy(dtype=float)
            / 60.0
        )
        if observed_minutes[-1] + 1e-12 < horizon:
            continue
        indices = np.searchsorted(observed_minutes, query_minutes, side="right") - 1
        if np.any(indices < 0):
            raise AssertionError("session must contain its own time-zero observation")
        exact_minute = np.isclose(observed_minutes[indices], query_minutes)
        if params.missing_data_treatment == "error" and not bool(np.all(exact_minute)):
            missing = query_minutes[~exact_minute].astype(int).tolist()
            raise ValueError(
                f"{csv_path}: split={split!r} session={group['_timestamp'].iloc[0].date()} "
                f"is missing minute(s) {missing}"
            )
        if params.missing_data_treatment not in ("error", "ffill"):
            raise ValueError("missing_data_treatment must be 'error' or 'ffill'")
        for symbol in params.symbols:
            if symbol not in group:
                raise ValueError(f"{csv_path}: missing symbol column {symbol!r}")
            raw = group[symbol].to_numpy(dtype=float)
            selected = raw[indices]
            if not np.isfinite(selected).all():
                raise ValueError(
                    f"{csv_path}: split={split!r}, symbol={symbol!r} contains missing/nonfinite values"
                )
            if selected[0] == 0.0:
                raise ValueError(f"{csv_path}: cannot normalize {symbol!r} from zero")
            paths[symbol].append(
                selected * (float(params.normalize_first) / float(selected[0]))
            )

    out: dict[str, np.ndarray] = {}
    for symbol, candidates in paths.items():
        if not candidates:
            raise ValueError(
                f"{csv_path}: split={split!r}, symbol={symbol!r} has no session "
                f"covering minute {horizon}"
            )
        out[symbol] = candidates[0] if len(candidates) == 1 else np.stack(candidates)
    return out


def build_midprice(
    cfg: ExperimentConfig,
    symbol: str | None = None,
    repo_root: str | Path = ".",
    data_split: str = "train",
) -> MidPriceModel:
    """Factory: the configured mid-price model for one experiment/symbol."""
    mp = cfg.midprice
    if mp.model == "rough_heston":
        if mp.rough_heston is None:
            raise ValueError("midprice.model=rough_heston but midprice.rough_heston missing")
        return RoughHestonMidPrice(mp.rough_heston, cfg.grid)
    if mp.model == "historical":
        if mp.historical is None:
            raise ValueError("midprice.model=historical but midprice.historical missing")
        sym = symbol if symbol is not None else mp.historical.symbols[0]
        paths = _load_regularized_mid_paths(
            mp.historical,
            repo_root=repo_root,
            split=data_split,
            horizon=cfg.grid.tau_op,
        )
        if sym not in paths:
            raise KeyError(f"symbol {sym!r} not in {list(paths)}")
        return HistoricalMidPrice(mp.historical, cfg.grid, paths[sym])
    raise ValueError(f"unknown midprice.model {mp.model!r}")
