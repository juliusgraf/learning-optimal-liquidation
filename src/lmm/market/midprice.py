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

    Kept EXACTLY as legacy ``main.py:198-250``: kernel
    K(u) = u^(H-1/2)/Gamma(H+1/2), (V)_+ truncation in the variance
    recursion, correlated log-price update (rho dW_v + sqrt(1-rho^2) dW_perp,
    drift -V_+/2), physical-time scaling
    dt_years = dt_grid * (T_physical/tau_cl) / seconds_per_year (one grid
    unit = T_physical/tau_cl seconds; legacy ``main.py:25-33, 557-566``).

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
        # Grid time -> physical years: one grid unit = T_physical/tau_cl
        # seconds (legacy dt = T/tau_cl), then seconds -> years.
        self._years_per_grid_unit = (
            grid.T_physical / float(grid.tau_cl)
        ) / params.seconds_per_year
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

    Row r of the (pre-normalized) path is the mid at decision time t = r,
    r in [0, tau_op): ``advance_to(t)`` returns ``path[clip(floor(t), 0,
    len(path)-1)]`` (legacy ``data.py:137-142``). The same realized path is
    reused for every episode; ``reset`` consumes no random draws. The frozen
    auction mid (the env calls ``advance_to(tau_op)``) is row tau_op - 1 =
    119, matching the legacy episode construction pinned in AUDIT A.12.
    """

    def __init__(self, params: HistoricalParams, grid: GridParams, path: np.ndarray) -> None:
        if len(path) == 0:
            raise ValueError("historical mid path is empty")
        self.params = params
        self.grid = grid
        self.path = np.asarray(path, dtype=float)
        self.mid = float(self.path[0])

    def reset(self, rng: np.random.Generator) -> float:  # rng unused (D13)
        self.mid = float(self.path[0])
        return self.mid

    def advance_to(self, t: float) -> float:
        idx = int(math.floor(t))
        idx = max(0, min(idx, len(self.path) - 1))
        self.mid = float(self.path[idx])
        return self.mid


def load_mid_paths(params: HistoricalParams, repo_root: str | Path = ".") -> dict[str, np.ndarray]:
    """Load the per-symbol historical mid paths (ruling D13).

    Reproduces the legacy construction (``data.py:1654-1666``): read the CSV,
    sort by ``Datetime``, take rows ``0..n_rows-1`` of each requested symbol,
    and normalize so the first row equals ``normalize_first`` (the
    ``--normalize first=100`` rule; identity on the pre-normalized committed
    ``legacy/data.csv``).
    """
    import pandas as pd

    csv_path = Path(repo_root) / params.csv_path
    df = pd.read_csv(csv_path, parse_dates=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    paths: dict[str, np.ndarray] = {}
    for sym in params.symbols:
        raw = df[sym].to_numpy(dtype=float)[: params.n_rows]
        if len(raw) < params.n_rows:
            raise ValueError(
                f"{csv_path}: symbol {sym!r} has {len(raw)} rows < n_rows={params.n_rows}"
            )
        paths[sym] = raw * (params.normalize_first / raw[0])
    return paths


def build_midprice(
    cfg: ExperimentConfig,
    symbol: str | None = None,
    repo_root: str | Path = ".",
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
        paths = load_mid_paths(mp.historical, repo_root=repo_root)
        if sym not in paths:
            raise KeyError(f"symbol {sym!r} not in {list(paths)}")
        return HistoricalMidPrice(mp.historical, cfg.grid, paths[sym])
    raise ValueError(f"unknown midprice.model {mp.model!r}")
