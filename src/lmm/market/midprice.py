"""Mid-price models (paper `sub:rough`, `sub:historical`; rulings D13, D14).

Phase 3 fills in the bodies. The env (not the model) freezes the mid at
tau_op during the auction phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lmm.config import GridParams, HistoricalParams, RoughHestonParams

__all__ = ["MidPriceModel", "RoughHestonMidPrice", "HistoricalMidPrice"]


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

    Kept EXACTLY as legacy `main.py:198-250`: kernel
    K(u) = u^(H-1/2)/Gamma(H+1/2), (V)_+ truncation in the variance
    recursion, correlated log-price update (rho dW_v + sqrt(1-rho^2) dW_perp),
    physical-time scaling dt_phys = grid-dt * (T/tau_cl) / seconds_per_year.
    Performance work may only VECTORIZE the same recursion (precomputed
    kernel weights), validated allclose against the naive loop.
    """

    def __init__(self, params: RoughHestonParams, grid: GridParams) -> None:
        self.params = params
        self.grid = grid

    def reset(self, rng: np.random.Generator) -> float:
        raise NotImplementedError("Phase 3")

    def advance_to(self, t: float) -> float:
        raise NotImplementedError("Phase 3")


class HistoricalMidPrice(MidPriceModel):
    """Fixed historical path replay (`sub:historical`; ruling D13).

    Row r of the (pre-normalized) path is the mid at decision time t = r,
    r in [0, tau_op); the same path is reused for every episode. The env
    freezes the mid at row tau_op - 1 during the auction.
    """

    def __init__(self, params: HistoricalParams, grid: GridParams, path: np.ndarray) -> None:
        self.params = params
        self.grid = grid
        self.path = path  # shape (tau_op,), e.g. data.csv rows 0..119 of one symbol

    def reset(self, rng: np.random.Generator) -> float:
        raise NotImplementedError("Phase 3")

    def advance_to(self, t: float) -> float:
        raise NotImplementedError("Phase 3")
