"""Pruned RL feature vectors (ruling D11; exact legacy lists, AUDIT A.7).

CLOB (8 dims; legacy `feat_clob` main.py:1443-1450):
    [X1/I_max, X3 (H_cl, raw), X10 (S_mid, raw), X4/Lc, X5/Lc,
     X13[0]/V_max, X14[0]/V_max, t_norm],  t_norm = clip(t/(tau_op-1), 0, 1).

Auction (7 dims; legacy `feat_auction` main.py:1452-1457):
    [X1/I_max, X3 (raw), X10 (raw), X6/La, X7/L_max, X8/L_max, t_norm],
    t_norm = clip((t-tau_op)/(tau_cl-tau_op), 0, 1).

Note (AUDIT A.7): legacy includes neither Z nor a theta summary in the
auction features; the lists above are the configurable defaults. X7/X8 use
the PAPER convention here (X7 = n_buy = BUY-market-order count; ruling D3 —
legacy's X7 counted sells, so the buy/sell feature order is swapped relative
to legacy's raw vector; pure relabeling).

The extractor reads lightweight env accessors only (never `paper_state()`,
which is test/documentation-only per ruling D11).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from lmm.config import ClobFlowParams, FeatureParams, GridParams
from lmm.config import AuctionFlowParams

__all__ = ["FeatureExtractor", "CLOB_FEATURE_DIM", "AUCTION_FEATURE_DIM"]

CLOB_FEATURE_DIM = 8
AUCTION_FEATURE_DIM = 7


class FeatureExtractor:
    """Maps env accessors to the configured (default: legacy-pruned) features.

    Each feature name resolves to a getter ``env -> float``; the configured
    name lists (``features.clob`` / ``features.auction``) select and order
    them. Unknown names raise at construction, not in the hot path.
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

        tau_op, tau_cl = grid.tau_op, grid.tau_cl
        I_max, Lc, V_max = grid.I_max, clob_flow.Lc, clob_flow.V_max
        La, L_max = auction_flow.La, auction_flow.L_max

        # Affine price normalization for h_cl_norm / s_mid_norm (continuous
        # agents): centered at the initial mid S0, scaled, then clipped. The
        # legacy raw h_cl / s_mid getters are untouched.
        S0 = grid.S0
        p_scale = max(float(params.price_norm_scale), 1e-12)
        p_clip = float(params.price_norm_clip)

        def price_norm(getter: Callable) -> Callable:
            return lambda env: float(np.clip((getter(env) - S0) / p_scale, -p_clip, p_clip))

        registry: dict[str, Callable] = {
            "inv_norm": lambda env: env.inventory / max(1.0, I_max),
            "h_cl": lambda env: env.h_cl,  # raw (~100), as legacy
            "s_mid": lambda env: env.s_mid,  # raw (~100), as legacy
            "h_cl_norm": price_norm(lambda env: env.h_cl),  # centered at S0, clipped
            "s_mid_norm": price_norm(lambda env: env.s_mid),
            "depth_ask_norm": lambda env: env.depth_ask / max(1, Lc),
            "depth_bid_norm": lambda env: env.depth_bid / max(1, Lc),
            "top_ask_norm": lambda env: env.top_ask / V_max,
            "top_bid_norm": lambda env: env.top_bid / V_max,
            "n_mm_norm": lambda env: env.n_mm / max(1, La),
            "n_buy_norm": lambda env: env.n_buy / max(1, L_max),
            "n_sell_norm": lambda env: env.n_sell / max(1, L_max),
            # Phase-specific time normalizations (legacy main.py:1445, 1454).
            "t_norm": None,  # placeholder; resolved per phase below
        }

        def t_norm_clob(env) -> float:
            return float(np.clip(env.t / max(1.0, tau_op - 1), 0.0, 1.0))

        def t_norm_auction(env) -> float:
            return float(np.clip((env.t - tau_op) / max(1.0, tau_cl - tau_op), 0.0, 1.0))

        def build(names: tuple[str, ...], t_norm: Callable) -> list[Callable]:
            getters = []
            for name in names:
                getter = t_norm if name == "t_norm" else registry.get(name)
                if getter is None:
                    raise ValueError(f"unknown feature name {name!r}")
                getters.append(getter)
            return getters

        self._clob_getters = build(params.clob, t_norm_clob)
        self._auction_getters = build(params.auction, t_norm_auction)

    def clob_features(self, env) -> np.ndarray:
        """Time-augmented CLOB feature vector (default 8 dims)."""
        return np.array([g(env) for g in self._clob_getters], dtype=np.float32)

    def auction_features(self, env) -> np.ndarray:
        """Time-augmented auction feature vector (default 7 dims)."""
        return np.array([g(env) for g in self._auction_getters], dtype=np.float32)
