"""Pruned RL feature vectors (ruling D11; exact legacy lists, AUDIT A.7).

CLOB (8 dims; legacy `feat_clob` main.py:1443-1450):
    [X1/I_max, X3 (H_cl, raw), X10 (S_mid, raw), X4/Lc, X5/Lc,
     X13[0]/V_max, X14[0]/V_max, t_norm],  t_norm = clip(t/(tau_op-1), 0, 1).

Auction (7 dims; legacy `feat_auction` main.py:1452-1457):
    [X1/I_max, X3 (raw), X10 (raw), X6/La, X7/L_max, X8/L_max, t_norm],
    t_norm = clip((t-tau_op)/(tau_cl-tau_op), 0, 1).

Note (AUDIT A.7): legacy includes neither Z nor a theta summary in the
auction features; the lists above are the configurable defaults. X7/X8 use
the PAPER convention here (X7 = buy-MO count; naming fixed per D3).

Phase 3 fills in the bodies.
"""

from __future__ import annotations

import numpy as np

from lmm.config import FeatureParams, GridParams

__all__ = ["FeatureExtractor", "CLOB_FEATURE_DIM", "AUCTION_FEATURE_DIM"]

CLOB_FEATURE_DIM = 8
AUCTION_FEATURE_DIM = 7


class FeatureExtractor:
    """Maps env accessors to the configured (default: legacy-pruned) features."""

    def __init__(self, params: FeatureParams, grid: GridParams) -> None:
        self.params = params
        self.grid = grid

    def clob_features(self, env) -> np.ndarray:
        """Time-augmented CLOB feature vector (default 8 dims)."""
        raise NotImplementedError("Phase 3")

    def auction_features(self, env) -> np.ndarray:
        """Time-augmented auction feature vector (default 7 dims)."""
        raise NotImplementedError("Phase 3")
