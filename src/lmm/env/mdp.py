"""MarketMakingEnv: the paper's MDP (`sec:MDP`) as a gymnasium-style env.

Timing (CLAUDE.md grid; fixes AUDIT N1): CLOB decisions at t in {0,...,n},
auction decisions at t in {n+1,...,m}; t = tau_cl is terminal (no action,
clearing + terminal reward only). The terminal reward is folded into the
final t_m transition with done=True and zero bootstrap (documented
equivalence, ruling D9).

Rewards: corrected three-regime definitions —
- CLOB (ruling D5): r_t = S*_t E_t f_c(k* alpha - (H_cl_t - S*_t)) with
  f_c(u) = (u)_+/(k* alpha), NO clamp of the multiplier at 1; H_cl_t is the
  LAGGED Algorithm-1 value (D2).
- Auction (rulings D1, D4): r_t = K^a_t H_cl_t (H_cl_t - S^a_t)
  + f_a(...) - d_t c_t with f_a(u) = -q(-u)_+, d_t = (t-n-1) d, scalar
  c_t in {0,1}; H_cl_t read from the end-of-(t-1) Eq. (2) cache (D1).
- Terminal (rulings D3, D8): Z_{tau_cl}, I_{tau_cl} = I_{tau_op} - Z_{tau_cl}
  (inventory frozen during the auction), r_{tau_cl} with NO inventory
  clipping (optional `numerical_guard`, default OFF).

State (ruling D11): EFFICIENT internal representation (inventory, cached
H_cl, book arrays, auction ledgers, theta, counters) + a `paper_state()`
accessor materializing X^1..X^17 for tests/documentation only (never in the
training hot path). Naming follows the paper sign convention (D3): X^7 = N^+
counts BUYING market orders.

Phase 3 fills in the bodies.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from lmm.config import ExperimentConfig
from lmm.market.generator import MarketGenerator
from lmm.market.midprice import MidPriceModel

__all__ = ["MarketMakingEnv"]


class MarketMakingEnv(gymnasium.Env):
    """The market-making MDP with CLOB and auction phases.

    Action encoding (see `lmm.env.action_spaces`): a discrete index into the
    phase grid; admissibility (Adm(x), CLAUDE.md) is exposed via
    ``action_mask()`` and enforced by agents through masking (not projection;
    AUDIT N12).

    ``info`` diagnostics per step: ``E_t``, ``H_cl``, ``S_cl`` (terminal),
    ``theta``, executed/cancel flags, degenerate-fallback indicators (D17).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: ExperimentConfig, midprice: MidPriceModel) -> None:
        super().__init__()
        self.cfg = cfg
        self.midprice = midprice
        self.generator = MarketGenerator(cfg.clob_flow, cfg.auction_flow, cfg.grid)

    # -- gymnasium API ------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode; seeds the env's PRIVATE generator only (D10)."""
        raise NotImplementedError("Phase 3")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """One decision step; the final auction step folds in the terminal
        clearing reward and returns terminated=True."""
        raise NotImplementedError("Phase 3")

    # -- accessors (D11) ----------------------------------------------------

    @property
    def phase(self) -> str:
        """'clob' for t <= n, 'auction' for n+1 <= t <= m."""
        raise NotImplementedError("Phase 3")

    @property
    def inventory(self) -> float:
        """X^1_t = I_t."""
        raise NotImplementedError("Phase 3")

    @property
    def h_cl(self) -> float:
        """X^3_t: the CACHED hypothetical clearing price (D1/D2 vintage)."""
        raise NotImplementedError("Phase 3")

    def action_mask(self) -> np.ndarray:
        """Boolean admissibility mask over the current phase's action grid:
        a^1 <= x^1, a^2 >= x^10/alpha (structural), a^5 <= C(x)."""
        raise NotImplementedError("Phase 3")

    def paper_state(self) -> dict[str, np.ndarray]:
        """Materialize the paper state X^1..X^17 (correctly shaped,
        zero-padded vectors; fixes AUDIT N5). Tests/documentation only —
        never called in the training hot path (D11)."""
        raise NotImplementedError("Phase 3")
