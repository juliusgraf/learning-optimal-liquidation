"""MarketMakingEnv: the paper's MDP (`sec:MDP`) as a gymnasium-style env.

TIMING CONTRACT (CLAUDE.md grid + rulings D1/D2; enforced by
tests/test_timing_conventions.py)
================================================================------------

Grid: 0 = t_0 < ... < t_n < tau_op = t_{n+1} < ... < t_m < tau_cl = t_{m+1},
n = tau_op - 1, m = tau_cl - 1. CLOB decisions at t in {0,...,n} (the grid
always reaches t_n exactly — fixes AUDIT N1), auction decisions at
t in {n+1,...,m}; t = tau_cl is terminal (no action; clearing + terminal
reward only, folded into the final t_m transition with done=True and zero
bootstrap — documented equivalence, ruling D9).

At decision time t the agent SEES the state before the step-t randomness:
X^7 = N^+_{t-}, X^8 = N^-_{t-} (counts as of end of t-1), X^9 = theta_t
(predictable: embeds c_{t-1}), and X^3 = H^cl_t — the value CACHED at the
end of step t-1. Nothing sampled or decided at t enters the time-t state or
reward. One cached H drives both:

- CLOB step t (ruling D2): place the agent's order -> realize the step's
  exogenous flow (executions E_t) -> Algorithm-1 update over the post-flow
  standing book (incl. the agent's unexecuted remainder, BEFORE the next
  refresh) -> reward from the OLD cache -> cache <- the new H (input to the
  time-(t+1) state/reward). H_0 = the configured initial value (= initial
  mid); the reward at t = 0 uses H_0.
- Auction step t (rulings D1, D3): reward from the cache -> sample the
  step-t exogenous events -> record the agent's time-t order (K^a = 0 ==
  abstain) -> apply c_t (kills orders submitted < t, i.e. theta_{t+1}) ->
  recompute corrected Eq. (2) on end-of-t information -> cache <- the root.
  At t_{n+1} the cache holds Algorithm 1's last CLOB output.
- At t = t_m the same end-of-step recompute IS corrected Eq. (1) (setting
  j = m+1 recovers it exactly; theta_{t_{m+1}} embeds c_{t_m}): S_cl := it,
  then Z_{tau_cl}, I_{tau_cl} = I_{tau_op} - Z_{tau_cl} (inventory frozen
  during the auction; NO clipping, ruling D8 — optional `numerical_guard`
  config, default OFF, logged loudly if it ever binds) and the terminal
  reward.

Rewards (corrected three-regime definitions; see env/rewards.py):
- CLOB (ruling D5): r_t = S*_t E_t f_c(k* alpha - (H^cl_t - S*_t)) with
  f_c(u) = (u)_+/(k* alpha), NO clamp of the multiplier at 1.
- Auction (rulings D1, D4): r_t = K^a_t H^cl_t (H^cl_t - S^a_t) + f_a(...)
  - d_t c_t with f_a(u) = -q(-u)_+, d_t = (t - n - 1) d, scalar c_t in {0,1}.
- Terminal (rulings D3, D8): see env/rewards.terminal_reward.

State (ruling D11): EFFICIENT internal representation (inventory, cached
H_cl, book arrays, auction ledgers, theta, counters) + a `paper_state()`
accessor materializing X^1..X^17 for tests/documentation only (never in the
training hot path). Naming follows the paper sign convention (D3): X^7 = N^+
counts BUYING market orders.

RNG (ruling D10): `reset(seed)` seeds only the env's PRIVATE generator; all
draws occur unconditionally or on exogenous-only conditions, so the event
stream is policy-independent (common random numbers across policies). Fixed
per-step draw order — reset: book refresh (2 Beta); CLOB step: tau^+, tau^-
(2 Exp), then (volume, next-interarrival) per processed order, then refresh
(2 Beta; skipped after the t_n step) and the mid-price update (2 Normal for
rough Heston, 0 historical); auction step: see market/auction.py.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import gymnasium
import numpy as np

from lmm.config import ExperimentConfig
from lmm.env.action_spaces import (
    AuctionAction,
    AuctionActionGrid,
    ClobAction,
    ClobActionGrid,
)
from lmm.env.features import FeatureExtractor
from lmm.env.rewards import auction_reward, clob_reward, terminal_reward
from lmm.market.auction import AgentOrderLedger
from lmm.market.clearing import Algo1Estimator, ClearingInputs, Eq2Cache
from lmm.market.generator import MarketGenerator
from lmm.market.midprice import MidPriceModel, build_midprice

__all__ = ["MarketMakingEnv", "make_env"]

logger = logging.getLogger(__name__)


class MarketMakingEnv(gymnasium.Env):
    """The market-making MDP with CLOB and auction phases.

    Actions: an integer index into the CURRENT phase's discrete grid
    (`lmm.env.action_spaces`), or a decoded :class:`ClobAction` /
    :class:`AuctionAction` instance (used by benchmarks and the Phase-5
    continuous relaxation, whose values may be off-grid). Admissibility
    Adm(x) is exposed via ``action_mask()`` for agent-side masking (AUDIT
    N12: stored action == executed action) and ENFORCED here: inadmissible
    submissions raise ValueError (the env never projects).

    ``action_space``/``observation_space`` are PER-PHASE and re-assigned at
    the auction open (documented deviation from static gymnasium spaces; the
    two-network RL design consumes per-phase dimensions anyway).

    ``info`` diagnostics per step: ``t``, ``phase``, ``H_used`` (the H^cl in
    the time-t state/reward), ``H_next`` (the freshly cached value),
    ``E_t``/``S_bullet`` (CLOB), exogenous-event flags and ``d_t`` (auction),
    degenerate-fallback counter (D17), and at the terminal ``S_cl``, ``Z``,
    ``I_final``, ``terminal_reward``.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: ExperimentConfig, midprice: MidPriceModel) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.grid
        self.midprice = midprice
        self.generator = MarketGenerator(cfg.clob_flow, cfg.auction_flow, cfg.grid)
        self.algo1 = Algo1Estimator(cfg.algo1, cfg.grid)
        self.eq2 = Eq2Cache(cfg.grid)
        self.features = FeatureExtractor(cfg.features, cfg.grid, cfg.clob_flow, cfg.auction_flow)
        self.clob_grid = ClobActionGrid(cfg.actions)
        self.auction_grid = AuctionActionGrid(cfg.actions)
        self._ledger = AgentOrderLedger(cfg.grid)

        self._clob_space = gymnasium.spaces.Discrete(len(self.clob_grid))
        self._auction_space = gymnasium.spaces.Discrete(len(self.auction_grid))
        self._clob_obs_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(len(cfg.features.clob),), dtype=np.float32
        )
        self._auction_obs_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(len(cfg.features.auction),), dtype=np.float32
        )
        self.action_space = self._clob_space
        self.observation_space = self._clob_obs_space

        self._n = cfg.grid.tau_op - 1  # n = tau_op - 1
        self._m = cfg.grid.tau_cl - 1  # m = tau_cl - 1
        self._done = True  # reset() required before step()

    # -- gymnasium API ------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode; seeds the env's PRIVATE generator only (D10)."""
        super().reset(seed=seed)
        g = self.grid

        self._t: float = 0.0
        self._phase: str = "clob"
        self._done = False
        self._inventory: float = float(g.I0)
        self._mid: float = self.midprice.reset(self.np_random)
        self._frozen_mid: float | None = None
        self._I_tau_op: float | None = None
        self._Z: float = 0.0
        self._S_cl: float | None = None

        h0 = self._mid if self.cfg.algo1.H0_from_mid else self.cfg.algo1.H0
        if h0 is None:
            raise ValueError("algo1.H0 must be set when H0_from_mid is false")
        self.algo1.reset(float(h0))
        self._h_cache: float = self.algo1.h
        self.eq2.reset(self._h_cache)

        self.generator.book.refresh(self.np_random)
        self.generator.book.k_mid = self._k_mid()
        self.generator.auction_flow.reset(self._mid)
        self._ledger.reset()

        self.action_space = self._clob_space
        self.observation_space = self._clob_obs_space

        info = {"t": self._t, "phase": self._phase, "H_used": self._h_cache}
        return self.features.clob_features(self), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """One decision step; the final auction step folds in the terminal
        clearing reward and returns terminated=True."""
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        if self._phase == "clob":
            return self._step_clob(action)
        return self._step_auction(action)

    # -- CLOB step (ruling D2; sequencing per the module docstring) ----------

    def _step_clob(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = self._decode_clob(action)
        if a.volume < 0.0:
            raise ValueError(f"volume must be >= 0, got {a.volume}")
        if a.volume > self._inventory:
            raise ValueError(
                f"inadmissible action: volume {a.volume} > inventory {self._inventory} (Adm: a^1 <= x^1)"
            )
        if a.delta < 0:
            raise ValueError(f"delta must be >= 0, got {a.delta} (Adm: a^2 >= x^10/alpha)")

        t = self._t
        book = self.generator.book
        k_mid = self._k_mid()
        book.k_mid = k_mid
        s_bullet = self.grid.alpha * (k_mid + a.delta)
        book.place_agent_order(a.volume, a.delta)

        is_final = t >= float(self._n)
        flow = self.generator.step_clob(self.np_random, t, final=is_final)
        executed = flow.executed_agent
        self._inventory -= executed

        # End-of-step Algorithm-1 update (D2): post-flow book incl. the
        # agent's remainder, BEFORE the refresh. Output = H_{t+1}.
        h_next = self.algo1.update(book.snapshot())
        h_used = self._h_cache
        reward = clob_reward(s_bullet, executed, h_used, self.cfg.reward.k_star, self.grid.alpha)
        self._h_cache = h_next
        book.clear_agent_order()

        info: dict[str, Any] = {
            "t": t,
            "phase": "clob",
            "action": a,
            "H_used": h_used,
            "H_next": h_next,
            "E_t": executed,
            "S_bullet": s_bullet,
            "n_buy_step": flow.n_buy,
            "n_sell_step": flow.n_sell,
            "t_next": flow.t_next,
        }

        if is_final:
            # Auction open: the mid advances to tau_op and FREEZES there
            # (Algorithm 2: S^i ~ S^mid_{tau_op} + ...); no book refresh —
            # the CLOB ceases to exist. Eq. (2) cache seeded with Algorithm
            # 1's last CLOB output (D1).
            self._t = float(self.grid.tau_op)
            self._phase = "auction"
            self._frozen_mid = self.midprice.advance_to(float(self.grid.tau_op))
            self._mid = self._frozen_mid
            self._I_tau_op = self._inventory
            self.generator.auction_flow.reset(self._frozen_mid)
            self._ledger.reset()
            self.eq2.reset(self._h_cache)
            self.action_space = self._auction_space
            self.observation_space = self._auction_obs_space
            obs = self.features.auction_features(self)
        else:
            self._t = flow.t_next
            book.refresh(self.np_random)
            self._mid = self.midprice.advance_to(self._t)
            book.k_mid = self._k_mid()
            obs = self.features.clob_features(self)

        return obs, reward, False, False, info

    # -- auction step (rulings D1, D3, D4; sequencing per the docstring) -----

    def _step_auction(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = self._decode_auction(action)
        if a.K_a < 0.0:
            raise ValueError(f"inadmissible action: K^a must be >= 0, got {a.K_a}")
        if a.cancel not in (0, 1):
            raise ValueError(f"c_t must be 0 or 1, got {a.cancel}")
        if a.cancel == 1 and not self._ledger.cancel_admissible():
            raise ValueError(
                "inadmissible action: cancel-all with no live prior K^a > 0 order (Adm: a^5 <= C(x))"
            )

        t = int(self._t)
        g = self.grid
        k_mid_frozen = int(math.floor(self._frozen_mid / g.alpha))
        s_a = g.alpha * (k_mid_frozen + a.offset)  # tick-snapped quote (AUDIT N4)
        d_t = (t - g.tau_op) * self.cfg.reward.d  # d_t = (t - n - 1) d (D4)

        # Reward FIRST, from the end-of-(t-1) cache (D1): nothing sampled or
        # decided at t may enter it.
        h_used = self._h_cache
        reward = auction_reward(a.K_a, s_a, h_used, self.cfg.reward.q, d_t, a.cancel)

        events = self.generator.step_auction(self.np_random)
        self._ledger.submit(t, a.K_a, s_a)
        if a.cancel == 1:
            self._ledger.apply_cancel_all(t)  # theta_{t+1}: kills orders < t

        # End-of-step corrected Eq. (2) recompute -> the time-(t+1) cache.
        n_degen_before = self.eq2.n_degenerate_fallbacks
        h_next = self.eq2.recompute(self._clearing_inputs())
        self._h_cache = h_next

        info: dict[str, Any] = {
            "t": float(t),
            "phase": "auction",
            "action": a,
            "H_used": h_used,
            "H_next": h_next,
            "S_a": s_a,
            "d_t": d_t,
            "events": events,
            "degenerate_fallback": self.eq2.n_degenerate_fallbacks > n_degen_before,
        }

        terminated = t == self._m
        if terminated:
            reward += self._terminal(info)
            self._t = float(g.tau_cl)
            self._done = True
        else:
            self._t = float(t + 1)

        return self.features.auction_features(self), reward, terminated, False, info

    def _terminal(self, info: dict[str, Any]) -> float:
        """Terminal clearing at tau_cl (corrected Eq. (1) = the end-of-t_m
        Eq. (2) recompute; rulings D3, D8, D17)."""
        r = self.cfg.reward
        s_cl = self._h_cache
        K_live, S_live = self._ledger.live_orders()
        z = float(np.sum(K_live * (s_cl - S_live)))
        i_final = self._I_tau_op - z

        if r.numerical_guard and abs(i_final) > r.numerical_guard_bound:
            logger.error(
                "numerical_guard BOUND (|I_final| = %.6g > %.6g): clipping. "
                "This should never happen on standard runs (ruling D8).",
                abs(i_final),
                r.numerical_guard_bound,
            )
            i_final = float(np.clip(i_final, -r.numerical_guard_bound, r.numerical_guard_bound))

        r_term = terminal_reward(K_live, S_live, s_cl, i_final, r.lambda_inv, r.q)
        self._S_cl = s_cl
        self._Z = z
        self._inventory = i_final
        info.update(S_cl=s_cl, Z=z, I_final=i_final, terminal_reward=r_term)
        return r_term

    # -- decoding / admissibility --------------------------------------------

    def _decode_clob(self, action) -> ClobAction:
        if isinstance(action, ClobAction):
            return action
        if isinstance(action, (int, np.integer)):
            return self.clob_grid.decode(int(action))
        raise TypeError(f"CLOB action must be an index or ClobAction, got {type(action).__name__}")

    def _decode_auction(self, action) -> AuctionAction:
        if isinstance(action, AuctionAction):
            return action
        if isinstance(action, (int, np.integer)):
            return self.auction_grid.decode(int(action))
        raise TypeError(
            f"auction action must be an index or AuctionAction, got {type(action).__name__}"
        )

    def action_mask(self) -> np.ndarray:
        """Boolean admissibility mask over the current phase's action grid:
        a^1 <= x^1, a^2 >= x^10/alpha (structural), a^5 <= C(x).

        The cancel-admissibility comes from the internal ledger's liveness
        flags — equivalent to evaluating C on ``paper_state()`` (asserted in
        tests/test_admissibility.py)."""
        if self._phase == "clob":
            return self.clob_grid.mask(self._inventory)
        return self.auction_grid.mask(self._ledger.cancel_admissible())

    def _clearing_inputs(self) -> ClearingInputs:
        flow = self.generator.auction_flow
        K_exo, S_exo = flow.supply_curves()
        K_agent, S_agent = self._ledger.live_orders()
        return ClearingInputs(
            K_exo=K_exo,
            S_exo=S_exo,
            K_agent=K_agent,
            S_agent=S_agent,
            net_market_volume=flow.net_market_volume(),
            fallback_mid=self._frozen_mid,  # D17: S^mid (frozen at tau_op)
        )

    def _k_mid(self) -> int:
        """Mid tick floor(S^mid/alpha) — ONE rounding convention (AUDIT N3)."""
        return int(math.floor(self._mid / self.grid.alpha))

    # -- lightweight accessors (D11; consumed by FeatureExtractor) -----------

    @property
    def t(self) -> float:
        """Current decision time (tau_cl after the terminal transition)."""
        return self._t

    @property
    def phase(self) -> str:
        """'clob' for t <= n, 'auction' for n+1 <= t (incl. the terminal)."""
        return self._phase

    @property
    def inventory(self) -> float:
        """X^1_t = I_t."""
        return self._inventory

    @property
    def h_cl(self) -> float:
        """X^3_t: the CACHED hypothetical clearing price (D1/D2 vintage)."""
        return self._h_cache

    @property
    def s_mid(self) -> float:
        """X^10_t = S^mid_t (frozen at tau_op during the auction)."""
        return self._mid

    @property
    def depth_ask(self) -> int:
        """X^4_t = L^+_t (0 in the auction phase)."""
        return self.generator.book.depth(+1) if self._phase == "clob" else 0

    @property
    def depth_bid(self) -> int:
        """X^5_t = L^-_t (0 in the auction phase)."""
        return self.generator.book.depth(-1) if self._phase == "clob" else 0

    @property
    def top_ask(self) -> float:
        """X^13_t[0] = V^{+,1}_t (0 in the auction phase)."""
        return float(self.generator.book.ask_volumes[0]) if self._phase == "clob" else 0.0

    @property
    def top_bid(self) -> float:
        """X^14_t[0] = V^{-,1}_t (0 in the auction phase)."""
        return float(self.generator.book.bid_volumes[0]) if self._phase == "clob" else 0.0

    @property
    def n_mm(self) -> int:
        """X^6_t = M_t (0 in the CLOB phase)."""
        return self.generator.auction_flow.n_mm if self._phase == "auction" else 0

    @property
    def n_buy(self) -> int:
        """X^7_t = N^+_{t-}: BUYING market orders (paper convention, D3)."""
        return self.generator.auction_flow.n_buy if self._phase == "auction" else 0

    @property
    def n_sell(self) -> int:
        """X^8_t = N^-_{t-}: selling market orders."""
        return self.generator.auction_flow.n_sell if self._phase == "auction" else 0

    # -- paper state (D11: tests/documentation only) --------------------------

    def paper_state(self) -> dict[str, Any]:
        """Materialize the paper state X^1..X^17 (correctly shaped,
        zero-padded vectors; fixes AUDIT N5). Tests/documentation only —
        never called in the training hot path (D11).

        Shapes: X^9, X^16, X^17 in R^{m-n} (component s-n for the order
        decided at t_s; predictable indexing — entries up to t-1 only);
        X^11/X^12 in R^{L_max}; X^13/X^14 in R^{Lc} (volumes zeroed above
        the depth); X^15 in R^{La x 2} (rows (K^i, S^i), zero-padded). The
        paper's script-N/script-L bounds are realized by the config caps.
        """
        g = self.grid
        in_clob = self._phase == "clob"
        in_auction = not in_clob
        book = self.generator.book
        flow = self.generator.auction_flow
        n_slots = g.tau_cl - g.tau_op

        S_hist, K_hist = self._ledger.history_at(self._t)
        theta = self._ledger.theta() if in_auction else np.zeros(n_slots)

        def book_side(vols: np.ndarray, depth: int) -> np.ndarray:
            out = np.zeros(self.cfg.clob_flow.Lc)
            if in_clob:
                out[:] = vols
                out[depth:] = 0.0  # X^13_j = V^{+,j} 1{j <= L^+} (legacy rule)
            return out

        x15 = np.zeros((self.cfg.auction_flow.La, 2))
        if in_auction:
            K_exo, S_exo = flow.supply_curves()
            m_show = min(len(K_exo), self.cfg.auction_flow.La)
            x15[:m_show, 0] = K_exo[:m_show]
            x15[:m_show, 1] = S_exo[:m_show]

        return {
            "X1": self._inventory,
            "X2": self._Z if self._t == float(g.tau_cl) else 0.0,
            "X3": self._h_cache,
            "X4": self.depth_ask,
            "X5": self.depth_bid,
            "X6": self.n_mm,
            "X7": self.n_buy,
            "X8": self.n_sell,
            "X9": theta,
            "X10": self._mid,
            "X11": flow.buy_volumes.copy() if in_auction else np.zeros(self.cfg.auction_flow.L_max),
            "X12": flow.sell_volumes.copy() if in_auction else np.zeros(self.cfg.auction_flow.L_max),
            "X13": book_side(book.ask_volumes, book.depth(+1)),
            "X14": book_side(book.bid_volumes, book.depth(-1)),
            "X15": x15,
            "X16": S_hist,
            "X17": K_hist,
            "time": self._t,
        }


def make_env(cfg: ExperimentConfig, symbol: str | None = None, repo_root: str = ".") -> MarketMakingEnv:
    """Build the env with the configured mid-price model (one call site for
    experiments and tests)."""
    return MarketMakingEnv(cfg, build_midprice(cfg, symbol=symbol, repo_root=repo_root))
