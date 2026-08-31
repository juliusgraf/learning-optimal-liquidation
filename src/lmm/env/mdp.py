"""Two-phase controlled market simulator matching the revised chronology.

The random CLOB decision grid and all CLOB exogenous data are sampled before
the first action.  Every returned observation is already a complete
pre-action state: a fresh exogenous CLOB snapshot in the continuous phase, or
the current accepted auction proposals combined with the lagged indicative
price in the auction phase.
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
from lmm.env.features import COMMON_FEATURES, FeatureExtractor
from lmm.env.rewards import auction_reward, clob_reward, f_a, terminal_reward
from lmm.market.auction import AgentOrderLedger, AuctionEvents
from lmm.market.clearing import (
    Algo1Diagnostics,
    Algo1Estimator,
    CarryoverCalibration,
    ClearingInputs,
    ClearingResult,
    Eq2Cache,
    TerminalAllocation,
    allocate_terminal,
    clear_linear,
    round_half_up_to_tick,
)
from lmm.market.generator import EpisodeGrid, MarketGenerator
from lmm.market.midprice import MidPriceModel, build_midprice

__all__ = ["MarketMakingEnv", "make_env"]

logger = logging.getLogger(__name__)
_EPS = 1e-9


class MarketMakingEnv(gymnasium.Env):
    """Gymnasium-style environment with phase-specific action spaces."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: ExperimentConfig, midprice: MidPriceModel) -> None:
        super().__init__()
        if cfg.grid.h != cfg.grid.tau_cl - cfg.grid.tau_op:
            raise ValueError("grid.h must equal tau_cl-tau_op")
        if cfg.rl.chi != 1.0:
            raise ValueError("the revised finite-horizon problem requires rl.chi=1")

        self.cfg = cfg
        self.grid = cfg.grid
        self.midprice = midprice
        self.generator = MarketGenerator(cfg.clob_flow, cfg.auction_flow, cfg.grid)
        self.algo1 = Algo1Estimator(cfg.algo1, cfg.grid)
        self.eq2 = Eq2Cache(cfg.grid)
        self.features = FeatureExtractor(
            cfg.features, cfg.grid, cfg.clob_flow, cfg.auction_flow
        )
        self.clob_grid = ClobActionGrid(cfg.actions)
        self.auction_grid = AuctionActionGrid(cfg.actions)
        self._ledger = AgentOrderLedger(cfg.grid)

        self._clob_space = gymnasium.spaces.Discrete(len(self.clob_grid))
        self._auction_space = gymnasium.spaces.Discrete(len(self.auction_grid))
        feature_shape = (len(COMMON_FEATURES),)
        self._clob_obs_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=feature_shape, dtype=np.float32
        )
        self._auction_obs_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=feature_shape, dtype=np.float32
        )
        self.action_space = self._clob_space
        self.observation_space = self._clob_obs_space
        self._done = True

    # ------------------------------------------------------------------
    # Episode preparation
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)

        # The grid must exist before the non-uniform mid-price path is built.
        self._episode_grid = self.generator.reset_episode(self.np_random)
        self._prepare_midprice_path()

        self._phase = "clob"
        self._done = False
        self._clob_index = 0
        self._auction_index = -1
        self._decision_index = 0
        self._t = float(self._episode_grid.clob_times[0])
        self._mid = float(self._clob_mid_values[0])
        self._frozen_mid: float | None = None
        self._inventory = float(self.grid.I0)
        self._I_tau_op: float | None = None
        self._Z = 0.0
        self._S_cl: float | None = None
        self._terminal_allocation: TerminalAllocation | None = None
        self._last_clearing_result: ClearingResult | None = None
        self._pending_auction_events: AuctionEvents | None = None
        self._current_algo1_diag: Algo1Diagnostics | None = None
        self._interim_shaping_by_slot = np.zeros(self.grid.h, dtype=float)

        self.algo1.reset(self.cfg.algo1.H0)
        self._h_cache = self.algo1.h
        self.eq2.reset(self._h_cache)
        self._ledger.reset()
        self.generator.prepare_clob_decision(0, self._k_mid())

        self._n = self._episode_grid.n
        self._m = self._episode_grid.m
        self.action_space = self._clob_space
        self.observation_space = self._clob_obs_space
        info = {
            "t": self._t,
            "decision_index": self._decision_index,
            "phase": self._phase,
            "H_used": self._h_cache,
            "episode_n": self._n,
            "episode_m": self._m,
        }
        return self.features.clob_features(self), info

    def _prepare_midprice_path(self) -> None:
        """Simulate/replay the complete revealed path and project half-up."""
        alpha = self.grid.alpha
        values = [round_half_up_to_tick(self.midprice.reset(self.np_random), alpha)]
        for t in self._episode_grid.clob_times[1:]:
            values.append(round_half_up_to_tick(self.midprice.advance_to(float(t)), alpha))
        frozen = round_half_up_to_tick(
            self.midprice.advance_to(float(self.grid.tau_op)), alpha
        )
        self._clob_mid_values = np.asarray(values, dtype=float)
        self._clob_mid_values.setflags(write=False)
        self._prepared_frozen_mid = float(frozen)

    # ------------------------------------------------------------------
    # Gym transition
    # ------------------------------------------------------------------

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        if self._phase == "clob":
            return self._step_clob(action)
        return self._step_auction(action)

    def _step_clob(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = self._decode_clob(action)
        self._validate_clob_action(a)

        t = self._t
        i = self._clob_index
        h_used = self._h_cache
        s_bullet = float(self._mid + self.grid.alpha * a.delta)
        book = self.generator.book
        book.place_agent_order(a.volume, a.delta)
        flow = self.generator.step_clob_index(i)
        executed = float(flow.executed_agent)
        self._inventory -= executed
        if -_EPS < self._inventory < 0.0:
            self._inventory = 0.0

        economic_cash = s_bullet * executed
        reward = clob_reward(
            s_bullet,
            executed,
            h_used,
            self.cfg.reward.k_star,
            self.grid.alpha,
            shaping_enabled=self.cfg.reward.shaping_enabled,
        )
        shaping_adjustment = reward - economic_cash
        book.clear_agent_order()  # every strategic CLOB order lasts one interval

        info: dict[str, Any] = {
            "t": t,
            "t_next": float(flow.t_next),
            "decision_index": self._decision_index,
            "phase": "clob",
            "action": a,
            "H_used": h_used,
            "E_t": executed,
            "S_bullet": s_bullet,
            "n_buy_step": flow.n_buy,
            "n_sell_step": flow.n_sell,
            "algo1": self._current_algo1_diag,
            "clob_economic_cash": economic_cash,
            "auction_economic_cash": 0.0,
            "cancellation_fee": 0.0,
            "residual_mark": 0.0,
            "terminal_penalty": 0.0,
            "clob_shaping_adjustment": shaping_adjustment,
            "auction_interim_shaping": 0.0,
            "auction_terminal_shaping": 0.0,
        }

        if i == self._episode_grid.n:
            residual = flow.residual_exogenous
            if residual is None:
                raise AssertionError("final CLOB interval did not retain its residual book")
            carryover = self.algo1.calibrate_carryover(
                residual, n=self._episode_grid.n
            )
            if not self.cfg.experiment.auction_enabled:
                reward += self._terminate_no_auction(carryover, info)
                info["H_next"] = self._h_cache
                info["training_reward"] = reward
                obs = self.features.clob_features(self)
                return obs, float(reward), True, False, info
            self._open_auction(carryover)
            obs = self.features.auction_features(self)
        else:
            self._prepare_next_clob_decision(i + 1)
            obs = self.features.clob_features(self)

        info["H_next"] = self._h_cache
        info["training_reward"] = reward
        return obs, float(reward), False, False, info

    def _terminate_no_auction(
        self,
        carryover: CarryoverCalibration,
        info: dict[str, Any],
    ) -> float:
        """No-auction comparator: mark and penalize at ``tau_op``."""
        frozen_mid = float(self._prepared_frozen_mid)
        i_final = float(self._inventory)
        residual_mark = frozen_mid * i_final
        penalty = self.cfg.reward.lambda_inv * i_final**2
        r_term = residual_mark - penalty
        self._frozen_mid = frozen_mid
        self._mid = frozen_mid
        self._I_tau_op = i_final
        self._S_cl = frozen_mid
        self._Z = 0.0
        self._t = float(self.grid.tau_op)
        self._decision_index = self._episode_grid.n + 1
        self._phase = "terminal"
        self._done = True
        empty_counts = {
            name: {
                "proposed": 0,
                "accepted": 0,
                "rejected": 0,
                "ineligible": 0,
            }
            for name in (
                "schedule_arrival",
                "schedule_cancel",
                "buy_arrival",
                "buy_cancel",
                "sell_arrival",
                "sell_cancel",
            )
        }
        info.update(
            t_next=float(self.grid.tau_op),
            S_cl=frozen_mid,
            Z=0.0,
            requested_Z=0.0,
            I_final=i_final,
            Q_supply=0.0,
            Q_demand=0.0,
            rho_supply=1.0,
            rho_demand=1.0,
            executed_supply=0.0,
            executed_demand=0.0,
            self_trade_count=0,
            terminal_reward=r_term,
            auction_economic_cash=0.0,
            residual_mark=residual_mark,
            terminal_penalty=penalty,
            auction_terminal_shaping=0.0,
            continuous_price=frozen_mid,
            tick_price=frozen_mid,
            clearing_residual=0.0,
            leave_agent_out_continuous_price=frozen_mid,
            leave_agent_out_tick_price=frozen_mid,
            agent_price_displacement=0.0,
            carryover_slope=float(carryover.total_slope),
            fallback_used=bool(carryover.total_slope < self.cfg.auction_flow.D_mu),
            persistent_schedule_id=-1,
            proposal_counts=empty_counts,
            no_auction_comparator=True,
        )
        return float(r_term)

    def _prepare_next_clob_decision(self, next_index: int) -> None:
        self._clob_index = int(next_index)
        self._decision_index = int(next_index)
        self._t = float(self._episode_grid.clob_times[next_index])
        self._mid = float(self._clob_mid_values[next_index])
        snapshot = self.generator.prepare_clob_decision(next_index, self._k_mid())
        self._current_algo1_diag = self.algo1.observe(next_index, snapshot)
        self._h_cache = self._current_algo1_diag.H

    def _open_auction(self, carryover: CarryoverCalibration) -> None:
        self._phase = "auction"
        self._auction_index = 0
        self._decision_index = self._episode_grid.n + 1
        self._t = float(self._episode_grid.auction_times[0])
        self._frozen_mid = self._prepared_frozen_mid
        self._mid = self._frozen_mid
        self._I_tau_op = self._inventory
        self._ledger.reset()
        self._interim_shaping_by_slot.fill(0.0)
        self.generator.auction_flow.reset(self._frozen_mid, carryover)
        self._carryover_slope = float(carryover.total_slope)
        self._fallback_used = any(
            rec.provenance == "fallback"
            for rec in self.generator.auction_flow.active_schedules
        )
        persistent = [
            rec for rec in self.generator.auction_flow.active_schedules if rec.persistent
        ]
        if len(persistent) != 1:
            raise AssertionError("auction initialization must designate one persistent schedule")
        self._persistent_schedule_id = persistent[0].schedule_id
        self.eq2.reset(self._h_cache)
        self._last_clearing_result = None
        self.action_space = self._auction_space
        self.observation_space = self._auction_obs_space
        self._prepare_current_auction_proposals()

    def _prepare_current_auction_proposals(self) -> None:
        """Accept current proposals before constructing the action state."""
        self._pending_auction_events = self.generator.step_auction(self.np_random)
        self.generator.auction_flow.assert_valid()

    def _step_auction(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = self._decode_auction(action)
        external = self._is_external_action(a)
        s_a = self._auction_reference(a)
        self._validate_auction_action(a, s_a, external)
        if self._pending_auction_events is None:
            raise AssertionError("auction proposals were not prepared before the action")

        t = self._t
        h_used = self._h_cache
        d_t = float(self._auction_index) * self.cfg.reward.d
        fee = d_t * int(a.cancel)
        interim_reward = auction_reward(
            a.K_a,
            s_a,
            h_used,
            self.cfg.reward.q,
            d_t,
            a.cancel,
            shaping_enabled=self.cfg.reward.shaping_enabled,
            external_policy=external,
        )
        interim_shaping = interim_reward + fee

        prior_live = self._ledger.live.copy()
        clawback = 0.0
        if a.cancel == 1:
            if self.cfg.reward.clawback_shaping:
                clawback = float(np.sum(self._interim_shaping_by_slot[prior_live]))
                interim_reward -= clawback
                self._interim_shaping_by_slot[prior_live] = 0.0
            self._ledger.apply_cancel_all(t)
        self._ledger.submit(
            t,
            a.K_a,
            s_a,
            one_sided=external,
            quantity_cap=a.quantity_cap,
            reference_price=s_a if external else None,
        )
        if a.K_a > 0.0 and self.cfg.reward.shaping_enabled and not external:
            self._interim_shaping_by_slot[self._auction_index] = interim_shaping

        inputs = self._clearing_inputs()
        external_schedule = self._ledger.external_schedule()
        result = self.eq2.recompute_result(
            inputs,
            self.cfg.auction_flow.D_mu,
            external_schedule=external_schedule,
        )
        self._last_clearing_result = result
        self._h_cache = result.tick_price

        events = self._pending_auction_events
        info: dict[str, Any] = {
            "t": t,
            "t_next": (
                float(self.grid.tau_cl)
                if self._auction_index == self.grid.h - 1
                else float(self._episode_grid.auction_times[self._auction_index + 1])
            ),
            "decision_index": self._decision_index,
            "phase": "auction",
            "action": a,
            "H_used": h_used,
            "H_next": result.tick_price,
            "S_a": s_a,
            "d_t": d_t,
            "events": events,
            "proposal_counts": self.generator.auction_flow.proposal_counts,
            "carryover_slope": self._carryover_slope,
            "fallback_used": self._fallback_used,
            "persistent_schedule_id": self._persistent_schedule_id,
            "D": result.D,
            "R": result.R,
            "continuous_price": result.continuous_price,
            "tick_price": result.tick_price,
            "clearing_residual": result.residual_at_tick,
            "nonlinear_clearing": result.nonlinear,
            "degenerate_fallback": False,
            "clob_economic_cash": 0.0,
            "auction_economic_cash": 0.0,
            "cancellation_fee": fee,
            "residual_mark": 0.0,
            "terminal_penalty": 0.0,
            "clob_shaping_adjustment": 0.0,
            "auction_interim_shaping": interim_shaping,
            "auction_shaping_clawback": clawback,
            "auction_terminal_shaping": 0.0,
        }

        terminated = self._auction_index == self.grid.h - 1
        total_reward = float(interim_reward)
        if terminated:
            total_reward += self._terminal(result, inputs, external_schedule, info)
            self._t = float(self.grid.tau_cl)
            self._decision_index = self._episode_grid.m + 1
            self._done = True
            self._pending_auction_events = None
        else:
            self._auction_index += 1
            self._decision_index += 1
            self._t = float(self._episode_grid.auction_times[self._auction_index])
            self._prepare_current_auction_proposals()

        info["training_reward"] = total_reward
        obs = self.features.auction_features(self)
        return obs, total_reward, terminated, False, info

    def _terminal(
        self,
        result: ClearingResult,
        inputs: ClearingInputs,
        external_schedule,
        info: dict[str, Any],
    ) -> float:
        if self._I_tau_op is None or self._frozen_mid is None:
            raise AssertionError("terminal clearing before auction initialization")
        allocation = allocate_terminal(
            inputs,
            result.tick_price,
            external_agent_schedule=external_schedule,
        )
        z = allocation.actual_agent
        i_final = float(self._I_tau_op - z)
        if (
            self.cfg.reward.numerical_guard
            and abs(i_final) > self.cfg.reward.numerical_guard_bound
        ):
            logger.warning(
                "terminal inventory %.12g exceeds diagnostic bound %.12g; "
                "the revised simulator does not clip it",
                i_final,
                self.cfg.reward.numerical_guard_bound,
            )

        external_policy = external_schedule is not None
        r_term = terminal_reward(
            result.tick_price,
            z,
            i_final,
            self._frozen_mid,
            self.cfg.reward.lambda_inv,
            self.cfg.reward.q,
            shaping_enabled=self.cfg.reward.shaping_enabled,
            external_policy=external_policy,
        )
        auction_cash = result.tick_price * z
        residual_mark = self._frozen_mid * i_final
        terminal_penalty = self.cfg.reward.lambda_inv * i_final**2
        terminal_shaping = (
            f_a(auction_cash, self.cfg.reward.q)
            if self.cfg.reward.shaping_enabled and not external_policy
            else 0.0
        )

        leave_agent_out_inputs = ClearingInputs(
            K_exo=inputs.K_exo,
            S_exo=inputs.S_exo,
            K_agent=np.zeros(0, dtype=float),
            S_agent=np.zeros(0, dtype=float),
            net_market_volume=inputs.net_market_volume,
            fallback_mid=inputs.fallback_mid,
            buy_market_volume=inputs.buy_market_volume,
            sell_market_volume=inputs.sell_market_volume,
        )
        leave_agent_out = clear_linear(
            leave_agent_out_inputs,
            self.grid.alpha,
            self.cfg.auction_flow.D_mu,
        )

        self._S_cl = result.tick_price
        self._Z = z
        self._inventory = i_final
        self._terminal_allocation = allocation
        info.update(
            S_cl=result.tick_price,
            Z=z,
            requested_Z=allocation.requested_agent,
            I_final=i_final,
            Q_supply=allocation.Q_supply,
            Q_demand=allocation.Q_demand,
            rho_supply=allocation.rho_supply,
            rho_demand=allocation.rho_demand,
            executed_supply=allocation.executed_supply,
            executed_demand=allocation.executed_demand,
            self_trade_count=allocation.self_trade_count,
            leave_agent_out_continuous_price=leave_agent_out.continuous_price,
            leave_agent_out_tick_price=leave_agent_out.tick_price,
            agent_price_displacement=(result.tick_price - leave_agent_out.tick_price),
            terminal_reward=r_term,
            auction_economic_cash=auction_cash,
            residual_mark=residual_mark,
            terminal_penalty=terminal_penalty,
            auction_terminal_shaping=terminal_shaping,
        )
        return float(r_term)

    # ------------------------------------------------------------------
    # Action decoding and admissibility
    # ------------------------------------------------------------------

    def _decode_clob(self, action) -> ClobAction:
        if isinstance(action, ClobAction):
            return action
        if isinstance(action, (int, np.integer)):
            return self.clob_grid.decode(int(action))
        raise TypeError(
            f"CLOB action must be an index or ClobAction, got {type(action).__name__}"
        )

    def _decode_auction(self, action) -> AuctionAction:
        if isinstance(action, AuctionAction):
            return action
        if isinstance(action, (int, np.integer)):
            return self.auction_grid.decode(int(action))
        raise TypeError(
            f"auction action must be an index or AuctionAction, got {type(action).__name__}"
        )

    def _validate_clob_action(self, a: ClobAction) -> None:
        if not math.isfinite(float(a.volume)) or a.volume < 0.0:
            raise ValueError(f"volume must be finite and nonnegative, got {a.volume}")
        if not math.isclose(float(a.volume), round(float(a.volume)), abs_tol=1e-10):
            raise ValueError("strategic CLOB submitted volume must be integer-valued")
        max_volume = min(
            self.cfg.actions.V_max,
            max(0, math.floor(float(self._inventory) + 1e-12)),
        )
        if a.volume > max_volume:
            raise ValueError(
                f"inadmissible CLOB volume {a.volume}; current maximum is {max_volume}"
            )
        if int(a.delta) != a.delta or not 0 <= int(a.delta) <= self.cfg.actions.L_max:
            raise ValueError(
                f"CLOB offset must be an integer in [0,{self.cfg.actions.L_max}]"
            )
        if a.volume == 0.0 and int(a.delta) != 0:
            raise ValueError("the canonical zero CLOB action has delta=0")

    @staticmethod
    def _is_external_action(a: AuctionAction) -> bool:
        return bool(
            a.one_sided or a.quantity_cap is not None or a.reference_price is not None
        )

    def _auction_reference(self, a: AuctionAction) -> float:
        if a.reference_price is not None:
            return float(a.reference_price)
        if self._frozen_mid is None:
            raise AssertionError("auction reference requested before auction open")
        return float(self._frozen_mid + self.grid.alpha * int(a.offset))

    def _validate_auction_action(
        self, a: AuctionAction, s_a: float, external: bool
    ) -> None:
        if not math.isfinite(float(a.K_a)) or a.K_a < 0.0:
            raise ValueError(f"K^a must be finite and nonnegative, got {a.K_a}")
        if not external and a.K_a > self.cfg.actions.auction_K_grid_max + _EPS:
            raise ValueError(
                f"K^a exceeds {self.cfg.actions.auction_K_grid_max}: {a.K_a}"
            )
        if int(a.offset) != a.offset:
            raise ValueError("auction offset must be integer-valued")
        if not external and abs(int(a.offset)) > self.cfg.actions.B_max:
            raise ValueError(
                f"auction offset must lie in [-{self.cfg.actions.B_max},{self.cfg.actions.B_max}]"
            )
        if a.K_a == 0.0 and int(a.offset) != 0:
            raise ValueError("the canonical zero-slope action has offset=0")
        if not math.isfinite(float(s_a)) or s_a < 0.0:
            raise ValueError(f"auction reference price must be nonnegative, got {s_a}")
        if a.cancel not in (0, 1):
            raise ValueError(f"cancel must be 0 or 1, got {a.cancel}")
        if a.cancel and self.cfg.actions.auction_cancel_mode == "never":
            raise ValueError("cancellation is disabled in this treatment")
        if a.cancel and not self._ledger.cancel_admissible():
            raise ValueError("cancel-all is ineligible without a live prior schedule")

    def action_mask(self) -> np.ndarray:
        if self._phase == "clob":
            return self.clob_grid.mask(self._inventory)
        if self._frozen_mid is None:
            raise AssertionError("auction mask requested before auction open")
        return self.auction_grid.mask(
            self.cancel_admissible,
            frozen_mid=self._frozen_mid,
            alpha=self.grid.alpha,
        )

    # ------------------------------------------------------------------
    # Clearing inputs and observable feature accessors
    # ------------------------------------------------------------------

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
            fallback_mid=float(self._frozen_mid if self._frozen_mid is not None else self._mid),
            buy_market_volume=flow.buy_market_volume(),
            sell_market_volume=flow.sell_market_volume(),
        )

    def _k_mid(self) -> int:
        return int(round(float(self._mid) / self.grid.alpha))

    @property
    def episode_grid(self) -> EpisodeGrid:
        return self._episode_grid

    @property
    def t(self) -> float:
        return float(self._t)

    @property
    def decision_index(self) -> int:
        return int(self._decision_index)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def inventory(self) -> float:
        return float(self._inventory)

    @property
    def h_cl(self) -> float:
        return float(self._h_cache)

    @property
    def s_mid(self) -> float:
        return float(self._mid)

    @property
    def depth_ask(self) -> int:
        return self.generator.book.depth(+1) if self._phase == "clob" else 0

    @property
    def depth_bid(self) -> int:
        return self.generator.book.depth(-1) if self._phase == "clob" else 0

    @property
    def top_ask(self) -> float:
        return (
            float(self.generator.book.ask_volumes[0]) if self._phase == "clob" else 0.0
        )

    @property
    def top_bid(self) -> float:
        return (
            float(self.generator.book.bid_volumes[0]) if self._phase == "clob" else 0.0
        )

    @property
    def n_mm(self) -> int:
        return self.generator.auction_flow.n_mm if self._phase == "auction" else 0

    @property
    def n_buy(self) -> int:
        return self.generator.auction_flow.n_buy if self._phase == "auction" else 0

    @property
    def n_sell(self) -> int:
        return self.generator.auction_flow.n_sell if self._phase == "auction" else 0

    @property
    def cancel_admissible(self) -> bool:
        return bool(
            self._phase == "auction"
            and self.cfg.actions.auction_cancel_mode == "enabled"
            and self._ledger.cancel_admissible()
        )

    @property
    def own_slope(self) -> float:
        return self._ledger.aggregates()[0] if self._phase == "auction" else 0.0

    @property
    def own_weighted_quote(self) -> float:
        return self._ledger.aggregates()[1] if self._phase == "auction" else 0.0

    @property
    def exogenous_slope(self) -> float:
        return (
            self.generator.auction_flow.aggregates()[0]
            if self._phase == "auction"
            else 0.0
        )

    @property
    def auction_imbalance(self) -> float:
        return (
            self.generator.auction_flow.aggregates()[2]
            if self._phase == "auction"
            else 0.0
        )

    @property
    def exogenous_weighted_quote(self) -> float:
        return (
            self.generator.auction_flow.aggregates()[1]
            if self._phase == "auction"
            else 0.0
        )

    # ------------------------------------------------------------------
    # Complete latent configuration for diagnostics only
    # ------------------------------------------------------------------

    def paper_state(self) -> dict[str, Any]:
        in_clob = self._phase == "clob"
        flow = self.generator.auction_flow
        S_hist, K_hist = self._ledger.history_at(self._t)
        K_exo, S_exo = flow.supply_curves()
        return {
            "X1": self._inventory,
            "X2": self._Z if self._done else 0.0,
            "X3": self._h_cache,
            "X4": self.depth_ask,
            "X5": self.depth_bid,
            "X6": self.n_mm,
            "X7": self.n_buy,
            "X8": self.n_sell,
            "X9": self._ledger.theta() if not in_clob else np.zeros(self.grid.h),
            "X10": self._mid,
            "X11": flow.buy_volumes.copy() if not in_clob else np.zeros(0),
            "X12": flow.sell_volumes.copy() if not in_clob else np.zeros(0),
            "X13": self.generator.book.ask_volumes.copy() if in_clob else np.zeros(0),
            "X14": self.generator.book.bid_volumes.copy() if in_clob else np.zeros(0),
            "X15": (
                np.column_stack((K_exo, S_exo)) if not in_clob else np.zeros((0, 2))
            ),
            "X16": S_hist,
            "X17": K_hist,
            "X18": self._decision_index,
            "time": self._t,
        }


def make_env(
    cfg: ExperimentConfig,
    symbol: str | None = None,
    repo_root: str = ".",
    data_split: str = "train",
) -> MarketMakingEnv:
    return MarketMakingEnv(
        cfg,
        build_midprice(
            cfg,
            symbol=symbol,
            repo_root=repo_root,
            data_split=data_split,
        ),
    )
