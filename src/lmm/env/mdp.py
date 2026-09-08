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
from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np

from lmm.config import ExperimentConfig
from lmm.env.action_spaces import (
    AuctionAction,
    AuctionActionGrid,
    ClobAction,
    ClobActionGrid,
    _BenchmarkAuctionAction,
    round_half_up,
)
from lmm.env.features import COMMON_FEATURES, FeatureExtractor
from lmm.env.rewards import (
    auction_fictive_reward,
    auction_reward,
    clob_reward,
    f_a,
    terminal_reward,
)
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
from lmm.market.midprice import MidPriceModel, RoughHestonMidPrice, build_midprice

__all__ = ["MarketMakingEnv", "make_env"]

logger = logging.getLogger(__name__)
_EPS = 1e-9


@dataclass(frozen=True)
class _ExecutedAuctionAction:
    """Private executable schedule after local ``ell`` resolves to ``b``."""

    K_a: float
    b: int
    cancel: int


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
        # A mid-price model may hold a complete preloaded historical path.
        # Keep it private so the agent-facing environment API exposes only
        # the currently revealed ``s_mid`` value.
        self._midprice = midprice
        # The pre-sampled episode realization is simulator-private: exposing
        # the generator would reveal future random decision times to a bound
        # policy before those times enter the filtration.
        self._generator = MarketGenerator(cfg.clob_flow, cfg.auction_flow, cfg.grid)
        self.algo1 = Algo1Estimator(cfg.algo1, cfg.grid)
        self.eq2 = Eq2Cache(cfg.grid, cfg.auction_flow.clearing_mechanism)
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
        self._episode_grid = self._generator.reset_episode(self.np_random)
        self._prepare_midprice_path()

        self._phase = "clob"
        self._done = False
        self._clob_index = 0
        self._auction_index = -1
        self._decision_index = 0
        self._t = float(self._episode_grid.clob_times[0])
        self._mid = float(self._clob_mid_values[0])
        self._initial_mid = self._mid
        self._frozen_mid: float | None = None
        self._inventory = float(self.grid.I0)
        self._I_tau_op: float | None = None
        self._Z = 0.0
        self._S_cl: float | None = None
        self._terminal_allocation: TerminalAllocation | None = None
        self._last_clearing_result: ClearingResult | None = None
        self._pending_auction_events: AuctionEvents | None = None
        self._current_algo1_diag: Algo1Diagnostics | None = None
        # Latent per-order state needed by r_t(X_t,A_t) under cancellation
        # clawback. It is intentionally not part of the reduced policy feature
        # map: the manuscript already treats that map as a feature-based,
        # potentially non-Markov approximation.
        self._auction_shaping_credit_by_slot = np.zeros(self.grid.h, dtype=float)

        self.algo1.reset(self.cfg.algo1.H0)
        self._h_cache = self.algo1.h
        # Complete latent X^3 history through the current decision time. The
        # reduced feature vector deliberately keeps only the last component.
        self._h_history = [float(self._h_cache)]
        self.eq2.reset(self._h_cache)
        self._ledger.reset()
        self._generator.prepare_clob_decision(0, self._k_mid())

        self._n = self._episode_grid.n
        self._m = self._episode_grid.m
        self.action_space = self._clob_space
        self.observation_space = self._clob_obs_space
        info = {
            "t": self._t,
            "decision_index": self._decision_index,
            "phase": self._phase,
            "H_used": self._h_cache,
        }
        return self.features.clob_features(self), info

    def _prepare_midprice_path(self) -> None:
        """Simulate/replay the complete revealed path and project half-up."""
        alpha = self.grid.alpha
        values = [round_half_up_to_tick(self._midprice.reset(self.np_random), alpha)]
        if isinstance(self._midprice, RoughHestonMidPrice):
            self._midprice.prepare_grid(self._episode_grid.clob_times, float(self.grid.tau_op))
        for t in self._episode_grid.clob_times[1:]:
            values.append(
                round_half_up_to_tick(self._midprice.advance_to(float(t)), alpha)
            )
        frozen = round_half_up_to_tick(
            self._midprice.advance_to(float(self.grid.tau_op)), alpha
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
        book = self._generator.book
        book.place_agent_order(a.volume, a.delta)
        flow = self._generator.step_clob_index(i)
        executed = float(flow.executed_agent)
        self._inventory -= executed
        if -_EPS < self._inventory < 0.0:
            self._inventory = 0.0

        economic_cash = s_bullet * executed
        uncentered_reward = clob_reward(
            s_bullet,
            executed,
            h_used,
            self.cfg.reward.k_star,
            self.grid.alpha,
            shaping_enabled=self.cfg.reward.effective_clob_shaping,
        )
        shaping_adjustment = uncentered_reward - economic_cash
        reward_baseline_adjustment = (
            -self._initial_mid * executed
            if self.cfg.reward.center_initial_inventory_value
            else 0.0
        )
        reward = uncentered_reward + reward_baseline_adjustment
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
            "reward_baseline_adjustment": reward_baseline_adjustment,
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
        terminal_baseline_adjustment = (
            -self._initial_mid * i_final
            if self.cfg.reward.center_initial_inventory_value
            else 0.0
        )
        r_term += terminal_baseline_adjustment
        self._frozen_mid = frozen_mid
        self._mid = frozen_mid
        self._I_tau_op = i_final
        self._S_cl = frozen_mid
        self._Z = 0.0
        self._t = float(self.grid.tau_op)
        self._decision_index = self._episode_grid.n + 1
        self._phase = "terminal"
        self._done = True
        self._h_history.append(float(self._h_cache))
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
            reward_baseline_adjustment=float(
                info.get("reward_baseline_adjustment", 0.0)
            )
            + terminal_baseline_adjustment,
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
        snapshot = self._generator.prepare_clob_decision(next_index, self._k_mid())
        self._current_algo1_diag = self.algo1.observe(next_index, snapshot)
        self._h_cache = self._current_algo1_diag.H
        weights = self.cfg.algo1.clob_forecast_weights
        if weights:
            # Raw Algorithm 1 and carryover calibration are unchanged. Only
            # its use as a future-close signal is reliability-adjusted.
            # The coefficients are frozen from independent training paths.
            bin_index = min(3, int(4*self._t/self.grid.tau_op))
            self._h_cache = self._mid + weights[bin_index]*(self._h_cache-self._mid)
        self._h_history.append(float(self._h_cache))

    def _open_auction(self, carryover: CarryoverCalibration) -> None:
        self._phase = "auction"
        self._auction_index = 0
        self._decision_index = self._episode_grid.n + 1
        self._t = float(self._episode_grid.auction_times[0])
        self._frozen_mid = self._prepared_frozen_mid
        self._mid = self._frozen_mid
        self._I_tau_op = self._inventory
        self._ledger.reset()
        self._auction_shaping_credit_by_slot.fill(0.0)
        self._h_history.append(float(self._h_cache))
        self._generator.auction_flow.reset(self._frozen_mid, carryover)
        self._carryover_slope = float(carryover.total_slope)
        self._fallback_used = any(
            rec.provenance == "fallback"
            for rec in self._generator.auction_flow.active_schedules
        )
        persistent = [
            rec for rec in self._generator.auction_flow.active_schedules if rec.persistent
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
        self._pending_auction_events = self._generator.step_auction(self.np_random)
        self._generator.auction_flow.assert_valid()

    def _step_auction(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = self._decode_auction(action)
        external = isinstance(a, _BenchmarkAuctionAction)
        if external:
            self._validate_benchmark_auction_action(a)
            executed: _ExecutedAuctionAction | _BenchmarkAuctionAction = a
        else:
            self._validate_auction_action(a)
            executed = self._resolve_auction_action(a)
            self._validate_executed_auction_action(executed)
        s_a = self._auction_reference(executed)
        K_a = float(executed.K_a)
        cancel = int(executed.cancel)
        if self._pending_auction_events is None:
            raise AssertionError("auction proposals were not prepared before the action")

        t = self._t
        h_used = self._h_cache
        auction_anchor_b = self._auction_anchor_b_ticks()
        if self._frozen_mid is None:
            raise AssertionError("auction action executed before auction open")
        auction_anchor_price = float(
            self._frozen_mid + self.grid.alpha * auction_anchor_b
        )
        d_t = float(self._auction_index) * self.cfg.reward.d
        fee = d_t * cancel
        prior_live = self._ledger.live.copy()
        clawback = 0.0
        if cancel == 1 and self.cfg.reward.clawback_shaping:
            # Reverse the exact signed credits assigned when the currently
            # live schedules were submitted. Re-marking them with H_t would
            # not telescope and would define a different reward.
            clawback = float(
                np.sum(self._auction_shaping_credit_by_slot[prior_live])
            )

        shaping_active = (
            self.cfg.reward.effective_auction_shaping and not external
        )
        interim_shaping = (
            self.cfg.reward.auction_shaping_weight * auction_fictive_reward(K_a, s_a, h_used, self.cfg.reward.q)
            if shaping_active
            else 0.0
        )
        interim_reward = auction_reward(
            K_a,
            s_a,
            h_used,
            self.cfg.reward.q,
            d_t,
            cancel,
            shaping_enabled=self.cfg.reward.effective_auction_shaping,
            external_policy=external,
            cancelled_interim_shaping=clawback,
            shaping_weight=self.cfg.reward.auction_shaping_weight,
        )

        if cancel == 1:
            # A canceled slot cannot be clawed back a second time, including
            # in the legacy no-clawback treatment.
            self._auction_shaping_credit_by_slot[prior_live] = 0.0
            self._ledger.apply_cancel_all(t)
        self._ledger.submit(
            t,
            K_a,
            s_a,
            one_sided=external,
            quantity_cap=a.quantity_cap if external else None,
            reference_price=s_a if external else None,
        )
        if K_a > 0.0 and shaping_active:
            self._auction_shaping_credit_by_slot[self._auction_index] = (
                interim_shaping
            )

        inputs = self._clearing_inputs()
        external_schedule = self._ledger.external_schedule()
        result = self.eq2.recompute_result(
            inputs,
            self.cfg.auction_flow.D_mu,
            external_schedule=external_schedule,
        )
        self._last_clearing_result = result
        self._h_cache = result.tick_price
        self._h_history.append(float(self._h_cache))

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
            "executed_action": executed,
            "executed_b": (
                int(executed.b)
                if isinstance(executed, _ExecutedAuctionAction)
                else round_half_up(
                    (float(s_a) - float(self._frozen_mid)) / self.grid.alpha
                )
            ),
            "auction_anchor": self.auction_anchor,
            "auction_anchor_b": auction_anchor_b,
            "auction_anchor_price": auction_anchor_price,
            "H_used": h_used,
            "H_next": result.tick_price,
            "S_a": s_a,
            "d_t": d_t,
            "events": events,
            "proposal_counts": self._generator.auction_flow.proposal_counts,
            "carryover_slope": self._carryover_slope,
            "fallback_used": self._fallback_used,
            "persistent_schedule_id": self._persistent_schedule_id,
            "D": result.D,
            "R": result.R,
            "continuous_price": result.continuous_price,
            "tick_price": result.tick_price,
            "clearing_residual": result.residual_at_tick,
            "nonlinear_clearing": result.nonlinear,
            "clearing_mechanism": self.cfg.auction_flow.clearing_mechanism,
            "clearing_tick_index": result.tick_index,
            "matched_volume": result.matched_volume,
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
            "reward_baseline_adjustment": 0.0,
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
            shaping_enabled=self.cfg.reward.effective_auction_shaping,
            external_policy=external_policy,
            shaping_weight=self.cfg.reward.auction_shaping_weight,
        )
        auction_cash = result.tick_price * z
        residual_mark = self._frozen_mid * i_final
        terminal_penalty = self.cfg.reward.lambda_inv * i_final**2
        terminal_shaping = (
            self.cfg.reward.auction_shaping_weight * f_a(auction_cash, self.cfg.reward.q)
            if self.cfg.reward.effective_auction_shaping and not external_policy
            else 0.0
        )
        terminal_baseline_adjustment = (
            -self._initial_mid * (z + i_final)
            if self.cfg.reward.center_initial_inventory_value
            else 0.0
        )
        r_term += terminal_baseline_adjustment

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
            mechanism=self.cfg.auction_flow.clearing_mechanism,
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
            reward_baseline_adjustment=terminal_baseline_adjustment,
        )
        return float(r_term)

    # ------------------------------------------------------------------
    # Action decoding and admissibility
    # ------------------------------------------------------------------

    def _decode_clob(self, action) -> ClobAction:
        if isinstance(action, ClobAction):
            return action
        if isinstance(action, (int, np.integer)):
            return self.clob_grid.decode(action)
        raise TypeError(
            f"CLOB action must be an index or ClobAction, got {type(action).__name__}"
        )

    def _decode_auction(
        self, action
    ) -> AuctionAction | _BenchmarkAuctionAction:
        if isinstance(action, AuctionAction):
            return action
        if isinstance(action, _BenchmarkAuctionAction):
            return action
        if isinstance(action, (int, np.integer)):
            return self.auction_grid.decode(action)
        raise TypeError(
            "auction action must be an index or local AuctionAction, got "
            f"{type(action).__name__}"
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

    def _auction_reference(
        self, a: _ExecutedAuctionAction | _BenchmarkAuctionAction
    ) -> float:
        if isinstance(a, _BenchmarkAuctionAction):
            return float(a.reference_price)
        if self._frozen_mid is None:
            raise AssertionError("auction reference requested before auction open")
        return float(self._frozen_mid + self.grid.alpha * int(a.b))

    def _validate_auction_action(self, a: AuctionAction) -> None:
        if not math.isfinite(float(a.K_a)) or a.K_a < 0.0:
            raise ValueError(f"K^a must be finite and nonnegative, got {a.K_a}")
        if a.K_a > self.cfg.actions.auction_K_grid_max + _EPS:
            raise ValueError(
                f"K^a exceeds {self.cfg.actions.auction_K_grid_max}: {a.K_a}"
            )
        slope_index = float(a.K_a) / float(self.cfg.actions.beta)
        if not math.isclose(
            slope_index,
            round(slope_index),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "strategic auction slope must lie on the beta lattice "
                "{beta*k: k=0,...,K_max}"
            )
        if int(a.ell) != a.ell:
            raise ValueError("auction ell must be integer-valued")
        if abs(int(a.ell)) > self.cfg.actions.B_max:
            raise ValueError(
                "auction ell must lie in "
                f"[-{self.cfg.actions.B_max},{self.cfg.actions.B_max}]"
            )
        if a.K_a == 0.0 and int(a.ell) != 0:
            raise ValueError("the canonical zero-slope action has ell=0")
        if a.cancel not in (0, 1):
            raise ValueError(f"cancel must be 0 or 1, got {a.cancel}")
        if a.cancel and self.cfg.actions.auction_cancel_mode == "never":
            raise ValueError("cancellation is disabled in this treatment")
        if a.cancel and not self._ledger.cancel_admissible():
            raise ValueError("cancel-all is ineligible without a live prior schedule")

    def _resolve_auction_action(self, a: AuctionAction) -> _ExecutedAuctionAction:
        b = (
            0
            if a.K_a == 0.0
            else self._auction_anchor_b_ticks() + int(a.ell)
        )
        return _ExecutedAuctionAction(a.K_a, b, a.cancel)

    def _validate_executed_auction_action(self, a: _ExecutedAuctionAction) -> None:
        if abs(int(a.b)) > self.cfg.actions.B_inf:
            raise ValueError(
                "resolved auction b must lie in "
                f"[-{self.cfg.actions.B_inf},{self.cfg.actions.B_inf}]"
            )
        s_a = self._auction_reference(a)
        if not math.isfinite(float(s_a)) or s_a < 0.0:
            raise ValueError(f"auction reference price must be nonnegative, got {s_a}")

    @staticmethod
    def _validate_benchmark_auction_action(a: _BenchmarkAuctionAction) -> None:
        if not math.isfinite(float(a.K_a)) or a.K_a < 0.0:
            raise ValueError(f"K^a must be finite and nonnegative, got {a.K_a}")
        if a.cancel != 0:
            raise ValueError("benchmark auction schedules cannot cancel")
        if not math.isfinite(float(a.quantity_cap)) or a.quantity_cap < 0.0:
            raise ValueError("benchmark quantity cap must be finite and nonnegative")
        if not math.isfinite(float(a.reference_price)) or a.reference_price < 0.0:
            raise ValueError(
                "benchmark reference price must be finite and nonnegative"
            )

    def action_mask(self) -> np.ndarray:
        if self._phase == "clob":
            return self.clob_grid.mask(self._inventory)
        if self._frozen_mid is None:
            raise AssertionError("auction mask requested before auction open")
        return self.auction_grid.mask(
            self.cancel_admissible,
            frozen_mid=self._frozen_mid,
            alpha=self.grid.alpha,
            anchor_b_ticks=self._auction_anchor_b_ticks(),
        )

    @property
    def auction_anchor(self) -> str:
        """Stable provenance label for the learned auction-price anchor."""
        return self.cfg.actions.auction_anchor

    def _auction_anchor_b_ticks(self) -> int:
        """Treatment-visible local-grid anchor in frozen-mid coordinates."""
        if self._frozen_mid is None:
            raise AssertionError("auction action center requested before auction open")
        if self.cfg.actions.auction_anchor == "frozen_mid":
            return 0
        return self._indicative_b_ticks()

    def _indicative_b_ticks(self) -> int:
        """Absolute ``b`` coordinate of current indicative price ``H_t^cl``."""
        if self._frozen_mid is None:
            raise AssertionError("indicative b requested before auction open")
        return round_half_up(
            (float(self._h_cache) - float(self._frozen_mid)) / self.grid.alpha
        )

    # ------------------------------------------------------------------
    # Clearing inputs and observable feature accessors
    # ------------------------------------------------------------------

    def _clearing_inputs(self) -> ClearingInputs:
        flow = self._generator.auction_flow
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
    def completed_episode_grid(self) -> EpisodeGrid:
        """Return the realized grid only after it can no longer inform actions."""
        if not self._done:
            raise RuntimeError(
                "the realized episode grid is unavailable until the episode completes"
            )
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
        return self._generator.book.depth(+1) if self._phase == "clob" else 0

    @property
    def depth_bid(self) -> int:
        return self._generator.book.depth(-1) if self._phase == "clob" else 0

    @property
    def top_ask(self) -> float:
        return (
            float(self._generator.book.ask_volumes[0]) if self._phase == "clob" else 0.0
        )

    @property
    def top_bid(self) -> float:
        return (
            float(self._generator.book.bid_volumes[0]) if self._phase == "clob" else 0.0
        )

    @property
    def n_mm(self) -> int:
        return self._generator.auction_flow.n_mm if self._phase == "auction" else 0

    @property
    def n_buy(self) -> int:
        return self._generator.auction_flow.n_buy if self._phase == "auction" else 0

    @property
    def n_sell(self) -> int:
        return self._generator.auction_flow.n_sell if self._phase == "auction" else 0

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
            self._generator.auction_flow.aggregates()[0]
            if self._phase == "auction"
            else 0.0
        )

    @property
    def auction_imbalance(self) -> float:
        return (
            self._generator.auction_flow.aggregates()[2]
            if self._phase == "auction"
            else 0.0
        )

    @property
    def exogenous_weighted_quote(self) -> float:
        return (
            self._generator.auction_flow.aggregates()[1]
            if self._phase == "auction"
            else 0.0
        )

    # ------------------------------------------------------------------
    # Complete latent configuration for diagnostics only
    # ------------------------------------------------------------------

    def paper_state(self) -> dict[str, Any]:
        in_clob = self._phase == "clob"
        flow = self._generator.auction_flow
        S_hist, K_hist = self._ledger.history_at(self._t)
        K_exo, S_exo = flow.supply_curves()
        return {
            "X1": self._inventory,
            "X2": self._Z if self._done else 0.0,
            "X3": np.asarray(self._h_history, dtype=float).copy(),
            "X4": self.depth_ask,
            "X5": self.depth_bid,
            "X6": self.n_mm,
            "X7": self.n_buy,
            "X8": self.n_sell,
            "X9": self._ledger.theta() if not in_clob else np.zeros(self.grid.h),
            "X10": self._mid,
            "X11": flow.buy_volumes.copy() if not in_clob else np.zeros(0),
            "X12": flow.sell_volumes.copy() if not in_clob else np.zeros(0),
            "X13": self._generator.book.ask_volumes.copy() if in_clob else np.zeros(0),
            "X14": self._generator.book.bid_volumes.copy() if in_clob else np.zeros(0),
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
