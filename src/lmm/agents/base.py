"""Agent interface shared by RL agents and benchmarks (ruling D9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Union

import numpy as np

from lmm.env.features import FeatureNormalizer

__all__ = [
    "BELLMAN_FACTOR",
    "REWARD_SCALE",
    "ENVIRONMENT_CONTRACT",
    "Transition",
    "Agent",
]


# The revised manuscript is an undiscounted finite-horizon problem.  Keeping
# these two numerical conventions in one place prevents the phase-specific
# implementations from silently drifting apart.
BELLMAN_FACTOR: Final[float] = 1.0
REWARD_SCALE: Final[float] = 1.0e-3
ENVIRONMENT_CONTRACT: Final[str] = "unified-minute-auction-mdp-2026-09-01-v6"

Action = Union[int, np.ndarray]  # discrete index (DQN) or continuous vector


@dataclass(frozen=True)
class Transition:
    """One environment transition as seen by ``Agent.observe``.

    ``next_phase`` marks the cross-phase junction: a CLOB
    transition whose next state is in the auction bootstraps from the AUCTION
    target network. On the terminal transition (``done=True``) the env returns
    the combined reward r_step + r_tau_cl and exposes r_tau_cl in
    ``info["terminal_reward"]``; the RL agents un-fold it and bootstrap the
    terminal clearing reward as the known absorbing-state value,
    ``y = c_r * r_step + c_r * r_tau_cl``.  It is not folded into the stored
    step reward and there is no terminal network bootstrap.

    ``action`` is the policy output. For DQN it is the exact executed discrete
    index (admissibility is enforced by selection-time masking, AUDIT N12).
    For a continuous agent it is the raw normalized proposal; the adapter
    projects it to the executable market order and records the clipped proposal in
    ``info["proposal_action_vec"]`` for continuous replay.

    ``next_mask`` is Adm(x') over the NEXT phase's action grid (None iff
    ``done``): the Bellman masked max needs it at update time because the
    auction cancel-admissibility C(x') is not recoverable from the pruned
    feature vector. ``info`` carries per-step env diagnostics (E_t, S_bullet,
    ...) for non-learning consumers (benchmark execution-price tracking); it
    is NEVER stored in replay.

    ``discount`` remains as a deprecated construction-time compatibility field.
    Learners deliberately ignore it: the Bellman factor is exactly one for
    every transition.
    """

    obs: np.ndarray
    action: Action
    reward: float
    next_obs: np.ndarray
    done: bool
    phase: str  # "clob" | "auction"
    next_phase: str  # phase of next_obs ("auction" for the junction)
    next_mask: np.ndarray | None = None  # Adm(x') on the next grid; None iff done
    info: dict[str, Any] | None = field(default=None, compare=False)
    discount: float | None = None  # deprecated and ignored; Bellman factor is always one


class Agent(ABC):
    """Minimal agent interface: act / observe / update / save / load,
    plus no-op lifecycle hooks (bind / start_episode / set_train)."""

    @abstractmethod
    def act(self, obs: np.ndarray, mask: np.ndarray, phase: str, *, eval_mode: bool = False) -> Action:
        """Select an action for ``obs`` in the given phase.

        ``mask`` is the env's admissibility mask Adm(x) over the phase grid.
        ``eval_mode=True`` disables exploration (epsilon = 0 / no noise).
        Discrete implementations must restrict BOTH the greedy argmax and any
        random draw to the admissible set (masking, never projection — AUDIT
        N12). Continuous implementations instead emit normalized proposals and
        use the documented adapter projection to an executable order.
        """

    @abstractmethod
    def observe(self, transition: Transition) -> None:
        """Record a transition (e.g. into replay). Benchmarks may no-op."""

    @abstractmethod
    def update(self, phase: str | None = None) -> dict[str, float]:
        """At most one learning update for the latest environment transition.

        ``phase`` identifies the buffer belonging to that transition.  ``None``
        is retained for callers that rely on the agent tracking the phase in
        :meth:`observe`.  An empty dict means no update was eligible.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Checkpoint parameters and optimizer/schedule state."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore a checkpoint written by :meth:`save`."""

    # -- lifecycle hooks (no-op by default) -----------------------------------

    def bind(self, env) -> None:
        """Give the agent a handle on its env (EVALUATION-time state access
        for the benchmarks: inventory, t, phase). RL agents ignore it."""

    def start_episode(self, episode: int) -> None:
        """Called before each episode (epsilon schedule, per-episode state)."""

    def set_train(self, training: bool) -> None:
        """Toggle train/eval mode (network .train()/.eval(), exploration)."""

    @property
    def checkpoint_update_counts(self) -> dict[str, int]:
        """Phase-specific optimizer steps that count toward model maturity.

        Most agents count every optimizer step.  DQN overrides the underlying
        counter for the auction phase so updates made while its behavior policy
        is locked to the safe no-order action do not make a checkpoint eligible.
        """
        counts = getattr(
            self,
            "_checkpoint_update_count",
            getattr(self, "_update_count", {}),
        )
        return {
            phase: int(counts.get(phase, 0))
            for phase in ("clob", "auction")
        }

    # -- common frozen observation transform --------------------------------

    def set_feature_normalizer(self, normalizer: FeatureNormalizer) -> None:
        """Attach a fitted, frozen training-only feature transform."""
        if not normalizer.frozen:
            raise ValueError("feature normalizer must be frozen before attachment")
        self._feature_normalizer = normalizer

    def preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """Return the exact representation consumed by policy and replay."""
        normalizer = getattr(self, "_feature_normalizer", None)
        raw = np.asarray(observation, dtype=np.float32)
        return raw if normalizer is None else normalizer.transform(raw)

    def _feature_normalizer_state(self) -> dict[str, Any] | None:
        normalizer = getattr(self, "_feature_normalizer", None)
        return None if normalizer is None else normalizer.state_dict()

    def _load_feature_normalizer_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            self._feature_normalizer = None
            return
        cfg = getattr(self, "cfg", None)
        if cfg is not None:
            if float(state["tau_cl"]) != float(cfg.grid.tau_cl):
                raise ValueError("checkpoint normalizer tau_cl does not match resolved config")
            expected_zero_h = not bool(cfg.rl.h_cl_feature_enabled)
            if bool(state.get("zero_h_cl", False)) != expected_zero_h:
                raise ValueError(
                    "checkpoint normalizer H_cl ablation does not match resolved config"
                )
        normalizer = FeatureNormalizer(
            float(state["tau_cl"]),
            zero_h_cl=bool(state.get("zero_h_cl", False)),
        )
        normalizer.load_state_dict(state)
        if not normalizer.frozen:
            raise ValueError("checkpoint feature normalizer is not frozen")
        self._feature_normalizer = normalizer
