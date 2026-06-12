"""Benchmark policies (paper `sec:benchmark`): Avellaneda-Stoikov and TWAP.

Both share the auction heuristic: a SINGLE order at auction open with the
one-sided supply z * q_{tau_op} * (p - S_tilde)_+, z = 10, S_tilde = average
of the mean and max executed CLOB prices (ruling D16: one-sided curve —
benchmarks only liquidate; the legacy linear curve could clear as a buyer,
AUDIT N7). Subsequent auction steps abstain (K^a = 0); dust threshold
q < dust_threshold => no order. One reward definition for all policies
(AUDIT C.4: the legacy penalty override is deleted).

Phase 5 fills in the bodies (constants in BenchmarkParams).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lmm.agents.base import Agent, Transition
from lmm.config import BenchmarkParams, ExperimentConfig

__all__ = ["ASBenchmarkAgent", "TWAPBenchmarkAgent"]


class ASBenchmarkAgent(Agent):
    """Avellaneda-Stoikov benchmark (`section:AS`; AUDIT A.9).

    Closed form v_q(t) = sum_{j<=q} (A e^{-1} (T-t))^j / j! and
    delta^{a,*}(t, q) = (1/(alpha k)) [1 + ln(v_q / v_{q-1})], gamma -> 0;
    discrete-time approximation v_t = q_t (expose whole inventory, capped by
    the volume grid). Calibration: A = lambda_0 / gamma_m; k = gamma_m * K_hat
    with K_hat the least-squares fit of ln Q = K_hat * Delta p over
    ``as_n_samples`` simulated executions; sigma per ``as_sigma_rule``
    (pooled training paths / single fixed path).
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.params: BenchmarkParams = cfg.benchmark

    def calibrate(self, env, rng: np.random.Generator) -> None:
        """Estimate A, k, sigma from the configured environment (AUDIT A.9)."""
        raise NotImplementedError("Phase 5")

    def act(self, obs: np.ndarray, *, eval_mode: bool = False) -> int:
        raise NotImplementedError("Phase 5")

    def observe(self, transition: Transition) -> None:  # benchmarks do not learn
        pass

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Phase 5")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("Phase 5")


class TWAPBenchmarkAgent(Agent):
    """TWAP benchmark (`section:TWAP`): v_t = ceil(q_t / (T - t + 1)) at
    delta = 1 (``twap_delta_mode = "min"``); same auction heuristic as AS."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.params: BenchmarkParams = cfg.benchmark

    def act(self, obs: np.ndarray, *, eval_mode: bool = False) -> int:
        raise NotImplementedError("Phase 5")

    def observe(self, transition: Transition) -> None:  # benchmarks do not learn
        pass

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Phase 5")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("Phase 5")
