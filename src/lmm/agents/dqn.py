"""Two-phase Double DQN for the manuscript's undiscounted control problem.

Design (full spec in docs/rl_design.md, from which the author rewrites paper
Section 4):
- TWO phase networks Q_phi (CLOB, 18-dim features) and Q_psi (auction, 18-dim
  features) on time-augmented features, matching the manuscript's
  phase-specific approximation.
- Uniform replay (one buffer per phase), exactly one eligible minibatch update
  from the current transition's phase, and phase-local target updates,
  epsilon-greedy with the configured schedule, configured loss/optimizer, gradient
  clipping, eval mode (epsilon = 0, no_grad), checkpointing, full seeding
  (D10: private exploration/replay generators from the SeedBundle; no global
  RNG).
- Cross-phase junction: a CLOB transition with next_phase == "auction"
  bootstraps from the AUCTION target network. The terminal tau_cl state has no
  decision, so its value is the KNOWN terminal reward r_tau_cl: the final
  auction transition is stored with done=True, the step reward only, and
  r_tau_cl carried in ``terminal_value``; the target is
  y = c_r*r_step + c_r*r_tau_cl exactly once, without a network bootstrap.
- Admissibility masking (Adm(x)) in BOTH the greedy argmax and the random
  draw (masking, never projection — AUDIT N12); the Bellman max runs over the
  admissible actions of x' via the stored ``next_mask``.
- The Bellman factor is exactly one on every row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from lmm.config import environment_contract, LEGACY_CLEARING
from lmm.agents.base import (
    BELLMAN_FACTOR,
    Agent,
    Transition,
)
from lmm.config import ExperimentConfig, build_hyperparams
from lmm.env.action_spaces import AuctionActionGrid, ClobActionGrid
from lmm.rl.networks import ACTIVATIONS, StructuredDiscreteQ
from lmm.rl.replay import ReplayBatch, ReplayBuffer
from lmm.rl.schedules import LinearEpsilonSchedule, learning_rate_factor, learning_conditioning_contract
from lmm.utils.seeding import SeedBundle

__all__ = ["DQNAgent", "DQNHyperparams"]


@dataclass(frozen=True)
class DQNHyperparams:
    """All DQN knobs; REQUIRED in configs/algo/dqn.yaml (no library defaults —
    configs/ is the single source of truth, engineering convention)."""

    buffer_size: int
    min_buffer: int
    batch_size: int
    lr: float
    loss: str  # "huber" | "mse"
    grad_clip_norm: float
    hidden_layers: tuple[int, ...]
    target_update_interval: int
    target_soft_tau: Optional[float]
    updates_per_env_step: int
    update_every: int
    reward_scale: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_episodes: float
    epsilon_warmup_episodes: int
    checkpoint_interval_episodes: int
    activation: str
    device: str
    action_embedding_dim: int
    # Double-DQN (van Hasselt et al. 2016): select a' with the ONLINE net,
    # evaluate it with the TARGET net, decoupling action selection from
    # evaluation to curb the max-bootstrap overestimation that destabilises
    # vanilla DQN here (oscillating eval, auction-head TD blow-up). Defaulted
    # (False = vanilla Mnih-2015) so configs predating this knob still load.
    double_q: bool = False
    # Optional phase-specific replay warm-ups.  Auction episodes contain far
    # fewer transitions than CLOB episodes, so a shared threshold can defer
    # auction learning until after the first validation checkpoint.  ``None``
    # preserves the historical shared ``min_buffer`` behavior.
    min_buffer_clob: Optional[int] = None
    min_buffer_auction: Optional[int] = None
    safe_auction_initialization: bool = False
    # The auction has many more consequential actions than the CLOB and only
    # 30 decisions per episode.  A phase-specific multiplier supports
    # conservative exploration without slowing CLOB exploration.
    epsilon_auction_scale: float = 1.0
    # Q-space prior against replacing the canonical auction no-op before data
    # supports doing so.  Units are replay-scaled reward units.
    safe_auction_noop_margin: float = 1e-3
    # Curriculum boundary.  Before this episode the behavior policy holds the
    # canonical no-op in the auction, while its Q-network still learns from
    # those transitions.  At the boundary the complete admissible auction
    # action/cancellation set is unlocked.
    auction_learning_start_episode: int = 0
    # Retired development options remain false-only. LayerNorm is the v18
    # bounded-representation repair and retains the exact discrete grid.
    fixed_action_features: bool = False
    layer_norm: bool = False
    dueling: bool = False
    weight_decay: float = 0.0
    # Conditional on epsilon exploration: reserve probability mass for the
    # economically distinct zero-slope controls (wait / cancel without a new
    # order). Uniform sampling over 1,346 grid entries almost never tries them.
    # The remaining mass samples every admissible action uniformly.
    auction_control_exploration_probability: float = 0.0
    reference_centered_phases: tuple[str, ...] = ()
    # Inactive rejected diagnostic, retained to reproduce its saved checkpoints.
    known_auction_fee: bool = False


class DQNAgent(Agent):
    """Two-phase DQN over the discrete action grids (Phase 4)."""

    def __init__(self, cfg: ExperimentConfig, seeds: SeedBundle) -> None:
        if cfg.algo is None or cfg.algo.name != "dqn":
            raise ValueError("DQNAgent requires algo.name == 'dqn' in the config")
        self.cfg = cfg
        hyperparams = dict(cfg.algo.hyperparams)
        # Read the temporary all-phase development setting without changing
        # the saved configuration of those archived diagnostic candidates.
        if 'reference_centered_advantage' in hyperparams:
            enabled = hyperparams.pop('reference_centered_advantage')
            if not isinstance(enabled, bool):
                raise ValueError('legacy reference_centered_advantage must be boolean')
            if 'reference_centered_phases' in hyperparams:
                raise ValueError('specify reference_centered_phases only')
            hyperparams['reference_centered_phases'] = ['clob','auction'] if enabled else []
        self.hp: DQNHyperparams = build_hyperparams(DQNHyperparams, hyperparams)
        if self.hp.fixed_action_features or self.hp.dueling:
            raise ValueError("retired development architectures are not supported")
        if not np.isfinite(self.hp.weight_decay) or self.hp.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.hp.auction_control_exploration_probability < 1:
            raise ValueError("auction_control_exploration_probability must lie in [0,1)")
        if (set(self.hp.reference_centered_phases) - {'clob','auction'}
                or len(set(self.hp.reference_centered_phases)) != len(self.hp.reference_centered_phases)):
            raise ValueError('reference_centered_phases must be a subset of clob, auction')
        if not 0.0 <= self.hp.epsilon_auction_scale <= 1.0:
            raise ValueError("epsilon_auction_scale must lie in [0,1]")
        if self.hp.safe_auction_noop_margin < 0.0:
            raise ValueError("safe_auction_noop_margin must be nonnegative")
        if self.hp.auction_learning_start_episode < 0:
            raise ValueError("auction_learning_start_episode must be nonnegative")
        if self.hp.action_embedding_dim <= 0:
            raise ValueError("action_embedding_dim must be positive")
        self.artifact_schema_version = int(cfg.experiment.artifact_schema_version)
        self.chi = BELLMAN_FACTOR  # public compatibility; intentionally ignores cfg.rl.chi
        if not np.isfinite(self.hp.reward_scale) or self.hp.reward_scale <= 0:
            raise ValueError("DQN reward_scale must be finite and positive")
        if self.hp.update_every != 1 or self.hp.updates_per_env_step != 1:
            raise ValueError("revised DQN performs exactly one eligible update per environment step")
        self.device = torch.device(self.hp.device)

        self.clob_grid = ClobActionGrid(cfg.actions)
        self.auction_grid = AuctionActionGrid(cfg.actions)
        self._auction_control_indices = np.array([
            i for i, action in enumerate(self.auction_grid.actions) if action.K_a == 0
        ], dtype=np.int64)
        self._obs_dim = {"clob": len(cfg.features.clob), "auction": len(cfg.features.auction)}
        self._n_actions = {"clob": len(self.clob_grid), "auction": len(self.auction_grid)}
        mask_dim = max(self._n_actions.values())
        next_obs_dim = max(self._obs_dim.values())

        act_cls = ACTIVATIONS[self.hp.activation]
        hidden = self.hp.hidden_layers

        action_coordinates = {
            "clob": np.asarray(
                [
                    (
                        float(a.volume) / float(max(1, cfg.actions.V_max)),
                        float(a.delta) / float(max(1, cfg.actions.L_max)),
                    )
                    for a in self.clob_grid.actions
                ],
                dtype=np.float32,
            ),
            "auction": np.asarray(
                [
                    (
                        float(a.K_a) / float(cfg.actions.beta * cfg.actions.K_max),
                        float(a.ell) / float(max(1, cfg.actions.B_max)),
                        float(a.cancel),
                    )
                    for a in self.auction_grid.actions
                ],
                dtype=np.float32,
            ),
        }

        def make_pair(phase: str) -> tuple[StructuredDiscreteQ, StructuredDiscreteQ]:
            # Q = learned residual - known immediate cancellation fee.
            # Both regression and Bellman action selection use this full Q;
            # rewards, admissibility and the optimization criterion are unchanged.
            known_cost = None
            if phase == 'auction' and self.hp.known_auction_fee:
                known_cost = (cfg.reward.d * cfg.grid.tau_cl * self.hp.reward_scale,
                              -cfg.reward.d * cfg.grid.tau_op * self.hp.reward_scale)
            safe_margin = (
                self.hp.safe_auction_noop_margin
                if phase == "auction" and self.hp.safe_auction_initialization
                else None
            )
            q = StructuredDiscreteQ(
                self._obs_dim[phase],
                hidden,
                action_coordinates[phase],
                self.hp.action_embedding_dim,
                act_cls,
                noop_index=0,
                initial_noop_margin=safe_margin,
                layer_norm=self.hp.layer_norm,
                reference_centered=phase in self.hp.reference_centered_phases,
                known_action_cost=known_cost,
            ).to(self.device)
            tgt = StructuredDiscreteQ(
                self._obs_dim[phase],
                hidden,
                action_coordinates[phase],
                self.hp.action_embedding_dim,
                act_cls,
                noop_index=0,
                initial_noop_margin=safe_margin,
                layer_norm=self.hp.layer_norm,
                reference_centered=phase in self.hp.reference_centered_phases,
                known_action_cost=known_cost,
            ).to(self.device)
            tgt.load_state_dict(q.state_dict())
            for p in tgt.parameters():
                p.requires_grad_(False)
            return q, tgt

        self.q = {}
        self.q_target = {}
        self.optim = {}
        for phase in ("clob", "auction"):
            self.q[phase], self.q_target[phase] = make_pair(phase)
            self.optim[phase] = torch.optim.AdamW(
                self.q[phase].parameters(), lr=self.hp.lr,
                weight_decay=self.hp.weight_decay,
            ) if self.hp.weight_decay else torch.optim.Adam(
                self.q[phase].parameters(), lr=self.hp.lr,
            )

        loss_fns = {"huber": nn.SmoothL1Loss(), "mse": nn.MSELoss()}
        if self.hp.loss not in loss_fns:
            raise ValueError(f"unknown loss {self.hp.loss!r}; expected one of {sorted(loss_fns)}")
        self._loss_fn = loss_fns[self.hp.loss]

        # Replay: the CLOB buffer's junction rows hold the (shorter) auction
        # next-obs and the auction-grid mask, zero-/False-padded to the
        # buffer-wide maxima; update() un-pads after splitting on `junction`.
        self.replay = {
            "clob": ReplayBuffer(
                self.hp.buffer_size,
                self._obs_dim["clob"],
                next_obs_dim,
                mask_dim,
                seeds.generators["replay_clob"],
            ),
            "auction": ReplayBuffer(
                self.hp.buffer_size,
                self._obs_dim["auction"],
                self._obs_dim["auction"],
                mask_dim,
                seeds.generators["replay_auction"],
            ),
        }
        self._explore_rng = seeds.generators["exploration"]
        self.schedule = LinearEpsilonSchedule(
            self.hp.epsilon_start,
            self.hp.epsilon_end,
            self.hp.epsilon_decay_episodes,
            self.hp.epsilon_warmup_episodes,
        )
        self._episode = 0
        self._env_steps = 0
        self._update_count = {phase: 0 for phase in ("clob", "auction")}
        # Keep a distinct checkpoint-maturity counter so an explicitly
        # configured future delayed-auction ablation cannot count pre-unlock
        # updates.  In the active configuration the unlock episode is zero,
        # so this counter equals raw optimizer activity from the outset.
        self._checkpoint_update_count = {
            phase: 0 for phase in ("clob", "auction")
        }
        self._pending_update_phase: str | None = None
        self._training = True

    # -- lifecycle -------------------------------------------------------------

    def start_episode(self, episode: int) -> None:
        self._episode = int(episode)

    def set_train(self, training: bool) -> None:
        self._training = bool(training)
        for phase in ("clob", "auction"):
            self.q[phase].train(self._training)

    @property
    def epsilon(self) -> float:
        """Current CLOB exploration rate (evaluation always uses zero)."""
        return self.schedule.value(self._episode)

    def epsilon_for_phase(self, phase: str) -> float:
        """Return the configured phase-specific exploration probability."""
        epsilon = self.schedule.value(self._episode)
        if phase == "auction":
            return epsilon * self.hp.epsilon_auction_scale
        if phase != "clob":
            raise ValueError(f"unknown phase {phase!r}")
        return epsilon

    # -- acting ----------------------------------------------------------------

    def act(self, obs: np.ndarray, mask: np.ndarray, phase: str, *, eval_mode: bool = False) -> int:
        """Masked epsilon-greedy: exploration over admissible actions, with
        optional auction control coverage, else the masked greedy argmax (-inf on
        inadmissible entries). Eval mode is fully greedy and consumes no
        exploration randomness (the stream stays aligned across eval calls).
        """
        if not bool(np.any(mask)):
            raise ValueError("empty admissible set Adm(x); the MDP guarantees it is nonempty")
        explore = not eval_mode and self._training
        if (
            explore
            and phase == "auction"
            and self._episode < self.hp.auction_learning_start_episode
        ):
            # Index zero is always the canonical (K,ell,c)=(0,0,0), and hence
            # admissible both before and after a live schedule exists.
            if not bool(mask[0]):
                raise AssertionError("canonical auction no-op was masked")
            return 0
        if explore and self._explore_rng.random() < self.epsilon_for_phase(phase):
            if (phase == "auction" and self.hp.auction_control_exploration_probability > 0
                    and self._explore_rng.random() < self.hp.auction_control_exploration_probability):
                controls = self._auction_control_indices[mask[self._auction_control_indices]]
                if len(controls):
                    return int(controls[self._explore_rng.integers(len(controls))])
            admissible = np.flatnonzero(mask)
            return int(admissible[self._explore_rng.integers(len(admissible))])
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q[phase](obs_t).squeeze(0)
            qvals = qvals.masked_fill(
                ~torch.as_tensor(mask, dtype=torch.bool, device=self.device), -torch.inf
            )
            return self._lexicographic_greedy_index(qvals)

    @staticmethod
    def _lexicographic_greedy_index(masked_q: torch.Tensor) -> int:
        """Return the first maximum in the lexicographically ordered grid.

        Both action grids are constructed in lexicographic tuple order.  The
        explicit first-winner rule makes evaluation deterministic even when
        several admissible actions have identical Q values.
        """
        best = torch.max(masked_q)
        winners = torch.nonzero(masked_q == best, as_tuple=False).flatten()
        return int(winners[0].item())

    # -- replay ------------------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        """Store the EXECUTED transition (AUDIT N12) in the phase buffer;
        rewards are scaled by ``reward_scale`` INSIDE replay only (reported
        metrics stay in paper units). Eval transitions are not stored.

        On the terminal transition the env returns the combined reward
        ``r_step + g``.  Replay stores the two scaled pieces separately so the
        target can add each exactly once."""
        if not self._training:
            return
        tr = transition
        junction = tr.phase == "clob" and tr.next_phase == "auction"
        step_reward = float(tr.reward)
        terminal_value = 0.0
        if tr.done and tr.info is not None and "terminal_reward" in tr.info:
            r_term = float(tr.info["terminal_reward"])
            step_reward -= r_term  # un-fold: store the step reward only
            terminal_value = r_term
        self.replay[tr.phase].add(
            obs=np.asarray(tr.obs, dtype=np.float32),
            action=int(tr.action),
            reward=step_reward * self.hp.reward_scale,
            next_obs=None if tr.done else np.asarray(tr.next_obs, dtype=np.float32),
            done=tr.done,
            junction=junction,
            next_mask=None if tr.done else tr.next_mask,
            terminal_value=terminal_value * self.hp.reward_scale,
            discount=BELLMAN_FACTOR,
        )
        self._env_steps += 1
        self._pending_update_phase = tr.phase

    # -- learning -----------------------------------------------------------------

    def update(self, phase: str | None = None) -> dict[str, float]:
        """Perform at most one update from the current transition's buffer."""
        if not self._training or self._env_steps == 0:
            return {}
        pending = self._pending_update_phase
        if pending is None:
            # The one update opportunity associated with the latest observed
            # transition has already been consumed.  Supplying ``phase``
            # explicitly must not manufacture another optimizer step.
            return {}
        if phase is None:
            phase = pending
        elif phase != pending:
            raise ValueError(f"update phase {phase!r} does not match observed phase {pending!r}")
        # Consume the per-transition update opportunity even when warm-up has
        # not completed, preventing repeated calls from updating one env step.
        self._pending_update_phase = None
        if phase not in self.replay:
            raise ValueError(f"unknown phase {phase!r}")
        if (self.cfg.rl.learning_starts_after_warmup
                and self._episode < self.cfg.rl.structured_warmup_episodes):
            return {}
        phase_min_buffer = getattr(self.hp, f"min_buffer_{phase}")
        if phase_min_buffer is None:
            phase_min_buffer = self.hp.min_buffer
        if len(self.replay[phase]) < int(phase_min_buffer):
            return {}
        stats = self._gradient_step(phase, self.replay[phase].sample(self.hp.batch_size))
        self._update_count[phase] += 1
        if (
            phase != "auction"
            or self._episode >= self.hp.auction_learning_start_episode
        ):
            self._checkpoint_update_count[phase] += 1
        self._sync_target_after_update(phase)
        return {
            **{f"{k}_{phase}": v for k, v in stats.items()},
            f"n_grad_steps_{phase}": 1.0,
        }

    def compute_targets(self, phase: str, batch: ReplayBatch) -> torch.Tensor:
        """Bellman targets y = r + (1 - done) max_{a' in Adm(x')}
        Q_target(x', a'), with the target network chosen by the PHASE of x':
        CLOB junction rows (next state at the auction open) bootstrap from
        the AUCTION target network. Done rows do NOT bootstrap from a network;
        instead they add the known terminal value ``g`` exactly once. Exposed
        for the hand-computed unit tests
        (tests/test_dqn.py)."""
        y = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).clone()
        done = torch.as_tensor(batch.done, dtype=torch.bool, device=self.device)
        junction = torch.as_tensor(batch.junction, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            for next_phase, rows in (
                (phase, ~done & ~junction),
                ("auction" if phase == "clob" else phase, ~done & junction),
            ):
                if not bool(rows.any()):
                    continue
                idx = rows.nonzero(as_tuple=True)[0]
                nobs = torch.as_tensor(
                    batch.next_obs[idx.cpu().numpy(), : self._obs_dim[next_phase]],
                    dtype=torch.float32,
                    device=self.device,
                )
                nmask = torch.as_tensor(
                    batch.next_mask[idx.cpu().numpy(), : self._n_actions[next_phase]],
                    dtype=torch.bool,
                    device=self.device,
                )
                if self.hp.double_q:
                    # Double-DQN: argmax over Adm(x') with the ONLINE net,
                    # value from the TARGET net (van Hasselt et al. 2016).
                    q_sel = self.q[next_phase](nobs).masked_fill(~nmask, -torch.inf)
                    a_star = q_sel.argmax(dim=1, keepdim=True)
                    q_eval = self.q_target[next_phase](nobs)
                    y[idx] = y[idx] + q_eval.gather(1, a_star).squeeze(1)
                else:
                    qt = self.q_target[next_phase](nobs).masked_fill(~nmask, -torch.inf)
                    y[idx] = y[idx] + qt.max(dim=1).values
        # Terminal value is added only to terminal rows; there is no network
        # bootstrap and no discount on the final transition.
        if batch.terminal_value is not None:
            g = torch.as_tensor(batch.terminal_value, dtype=torch.float32, device=self.device)
            y[done] = y[done] + g[done]
        return y

    def _gradient_step(self, phase: str, batch: ReplayBatch) -> dict[str, float]:
        y = self.compute_targets(phase, batch)
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device)
        q_pred = self.q[phase](obs, actions)
        loss = self._loss_fn(q_pred, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite DQN {phase} loss; refusing a corrupt update")
        self.optim[phase].zero_grad()
        for group in self.optim[phase].param_groups:
            group['lr'] = self.hp.lr * learning_rate_factor(self.cfg.rl, self._episode)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q[phase].parameters(), self.hp.grad_clip_norm)
        self.optim[phase].step()
        td = (q_pred - y).detach()
        return {
            "loss": float(loss.item()),
            "learning_rate": self.optim[phase].param_groups[0]['lr'],
            "grad_norm": float(grad_norm.item()),
            "td_abs_mean": float(td.abs().mean().item()),
            "td_abs_max": float(td.abs().max().item()),
        }

    def _sync_target_after_update(self, phase: str) -> None:
        """Update only the target paired with the optimizer step just taken."""
        if self.hp.target_soft_tau is not None:
            tau = self.hp.target_soft_tau
            with torch.no_grad():
                for p, pt in zip(self.q[phase].parameters(), self.q_target[phase].parameters()):
                    pt.mul_(1.0 - tau).add_(p, alpha=tau)
        elif self._update_count[phase] % self.hp.target_update_interval == 0:
            self.q_target[phase].load_state_dict(self.q[phase].state_dict())

    # -- checkpointing (resumable; D10) -------------------------------------------

    def _checkpoint_contract(self) -> dict[str, object]:
        """Configuration that gives saved Q values their exact semantics.

        Tensor shapes alone cannot distinguish, for example, two auction
        grids with the same number of entries but different slope units. The
        publication entry points already compare resolved configs; retaining
        this contract in the agent checkpoint makes direct API use fail closed
        as well. ``device`` is excluded so a checkpoint remains portable.
        """
        hyperparams = {
            key: value
            for key, value in self.hp.__dict__.items()
            if key != "device"
        }
        return {
            "algorithm": "dqn",
            # Operational identity (run name/path, seeds, episode budget and
            # validation cadence) is deliberately excluded. Everything that
            # changes trajectories, observations, rewards, or Bellman values
            # is included, including the shaped-vs-economic objective switch.
            "auction_enabled": bool(self.cfg.experiment.auction_enabled),
            "grid": asdict(self.cfg.grid),
            "clob_flow": asdict(self.cfg.clob_flow),
            "auction_flow": asdict(self.cfg.auction_flow),
            "midprice": asdict(self.cfg.midprice),
            "algo1": asdict(self.cfg.algo1),
            "reward": asdict(self.cfg.reward),
            "bellman": {
                "chi": float(self.cfg.rl.chi),
                "discount_mode": self.cfg.rl.discount_mode,
            },
            "actions": asdict(self.cfg.actions),
            "features": asdict(self.cfg.features),
            "h_cl_feature_enabled": bool(self.cfg.rl.h_cl_feature_enabled),
            "relative_price_features": self.cfg.rl.relative_price_features,
            "n_step": self.cfg.rl.n_step,
            **learning_conditioning_contract(self.cfg.rl),
            "hyperparams_except_device": hyperparams,
        }

    def save(self, path: str | Path, *, include_replay: bool = False) -> None:
        """Full checkpoint: networks, targets, optimizers, counters, RNG
        states (exploration numpy + torch global), optionally the replay
        contents (needed for bit-identical --resume; omitted from the
        initial/best/final snapshots to keep them small)."""
        state = {
            "artifact_schema_version": self.artifact_schema_version,
            "environment_contract": environment_contract(self.cfg),
            "q_architecture": self.q["clob"].architecture,
            "q_architectures": {phase: network.architecture for phase, network in self.q.items()},
            "dqn_contract": self._checkpoint_contract(),
            "feature_normalizer": self._feature_normalizer_state(),
            "hyperparams": self.hp.__dict__,
            "episode": self._episode,
            "env_steps": self._env_steps,
            "update_count": dict(self._update_count),
            "checkpoint_update_count": dict(self._checkpoint_update_count),
            "pending_update_phase": self._pending_update_phase,
            "q": {p: self.q[p].state_dict() for p in self.q},
            "q_target": {p: self.q_target[p].state_dict() for p in self.q_target},
            "optim": {p: self.optim[p].state_dict() for p in self.optim},
            "explore_rng_state": self._explore_rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "replay": (
                {p: self.replay[p].state_dict() for p in self.replay} if include_replay else None
            ),
        }
        torch.save(state, Path(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        saved_schema = state.get("artifact_schema_version")
        if saved_schema is None:
            raise ValueError(
                "checkpoint is missing artifact_schema_version; old checkpoints "
                "cannot be loaded into the revised DQN agent"
            )
        if int(saved_schema) != self.artifact_schema_version:
            raise ValueError(
                "checkpoint artifact_schema_version mismatch: "
                f"expected {self.artifact_schema_version}, got {saved_schema}"
            )
        if state.get("environment_contract") != environment_contract(self.cfg):
            raise ValueError(
                "checkpoint environment contract mismatch; old auction/grid "
                "checkpoints cannot be loaded by the revised pipeline"
            )
        if state.get("q_architecture") != self.q["clob"].architecture:
            raise ValueError(
                "checkpoint DQN architecture mismatch; this checkpoint predates "
                "the action-coordinate-conditioned Q network"
            )
        if ('q_architectures' in state and state['q_architectures'] !=
                {phase: network.architecture for phase, network in self.q.items()}):
            raise ValueError('checkpoint DQN phase architecture mismatch')
        contract = state.get("dqn_contract")
        if isinstance(contract, dict) and isinstance(contract.get('algo1'), dict):
            contract['algo1'].setdefault('clob_forecast_weights', ())
            contract['algo1'].setdefault('clob_forecast_mechanism', LEGACY_CLEARING)
        if isinstance(contract, dict) and isinstance(contract.get('auction_flow'), dict):
            contract['auction_flow'].setdefault('clearing_mechanism', LEGACY_CLEARING)
        if isinstance(contract, dict):
            contract.get('hyperparams_except_device', {}).setdefault('weight_decay', 0.0)
            contract.get('hyperparams_except_device', {}).setdefault('auction_control_exploration_probability', 0.0)
            contract.get('hyperparams_except_device', {}).setdefault('known_auction_fee', False)
            hp_contract = contract.get('hyperparams_except_device', {})
            legacy_centering = hp_contract.pop('reference_centered_advantage', False)
            hp_contract.setdefault('reference_centered_phases', ('clob','auction') if legacy_centering else ())
            historical = contract.get('midprice', {}).get('historical')
            if isinstance(historical, dict):
                historical.setdefault('training_pool', 'symbol')
        if isinstance(contract, dict) and isinstance(contract.get('reward'), dict):
            contract['reward'].setdefault('auction_shaping_weight', 1.0)
        if contract != self._checkpoint_contract():
            raise ValueError(
                "checkpoint DQN configuration contract mismatch; simulator, "
                "reward, action, feature, or hyperparameter semantics differ"
            )
        if state.get("feature_normalizer") is None:
            raise ValueError(
                "checkpoint predates the frozen feature-normalization contract"
            )
        self._load_feature_normalizer_state(state["feature_normalizer"])
        self._episode = int(state["episode"])
        self._env_steps = int(state["env_steps"])
        self._update_count = {p: int(state["update_count"][p]) for p in ("clob", "auction")}
        self._checkpoint_update_count = {
            p: int(state["checkpoint_update_count"][p])
            for p in ("clob", "auction")
        }
        self._pending_update_phase = state.get("pending_update_phase")
        for p in ("clob", "auction"):
            self.q[p].load_state_dict(state["q"][p])
            self.q_target[p].load_state_dict(state["q_target"][p])
            self.optim[p].load_state_dict(state["optim"][p])
        self._explore_rng.bit_generator.state = state["explore_rng_state"]
        torch.set_rng_state(state["torch_rng_state"])
        if state.get("replay") is not None:
            for p in ("clob", "auction"):
                self.replay[p].load_state_dict(state["replay"][p])
