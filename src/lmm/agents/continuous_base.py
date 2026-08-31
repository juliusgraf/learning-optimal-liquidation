"""Shared scaffolding for projected continuous-control agents.

``ContinuousActorCriticAgent`` implements the per-phase actor/critic training
common to DDPG, TD3 and SAC; the three differ only in a handful of overridable
hooks (critic count, target action, actor loss, exploration). The full spec is
in docs/continuous_action_extension.md. This is the SEPARATE, clearly labeled
relaxation — the discrete DQN setting is untouched.

Design mirrors the DQN (docs/rl_design.md):
- TWO phase networks (CLOB and auction) with their own actor/critic(s), target
  networks, optimizers and replay buffer — a documented design choice.
- Cross-phase junction: a CLOB transition whose next state is the auction open
  bootstraps from the AUCTION networks; a terminal row adds its known terminal
  value exactly once and never bootstraps from a network.
- Exactly one eligible optimizer update is sampled from the current phase;
  phase-local Polyak updates occur only after the corresponding update.
- Actors, behavior noise, critics, replay, and target smoothing all use raw
  normalized proposals in ``[-1, 1]``.  The environment adapter alone applies
  the state-dependent market-action projection.
- Seeding (D10): exploration noise from the numpy ``exploration`` generator,
  replay sampling from ``replay_clob`` / ``replay_auction``, networks + any
  reparameterized sampling from the seeded torch global RNG. No global numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lmm.agents.base import (
    BELLMAN_FACTOR,
    ENVIRONMENT_CONTRACT,
    REWARD_SCALE,
    Agent,
    Transition,
)
from lmm.config import ExperimentConfig
from lmm.env.action_spaces import continuous_action_specs
from lmm.rl.continuous_replay import ContinuousReplayBatch, ContinuousReplayBuffer
from lmm.rl.networks import ACTIVATIONS, Critic, DeterministicActor
from lmm.utils.seeding import SeedBundle

__all__ = ["ContinuousHyperparams", "ContinuousActorCriticAgent"]

_OU_THETA = 0.15  # Ornstein-Uhlenbeck mean-reversion (standard DDPG default)


@dataclass(frozen=True)
class ContinuousHyperparams:
    """Knobs shared by DDPG/TD3/SAC; REQUIRED in configs/algo/*.yaml (configs
    are the single source of truth, ruling D6 convention)."""

    buffer_size: int
    min_buffer: int
    batch_size: int
    actor_lr: float
    critic_lr: float
    hidden_layers: tuple[int, ...]
    target_soft_tau: float
    grad_clip_norm: float
    updates_per_env_step: int
    update_every: int
    reward_scale: float
    activation: str
    device: str
    checkpoint_interval_episodes: int


class ContinuousActorCriticAgent(Agent):
    """Two-phase off-policy actor-critic base (DDPG/TD3/SAC)."""

    # Subclass overrides:
    n_critics: int = 1  # DDPG=1, TD3/SAC=2
    uses_target_actor: bool = True  # SAC has no target actor

    PHASES = ("clob", "auction")

    def __init__(self, cfg: ExperimentConfig, seeds: SeedBundle, hp: ContinuousHyperparams) -> None:
        self.cfg = cfg
        self.hp = hp
        self.artifact_schema_version = int(cfg.experiment.artifact_schema_version)
        self.chi = BELLMAN_FACTOR  # public compatibility; intentionally ignores cfg.rl.chi
        if not np.isclose(hp.reward_scale, REWARD_SCALE, rtol=0.0, atol=1e-15):
            raise ValueError(
                f"revised continuous agents require reward_scale={REWARD_SCALE:g}, "
                f"got {hp.reward_scale:g}"
            )
        if hp.update_every != 1 or hp.updates_per_env_step != 1:
            raise ValueError(
                "revised continuous agents perform exactly one eligible update per environment step"
            )
        self.device = torch.device(hp.device)
        self.specs = continuous_action_specs(cfg)
        self.continuous_cancel = (
            "threshold" if self.specs["auction"].dim == 3 else "never"
        )
        self._obs_dim = {"clob": len(cfg.features.clob), "auction": len(cfg.features.auction)}
        self._act_dim = {p: self.specs[p].dim for p in self.PHASES}
        # Every learning-side action is the raw normalized proposal.  Physical
        # action bounds live exclusively in the environment adapter.
        self._low = {
            p: -torch.ones(self._act_dim[p], dtype=torch.float32, device=self.device)
            for p in self.PHASES
        }
        self._high = {
            p: torch.ones(self._act_dim[p], dtype=torch.float32, device=self.device)
            for p in self.PHASES
        }

        self._setup_networks()

        next_obs_dim = max(self._obs_dim.values())
        self.replay = {
            "clob": ContinuousReplayBuffer(
                hp.buffer_size, self._obs_dim["clob"], self._act_dim["clob"], next_obs_dim,
                seeds.generators["replay_clob"],
            ),
            "auction": ContinuousReplayBuffer(
                hp.buffer_size, self._obs_dim["auction"], self._act_dim["auction"], self._obs_dim["auction"],
                seeds.generators["replay_auction"],
            ),
        }
        self._explore_rng = seeds.generators["exploration"]
        self._episode = 0
        self._env_steps = 0
        self._update_count = {p: 0 for p in self.PHASES}
        self._pending_update_phase: str | None = None
        self._ou_state: dict[str, np.ndarray] = {}
        self._training = True
        self._exploration_scale = 0.0  # reported as the "epsilon" metric; subclass sets it
        self._post_setup(seeds)

    # -- construction hooks ---------------------------------------------------

    def _make_actor(self, phase: str, obs_dim: int, hidden, act_cls) -> nn.Module:
        """Default deterministic actor (DDPG/TD3); SAC overrides."""
        dim = self._act_dim[phase]
        return DeterministicActor(
            obs_dim,
            -np.ones(dim, dtype=np.float32),
            np.ones(dim, dtype=np.float32),
            hidden,
            act_cls,
        )

    def _setup_networks(self) -> None:
        hidden = self.hp.hidden_layers
        act_cls = ACTIVATIONS[self.hp.activation]
        self.actor, self.actor_target, self.actor_optim = {}, {}, {}
        self.critics, self.critic_targets, self.critic_optim = {}, {}, {}
        for phase in self.PHASES:
            obs_dim = self._obs_dim[phase]
            actor = self._make_actor(phase, obs_dim, hidden, act_cls).to(self.device)
            self.actor[phase] = actor
            self.actor_optim[phase] = torch.optim.Adam(actor.parameters(), lr=self.hp.actor_lr)
            if self.uses_target_actor:
                tgt = self._make_actor(phase, obs_dim, hidden, act_cls).to(self.device)
                tgt.load_state_dict(actor.state_dict())
                for p in tgt.parameters():
                    p.requires_grad_(False)
                self.actor_target[phase] = tgt
            critics = [Critic(obs_dim, self._act_dim[phase], hidden, act_cls).to(self.device) for _ in range(self.n_critics)]
            ctgts = [Critic(obs_dim, self._act_dim[phase], hidden, act_cls).to(self.device) for _ in range(self.n_critics)]
            for c, ct in zip(critics, ctgts):
                ct.load_state_dict(c.state_dict())
                for p in ct.parameters():
                    p.requires_grad_(False)
            self.critics[phase] = critics
            self.critic_targets[phase] = ctgts
            params = [p for c in critics for p in c.parameters()]
            self.critic_optim[phase] = torch.optim.Adam(params, lr=self.hp.critic_lr)

    def _post_setup(self, seeds: SeedBundle) -> None:
        """Hook for subclass extras (e.g. SAC temperature). Default: no-op."""

    # -- lifecycle ------------------------------------------------------------

    def start_episode(self, episode: int) -> None:
        self._episode = int(episode)
        self._ou_state = {}  # reset OU exploration state per episode

    def set_train(self, training: bool) -> None:
        self._training = bool(training)
        for phase in self.PHASES:
            self.actor[phase].train(self._training)
            for c in self.critics[phase]:
                c.train(self._training)

    @property
    def epsilon(self) -> float:
        """Reported in the metrics ``epsilon`` column: the exploration-noise
        scale (0 for SAC, whose policy is intrinsically stochastic)."""
        return float(self._exploration_scale)

    # -- acting ---------------------------------------------------------------

    def act(self, obs: np.ndarray, mask: np.ndarray, phase: str, *, eval_mode: bool = False) -> np.ndarray:
        """Return a raw normalized proposal in ``[-1, 1]``.

        ``mask`` is ignored for selection: the adapter is solely responsible
        for the state-dependent projection and executable market action.
        """
        greedy = eval_mode or not self._training
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self._select_action(phase, obs_t, greedy=greedy)

    def _select_action(self, phase: str, obs_t: torch.Tensor, *, greedy: bool) -> np.ndarray:
        """Deterministic actor plus normalized-coordinate exploration noise.

        The adapter performs and logs the final box clipping.  Keeping the
        pre-clip noisy proposal until that boundary is what makes clipping and
        saturation diagnostics observable without changing the replay action:
        replay still receives the adapter's committed ``[-1,1]`` proposal.
        """
        a = self.actor[phase](obs_t).squeeze(0).cpu().numpy()
        if not greedy:
            a = a + self._sample_noise(phase)
        return np.asarray(a, dtype=np.float32)

    def _sample_noise(self, phase: str) -> np.ndarray:
        """Action-space exploration noise from the seeded numpy generator.
        DDPG/TD3 set ``exploration_noise`` / ``exploration_noise_std``."""
        dim = self._act_dim[phase]
        scale = float(getattr(self.hp, "exploration_noise_std", 0.0))
        kind = getattr(self.hp, "exploration_noise", "gaussian")
        if kind == "gaussian":
            return self._explore_rng.normal(0.0, scale, size=dim)
        if kind == "ou":
            ou = self._ou_state.get(phase, np.zeros(dim))
            ou = ou - _OU_THETA * ou + self._explore_rng.normal(0.0, scale, size=dim)
            self._ou_state[phase] = ou
            return ou
        raise ValueError(f"exploration_noise must be 'gaussian' or 'ou', got {kind!r}")

    # -- replay ---------------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        if not self._training:
            return
        tr = transition
        junction = tr.phase == "clob" and tr.next_phase == "auction"
        if tr.info is not None and "proposal_action_vec" in tr.info:
            action_vec = np.asarray(tr.info["proposal_action_vec"], dtype=np.float32)
        elif tr.info is not None and "executed_action_vec" in tr.info:
            # Name compatibility only; revised adapters store the raw
            # normalized proposal under either key.
            action_vec = np.asarray(tr.info["executed_action_vec"], dtype=np.float32)
        else:
            action_vec = np.asarray(tr.action, dtype=np.float32)
        # next_cancel_admissible: C(x') > 0 when x' is an auction state. The
        # adapter's mask is all-True iff a cancel-all is admissible there.
        next_cancel_adm = False
        if (
            self.continuous_cancel == "threshold"
            and not tr.done
            and tr.next_phase == "auction"
            and tr.next_mask is not None
        ):
            next_cancel_adm = bool(np.asarray(tr.next_mask).all())
        # Un-fold the terminal clearing reward and scale both components once.
        step_reward = float(tr.reward)
        terminal_value = 0.0
        if tr.done and tr.info is not None and "terminal_reward" in tr.info:
            r_term = float(tr.info["terminal_reward"])
            step_reward -= r_term
            terminal_value = r_term
        reward = step_reward * self.hp.reward_scale
        tv = terminal_value * self.hp.reward_scale
        self.replay[tr.phase].add(
            obs=np.asarray(tr.obs, dtype=np.float32),
            action=action_vec,
            reward=reward,
            next_obs=None if tr.done else np.asarray(tr.next_obs, dtype=np.float32),
            done=tr.done,
            junction=junction,
            next_cancel_admissible=next_cancel_adm,
            terminal_value=tv,
            discount=BELLMAN_FACTOR,
        )
        self._env_steps += 1
        self._pending_update_phase = tr.phase

    # -- learning -------------------------------------------------------------

    def update(self, phase: str | None = None) -> dict[str, float]:
        if not self._training or self._env_steps == 0:
            return {}
        pending = self._pending_update_phase
        if pending is None:
            # One transition grants at most one optimizer step.  An explicit
            # phase argument cannot reopen an already-consumed opportunity.
            return {}
        if phase is None:
            phase = pending
        elif phase != pending:
            raise ValueError(f"update phase {phase!r} does not match observed phase {pending!r}")
        self._pending_update_phase = None
        if phase not in self.replay:
            raise ValueError(f"unknown phase {phase!r}")
        if len(self.replay[phase]) < self.hp.min_buffer:
            return {}
        stats = self._gradient_step(phase, self.replay[phase].sample(self.hp.batch_size))
        return {
            **{f"{k}_{phase}": v for k, v in stats.items()},
            f"n_grad_steps_{phase}": 1.0,
        }

    def _gradient_step(self, phase: str, batch: ContinuousReplayBatch) -> dict[str, float]:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.compute_targets(phase, batch)
        out = self._update_critics(phase, obs, action, y)
        self._update_count[phase] += 1
        if self._should_update_actor(phase):
            out.update(self._update_actor(phase, obs))
            self._soft_update_targets(phase)
        return out

    def compute_targets(self, phase: str, batch: ContinuousReplayBatch) -> torch.Tensor:
        """Undiscounted targets with the cross-phase junction rule.

        Target networks are chosen by the phase of the next state. Terminal
        rows add their known terminal value exactly once and never use a
        network bootstrap.
        """
        y = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).clone()
        done = batch.done
        junction = batch.junction
        groups = (
            (phase, (~done) & (~junction)),
            ("auction" if phase == "clob" else phase, (~done) & junction),
        )
        with torch.no_grad():
            for next_phase, row_mask in groups:
                rows = np.flatnonzero(row_mask)
                if rows.size == 0:
                    continue
                nobs = torch.as_tensor(
                    batch.next_obs[rows, : self._obs_dim[next_phase]],
                    dtype=torch.float32,
                    device=self.device,
                )
                cadm = batch.next_cancel_admissible[rows]
                a_next, logp_next = self._target_next_action(next_phase, nobs, cadm)
                q_next = self._target_q(next_phase, nobs, a_next)
                if logp_next is not None:  # SAC entropy term
                    q_next = q_next - self._alpha(next_phase) * logp_next
                idx = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
                y[idx] = y[idx] + q_next
        if batch.terminal_value is not None:
            g = torch.as_tensor(batch.terminal_value, dtype=torch.float32, device=self.device)
            done_t = torch.as_tensor(done, dtype=torch.bool, device=self.device)
            y[done_t] = y[done_t] + g[done_t]
        return y

    # -- algorithm hooks (overridable) ----------------------------------------

    def _should_update_actor(self, phase: str) -> bool:
        """DDPG/SAC: every step. TD3 overrides for delayed updates."""
        return True

    def _target_next_action(self, phase: str, next_obs: torch.Tensor, cadm: np.ndarray):
        """Deterministic target action (DDPG); TD3 adds smoothing, SAC samples.
        Returns (action, logp or None).  The action remains a normalized proposal;
        the environment map Gamma_x applies cancellation admissibility, just
        as it does for behavior/replay actions."""
        a = self.actor_target[phase](next_obs)
        return a, None

    def _target_q(self, phase: str, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        qs = [ct(obs, action) for ct in self.critic_targets[phase]]
        q = qs[0]
        for qi in qs[1:]:
            q = torch.minimum(q, qi)
        return q

    def _alpha(self, phase: str) -> torch.Tensor:
        """SAC temperature; base (no entropy) returns 0."""
        return torch.zeros((), device=self.device)

    def _update_critics(self, phase: str, obs: torch.Tensor, action: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.critic_optim[phase].zero_grad()
        losses, td_abs = [], []
        for c in self.critics[phase]:
            q = c(obs, action)
            losses.append(F.mse_loss(q, y))
            td_abs.append((q - y).detach().abs())
        total = sum(losses)
        total.backward()
        params = [p for c in self.critics[phase] for p in c.parameters()]
        gnorm = nn.utils.clip_grad_norm_(params, self.hp.grad_clip_norm)
        self.critic_optim[phase].step()
        td = torch.cat(td_abs)
        return {
            "loss": float((total / len(losses)).item()),  # mean critic loss -> CSV loss_{phase}
            "grad_norm": float(gnorm.item()),
            "td_abs_mean": float(td.mean().item()),
            "td_abs_max": float(td.max().item()),
        }

    def _update_actor(self, phase: str, obs: torch.Tensor) -> dict[str, float]:
        """Deterministic policy gradient (DDPG/TD3): maximize Q1(x, mu(x))."""
        a = self.actor[phase](obs)
        actor_loss = -self.critics[phase][0](obs, a).mean()
        self.actor_optim[phase].zero_grad()
        actor_loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.actor[phase].parameters(), self.hp.grad_clip_norm)
        self.actor_optim[phase].step()
        return {"actor_loss": float(actor_loss.item()), "actor_grad_norm": float(gnorm.item())}

    # -- helpers --------------------------------------------------------------

    def _soft_update_targets(self, phase: str) -> None:
        tau = self.hp.target_soft_tau
        with torch.no_grad():
            if self.uses_target_actor:
                self._polyak(self.actor[phase], self.actor_target[phase], tau)
            for c, ct in zip(self.critics[phase], self.critic_targets[phase]):
                self._polyak(c, ct, tau)

    @staticmethod
    def _polyak(src: nn.Module, dst: nn.Module, tau: float) -> None:
        for p, pt in zip(src.parameters(), dst.parameters()):
            pt.mul_(1.0 - tau).add_(p, alpha=tau)

    # -- checkpointing (resumable; D10) ---------------------------------------

    def save(self, path: str | Path, *, include_replay: bool = False) -> None:
        state = {
            "artifact_schema_version": self.artifact_schema_version,
            "environment_contract": ENVIRONMENT_CONTRACT,
            "feature_normalizer": self._feature_normalizer_state(),
            "hyperparams": self.hp.__dict__,
            "episode": self._episode,
            "env_steps": self._env_steps,
            "update_count": dict(self._update_count),
            "pending_update_phase": self._pending_update_phase,
            "actor": {p: self.actor[p].state_dict() for p in self.PHASES},
            "actor_optim": {p: self.actor_optim[p].state_dict() for p in self.PHASES},
            "critics": {p: [c.state_dict() for c in self.critics[p]] for p in self.PHASES},
            "critic_optim": {p: self.critic_optim[p].state_dict() for p in self.PHASES},
            "explore_rng_state": self._explore_rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "extra": self._extra_state(),
            "replay": (
                {p: self.replay[p].state_dict() for p in self.PHASES} if include_replay else None
            ),
        }
        if self.uses_target_actor:
            state["actor_target"] = {p: self.actor_target[p].state_dict() for p in self.PHASES}
        state["critic_targets"] = {p: [c.state_dict() for c in self.critic_targets[p]] for p in self.PHASES}
        torch.save(state, Path(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        saved_schema = state.get("artifact_schema_version")
        if saved_schema is None:
            raise ValueError(
                "checkpoint is missing artifact_schema_version; old checkpoints "
                "cannot be loaded into the revised continuous-control agent"
            )
        if int(saved_schema) != self.artifact_schema_version:
            raise ValueError(
                "checkpoint artifact_schema_version mismatch: "
                f"expected {self.artifact_schema_version}, got {saved_schema}"
            )
        if state.get("environment_contract") != ENVIRONMENT_CONTRACT:
            raise ValueError(
                "checkpoint environment contract mismatch; old auction/grid "
                "checkpoints cannot be loaded by the revised pipeline"
            )
        if state.get("feature_normalizer") is None:
            raise ValueError(
                "checkpoint predates the frozen feature-normalization contract"
            )
        self._load_feature_normalizer_state(state["feature_normalizer"])
        self._episode = int(state["episode"])
        self._env_steps = int(state["env_steps"])
        self._update_count = dict(state["update_count"])
        self._pending_update_phase = state.get("pending_update_phase")
        for p in self.PHASES:
            self.actor[p].load_state_dict(state["actor"][p])
            self.actor_optim[p].load_state_dict(state["actor_optim"][p])
            for c, sd in zip(self.critics[p], state["critics"][p]):
                c.load_state_dict(sd)
            self.critic_optim[p].load_state_dict(state["critic_optim"][p])
            for c, sd in zip(self.critic_targets[p], state["critic_targets"][p]):
                c.load_state_dict(sd)
            if self.uses_target_actor:
                self.actor_target[p].load_state_dict(state["actor_target"][p])
        self._explore_rng.bit_generator.state = state["explore_rng_state"]
        torch.set_rng_state(state["torch_rng_state"])
        self._load_extra_state(state.get("extra"))
        if state.get("replay") is not None:
            for p in self.PHASES:
                self.replay[p].load_state_dict(state["replay"][p])

    def _extra_state(self):
        """Subclass extra checkpoint state (SAC temperature). Default: None."""
        return None

    def _load_extra_state(self, extra) -> None:
        """Restore subclass extra checkpoint state. Default: no-op."""
