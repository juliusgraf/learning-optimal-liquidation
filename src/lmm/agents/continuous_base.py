"""Shared scaffolding for the continuous-action agents (Phase 5; ruling D9).

``ContinuousActorCriticAgent`` implements the per-phase actor/critic training
common to DDPG, TD3 and SAC; the three differ only in a handful of overridable
hooks (critic count, target action, actor loss, exploration). The full spec is
in docs/continuous_action_extension.md. This is the SEPARATE, clearly labeled
relaxation — the discrete DQN setting is untouched.

Design mirrors the DQN (docs/rl_design.md):
- TWO phase networks (CLOB and auction) with their own actor/critic(s), target
  networks, optimizers and replay buffer — a documented design choice.
- Cross-phase junction: a CLOB transition whose next state is the auction open
  bootstraps from the AUCTION target networks; the terminal tau_cl reward is
  folded into the final auction transition with done=True and ZERO bootstrap.
- Per-env-step updates gated on ``update_every`` / ``min_buffer``; Polyak
  target updates (``target_soft_tau``); reward scaled inside replay only.
- Seeding (D10): exploration noise from the numpy ``exploration`` generator,
  replay sampling from ``replay_clob`` / ``replay_auction``, networks + any
  reparameterized sampling from the seeded torch global RNG. No global numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lmm.agents.base import Agent, Transition
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
    continuous_cancel: str  # "threshold" | "never"
    eval_interval_episodes: int
    eval_n_seeds: int
    final_eval_n_seeds: int
    checkpoint_interval_episodes: int
    # Optional symmetric clip on the SCALED reward stored in replay (None =
    # off). The paper's per-step fictive auction reward is unbounded; the
    # deterministic/gradient actors exploit it, so the MSE critic targets blow
    # up even after reward_scale. Clipping bounds the regression target
    # (replay-only; reported metrics stay in paper units). kw_only so the
    # default does not collide with subclasses' required positional fields
    # (DDPG/TD3/SAC add their own); the default keeps frozen fixture configs
    # (which predate this knob) loading.
    reward_clip: Optional[float] = field(default=None, kw_only=True)


class ContinuousActorCriticAgent(Agent):
    """Two-phase off-policy actor-critic base (DDPG/TD3/SAC)."""

    # Subclass overrides:
    n_critics: int = 1  # DDPG=1, TD3/SAC=2
    uses_target_actor: bool = True  # SAC has no target actor

    PHASES = ("clob", "auction")

    def __init__(self, cfg: ExperimentConfig, seeds: SeedBundle, hp: ContinuousHyperparams) -> None:
        self.cfg = cfg
        self.hp = hp
        self.chi = cfg.rl.chi
        self.device = torch.device(hp.device)
        self.continuous_cancel = hp.continuous_cancel
        self.specs = continuous_action_specs(cfg, hp.continuous_cancel)
        self._obs_dim = {"clob": len(cfg.features.clob), "auction": len(cfg.features.auction)}
        self._act_dim = {p: self.specs[p].dim for p in self.PHASES}
        # Box bounds as device tensors (for clamping target actions).
        self._low = {p: torch.as_tensor(self.specs[p].low, dtype=torch.float32, device=self.device) for p in self.PHASES}
        self._high = {p: torch.as_tensor(self.specs[p].high, dtype=torch.float32, device=self.device) for p in self.PHASES}

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
        self._ou_state: dict[str, np.ndarray] = {}
        self._training = True
        self._exploration_scale = 0.0  # reported as the "epsilon" metric; subclass sets it
        self._post_setup(seeds)

    # -- construction hooks ---------------------------------------------------

    def _make_actor(self, phase: str, obs_dim: int, hidden, act_cls) -> nn.Module:
        """Default deterministic actor (DDPG/TD3); SAC overrides."""
        spec = self.specs[phase]
        return DeterministicActor(obs_dim, spec.low, spec.high, hidden, act_cls)

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
        """Return a continuous action in the phase's Box (the adapter then
        projects/snaps it). ``mask`` is ignored for selection: admissibility is
        enforced by the adapter's projection (volume, tick) and cancel mask."""
        greedy = eval_mode or not self._training
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self._select_action(phase, obs_t, greedy=greedy)

    def _select_action(self, phase: str, obs_t: torch.Tensor, *, greedy: bool) -> np.ndarray:
        """Deterministic actor + exploration noise (DDPG/TD3); SAC overrides."""
        a = self.actor[phase](obs_t).squeeze(0).cpu().numpy()
        if not greedy:
            a = a + self._sample_noise(phase)
        spec = self.specs[phase]
        return np.clip(a, spec.low, spec.high).astype(np.float32)

    def _sample_noise(self, phase: str) -> np.ndarray:
        """Action-space exploration noise from the seeded numpy generator.
        DDPG/TD3 set ``exploration_noise`` / ``exploration_noise_std``."""
        spec = self.specs[phase]
        scale = float(getattr(self.hp, "exploration_noise_std", 0.0)) * (spec.high - spec.low) / 2.0
        kind = getattr(self.hp, "exploration_noise", "gaussian")
        if kind == "gaussian":
            return self._explore_rng.normal(0.0, 1.0, size=spec.dim) * scale
        if kind == "ou":
            ou = self._ou_state.get(phase, np.zeros(spec.dim))
            ou = ou - _OU_THETA * ou + scale * self._explore_rng.normal(0.0, 1.0, size=spec.dim)
            self._ou_state[phase] = ou
            return ou
        raise ValueError(f"exploration_noise must be 'gaussian' or 'ou', got {kind!r}")

    # -- replay ---------------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        if not self._training:
            return
        tr = transition
        junction = tr.phase == "clob" and tr.next_phase == "auction"
        if tr.info is not None and "executed_action_vec" in tr.info:
            action_vec = np.asarray(tr.info["executed_action_vec"], dtype=np.float32)
        else:
            action_vec = np.asarray(tr.action, dtype=np.float32)
        # next_cancel_admissible: C(x') > 0 when x' is an auction state. The
        # adapter's mask is all-True iff a cancel-all is admissible there.
        next_cancel_adm = False
        if not tr.done and tr.next_phase == "auction" and tr.next_mask is not None:
            next_cancel_adm = bool(np.asarray(tr.next_mask).all())
        reward = float(tr.reward) * self.hp.reward_scale
        if self.hp.reward_clip is not None:
            reward = float(np.clip(reward, -self.hp.reward_clip, self.hp.reward_clip))
        self.replay[tr.phase].add(
            obs=np.asarray(tr.obs, dtype=np.float32),
            action=action_vec,
            reward=reward,
            next_obs=None if tr.done else np.asarray(tr.next_obs, dtype=np.float32),
            done=tr.done,
            junction=junction,
            next_cancel_admissible=next_cancel_adm,
        )
        self._env_steps += 1

    # -- learning -------------------------------------------------------------

    def update(self) -> dict[str, float]:
        if not self._training or self._env_steps == 0:
            return {}
        if self._env_steps % self.hp.update_every != 0:
            return {}
        diag: dict[str, float] = {}
        for phase in self.PHASES:
            if len(self.replay[phase]) < self.hp.min_buffer:
                continue
            for _ in range(self.hp.updates_per_env_step):
                stats = self._gradient_step(phase, self.replay[phase].sample(self.hp.batch_size))
                for k, v in stats.items():
                    diag[f"{k}_{phase}"] = v
                diag[f"n_grad_steps_{phase}"] = diag.get(f"n_grad_steps_{phase}", 0.0) + 1.0
        return diag

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
        """Bellman targets with the cross-phase junction rule (the target
        networks are chosen by the PHASE of x'); done rows get ZERO bootstrap
        (terminal fold). Exposed for the hand-computed unit tests."""
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
                y[idx] = y[idx] + self.chi * q_next
        return y

    # -- algorithm hooks (overridable) ----------------------------------------

    def _should_update_actor(self, phase: str) -> bool:
        """DDPG/SAC: every step. TD3 overrides for delayed updates."""
        return True

    def _target_next_action(self, phase: str, next_obs: torch.Tensor, cadm: np.ndarray):
        """Deterministic target action (DDPG); TD3 adds smoothing, SAC samples.
        Returns (action, logp or None). Clamps the cancel coordinate at states
        that cannot cancel (matches the env mask)."""
        a = self.actor_target[phase](next_obs)
        a = self._clamp_cancel(phase, a, cadm)
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

    def _clamp_cancel(self, phase: str, action: torch.Tensor, cadm: np.ndarray) -> torch.Tensor:
        """Force the cancel logit to the no-cancel region (< 0.5) at auction
        next states that cannot cancel (threshold mode only). Matches the env's
        admissibility mask; no-op in 'never' mode or for non-auction phases."""
        if phase != "auction" or self.continuous_cancel != "threshold":
            return action
        cadm_t = torch.as_tensor(cadm, dtype=torch.bool, device=self.device)
        action = action.clone()
        action[:, 2] = torch.where(cadm_t, action[:, 2], torch.zeros_like(action[:, 2]))
        return action

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
            "hyperparams": self.hp.__dict__,
            "episode": self._episode,
            "env_steps": self._env_steps,
            "update_count": dict(self._update_count),
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
        self._episode = int(state["episode"])
        self._env_steps = int(state["env_steps"])
        self._update_count = dict(state["update_count"])
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
