"""Library DDPG/TD3/SAC with a replay-only bridge between market phases.

All optimizer steps, losses, entropy updates and target synchronization are
Stable-Baselines3's implementations. The bridge replaces the continuation of
a CLOB junction row with the current auction target value at sampling time.
It never treats auction opening as termination of the economic episode.
"""
from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from lmm.agents.base import Agent, ENVIRONMENT_CONTRACT
from lmm.env.action_spaces import continuous_action_specs
from lmm.rl.schedules import learning_rate_factor, learning_conditioning_contract


class _SpacesOnlyEnv(gym.Env):
    def __init__(self, obs_dim, action_dim):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("The shared market episode loop owns environment interaction")

    def step(self, action):
        raise RuntimeError("The shared market episode loop owns environment interaction")


class PhaseReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.junction = np.zeros(self.buffer_size, dtype=bool)
        self.rng = None
        self.continuation = None

    def __len__(self):
        return self.size()

    def add(self, obs, next_obs, action, reward, done, infos):
        self.junction[self.pos] = bool(infos[0].get("junction", False))
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size, env=None):
        assert env is None and self.n_envs == 1 and self.rng is not None
        idx = self.rng.integers(0, self.size(), size=batch_size)
        obs, action, nxt, done, reward = tuple(map(self.to_torch, (
            self.observations[idx, 0], self.actions[idx, 0],
            self.next_observations[idx, 0], self.dones[idx, 0, None],
            self.rewards[idx, 0, None],
        )))
        junction = self.to_torch(self.junction[idx])
        if bool(junction.any()):
            with torch.no_grad():
                reward[junction] += self.continuation(nxt[junction])
            # Prevent SB3 from adding the *CLOB* continuation as well.
            done[junction] = 1.0
        return ReplayBufferSamples(obs, action, nxt, done, reward)


class SB3Agent(Agent):
    PHASES = ("clob", "auction")

    def __init__(self, cfg, seeds):
        self.cfg = cfg
        self.hp = SimpleNamespace(**cfg.algo.hyperparams)
        self.name = cfg.algo.name
        self.device = torch.device(self.hp.device)
        self._training = True
        self._episode = 0
        self._pending_update_phase = None
        self._explore_rng = seeds.generators["exploration"]
        self._update_count = dict.fromkeys(self.PHASES, 0)
        self._seen = dict.fromkeys(self.PHASES, 0)
        self.models, self.replay = {}, {}
        specs = continuous_action_specs(cfg)
        cls = {"ddpg": DDPG, "td3": TD3, "sac": SAC}[self.name]
        for phase in self.PHASES:
            kwargs = {}
            if self.name == "sac":
                kwargs.update(ent_coef=self.hp.ent_coef, target_entropy=self.hp.target_entropy)
            elif self.name == "td3":
                kwargs.update(policy_delay=self.hp.policy_delay,
                              target_policy_noise=self.hp.target_noise_std,
                              target_noise_clip=self.hp.target_noise_clip)
            model = cls(
                "MlpPolicy", _SpacesOnlyEnv(len(cfg.features.clob), specs[phase].dim),
                learning_rate=self.hp.learning_rate, buffer_size=self.hp.buffer_size,
                learning_starts=getattr(self.hp, f"min_buffer_{phase}"),
                batch_size=self.hp.batch_size, tau=self.hp.target_soft_tau, gamma=1.0,
                train_freq=1, gradient_steps=1, replay_buffer_class=PhaseReplayBuffer,
                policy_kwargs=dict(net_arch=list(self.hp.hidden_layers)),
                device=self.hp.device, seed=None, **kwargs,
            )
            model.set_logger(configure(folder=None, format_strings=[]))
            self._install_gradient_clip(model)
            self.models[phase] = model
            self.replay[phase] = model.replay_buffer
            model.replay_buffer.rng = seeds.generators[f"replay_{phase}"]
        self.replay["clob"].continuation = self._auction_continuation

    def _install_gradient_clip(self, model):
        limit = getattr(self.hp, 'grad_clip_norm', None)
        actor_rate = getattr(self.hp, 'actor_learning_rate', None)
        decay = getattr(self.hp, 'critic_weight_decay', 0.)
        if actor_rate is not None and (not np.isfinite(actor_rate) or actor_rate <= 0):
            raise ValueError('actor_learning_rate must be finite and positive')
        if not np.isfinite(decay) or decay < 0:
            raise ValueError('critic_weight_decay must be finite and nonnegative')
        for group in model.critic.optimizer.param_groups:
            group['weight_decay'] = decay
        if limit is not None and (not np.isfinite(limit) or limit <= 0):
            raise ValueError('grad_clip_norm must be finite and positive')
        # SB3 owns backward(), loss construction and all updates. Optimizer
        # hooks provide the common norm bound immediately before each step.
        model._lmm_grad_norms = {}
        for label, network in (('actor', model.actor), ('critic', model.critic)):
            parameters = tuple(network.parameters())
            def clip(optimizer, args, kwargs, parameters=parameters, label=label):
                if label == 'actor' and actor_rate is not None:
                    for group in optimizer.param_groups:
                        group['lr'] = actor_rate*learning_rate_factor(self.cfg.rl, self._episode)
                if limit is not None:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True)
                    model._lmm_grad_norms[label] = float(norm)
            network.optimizer.register_step_pre_hook(clip)

    def _auction_continuation(self, obs):
        model = self.models["auction"]
        if self.name == "sac":
            action, logp = model.actor.action_log_prob(obs)
            alpha = model.log_ent_coef.exp().detach() if model.log_ent_coef is not None else model.ent_coef_tensor
        else:
            action = model.actor_target(obs)
            if self.name == "td3":
                noise = torch.randn_like(action) * model.target_policy_noise
                action = (action + noise.clamp(-model.target_noise_clip, model.target_noise_clip)).clamp(-1, 1)
        q = torch.cat(model.critic_target(obs, action), dim=1).min(dim=1, keepdim=True).values
        return q - alpha * logp.reshape(-1, 1) if self.name == "sac" else q

    def start_episode(self, episode):
        self._episode = int(episode)

    def set_train(self, training):
        self._training = bool(training)
        for model in self.models.values():
            model.policy.set_training_mode(training)

    @property
    def epsilon(self):
        return float(getattr(self.hp, "exploration_noise_std", 0.0))

    def act(self, obs, mask, phase, *, eval_mode=False):
        greedy = eval_mode or not self._training
        model = self.models[phase]
        if not greedy and self._seen[phase] < model.learning_starts:
            return self._explore_rng.uniform(-1, 1, model.action_space.shape).astype(np.float32)
        action, _ = model.predict(obs, deterministic=greedy or self.name != "sac")
        if not greedy and self.name != "sac":
            action = action + self._explore_rng.normal(0, self.hp.exploration_noise_std, action.shape)
        return np.asarray(action, dtype=np.float32)

    def observe(self, tr):
        if not self._training:
            return
        action = np.asarray(tr.info["proposal_action_vec"], dtype=np.float32)
        self.replay[tr.phase].add(
            tr.obs[None], tr.next_obs[None], action[None],
            np.array([tr.reward * self.hp.reward_scale], dtype=np.float32),
            np.array([tr.done]), [dict(junction=tr.phase == "clob" and tr.next_phase == "auction")],
        )
        self._seen[tr.phase] += 1
        self._pending_update_phase = tr.phase

    def update(self, phase=None):
        if not self._training or self._pending_update_phase is None:
            return {}
        pending = self._pending_update_phase
        if phase is not None and phase != pending:
            raise ValueError("update phase differs from the observed transition")
        self._pending_update_phase = None
        model = self.models[pending]
        if len(self.replay[pending]) < model.learning_starts:
            return {}
        # TD3 does not update the actor on every critic step; avoid logging
        # an old actor loss as though it came from the current update.
        model.logger.name_to_value.pop("train/actor_loss", None)
        rate = self.hp.learning_rate * learning_rate_factor(self.cfg.rl, self._episode)
        model.lr_schedule = lambda progress, rate=rate: rate
        model.train(gradient_steps=1, batch_size=self.hp.batch_size)
        self._update_count[pending] = model._n_updates
        loss = float(model.logger.name_to_value["train/critic_loss"])
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {self.name} {pending} critic loss")
        stats = {f"loss_{pending}": loss, f"n_grad_steps_{pending}": 1.0,
                 f"learning_rate_{pending}": rate,
                 f"actor_learning_rate_{pending}": model.actor.optimizer.param_groups[0]['lr']}
        for metric in ("actor_loss", "ent_coef"):
            value = model.logger.name_to_value.get(f"train/{metric}")
            if value is not None:
                stats[f"{metric}_{pending}"] = float(value)
        if 'critic' in getattr(model, '_lmm_grad_norms', {}):
            stats[f'grad_norm_{pending}'] = model._lmm_grad_norms['critic']
        return stats

    def _contract(self):
        return {
            "environment": ENVIRONMENT_CONTRACT,
            "schema": self.cfg.experiment.artifact_schema_version,
            "auction_enabled": self.cfg.experiment.auction_enabled,
            **{key: asdict(getattr(self.cfg, key)) for key in (
                "grid", "clob_flow", "auction_flow", "midprice", "algo1", "reward", "actions", "features", "algo")},
            "h_cl_feature_enabled": self.cfg.rl.h_cl_feature_enabled,
            "relative_price_features": self.cfg.rl.relative_price_features,
            "n_step": self.cfg.rl.n_step,
            **learning_conditioning_contract(self.cfg.rl),
        }

    def save(self, path, *, include_replay=False):
        models, buffers = {}, {}
        for phase, model in self.models.items():
            data = io.BytesIO()
            model.save(data)
            models[phase] = data.getvalue()
            buffer = self.replay[phase]
            if include_replay:
                buffers[phase] = {k: v for k, v in vars(buffer).items()
                                  if k not in ("rng", "continuation")}
        torch.save(dict(
            contract=self._contract(), models=models, buffers=buffers,
            feature_normalizer=self._feature_normalizer_state(), episode=self._episode,
            seen=self._seen, update_count=self._update_count,
            explore_rng=self._explore_rng.bit_generator.state,
            replay_rng={p: b.rng.bit_generator.state for p, b in self.replay.items()},
            torch_rng=torch.get_rng_state(), pending=self._pending_update_phase,
        ), Path(path))

    def load(self, path):
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        contract = state.get("contract")
        if isinstance(contract, dict) and isinstance(contract.get('reward'), dict):
            contract['reward'].setdefault('auction_shaping_weight', 1.0)
        if contract != self._contract():
            raise ValueError("SB3 checkpoint configuration/environment contract mismatch")
        if state.get("feature_normalizer") is None:
            raise ValueError("SB3 checkpoint is missing its fitted normalizer")
        self._load_feature_normalizer_state(state["feature_normalizer"])
        for phase, previous in self.models.items():
            model = type(previous).load(io.BytesIO(state["models"][phase]), device=self.hp.device)
            model.set_logger(configure(folder=None, format_strings=[]))
            self._install_gradient_clip(model)
            buffer = self.replay[phase]
            if phase in state["buffers"]:
                vars(buffer).update(state["buffers"][phase])
            buffer.rng.bit_generator.state = state["replay_rng"][phase]
            model.replay_buffer = buffer
            self.models[phase] = model
        self._episode, self._seen = state["episode"], state["seen"]
        self._update_count = state["update_count"]
        self._pending_update_phase = state["pending"]
        self._explore_rng.bit_generator.state = state["explore_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
