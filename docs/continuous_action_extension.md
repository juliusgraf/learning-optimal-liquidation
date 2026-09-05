# Continuous-control implementation

The active DDPG, TD3 and SAC implementation is `src/lmm/agents/sb3.py`, selected
by `algo.backend: sb3`. It uses the pinned Stable-Baselines3 2.7.1 package.
`docs/rl_design.md` specifies the common observation, proposal mapping, rewards,
phase junction, checkpoint selection and treatment contracts.

Two library models allow the structurally different CLOB and auction action
spaces without padding an inactive action into SAC entropy. Their environment
objects provide spaces only; the shared market episode loop owns interaction.
The replay bridge is the only phase-specific Bellman adaptation. It computes
junction continuation at sampling time, never caches a stale auction value,
and prevents the CLOB library model from bootstrapping the same row again.
All ordinary and terminal training updates use the library implementations.

Private NumPy generators sample replay and exploration, and the seeded Torch
stream drives initialization, SAC reparameterization and TD3 target noise.
Checkpoint files include native SB3 models, optional replay contents, replay
and exploration generator states, Torch state, counters, and the frozen feature
normalizer. Loading validates the algorithm, economic, action and observation
contracts. Optimizer hooks, when configured, are reinstalled on loaded models.

The older handwritten `continuous_base.py`, `ddpg.py`, `td3.py`, and `sac.py`
are retained for characterization tests and provenance. Their internal details
must not be described as the active headline implementation.

Primary library references:

- [SAC implementation](https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/sac/sac.html)
- [TD3 implementation](https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/td3/td3.html)
- [DDPG implementation](https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/ddpg/ddpg.html)
