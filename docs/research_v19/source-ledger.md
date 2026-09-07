# Source ledger — v19 diagnosis

Research retrieved September 5, 2026. Theory, borrowed design precedent and
local empirical evidence are kept distinct. The duplicate frozen report source is [archived in Git](https://github.com/juliusgraf/Learning-Market-Making/blob/83c35645edfe2d15733f91ede4a54e1bd9660fde/docs/research_v19/report-source.md);
the maintained repository report is [pathology_repair_v19.md](../pathology_repair_v19.md).
See [local recovery instructions](../cleanup.md) if the archived revision has not been pushed.

| Source | Primary evidence used | Scope / limitation |
|---|---|---|
| [Ng, Harada & Russell, ICML 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) | Potential-difference rewards preserve optimal-policy ordering under the stated MDP transformation | Does not imply faster finite neural training; use undiscounted telescoping algebra for this finite-horizon implementation |
| [Grześ, AAMAS 2017](https://kar.kent.ac.uk/id/document/97755) | Episodic shaping requires attention to terminal potential | Supports retaining the terminal-zero correction in both credit arms |
| [Fujimoto, van Hoof & Meger, ICML 2018](https://proceedings.mlr.press/v80/fujimoto18a.html) | Actor-critic approximation error and overestimation feedback | Supports the diagnostic hypothesis, not a proof of this project's failure; retain DDPG identity rather than quietly add TD3 mechanisms |
| [Ball et al., ICML 2023, §4.2](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf) | Critic LayerNorm as an intervention against extrapolation error | Different SAC/offline-assisted setting; transfer to DDPG tested locally, no universal guarantee |
| [SB3 2.7.1 DDPG source](https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/ddpg/ddpg.html) | Native DDPG is the single-critic, delay-one, unsmoothed construction on the library TD3 infrastructure | Local tests verify those settings and exact save/resume update behavior with the custom critic factory |
| [Lillicrap et al., ICLR 2016](https://arxiv.org/abs/1509.02971) | Original DDPG algorithm and normalization context | Architectural precedent is not evidence that a particular normalization will help here |
| Repository v18/v19 artifacts | Phase forecast errors, frozen critic calibration, all 43 bounded learning runs, same-path economic comparisons | Adaptive development, maximum 200 episodes; not new ten-seed production evidence and not untouched historical generalization |

No external source supplies or validates the four fitted H weights. They are
estimated only from this model's training paths. No current exchange-liquidity
claim is inferred from a reinforcement-learning paper. The normalized market
calibration is unchanged in this repair and remains a stylized scenario.
