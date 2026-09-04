# Design documentation

- `rl_design.md` documents the active learner representation, action mapping,
  reward/accounting split, and validation protocol.
- `continuous_action_extension.md` documents the DDPG/TD3/SAC projected
  continuous-proposal methods and their relationship to executable actions.
- `metrics_schema.md` is the field-level schema for training and evaluation
  artifacts.

Active numerical values come from `configs/base.yaml` plus setting, learner,
and treatment overlays. Reports under `audit/` characterize superseded code.
