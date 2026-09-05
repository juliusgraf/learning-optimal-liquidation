#!/usr/bin/env bash
# Run the non-headline arms of the matched synthetic treatment matrix for all
# four learners: three cells of the 2x2 H/anchor-bundle x shaping ablation, the
# no-auction comparator, and the no-cancellation sensitivity. The fourth 2x2
# cell (H-on/shaping-on) is exactly the canonical synthetic headline
# configuration; reporting reuses it instead of retraining an alias.
#
# Usage:
#   scripts/run_synthetic_treatments.sh [--seed N] [--smoke] [--no-resume-ckpt]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
parse_common_args "$@"

ARMS=(
  ablation_h_off_shaping_off
  ablation_h_off_shaping_on
  ablation_h_on_shaping_off
  no_auction
  no_cancellation
)
ALGOS=(dqn ddpg td3 sac)

for arm in "${ARMS[@]}"; do
  EXTRA_CONFIGS=("configs/treatment/${arm}.yaml")
  EXTRA_OVERRIDES=(-o "experiment.name=synthetic_rough_heston__${arm}")
  RUN_NAME_SUFFIX="__${arm}"
  for algo in "${ALGOS[@]}"; do
    run_experiment \
      "synthetic_rough_heston__${arm}" \
      configs/synthetic_rough_heston.yaml \
      "configs/algo/${algo}.yaml" \
      "$algo"
  done
done
