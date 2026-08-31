#!/usr/bin/env bash
# Run the matched synthetic treatment matrix for all four learners:
# the 2x2 H_cl x shaping ablation, no-auction comparator, and no-cancellation
# sensitivity. This launcher deliberately does not start unless invoked.
#
# Usage:
#   scripts/run_synthetic_treatments.sh [--seed N] [--smoke] [--no-resume-ckpt]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
parse_common_args "$@"

ARMS=(
  ablation_h_off_shaping_off
  ablation_h_on_shaping_off
  ablation_h_off_shaping_on
  ablation_h_on_shaping_on
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
