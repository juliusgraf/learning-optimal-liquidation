#!/usr/bin/env bash
# Run five non-headline synthetic mechanisms for all four learners.
# The canonical synthetic headline is reused. Controls isolate auction credit,
# H observation, auction anchor, combined shaped preferences and auction access.
#
# Usage:
#   scripts/run_synthetic_treatments.sh [--seed N] [--smoke] [--no-resume-ckpt]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
parse_common_args "$@"

ARMS=(
  mechanism_fixed_anchor
  mechanism_economic_dense
  mechanism_economic_sparse
  mechanism_h_feature_off
  no_auction
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
