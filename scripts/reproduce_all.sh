#!/usr/bin/env bash
# Reproduce every reported result from scratch: train + evaluate + regret for
# all algorithms in both settings, then regenerate all figures/tables.
#
# Usage:
#   scripts/reproduce_all.sh            # full paper runs (long!)
#   scripts/reproduce_all.sh --smoke    # tiny CI smoke of the whole pipeline
#   scripts/reproduce_all.sh --seed 7   # override the master seed
#
# Flags are forwarded to every run_*.sh (see scripts/_common.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")

RUNNERS=(
  run_synthetic_dqn run_synthetic_ddpg run_synthetic_td3 run_synthetic_sac
  run_historical_dqn run_historical_ddpg run_historical_td3 run_historical_sac
)
for s in "${RUNNERS[@]}"; do
  echo "### ${s} ${ARGS[*]}"
  bash "$HERE/${s}.sh" "${ARGS[@]}"
done

bash "$HERE/make_all_outputs.sh"
echo "### reproduce_all complete"
