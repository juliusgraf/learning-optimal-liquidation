#!/usr/bin/env bash
# Reproduce every reported result from scratch: train + evaluate + paired
# fixed-policy differences for
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
OUTPUT_ARGS=()
REQUIRE_COMPLETE=1

# ``make_all_outputs.sh`` accepts only the seed-selection flag, so extract it
# from the runner arguments instead of dropping it (or forwarding ``--smoke``
# and ``--symbol``, which that script intentionally does not accept).
for ((i = 0; i < ${#ARGS[@]}; i += 1)); do
  case "${ARGS[$i]}" in
    --seed)
      if ((i + 1 >= ${#ARGS[@]})); then
        echo "--seed requires a value" >&2
        exit 2
      fi
      OUTPUT_ARGS=(--seed "${ARGS[$((i + 1))]}")
      i=$((i + 1))
      ;;
    --seed=*) OUTPUT_ARGS=(--seed "${ARGS[$i]#*=}") ;;
    --symbol|--symbol=*) REQUIRE_COMPLETE=0 ;;
  esac
done
if [[ "$REQUIRE_COMPLETE" -eq 1 ]]; then
  OUTPUT_ARGS+=(--require-complete)
fi

RUNNERS=(
  run_synthetic_dqn run_synthetic_ddpg run_synthetic_td3 run_synthetic_sac
  run_historical_dqn run_historical_ddpg run_historical_td3 run_historical_sac
)
for s in "${RUNNERS[@]}"; do
  echo "### ${s} ${ARGS[*]}"
  bash "$HERE/${s}.sh" "${ARGS[@]}"
done

bash "$HERE/make_all_outputs.sh" ${OUTPUT_ARGS[@]+"${OUTPUT_ARGS[@]}"}
echo "### reproduce_all complete"
