#!/usr/bin/env bash
# Reproduce every reported result from scratch: train + evaluate + paired
# fixed-policy differences for
# all algorithms in both settings, then regenerate all figures/tables.
#
# Usage:
#   scripts/reproduce_all.sh            # full paper runs (long!)
#   scripts/reproduce_all.sh --smoke    # tiny CI smoke of the whole pipeline
#   scripts/reproduce_all.sh --seed 7   # override the master seed
#   scripts/reproduce_all.sh --skip-output-generation  # orchestration workers
#
# Flags are forwarded to every run_*.sh (see scripts/_common.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
ARGS=()
OUTPUT_ARGS=()
REQUIRE_COMPLETE=1
SKIP_OUTPUT_GENERATION=0

# The multiseed orchestrator runs seed workers concurrently, then performs all
# shared output generation once after every worker has finished.  Strip its
# private switch rather than forwarding it to the individual run scripts.
for arg in "$@"; do
  if [[ "$arg" == "--skip-output-generation" ]]; then
    SKIP_OUTPUT_GENERATION=1
  else
    ARGS+=("$arg")
  fi
done

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

if [[ "$SKIP_OUTPUT_GENERATION" -eq 0 ]]; then
  bash "$HERE/make_all_outputs.sh" ${OUTPUT_ARGS[@]+"${OUTPUT_ARGS[@]}"}
fi
echo "### reproduce_all complete"
