#!/usr/bin/env bash
# Shared helpers for the Phase 7 experiment launchers. Sourced by run_*.sh.
#
# Each run_*.sh:
#   source "$(dirname "$0")/_common.sh"
#   parse_common_args "$@"
#   run_experiment <exp_name> <setting_cfg> <algo_cfg> <algo_name> [symbol]
#
# Common flags:
#   --seed N   override experiment.master_seed (default: the config value, 42)
#   --smoke    tiny episode counts for CI (4 train / 3 eval episodes)
#   --symbol T historical only: restrict to a single ticker
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEED=""
SMOKE=0
SYMBOL=""
EXTRA_OVERRIDES=()   # optional per-runner `-o key=value` train overrides

parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --seed) SEED="$2"; shift 2 ;;
      --seed=*) SEED="${1#*=}"; shift ;;
      --smoke) SMOKE=1; shift ;;
      --symbol) SYMBOL="$2"; shift 2 ;;
      --symbol=*) SYMBOL="${1#*=}"; shift ;;
      -h|--help) echo "usage: $0 [--seed N] [--smoke] [--symbol TICKER]"; exit 0 ;;
      *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  SEED_FOR_NAME="${SEED:-42}"  # runs are named with the effective seed
}

# run_experiment EXP_NAME SETTING_CFG ALGO_CFG ALGO_NAME [SYMBOL]
run_experiment() {
  local exp_name="$1" setting_cfg="$2" algo_cfg="$3" algo_name="$4" symbol="${5:-}"

  local run_name="${algo_name}_seed${SEED_FOR_NAME}"
  [[ -n "$symbol" ]] && run_name="${algo_name}_${symbol}_seed${SEED_FOR_NAME}"
  local run_dir="results/${exp_name}/${run_name}"

  local cfgs=(--config configs/base.yaml --config "$setting_cfg" --config "$algo_cfg")
  local seed_args=() sym_args=() train_over=() eval_args=()
  [[ -n "$SEED" ]] && seed_args=(--seed "$SEED")
  [[ -n "$symbol" ]] && sym_args=(--symbol "$symbol")

  if [[ "$SMOKE" -eq 1 ]]; then
    train_over=(
      -o experiment.episodes=4
      -o algo.hyperparams.final_eval_n_seeds=3
      -o algo.hyperparams.eval_interval_episodes=2
      -o algo.hyperparams.checkpoint_interval_episodes=2
      -o algo.hyperparams.min_buffer=1
    )
    eval_args=(--n-episodes 3 --trace-episodes 1)
  else
    eval_args=(--trace-episodes 1)
  fi

  # Per-runner extra train overrides (e.g. the historical DQN epsilon schedule,
  # which must track the shorter episode budget). Set by the caller before
  # run_experiment/run_historical; appended AFTER any smoke overrides.
  train_over+=( ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} )

  # Guard empty-array expansion for bash 3.2 (macOS) under `set -u`.
  echo ">>> train ${run_dir}"
  python3 -m lmm.experiments.train "${cfgs[@]}" --run-name "$run_name" \
    ${seed_args[@]+"${seed_args[@]}"} ${sym_args[@]+"${sym_args[@]}"} ${train_over[@]+"${train_over[@]}"}

  echo ">>> evaluate ${run_dir}"
  python3 -m lmm.experiments.evaluate --run-dir "$run_dir" \
    ${sym_args[@]+"${sym_args[@]}"} "${eval_args[@]}"

  echo ">>> regret ${run_dir}"
  python3 -m lmm.experiments.regret --run-dir "$run_dir" --benchmark as
  python3 -m lmm.experiments.regret --run-dir "$run_dir" --benchmark twap

  echo ">>> done ${run_dir}"
}

# Loop the five paper tickers (or a single --symbol) for the historical setting.
HIST_TICKERS=(MSFT JPM PG GOOGL CAT)
run_historical() {  # run_historical <algo_cfg> <algo_name>
  local algo_cfg="$1" algo_name="$2"
  local tickers=("${HIST_TICKERS[@]}")
  [[ -n "$SYMBOL" ]] && tickers=("$SYMBOL")
  for ticker in "${tickers[@]}"; do
    run_experiment historical_sp500 configs/historical_sp500.yaml "$algo_cfg" "$algo_name" "$ticker"
  done
}
