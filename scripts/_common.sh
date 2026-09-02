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
RESULTS_ROOT="results/revision_v10"

SEED=""
SMOKE=0
SYMBOL=""
NO_RESUME_CKPT=0     # 1 => disable periodic resume checkpoints (disk-safe)
EXTRA_OVERRIDES=()   # optional per-runner `-o key=value` train overrides
EXTRA_CONFIGS=()     # optional treatment overlays, appended after algo config
RUN_NAME_SUFFIX=""  # optional stable treatment label in the run directory

parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --seed) SEED="$2"; shift 2 ;;
      --seed=*) SEED="${1#*=}"; shift ;;
      --smoke) SMOKE=1; shift ;;
      --symbol) SYMBOL="$2"; shift 2 ;;
      --symbol=*) SYMBOL="${1#*=}"; shift ;;
      --no-resume-ckpt) NO_RESUME_CKPT=1; shift ;;
      -h|--help) echo "usage: $0 [--seed N] [--smoke] [--symbol TICKER] [--no-resume-ckpt]"; exit 0 ;;
      *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  SEED_FOR_NAME="${SEED:-42}"  # runs are named with the effective seed
}

# run_experiment EXP_NAME SETTING_CFG ALGO_CFG ALGO_NAME [SYMBOL]
run_experiment() {
  local exp_name="$1" setting_cfg="$2" algo_cfg="$3" algo_name="$4" symbol="${5:-}"

  local run_name="${algo_name}${RUN_NAME_SUFFIX}_seed${SEED_FOR_NAME}"
  [[ -n "$symbol" ]] && run_name="${algo_name}_${symbol}${RUN_NAME_SUFFIX}_seed${SEED_FOR_NAME}"
  local run_dir="${RESULTS_ROOT}/${exp_name}/${run_name}"

  local cfgs=(--config configs/base.yaml --config "$setting_cfg" --config "$algo_cfg")
  local extra_cfg
  for extra_cfg in ${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}; do
    cfgs+=(--config "$extra_cfg")
  done
  local seed_args=() sym_args=() train_over=() eval_args=()
  [[ -n "$SEED" ]] && seed_args=(--seed "$SEED")
  [[ -n "$symbol" ]] && sym_args=(--symbol "$symbol")

  if [[ "$SMOKE" -eq 1 ]]; then
    train_over=(
      -o experiment.episodes=4
      -o rl.test_size=3
      -o rl.validation_size=2
      -o rl.validation_frequency_episodes=2
      -o rl.validation_patience_evals=10
      -o rl.checkpoint_min_clob_updates=0
      -o rl.checkpoint_min_auction_updates=0
      -o rl.checkpoint_require_initial_improvement=false
      -o rl.normalizer_fit_episodes=2
      -o algo.hyperparams.checkpoint_interval_episodes=2
      -o algo.hyperparams.min_buffer=1
      -o algo.hyperparams.min_buffer_clob=1
      -o algo.hyperparams.min_buffer_auction=1
    )
    eval_args=(--n-episodes 3 --trace-episodes 1)
  else
    eval_args=(--trace-episodes 1)
  fi

  # Optional per-runner/treatment overrides, appended after smoke overrides.
  train_over+=( ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} )

  # Disk-safe: drop periodic resume checkpoints (replay-heavy ckpt_ep*.pt). The
  # best/final/initial.pt snapshots are still written, so early stopping and the
  # evaluation pipeline are unaffected. Applied LAST so it overrides the smoke
  # checkpoint cadence. Used by the multi-seed launcher (N x reproduction).
  [[ "$NO_RESUME_CKPT" -eq 1 ]] && \
    train_over+=( -o algo.hyperparams.checkpoint_interval_episodes=10000000 )

  # Guard empty-array expansion for bash 3.2 (macOS) under `set -u`.
  echo ">>> train ${run_dir}"
  python3 -m lmm.experiments.train "${cfgs[@]}" --run-name "$run_name" \
    ${seed_args[@]+"${seed_args[@]}"} ${sym_args[@]+"${sym_args[@]}"} ${train_over[@]+"${train_over[@]}"}

  echo ">>> evaluate ${run_dir}"
  python3 -m lmm.experiments.evaluate --run-dir "$run_dir" \
    ${sym_args[@]+"${sym_args[@]}"} "${eval_args[@]}"

  echo ">>> paired policy differences ${run_dir}"
  python3 -m lmm.experiments.policy_differences --run-dir "$run_dir" --benchmark as
  python3 -m lmm.experiments.policy_differences --run-dir "$run_dir" --benchmark twap

  echo ">>> done ${run_dir}"
}

# Loop the five paper tickers (or a single --symbol) for the true-midquote
# historical setting.
HIST_TICKERS=(MSFT JPM PG GOOGL CAT)
run_historical() {  # run_historical <algo_cfg> <algo_name>
  local algo_cfg="$1" algo_name="$2"
  local data="data/historical_sp500_midquotes_1m.csv"
  if [[ ! -f "$data" || ! -f "${data}.meta.json" ]]; then
    echo "missing true-midquote artifact or sidecar; build it using data/README.md" >&2
    return 2
  fi
  local tickers=("${HIST_TICKERS[@]}")
  [[ -n "$SYMBOL" ]] && tickers=("$SYMBOL")
  for ticker in "${tickers[@]}"; do
    run_experiment historical_sp500_midquotes configs/historical_sp500_midquotes.yaml "$algo_cfg" "$algo_name" "$ticker"
  done
}
