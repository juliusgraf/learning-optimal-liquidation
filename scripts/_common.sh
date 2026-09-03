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
#   --skip-completed validate and skip a run whose entire per-run pipeline ended
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# A scratch/CI caller may isolate artifacts without editing the publication
# configuration.  The same value is also forwarded into the resolved config
# below, so the launcher's filesystem path and train.py's path cannot diverge.
RESULTS_ROOT="${LMM_RESULTS_ROOT:-results/revision_v12}"

# Keep Matplotlib's font/config cache reusable across the many short reporting
# processes launched by a full matrix.  This also makes the pipeline work when
# the account's default Matplotlib directory is read-only (for example in a
# container or batch worker) instead of rebuilding a temporary font cache for
# every paired-difference command.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

SEED=""
SMOKE=0
SYMBOL=""
NO_RESUME_CKPT=0     # 1 => disable periodic resume checkpoints (disk-safe)
SKIP_COMPLETED=0     # 1 => skip only exact-config, explicitly completed runs
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
      --skip-completed) SKIP_COMPLETED=1; shift ;;
      -h|--help) echo "usage: $0 [--seed N] [--smoke] [--symbol TICKER] [--no-resume-ckpt] [--skip-completed]"; exit 0 ;;
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

  # Keep the shell-computed run path and the application's resolved output root
  # identical.  Append last so LMM_RESULTS_ROOT is authoritative even when a
  # treatment supplies other experiment overrides.
  train_over+=( -o "experiment.results_root=$RESULTS_ROOT" )

  # Disk-safe: drop periodic resume checkpoints (replay-heavy ckpt_ep*.pt). The
  # best/final/initial.pt snapshots are still written, so early stopping and the
  # evaluation pipeline are unaffected. Applied LAST so it overrides the smoke
  # checkpoint cadence. Used by the multi-seed launcher (N x reproduction).
  [[ "$NO_RESUME_CKPT" -eq 1 ]] && \
    train_over+=( -o algo.hyperparams.checkpoint_interval_episodes=10000000 )

  if [[ "$SKIP_COMPLETED" -eq 1 && -e "$run_dir" ]]; then
    local validation_args=(
      --validate-completed-run "$run_dir"
      "${cfgs[@]}"
      --seed "$SEED_FOR_NAME"
      ${train_over[@]+"${train_over[@]}"}
    )
    [[ -n "$symbol" ]] && validation_args+=(--symbol "$symbol")
    if python3 -m lmm.experiments.publication "${validation_args[@]}"; then
      echo ">>> skip validated completed run ${run_dir}"
      return 0
    fi
    echo "ERROR: $run_dir exists but is not a validated exact-config completed run." >&2
    echo "Move it aside for diagnosis, then rerun; automatic crash resume is intentionally disabled." >&2
    return 2
  fi

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

  echo ">>> bind completed pipeline artifacts ${run_dir}"
  python3 -m lmm.experiments.publication \
    --write-completion-manifest "$run_dir"

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
