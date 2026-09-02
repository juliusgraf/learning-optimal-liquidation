#!/usr/bin/env bash
# Multi-seed reproduction: run the full headline, historical, and synthetic
# treatment matrix, then build the cross-seed IQM/bootstrap-CI aggregate tables,
# figures, and paired cross-treatment contrasts. Reported policy is the
# best-validation checkpoint (early stopping; evaluate.py default).
#
# DISK-SAFE: runs with --no-resume-ckpt so the replay-heavy periodic ckpt_ep*.pt
# are NOT written (five-seed reproduction would otherwise ENOSPC). best/final/initial.pt
# are still written, so early stopping and evaluation are unaffected.
#
# Usage:
#   scripts/run_multiseed.sh [--seeds "42 7 99 123 2024"] [--jobs N]
#                            [--threads-per-job N] [--smoke] [--symbol TICKER]
# Seed workers may run concurrently. Shared table/figure generation starts only
# after all workers finish, so no process writes a combined artifact concurrently.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

SEEDS="42 7 99 123 2024"
SEEDS_EXPLICIT=0
JOBS=1
THREADS_PER_JOB=2
PASS=()             # --smoke / --symbol forwarded to headline/historical runs
TREATMENT_PASS=()   # treatments are synthetic, so --symbol is not forwarded
AGGREGATE_SYMBOL=""
SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; SEEDS_EXPLICIT=1; shift 2 ;;
    --seeds=*) SEEDS="${1#*=}"; SEEDS_EXPLICIT=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --threads-per-job) THREADS_PER_JOB="$2"; shift 2 ;;
    --threads-per-job=*) THREADS_PER_JOB="${1#*=}"; shift ;;
    --smoke) SMOKE=1; PASS+=(--smoke); TREATMENT_PASS+=(--smoke); shift ;;
    --symbol) AGGREGATE_SYMBOL="$2"; PASS+=(--symbol "$2"); shift 2 ;;
    --symbol=*) AGGREGATE_SYMBOL="${1#*=}"; PASS+=(--symbol "$AGGREGATE_SYMBOL"); shift ;;
    -h|--help)
      echo "usage: $0 [--seeds \"42 7 99 123 2024\"] [--jobs N] [--threads-per-job N] [--smoke] [--symbol TICKER]"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Smoke and publication artifacts intentionally use disjoint master-seed names,
# because run directories are keyed by seed and training never overwrites them.
if [[ "$SMOKE" -eq 1 && "$SEEDS_EXPLICIT" -eq 0 ]]; then
  SEEDS="9001 9002"
fi

if [[ "$SMOKE" -eq 0 && -n "$AGGREGATE_SYMBOL" ]]; then
  echo "full publication mode requires all five historical tickers; --symbol is smoke/development only" >&2
  exit 2
fi

case "$JOBS" in
  ''|*[!0-9]*|0) echo "--jobs must be a positive integer" >&2; exit 2 ;;
esac
case "$THREADS_PER_JOB" in
  ''|*[!0-9]*|0) echo "--threads-per-job must be a positive integer" >&2; exit 2 ;;
esac
seed_count=0
first_seed=""
seen_seeds=""
for s in $SEEDS; do
  case "$s" in
    ''|*[!0-9]*) echo "invalid nonnegative integer seed: $s" >&2; exit 2 ;;
  esac
  case " $seen_seeds " in
    *" $s "*) echo "duplicate master seed: $s" >&2; exit 2 ;;
  esac
  seen_seeds="$seen_seeds $s"
  if [[ "$SMOKE" -eq 1 ]]; then
    case "$s" in
      42|7|99|123|2024)
        echo "smoke runs must use noncanonical seeds (for example 9001 9002)" >&2
        exit 2
        ;;
    esac
  fi
  [[ -z "$first_seed" ]] && first_seed="$s"
  seed_count=$((seed_count + 1))
done
[[ "$seed_count" -ge 2 ]] || { echo "--seeds must contain at least two seeds" >&2; exit 2; }
if [[ "$SMOKE" -eq 0 ]]; then
  canonical_count=0
  for canonical_seed in 42 7 99 123 2024; do
    case " $seen_seeds " in
      *" $canonical_seed "*) canonical_count=$((canonical_count + 1)) ;;
    esac
  done
  if [[ "$seed_count" -ne 5 || "$canonical_count" -ne 5 ]]; then
    echo "full publication mode requires exactly the canonical seeds: 42 7 99 123 2024" >&2
    exit 2
  fi
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "full publication mode requires a clean git worktree so git_sha.txt fully identifies the code" >&2
    exit 2
  fi
fi

run_seed_worker() {
  local s="$1"
  echo "############ multi-seed: master seed ${s} ############"
  OMP_NUM_THREADS="$THREADS_PER_JOB" \
  MKL_NUM_THREADS="$THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
  VECLIB_MAXIMUM_THREADS="$THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
  LMM_TORCH_INTRAOP_THREADS="$THREADS_PER_JOB" \
  LMM_TORCH_INTEROP_THREADS=1 \
    bash "$HERE/reproduce_all.sh" --seed "$s" --no-resume-ckpt \
      --skip-output-generation --skip-completed ${PASS[@]+"${PASS[@]}"}
  OMP_NUM_THREADS="$THREADS_PER_JOB" \
  MKL_NUM_THREADS="$THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
  VECLIB_MAXIMUM_THREADS="$THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
  LMM_TORCH_INTRAOP_THREADS="$THREADS_PER_JOB" \
  LMM_TORCH_INTEROP_THREADS=1 \
    bash "$HERE/run_synthetic_treatments.sh" --seed "$s" --no-resume-ckpt \
      --skip-completed ${TREATMENT_PASS[@]+"${TREATMENT_PASS[@]}"}
}

# Bash 3.2-compatible bounded parallelism: launch one batch of at most JOBS
# seed workers, wait for the full batch, then launch the next batch.
pids=()
for s in $SEEDS; do
  run_seed_worker "$s" &
  pids+=("$!")
  if [[ ${#pids[@]} -ge "$JOBS" ]]; then
    batch_failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || batch_failed=1
    done
    [[ "$batch_failed" -eq 0 ]] || exit 1
    pids=()
  fi
done
if [[ ${#pids[@]} -gt 0 ]]; then
  batch_failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || batch_failed=1
  done
  [[ "$batch_failed" -eq 0 ]] || exit 1
fi

echo "############ serial per-run/combined output generation ############"
SINGLE_SEED_OUTPUT_ARGS=(--seed "$first_seed")
[[ -z "$AGGREGATE_SYMBOL" ]] && SINGLE_SEED_OUTPUT_ARGS+=(--require-complete)
bash "$HERE/make_all_outputs.sh" "${SINGLE_SEED_OUTPUT_ARGS[@]}"

echo "############ cross-seed IQM/CI aggregation ############"
AGGREGATE_ARGS=(--seeds "$SEEDS" --require-complete)
[[ -n "$AGGREGATE_SYMBOL" ]] && AGGREGATE_ARGS+=(--symbol "$AGGREGATE_SYMBOL")
[[ "$SMOKE" -eq 0 ]] && AGGREGATE_ARGS+=(--publication)
bash "$HERE/make_multiseed_outputs.sh" "${AGGREGATE_ARGS[@]}"
echo "### run_multiseed complete (seeds: $SEEDS)"
