#!/usr/bin/env bash
# Revised clearing: fit forecasts, then run the 440 headline/historical/control
# treatment matrix, then build the focused economic research report with
# seed-level uncertainty and paired treatment contrasts. Reported policy is the
# best-validation checkpoint (early stopping; evaluate.py default).
#
# DISK-SAFE: runs with --no-resume-ckpt so the replay-heavy periodic ckpt_ep*.pt
# are NOT written (ten-seed reproduction would otherwise ENOSPC). best/final/initial.pt
# are still written, so early stopping and evaluation are unaffected.
#
# Usage:
#   scripts/run_multiseed.sh [--seeds "42 7 99 123 2024 314 577 811 1618 2718"] [--jobs N]
#                            [--threads-per-job N] [--smoke] [--symbol TICKER] [--dry-run]
# Individual runs share a bounded queue across all seeds, algorithms and arms.
# Report generation starts once, after every run succeeds.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

SEEDS="42 7 99 123 2024 314 577 811 1618 2718"
SEEDS_EXPLICIT=0
JOBS=1
THREADS_PER_JOB=2
AGGREGATE_SYMBOL=""
SMOKE=0
DRY_RUN=0
LEGACY_CLEARING=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; SEEDS_EXPLICIT=1; shift 2 ;;
    --seeds=*) SEEDS="${1#*=}"; SEEDS_EXPLICIT=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --threads-per-job) THREADS_PER_JOB="$2"; shift 2 ;;
    --threads-per-job=*) THREADS_PER_JOB="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --legacy-clearing) LEGACY_CLEARING=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --symbol) AGGREGATE_SYMBOL="$2"; shift 2 ;;
    --symbol=*) AGGREGATE_SYMBOL="${1#*=}"; shift ;;
    -h|--help)
      echo "usage: $0 [--seeds \"42 7 99 123 2024 314 577 811 1618 2718\"] [--jobs N] [--threads-per-job N] [--smoke] [--symbol TICKER] [--dry-run] [--legacy-clearing]"
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
      42|7|99|123|2024|314|577|811|1618|2718)
        echo "smoke runs must use noncanonical seeds (for example 9001 9002)" >&2
        exit 2
        ;;
    esac
  fi
  seed_count=$((seed_count + 1))
done
[[ "$seed_count" -ge 2 ]] || { echo "--seeds must contain at least two seeds" >&2; exit 2; }
if [[ "$SMOKE" -eq 0 ]]; then
  canonical_count=0
  for canonical_seed in 42 7 99 123 2024 314 577 811 1618 2718; do
    case " $seen_seeds " in
      *" $canonical_seed "*) canonical_count=$((canonical_count + 1)) ;;
    esac
  done
  if [[ "$seed_count" -ne 10 || "$canonical_count" -ne 10 ]]; then
    echo "full publication mode requires exactly the canonical seeds: 42 7 99 123 2024 314 577 811 1618 2718" >&2
    exit 2
  fi
  if [[ "$DRY_RUN" -eq 0 && -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "full publication mode requires a clean git worktree so git_sha.txt fully identifies the code" >&2
    exit 2
  fi
fi

DEFAULT_ROOT=results/revision_v20
[[ "$LEGACY_CLEARING" -eq 1 ]] && DEFAULT_ROOT=results/revision_v19
# Make the advertised command usable without first activating the local venv.
# An explicit interpreter remains available for batch hosts and launcher tests.
if [[ -z "${LMM_PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    LMM_PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    LMM_PYTHON="$(command -v python3)"
  fi
fi
export LMM_PYTHON
export PATH="$(dirname "$LMM_PYTHON"):$PATH"
QUEUE_ARGS=(--root "${LMM_RESULTS_ROOT:-$DEFAULT_ROOT}" --seeds $SEEDS
            --jobs "$JOBS" --threads-per-job "$THREADS_PER_JOB")
[[ "$SMOKE" -eq 1 ]] && QUEUE_ARGS+=(--smoke)
[[ "$DRY_RUN" -eq 1 ]] && QUEUE_ARGS+=(--dry-run)
[[ -n "$AGGREGATE_SYMBOL" ]] && QUEUE_ARGS+=(--symbol "$AGGREGATE_SYMBOL")
[[ "$LEGACY_CLEARING" -eq 1 ]] && QUEUE_ARGS+=(--legacy-clearing)
"$LMM_PYTHON" -m lmm.experiments.run_matrix "${QUEUE_ARGS[@]}"
