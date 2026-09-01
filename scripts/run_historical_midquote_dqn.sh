#!/usr/bin/env bash
# Minute-clock true-midquote DQN launcher. The default is a bounded pilot;
# --confirm explicitly selects the longer confirmation budget.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEED=42
SYMBOL=""
MODE=pilot
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --seed=*) SEED="${1#*=}"; shift ;;
    --symbol) SYMBOL="$2"; shift 2 ;;
    --symbol=*) SYMBOL="${1#*=}"; shift ;;
    --confirm) MODE=confirm; shift ;;
    -h|--help)
      echo "usage: $0 [--seed N] [--symbol TICKER] [--confirm]"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

DATA=data/historical_sp500_midquotes_1m.csv
if [[ ! -f "$DATA" || ! -f "${DATA}.meta.json" ]]; then
  echo "missing true-midquote artifact or sidecar; build it using data/README.md" >&2
  exit 2
fi

TICKERS=(MSFT JPM PG GOOGL CAT)
if [[ -n "$SYMBOL" ]]; then
  TICKERS=("$SYMBOL")
fi

BASE_CONFIGS=(
  --config configs/base.yaml
  --config configs/historical_sp500_midquotes.yaml
  --config configs/algo/dqn.yaml
  --config configs/pilot/v3_economic_compact.yaml
  --config configs/pilot/dqn_exploration.yaml
)
if [[ "$MODE" == confirm ]]; then
  BASE_CONFIGS+=(--config configs/pilot/v3_confirmation.yaml)
fi

python3 -m lmm.experiments.diagnose_simulator \
  --config configs/base.yaml \
  --config configs/historical_sp500_midquotes.yaml \
  --symbols "${TICKERS[@]}" --episodes 20 --assert-ready

for ticker in "${TICKERS[@]}"; do
  run_name="dqn_${ticker}_v5_${MODE}_seed${SEED}"
  if [[ "$MODE" == confirm ]]; then
    run_dir="results/revision_v5/historical_sp500_midquotes/${run_name}"
  else
    run_dir="results/pilots_v5/historical_sp500_midquotes/${run_name}"
  fi
  python3 -m lmm.experiments.train "${BASE_CONFIGS[@]}" \
    --run-name "$run_name" --seed "$SEED" --symbol "$ticker"
  python3 -m lmm.experiments.evaluate --run-dir "$run_dir" \
    --symbol "$ticker" --trace-episodes 1
  python3 -m lmm.experiments.policy_differences \
    --run-dir "$run_dir" --benchmark as
  python3 -m lmm.experiments.policy_differences \
    --run-dir "$run_dir" --benchmark twap
  python3 -m lmm.experiments.make_tables --run-dir "$run_dir"
done
