#!/usr/bin/env bash
# Multi-seed reproduction: run the full pipeline under several master seeds, then
# build the cross-seed IQM/bootstrap-CI aggregate tables + figures (rliable-style;
# the credible RL reporting per Henderson 2018 / Agarwal 2021). Reported policy
# is the best-validation checkpoint (early stopping; evaluate.py default).
#
# DISK-SAFE: runs with --no-resume-ckpt so the replay-heavy periodic ckpt_ep*.pt
# are NOT written (3 x reproduction would otherwise ENOSPC). best/final/initial.pt
# are still written, so early stopping and evaluation are unaffected.
#
# Usage:
#   scripts/run_multiseed.sh [--seeds "42 7 99"] [--smoke] [--symbol TICKER]
# ~6-9 CPU-hours for 3 seeds (full); a few minutes with --smoke.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS="42 7 99"
PASS=()   # --smoke / --symbol forwarded to reproduce_all -> run_*.sh
AGGREGATE_SYMBOL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --seeds=*) SEEDS="${1#*=}"; shift ;;
    --smoke) PASS+=(--smoke); shift ;;
    --symbol) AGGREGATE_SYMBOL="$2"; PASS+=(--symbol "$2"); shift 2 ;;
    --symbol=*) AGGREGATE_SYMBOL="${1#*=}"; PASS+=(--symbol "$AGGREGATE_SYMBOL"); shift ;;
    -h|--help) echo "usage: $0 [--seeds \"42 7 99\"] [--smoke] [--symbol TICKER]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for s in $SEEDS; do
  echo "############ multi-seed: master seed ${s} ############"
  bash "$HERE/reproduce_all.sh" --seed "$s" --no-resume-ckpt ${PASS[@]+"${PASS[@]}"}
done

echo "############ cross-seed IQM/CI aggregation ############"
AGGREGATE_ARGS=(--seeds "$SEEDS" --require-complete)
[[ -n "$AGGREGATE_SYMBOL" ]] && AGGREGATE_ARGS+=(--symbol "$AGGREGATE_SYMBOL")
bash "$HERE/make_multiseed_outputs.sh" "${AGGREGATE_ARGS[@]}"
echo "### run_multiseed complete (seeds: $SEEDS)"
