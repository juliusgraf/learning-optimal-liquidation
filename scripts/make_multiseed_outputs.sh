#!/usr/bin/env bash
# Cross-seed aggregation: build the IQM/bootstrap-CI tables + figure from runs
# spanning several master seeds, per setting, into
# results/revision_v5/<setting>/_multiseed/.
# Reads each run's seed from seed.txt and includes only the requested seeds.
#
# Usage: scripts/make_multiseed_outputs.sh [--seeds "42 7 99"]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="42 7 99"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --seeds=*) SEEDS="${1#*=}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

run_seed() { tr -dc '0-9' < "$1/seed.txt" 2>/dev/null || true; }

RESULTS_ROOT="results/revision_v5"
[[ -d "$RESULTS_ROOT" ]] || { echo "no $RESULTS_ROOT directory" >&2; exit 1; }

for setting_dir in "$RESULTS_ROOT"/*/; do
  setting="$(basename "$setting_dir")"
  [[ "$setting" == _* ]] && continue
  group=() seen_seeds=""
  for rd in "${setting_dir}"*/; do
    base="$(basename "$rd")"
    [[ "$base" == _* ]] && continue
    [[ -f "${rd}eval/records.csv" ]] || continue
    s="$(run_seed "${rd%/}")"
    for want in $SEEDS; do
      if [[ "$s" == "$want" ]]; then
        group+=("${rd%/}")
        case " $seen_seeds " in *" $s "*) ;; *) seen_seeds="$seen_seeds $s" ;; esac
      fi
    done
  done
  n_seeds=$(echo $seen_seeds | wc -w | tr -d ' ')
  if [[ "$n_seeds" -lt 2 ]]; then
    echo "== $setting: only $n_seeds seed(s) present ($seen_seeds); need >=2; skipping" >&2
    continue
  fi
  echo "== multiseed aggregate: $setting  (${#group[@]} runs, seeds:$seen_seeds)"
  python3 -m lmm.experiments.make_tables  --multiseed --run-dir "${group[@]}" \
    --out "$RESULTS_ROOT/${setting}/_multiseed/tables"
  python3 -m lmm.experiments.make_figures --multiseed --run-dir "${group[@]}" \
    --out "$RESULTS_ROOT/${setting}/_multiseed/figures"
done
echo "### make_multiseed_outputs complete (seeds: $SEEDS)"
