#!/usr/bin/env bash
# Regenerate every figure and table from the saved outputs of all FINISHED runs
# (those with checkpoints/final.pt). Backfills regret CSVs if missing, produces
# per-run figures/tables, then the cross-algorithm combined figures/tables per
# setting under results/<setting>/_combined/. Does NOT train.
#
# Usage: scripts/make_all_outputs.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d results ]]; then
  echo "no results/ directory; run a run_*.sh script first" >&2
  exit 1
fi

echo "== discovering finished runs (checkpoints/final.pt) =="
finished=()
while IFS= read -r ckpt; do
  finished+=("$(dirname "$(dirname "$ckpt")")")
done < <(find results -type f -path '*/checkpoints/final.pt' 2>/dev/null | sort)

if [[ ${#finished[@]} -eq 0 ]]; then
  echo "no finished runs found (need checkpoints/final.pt)" >&2
  exit 1
fi
printf '  %s\n' "${finished[@]}"

for rd in "${finished[@]}"; do
  echo "== per-run outputs: $rd =="
  if [[ -f "$rd/eval/records.csv" ]]; then
    [[ -f "$rd/eval/regret_as.csv" ]]   || python3 -m lmm.experiments.regret --run-dir "$rd" --benchmark as || true
    [[ -f "$rd/eval/regret_twap.csv" ]] || python3 -m lmm.experiments.regret --run-dir "$rd" --benchmark twap || true
  else
    echo "  WARNING: no eval/records.csv in $rd (re-run its run_*.sh)" >&2
  fi
  python3 -m lmm.experiments.make_figures --run-dir "$rd"
  python3 -m lmm.experiments.make_tables  --run-dir "$rd"
done

echo "== combined cross-algorithm outputs per setting =="
for setting_dir in results/*/; do
  setting="$(basename "$setting_dir")"
  # Only runs WITH eval records feed the combined outputs (figures a-e read the
  # first run dir, figure f / multi-algo tables read all of them).
  group=()
  for rd in "${finished[@]}"; do
    [[ "$(basename "$(dirname "$rd")")" == "$setting" && -f "$rd/eval/records.csv" ]] && group+=("$rd")
  done
  [[ ${#group[@]} -eq 0 ]] && continue
  echo "  setting=$setting (${#group[@]} runs)"
  python3 -m lmm.experiments.make_figures --run-dir "${group[@]}" --out "results/${setting}/_combined/figures"
  python3 -m lmm.experiments.make_tables  --run-dir "${group[@]}" --out "results/${setting}/_combined/tables"
done

echo "== make_all_outputs complete =="
