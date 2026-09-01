#!/usr/bin/env bash
# Regenerate every figure and table from the saved outputs of all FINISHED runs
# (those with checkpoints/final.pt). Backfills paired-difference CSVs if missing, produces
# per-run figures/tables, then the cross-algorithm combined figures/tables per
# setting under results/<setting>/_combined/. Does NOT train.
#
# The combined (cross-algorithm) outputs require COMMON RANDOM NUMBERS across
# the algorithms being compared: every run in a combined group must share one
# master seed, or the eval-summary / historical tables abort with a CRN
# violation (tables.py: "env seeds differ between policies"). results/ often
# accumulates runs from several seeds (e.g. dqn_seed42, dqn_seed97, sweeps,
# acceptance runs); mixing them is exactly what breaks the combined tables.
# Therefore the COMBINED step is scoped to a single master seed (read from each
# run's seed.txt). Per-run outputs are still produced for every finished run.
#
# Usage: scripts/make_all_outputs.sh [--seed N]
#   --seed N : master seed for the combined groups. Default: 42 if present,
#              else the only seed present, else the smallest (with a warning).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMBINED_SEED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) COMBINED_SEED="${2:-}"; shift 2 ;;
    --seed=*) COMBINED_SEED="${1#*=}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

RESULTS_ROOT="results/revision_v5"
if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "no $RESULTS_ROOT directory; run a run_*.sh script first" >&2
  exit 1
fi

# Read a run's master seed from seed.txt (the authoritative provenance file).
run_seed() { tr -dc '0-9' < "$1/seed.txt" 2>/dev/null || true; }

echo "== discovering finished runs (checkpoints/final.pt) =="
finished=()
while IFS= read -r ckpt; do
  finished+=("$(dirname "$(dirname "$ckpt")")")
done < <(find "$RESULTS_ROOT" -type f -path '*/checkpoints/final.pt' 2>/dev/null | sort)

if [[ ${#finished[@]} -eq 0 ]]; then
  echo "no finished runs found (need checkpoints/final.pt)" >&2
  exit 1
fi
printf '  %s\n' "${finished[@]}"

# -- per-run outputs (every finished run; no cross-run mixing here) -----------
for rd in "${finished[@]}"; do
  echo "== per-run outputs: $rd =="
  if [[ -f "$rd/eval/records.csv" ]]; then
    [[ -f "$rd/eval/policy_difference_as.csv" ]]   || python3 -m lmm.experiments.policy_differences --run-dir "$rd" --benchmark as || true
    [[ -f "$rd/eval/policy_difference_twap.csv" ]] || python3 -m lmm.experiments.policy_differences --run-dir "$rd" --benchmark twap || true
  else
    echo "  WARNING: no eval/records.csv in $rd (re-run its run_*.sh)" >&2
  fi
  python3 -m lmm.experiments.make_figures --run-dir "$rd"
  python3 -m lmm.experiments.make_tables  --run-dir "$rd"
done

# -- combined cross-algorithm outputs per setting (single seed, CRN-safe) -----
echo "== combined cross-algorithm outputs per setting =="
for setting_dir in "$RESULTS_ROOT"/*/; do
  setting="$(basename "$setting_dir")"
  [[ "$setting" == "_combined" ]] && continue

  # Candidate runs: finished, in this setting, WITH eval records.
  candidates=()
  seeds_present=""
  for rd in "${finished[@]}"; do
    [[ "$(basename "$(dirname "$rd")")" == "$setting" ]] || continue
    [[ -f "$rd/eval/records.csv" ]] || continue
    candidates+=("$rd")
    s="$(run_seed "$rd")"
    [[ -n "$s" ]] && seeds_present="$seeds_present $s"
  done
  [[ ${#candidates[@]} -eq 0 ]] && continue

  uniq_seeds=$(echo "$seeds_present" | tr ' ' '\n' | grep -v '^$' | sort -un)

  # Pick the combined seed: explicit --seed, else single present, else 42, else smallest.
  target="$COMBINED_SEED"
  if [[ -z "$target" ]]; then
    n_seeds=$(echo "$uniq_seeds" | grep -c .)
    if [[ "$n_seeds" -eq 1 ]]; then
      target="$uniq_seeds"
    elif echo "$uniq_seeds" | grep -qx 42; then
      target=42
    else
      target=$(echo "$uniq_seeds" | head -n1)
    fi
  fi

  # Filter candidates to the target seed (CRN: combined group shares one seed).
  group=()
  skipped=""
  for rd in "${candidates[@]}"; do
    if [[ "$(run_seed "$rd")" == "$target" ]]; then
      group+=("$rd")
    else
      skipped="$skipped $(basename "$rd")(seed=$(run_seed "$rd"))"
    fi
  done

  if [[ ${#group[@]} -eq 0 ]]; then
    echo "  setting=$setting: no runs at seed=$target (present:$(echo $uniq_seeds | tr '\n' ' ')); skipping combined" >&2
    continue
  fi
  echo "  setting=$setting seed=$target (${#group[@]} runs)"
  [[ -n "$skipped" ]] && echo "    excluded from combined (other seeds):$skipped (pass --seed to choose)"
  python3 -m lmm.experiments.make_figures --run-dir "${group[@]}" --out "$RESULTS_ROOT/${setting}/_combined/figures"
  python3 -m lmm.experiments.make_tables  --run-dir "${group[@]}" --out "$RESULTS_ROOT/${setting}/_combined/tables"
done

echo "== make_all_outputs complete =="
