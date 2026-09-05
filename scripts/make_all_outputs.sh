#!/usr/bin/env bash
# Generate a focused single-seed development report. --diagnostics enables
# the comprehensive legacy figures/tables from all FINISHED runs
# (those with a reportable checkpoints/best.pt). Backfills paired-difference CSVs if missing, produces
# per-run figures/tables, then the cross-algorithm combined figures/tables per
# setting under results/<setting>/_combined/. Does NOT train.
#
# The combined (cross-algorithm) outputs require COMMON RANDOM NUMBERS across
# the algorithms being compared: every run in a combined group must share one
# master seed, or the eval-summary / historical tables abort with a CRN
# violation (tables.py: "env seeds differ between policies"). results/ often
# accumulates runs from several seeds (e.g. dqn_seed42, dqn_seed97, sweeps,
# acceptance runs); mixing them is exactly what breaks the combined tables.
# Therefore the combined step is scoped to a single master seed (read from each
# run's seed.txt). When ``--seed`` is explicit, per-run outputs are scoped to
# that seed too; multiseed publication summaries do not need hundreds of
# duplicate per-run plots.
#
# Usage: scripts/make_all_outputs.sh [--seed N] [--require-complete] [--diagnostics]
#   --seed N : master seed for per-run and combined outputs. By default,
#              per-run outputs cover every finished run; combined groups use
#              42 if present, else the only seed present, else the smallest.
#   --require-complete : require all four algorithms (and all five historical
#              tickers) at the selected seed before writing combined outputs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

COMBINED_SEED=""
COMBINED_SEED_EXPLICIT=0
REQUIRE_COMPLETE=0
DIAGNOSTICS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) COMBINED_SEED="${2:-}"; COMBINED_SEED_EXPLICIT=1; shift 2 ;;
    --seed=*) COMBINED_SEED="${1#*=}"; COMBINED_SEED_EXPLICIT=1; shift ;;
    --require-complete) REQUIRE_COMPLETE=1; shift ;;
    --diagnostics) DIAGNOSTICS=1; shift ;;
    -h|--help)
      echo "usage: $0 [--seed N] [--require-complete] [--diagnostics]"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

RESULTS_ROOT="${LMM_RESULTS_ROOT:-results/revision_v17}"
if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "no $RESULTS_ROOT directory; run a run_*.sh script first" >&2
  exit 1
fi

# Read a run's master seed from seed.txt (the authoritative provenance file).
run_seed() { tr -dc '0-9' < "$1/seed.txt" 2>/dev/null || true; }

echo "== discovering successful runs (checkpoints/best.pt) =="
finished=()
while IFS= read -r ckpt; do
  finished+=("$(dirname "$(dirname "$ckpt")")")
done < <(find "$RESULTS_ROOT" -type f -path '*/checkpoints/best.pt' 2>/dev/null | sort)

if [[ ${#finished[@]} -eq 0 ]]; then
  echo "no successful runs found (need a mature, reportable checkpoints/best.pt)" >&2
  exit 1
fi
if [[ "$COMBINED_SEED_EXPLICIT" -eq 1 ]]; then
  selected=()
  for rd in "${finished[@]}"; do
    [[ "$(run_seed "$rd")" == "$COMBINED_SEED" ]] && selected+=("$rd")
  done
  finished=("${selected[@]}")
  if [[ ${#finished[@]} -eq 0 ]]; then
    echo "no successful runs found for explicit seed=$COMBINED_SEED" >&2
    exit 1
  fi
fi
printf '  %s\n' "${finished[@]}"

if [[ "$DIAGNOSTICS" -eq 0 ]]; then
  if [[ -z "$COMBINED_SEED" ]]; then
    COMBINED_SEED="$(for rd in "${finished[@]}"; do run_seed "$rd"; echo; done | sort -un | head -1)"
    for rd in "${finished[@]}"; do
      [[ "$(run_seed "$rd")" == "42" ]] && COMBINED_SEED=42
    done
  fi
  REPORT_ARGS=(--root "$RESULTS_ROOT" --seeds "$COMBINED_SEED")
  [[ "$REQUIRE_COMPLETE" -eq 1 ]] && REPORT_ARGS+=(--require-complete)
  python3 -m lmm.experiments.make_report "${REPORT_ARGS[@]}"
  exit 0
fi

# -- per-run outputs (selected finished runs; no cross-run mixing here) -------
for rd in "${finished[@]}"; do
  echo "== per-run outputs: $rd =="
  if [[ ! -f "$rd/eval/records.csv" ]]; then
    echo "ERROR: no eval/records.csv in $rd (re-run its run_*.sh)" >&2
    exit 1
  fi
  if [[ ! -f "$rd/seed.txt" ]]; then
    echo "ERROR: no seed.txt in $rd; provenance is incomplete" >&2
    exit 1
  fi
  # Always rebuild derived paired differences. Evaluation may have been rerun
  # in place, in which case a pre-existing CSV is not evidence of freshness.
  python3 -m lmm.experiments.policy_differences --run-dir "$rd" --benchmark as
  python3 -m lmm.experiments.policy_differences --run-dir "$rd" --benchmark twap
  python3 -m lmm.experiments.make_figures --run-dir "$rd"
  python3 -m lmm.experiments.make_tables  --run-dir "$rd"
done

# -- combined cross-algorithm outputs per setting (single seed, CRN-safe) -----
echo "== combined cross-algorithm outputs per setting =="
complete_settings=""
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
    echo "  setting=$setting: no runs at seed=$target (present:$(echo $uniq_seeds | tr '\n' ' '))" >&2
    if [[ "$COMBINED_SEED_EXPLICIT" -eq 1 || "$REQUIRE_COMPLETE" -eq 1 ]]; then
      exit 1
    fi
    echo "  skipping combined output for this setting" >&2
    continue
  fi

  if [[ "$REQUIRE_COMPLETE" -eq 1 ]]; then
    required=()
    if [[ "$setting" == "synthetic_rough_heston" ]]; then
      for algo in dqn ddpg td3 sac; do
        required+=("${algo}_seed${target}")
      done
    elif [[ "$setting" == "historical_sp500_midquotes" ]]; then
      for algo in dqn ddpg td3 sac; do
        for ticker in MSFT JPM PG GOOGL CAT; do
          required+=("${algo}_${ticker}_seed${target}")
        done
      done
    fi
    missing=()
    for run_name in ${required[@]+"${required[@]}"}; do
      run_path="${setting_dir}${run_name}"
      if [[ ! -f "$run_path/checkpoints/best.pt" \
            || ! -f "$run_path/eval/records.csv" \
            || "$(run_seed "$run_path")" != "$target" ]]; then
        missing+=("$run_name")
      fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
      echo "ERROR: setting=$setting seed=$target lacks required publication runs:" >&2
      printf '  %s\n' "${missing[@]}" >&2
      exit 1
    fi
    if [[ "$setting" == "synthetic_rough_heston" \
          || "$setting" == "historical_sp500_midquotes" ]]; then
      complete_settings="$complete_settings $setting"
    fi
  fi
  echo "  setting=$setting seed=$target (${#group[@]} runs)"
  [[ -n "$skipped" ]] && echo "    excluded from combined (other seeds):$skipped (pass --seed to choose)"
  python3 -m lmm.experiments.make_figures --run-dir "${group[@]}" --out "$RESULTS_ROOT/${setting}/_combined/figures"
  python3 -m lmm.experiments.make_tables  --run-dir "${group[@]}" --out "$RESULTS_ROOT/${setting}/_combined/tables"
done

if [[ "$REQUIRE_COMPLETE" -eq 1 ]]; then
  for required_setting in synthetic_rough_heston historical_sp500_midquotes; do
    case " $complete_settings " in
      *" $required_setting "*) ;;
      *) echo "ERROR: missing complete publication setting: $required_setting" >&2; exit 1 ;;
    esac
  done
fi

echo "== make_all_outputs complete =="
