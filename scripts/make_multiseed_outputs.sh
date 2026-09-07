#!/usr/bin/env bash
# Cross-seed research report: four focused figures and three tables, with
# auditable seed estimates. Legacy comprehensive outputs are opt-in diagnostics.
# Reads each run's seed from seed.txt and includes only the requested seeds.
#
# Usage: scripts/make_multiseed_outputs.sh [--seeds "42 7 99 123 2024 314 577 811 1618 2718"]
#          [--require-complete] [--publication] [--symbol TICKER] [--diagnostics]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

SEEDS="42 7 99 123 2024 314 577 811 1618 2718"
REQUIRE_COMPLETE=0
PUBLICATION=0
SYMBOL=""
DIAGNOSTICS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --seeds=*) SEEDS="${1#*=}"; shift ;;
    --require-complete) REQUIRE_COMPLETE=1; shift ;;
    --publication) PUBLICATION=1; REQUIRE_COMPLETE=1; shift ;;
    --diagnostics) DIAGNOSTICS=1; shift ;;
    --symbol) SYMBOL="${2:-}"; shift 2 ;;
    --symbol=*) SYMBOL="${1#*=}"; shift ;;
    -h|--help)
      echo "usage: $0 [--seeds \"42 7 99 123 2024 314 577 811 1618 2718\"] [--require-complete] [--publication] [--symbol TICKER] [--diagnostics]"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$SYMBOL" ]]; then
  case " $SYMBOL " in
    " MSFT "|" JPM "|" PG "|" GOOGL "|" CAT ") ;;
    *) echo "unsupported historical symbol: $SYMBOL" >&2; exit 2 ;;
  esac
fi
if [[ "$PUBLICATION" -eq 1 && -n "$SYMBOL" ]]; then
  echo "--publication requires the complete five-ticker historical matrix; --symbol is not allowed" >&2
  exit 2
fi

run_seed() { tr -dc '0-9' < "$1/seed.txt" 2>/dev/null || true; }

for requested_seed in $SEEDS; do
  case "$requested_seed" in
    ''|*[!0-9]*) echo "invalid nonnegative integer seed: $requested_seed" >&2; exit 2 ;;
  esac
done
SEEDS=$(echo "$SEEDS" | tr ' ' '\n' | grep -v '^$' | sort -un | tr '\n' ' ')
requested_seed_count=$(echo "$SEEDS" | wc -w | tr -d ' ')
if [[ "$requested_seed_count" -lt 2 ]]; then
  echo "--seeds must contain at least two distinct master seeds" >&2
  exit 2
fi
if [[ "$PUBLICATION" -eq 1 ]]; then
  canonical_count=0
  for canonical_seed in 42 7 99 123 2024 314 577 811 1618 2718; do
    case " $SEEDS " in
      *" $canonical_seed "*) canonical_count=$((canonical_count + 1)) ;;
    esac
  done
  if [[ "$requested_seed_count" -ne 10 || "$canonical_count" -ne 10 ]]; then
    echo "--publication requires exactly the canonical seeds: 42 7 99 123 2024 314 577 811 1618 2718" >&2
    exit 2
  fi
fi

RESULTS_ROOT="${LMM_RESULTS_ROOT:-results/revision_v19}"
[[ -d "$RESULTS_ROOT" ]] || { echo "no $RESULTS_ROOT directory" >&2; exit 1; }
if [[ "$DIAGNOSTICS" -eq 0 ]]; then
  REPORT_ARGS=(--root "$RESULTS_ROOT" --seeds $SEEDS)
  [[ "$REQUIRE_COMPLETE" -eq 1 ]] && REPORT_ARGS+=(--require-complete --include-treatments)
  [[ "$PUBLICATION" -eq 1 ]] && REPORT_ARGS+=(--publication)
  [[ -n "$SYMBOL" ]] && REPORT_ARGS+=(--symbol "$SYMBOL")
  python3 -m lmm.experiments.make_report "${REPORT_ARGS[@]}"
  exit 0
fi
aggregates_written=0
complete_settings=""

for setting_dir in "$RESULTS_ROOT"/*/; do
  setting="$(basename "$setting_dir")"
  [[ "$setting" == _* ]] && continue
  if [[ "$REQUIRE_COMPLETE" -eq 1 \
        && "$setting" != "synthetic_rough_heston" \
        && "$setting" != "historical_sp500_midquotes" ]]; then
    continue
  fi
  group=() seen_seeds=""
  for rd in "${setting_dir}"*/; do
    base="$(basename "$rd")"
    [[ "$base" == _* ]] && continue
    if [[ -n "$SYMBOL" \
          && "$setting" == "historical_sp500_midquotes" \
          && "$base" != *_"${SYMBOL}"_seed* ]]; then
      continue
    fi
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
  # Ignore settings for which none of the explicitly requested seeds exists;
  # this lets unrelated treatment directories coexist in the results root.
  [[ "$n_seeds" -eq 0 ]] && continue
  if [[ "$n_seeds" -ne "$requested_seed_count" ]]; then
    echo "ERROR: $setting has only $n_seeds of $requested_seed_count requested seeds ($seen_seeds)" >&2
    exit 1
  fi

  # Each seed must contribute the same run identities (algorithm and, for the
  # historical setting, ticker). This allows intentionally symbol-scoped runs
  # while rejecting a silently unbalanced comparison matrix.
  reference_ids=""
  for want in $SEEDS; do
    ids=""
    for rd in ${group[@]+"${group[@]}"}; do
      [[ "$(run_seed "$rd")" == "$want" ]] || continue
      ids="$ids
$(basename "$rd" | sed "s/_seed${want}$//")"
    done
    ids=$(echo "$ids" | grep -v '^$' | sort -u)
    if [[ -z "$reference_ids" ]]; then
      reference_ids="$ids"
    elif [[ "$ids" != "$reference_ids" ]]; then
      echo "ERROR: $setting has an unbalanced run matrix at seed=$want" >&2
      exit 1
    fi
  done
  if [[ -z "$reference_ids" ]]; then
    echo "ERROR: $setting has no balanced run identities for seeds: $SEEDS" >&2
    exit 1
  fi
  if [[ "$REQUIRE_COMPLETE" -eq 1 ]]; then
    expected_ids=""
    if [[ "$setting" == "synthetic_rough_heston" ]]; then
      for algo in dqn ddpg td3 sac; do
        expected_ids="$expected_ids
$algo"
      done
    else
      tickers=(MSFT JPM PG GOOGL CAT)
      [[ -n "$SYMBOL" ]] && tickers=("$SYMBOL")
      for algo in dqn ddpg td3 sac; do
        for ticker in "${tickers[@]}"; do
          expected_ids="$expected_ids
${algo}_${ticker}"
        done
      done
    fi
    expected_ids=$(echo "$expected_ids" | grep -v '^$' | sort -u)
    if [[ "$reference_ids" != "$expected_ids" ]]; then
      echo "ERROR: $setting does not contain the required publication matrix" >&2
      echo "Expected:" >&2
      printf '  %s\n' $expected_ids >&2
      echo "Observed:" >&2
      printf '  %s\n' $reference_ids >&2
      exit 1
    fi
    complete_settings="$complete_settings $setting"
  fi
  echo "== multiseed aggregate: $setting  (${#group[@]} runs, seeds:$seen_seeds)"
  TABLE_ARGS=(--multiseed --run-dir "${group[@]}" \
    --out "$RESULTS_ROOT/${setting}/_multiseed/tables")
  [[ "$PUBLICATION" -eq 1 ]] && TABLE_ARGS+=(--publication)
  python3 -m lmm.experiments.make_tables "${TABLE_ARGS[@]}"
  python3 -m lmm.experiments.make_figures --multiseed --run-dir "${group[@]}" \
    --out "$RESULTS_ROOT/${setting}/_multiseed/figures"
  aggregates_written=$((aggregates_written + 1))
done
if [[ "$REQUIRE_COMPLETE" -eq 1 ]]; then
  for required_setting in synthetic_rough_heston historical_sp500_midquotes; do
    case " $complete_settings " in
      *" $required_setting "*) ;;
      *) echo "missing complete multiseed setting: $required_setting" >&2; exit 1 ;;
    esac
  done
fi
if [[ "$aggregates_written" -eq 0 ]]; then
  echo "no settings contained the requested seed set: $SEEDS" >&2
  exit 1
fi
if [[ "$REQUIRE_COMPLETE" -eq 1 ]]; then
  TREATMENT_ARGS=(--seeds "$SEEDS")
  [[ "$PUBLICATION" -eq 1 ]] && TREATMENT_ARGS+=(--publication)
  bash scripts/make_treatment_outputs.sh "${TREATMENT_ARGS[@]}"
fi
echo "### make_multiseed_outputs complete (seeds: $SEEDS)"
