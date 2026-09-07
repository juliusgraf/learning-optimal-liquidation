"""Paired fixed-policy economic differences from saved evaluation records."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import yaml

from lmm.config import environment_contract, LEGACY_CLEARING
from lmm.config import load_config
from lmm.experiments import plotting as P

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-policy-differences",
        description="Compute paired fixed-policy outcome differences under CRN.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmark", default="as", choices=["as", "twap"])
    parser.add_argument(
        "--policy",
        default=None,
        help="learned policy label (default: algo.name from the resolved config)",
    )
    parser.add_argument(
        "--metric",
        default="risk_adjusted_pnl",
        choices=["risk_adjusted_pnl", "pnl"],
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    metadata_path = run_dir / "eval" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get('clearing_mechanism', LEGACY_CLEARING) != cfg.auction_flow.clearing_mechanism:
        raise SystemExit('evaluation clearing mechanism disagrees with resolved config')
    if metadata.get("environment_contract") != environment_contract(cfg):
        raise SystemExit(
            "evaluation artifact predates the revised environment contract; "
            "old result files cannot be compared"
        )
    metadata_schema = metadata.get("artifact_schema_version")
    if metadata_schema is None or int(metadata_schema) != cfg.experiment.artifact_schema_version:
        raise SystemExit(
            "evaluation artifact_schema_version mismatch: expected active schema "
            f"{cfg.experiment.artifact_schema_version}, got {metadata_schema!r}"
        )
    if int(metadata_schema) != cfg.experiment.artifact_schema_version:
        raise SystemExit(
            "evaluation artifact_schema_version disagrees with config_resolved.yaml"
        )
    seed_path = run_dir / "seed.txt"
    if not seed_path.exists():
        raise SystemExit("run artifact is missing seed.txt")
    master_seed = int(seed_path.read_text().strip())
    if master_seed != cfg.experiment.master_seed:
        raise SystemExit(
            "seed.txt disagrees with config_resolved.yaml experiment.master_seed"
        )
    metadata_seed = metadata.get("master_seed")
    if metadata_seed is not None and int(metadata_seed) != master_seed:
        raise SystemExit("evaluation metadata master_seed disagrees with seed.txt")
    records = P.read_records(run_dir)
    try:
        P.validate_evaluation_records(run_dir, cfg, metadata, records)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if cfg.algo is None:
        raise ValueError("resolved run config has no learned algorithm")
    policy = args.policy or cfg.algo.name

    by_policy: dict[str, dict[int, dict[str, str]]] = {}
    with (run_dir / "eval" / "records.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            policy_rows = by_policy.setdefault(row["policy"], {})
            episode = int(row["episode"])
            if episode in policy_rows:
                raise SystemExit(
                    f"records.csv contains duplicate policy/episode row: "
                    f"{row['policy']!r}/{episode}"
                )
            policy_rows[episode] = row
    benchmark_rows = by_policy.get(args.benchmark)
    policy_rows = by_policy.get(policy)
    if not benchmark_rows or not policy_rows:
        raise SystemExit(
            f"records.csv lacks {args.benchmark!r}/{policy!r}; run evaluation first"
        )
    episodes = sorted(benchmark_rows)
    if episodes != sorted(policy_rows):
        raise SystemExit("paired policy episode sets differ")

    out_path = run_dir / "eval" / f"policy_difference_{args.benchmark}.csv"
    cumulative = 0.0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "env_seed",
                "benchmark_value",
                "policy_value",
                "policy_minus_benchmark",
                "cumulative_policy_minus_benchmark",
            ]
        )
        for episode in episodes:
            rb, rp = benchmark_rows[episode], policy_rows[episode]
            if rb["env_seed"] != rp["env_seed"]:
                raise SystemExit(f"CRN violation at evaluation episode {episode}")
            benchmark_value = float(rb[args.metric])
            policy_value = float(rp[args.metric])
            difference = policy_value - benchmark_value
            cumulative += difference
            writer.writerow(
                [
                    episode,
                    rb["env_seed"],
                    *[
                        format(value, ".17g")
                        for value in (
                            benchmark_value,
                            policy_value,
                            difference,
                            cumulative,
                        )
                    ],
                ]
            )
    print(
        f"cumulative paired {args.metric} difference = {cumulative:.6f} "
        f"[policy={policy}, benchmark={args.benchmark}, E={len(episodes)}]"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
