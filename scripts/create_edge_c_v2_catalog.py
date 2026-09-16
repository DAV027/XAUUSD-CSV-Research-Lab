from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from xau_lab.experiments.edge_c_v2_sampler import (
    EDGE_C_V2_DEFAULT_BUDGET,
    EDGE_C_V2_DEFAULT_SEED,
    generate_edge_c_v2_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the deterministic Edge C v2 volatility-scaled catalog"
    )
    parser.add_argument("--budget", type=int, default=EDGE_C_V2_DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=EDGE_C_V2_DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edge_c_v2/results/EXPERIMENT_CATALOG.csv"),
    )
    return parser


def write_edge_c_v2_catalog(path: Path, *, budget: int, seed: int) -> int:
    catalog = generate_edge_c_v2_catalog(total_budget=budget, seed=seed)
    rows = [experiment.to_dict() for experiment in catalog]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return len(rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget must be positive")

    count = write_edge_c_v2_catalog(args.output, budget=args.budget, seed=args.seed)
    print(f"Created {count} Edge C v2 complete experiments at {args.output}")


if __name__ == "__main__":
    main()
