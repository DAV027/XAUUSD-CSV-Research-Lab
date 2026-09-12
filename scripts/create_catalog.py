from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from xau_lab.experiments.sampler import generate_catalog


def write_catalog(path: Path, *, budget: int, seed: int) -> int:
    catalog = generate_catalog(total_budget=budget, seed=seed)
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
    parser = argparse.ArgumentParser(description="Create a deterministic complete-experiment catalog")
    parser.add_argument("--budget", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=9_215_000)
    parser.add_argument("--output", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget must be positive")

    count = write_catalog(args.output, budget=args.budget, seed=args.seed)
    print(f"Created {count} complete experiments at {args.output}")


if __name__ == "__main__":
    main()
