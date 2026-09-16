from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from xau_lab.validation.edge_b_v2 import select_edge_b_v2_candidates


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Edge B v2 TOP_CANDIDATES.csv requires a header")
        return list(reader.fieldnames), list(reader)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen Edge B v2 after-cost daily-profit objective to robust candidates"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("edge_b_v2/results/TOP_CANDIDATES.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edge_b_v2/results/EDGE_B_V2_SHORTLIST_CANDIDATES.csv"),
    )
    parser.add_argument("--min-profit-per-active-day", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=6)
    args = parser.parse_args()

    fieldnames, rows = _read_rows(args.input)
    selected = select_edge_b_v2_candidates(
        rows,
        min_profit_per_active_day=args.min_profit_per_active_day,
        max_candidates=args.max_candidates,
    )
    _write_rows(args.output, fieldnames, selected)
    print(f"ROBUST_INPUT_ROWS={len(rows)}")
    print(f"DAILY_PROFIT_ELIGIBLE_SELECTED={len(selected)}")
    print(f"MIN_PROFIT_PER_ACTIVE_DAY={args.min_profit_per_active_day}")
    print("TRADE_FREQUENCY_CONSTRAINT=none")
    print(f"OUTPUT={args.output}")
    if not selected:
        print("EDGE_B_V2_STATUS=NO_DAILY_TARGET_SURVIVORS")
    else:
        print("EDGE_B_V2_STATUS=SHORTLIST_CANDIDATES_READY_FOR_FREEZE_REVIEW")


if __name__ == "__main__":
    main()
