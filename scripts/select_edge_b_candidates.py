from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from xau_lab.validation.edge_b import select_edge_b_candidates


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Edge B TOP_CANDIDATES.csv requires a header")
        return list(reader.fieldnames), list(reader)


def _write_rows(path: Path, fieldnames: list[str], rows: tuple[dict[str, object], ...]) -> None:
    output_fields = list(fieldnames)
    for required in ("edge_b_frequency_trades_per_week", "approval_scope"):
        if required not in output_fields:
            output_fields.append(required)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen Edge B trade-frequency constraint to robust candidates"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("edge_b/results/TOP_CANDIDATES.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edge_b/results/EDGE_B_SHORTLIST_CANDIDATES.csv"),
    )
    parser.add_argument("--min-trades-per-week", type=float, default=3.0)
    parser.add_argument("--max-trades-per-week", type=float, default=10.0)
    parser.add_argument("--max-candidates", type=int, default=6)
    args = parser.parse_args()

    fieldnames, rows = _read_rows(args.input)
    selected = select_edge_b_candidates(
        rows,
        min_trades_per_week=args.min_trades_per_week,
        max_trades_per_week=args.max_trades_per_week,
        max_candidates=args.max_candidates,
    )
    _write_rows(args.output, fieldnames, selected)
    print(f"ROBUST_INPUT_ROWS={len(rows)}")
    print(f"FREQUENCY_ELIGIBLE_SELECTED={len(selected)}")
    print(f"OUTPUT={args.output}")
    if len(selected) < 3:
        print("EDGE_B_STATUS=INSUFFICIENT_QUALIFYING_CANDIDATES")
    else:
        print("EDGE_B_STATUS=SHORTLIST_CANDIDATES_READY_FOR_FREEZE_REVIEW")


if __name__ == "__main__":
    main()
