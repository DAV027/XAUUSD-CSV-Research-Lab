from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from xau_lab.validation.edge_c_v2 import select_edge_c_v2_candidates


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Edge C v2 TOP_CANDIDATES.csv requires a header")
        return list(reader.fieldnames), list(reader)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    output_fields = list(fieldnames)
    if "approval_scope" not in output_fields:
        output_fields.append("approval_scope")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row.setdefault("approval_scope", "edge_c_v2_viability")
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply frozen Edge C v2 economic viability gates to robust candidates"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("edge_c_v2/results/TOP_CANDIDATES.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edge_c_v2/results/EDGE_C_V2_SHORTLIST_CANDIDATES.csv"),
    )
    parser.add_argument("--max-candidates", type=int, default=6)
    args = parser.parse_args()

    fieldnames, rows = _read_rows(args.input)
    selected = select_edge_c_v2_candidates(rows, max_candidates=args.max_candidates)
    _write_rows(args.output, fieldnames, selected)

    print(f"ROBUST_INPUT_ROWS={len(rows)}")
    print(f"ECONOMICALLY_ELIGIBLE_SELECTED={len(selected)}")
    print(f"OUTPUT={args.output}")
    if selected:
        print("EDGE_C_V2_STATUS=VIABLE_CANDIDATES_READY_FOR_COVERAGE_REVIEW")
    else:
        print("EDGE_C_V2_STATUS=NO_VIABLE_CANDIDATES_VALID_NEGATIVE_RESULT")


if __name__ == "__main__":
    main()
