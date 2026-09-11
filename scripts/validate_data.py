from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from xau_lab.data.schema import DataPaths
from xau_lab.data.validate import validate_bars


def _atomic_write_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.write_csv(tmp)
    tmp.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and clean canonical XAUUSD M1 data")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    paths = DataPaths(args.root)
    if not paths.raw_csv.exists():
        raise FileNotFoundError(f"raw data not found: {paths.raw_csv}")

    raw = pl.read_csv(paths.raw_csv, try_parse_dates=False)
    result = validate_bars(raw)

    _atomic_write_csv(result.clean, paths.clean_csv)
    _atomic_write_csv(result.issues, paths.quality_csv)

    metadata = {}
    if paths.metadata_json.exists():
        metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    metadata.update(
        {
            "raw_rows": result.summary["input_rows"],
            "clean_rows": result.summary["clean_rows"],
            "removed_rows": result.summary["removed_rows"],
            "non_monotonic_input": result.summary["non_monotonic_input"],
            "validation_issue_counts": result.summary["issue_counts"],
        }
    )
    _atomic_write_json(metadata, paths.metadata_json)

    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
