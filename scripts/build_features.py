from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from xau_lab.data.features import build_shared_features
from xau_lab.data.schema import DataPaths
from xau_lab.data.sessions import SessionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anti-lookahead shared XAUUSD features")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--session-config", type=Path)
    args = parser.parse_args()

    paths = DataPaths(args.root)
    if not paths.clean_csv.exists():
        raise FileNotFoundError(f"clean data not found: {paths.clean_csv}")

    session_config = None
    if args.session_config is not None:
        payload = json.loads(args.session_config.read_text(encoding="utf-8"))
        session_config = SessionConfig(**payload)

    clean = pl.read_csv(paths.clean_csv, try_parse_dates=False)
    features = build_shared_features(clean, session_config=session_config)

    paths.features_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.features_parquet.with_name(paths.features_parquet.name + ".tmp")
    features.write_parquet(tmp, compression="zstd")
    tmp.replace(paths.features_parquet)

    print(f"wrote {features.height} rows -> {paths.features_parquet}")


if __name__ == "__main__":
    main()
