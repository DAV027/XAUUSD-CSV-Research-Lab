from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from xau_lab.data.features import build_shared_features, filter_research_window
from xau_lab.data.schema import DataPaths
from xau_lab.data.sessions import SessionConfig


DEFAULT_SESSION_CONFIG = Path(__file__).resolve().parents[1] / "config" / "research_sessions_v1.json"
_SESSION_FIELDS = {
    "broker_timezone",
    "asia_timezone",
    "asia_start",
    "asia_end",
    "london_timezone",
    "london_start",
    "london_end",
    "new_york_timezone",
    "new_york_start",
    "new_york_end",
}


def load_session_config(path: Path | None) -> SessionConfig:
    target = DEFAULT_SESSION_CONFIG if path is None else Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"session config must be a JSON object: {target}")
    if payload.get("version") != "v1":
        raise ValueError(f"unsupported session config version in {target}: {payload.get('version')!r}")
    missing = sorted(field for field in _SESSION_FIELDS if field not in payload)
    if missing:
        raise ValueError(f"session config missing required fields: {missing}")
    values = {field: payload[field] for field in _SESSION_FIELDS}
    return SessionConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anti-lookahead shared XAUUSD features")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--session-config", type=Path)
    args = parser.parse_args()

    paths = DataPaths(args.root)
    if not paths.clean_csv.exists():
        raise FileNotFoundError(f"clean data not found: {paths.clean_csv}")

    session_config = load_session_config(args.session_config)
    clean = filter_research_window(pl.read_csv(paths.clean_csv, try_parse_dates=False))
    features = build_shared_features(clean, session_config=session_config)

    paths.features_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.features_parquet.with_name(paths.features_parquet.name + ".tmp")
    features.write_parquet(tmp, compression="zstd")
    tmp.replace(paths.features_parquet)

    print(f"wrote {features.height} rows -> {paths.features_parquet}")


if __name__ == "__main__":
    main()
