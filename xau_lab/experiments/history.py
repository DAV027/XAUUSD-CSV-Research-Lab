from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import polars as pl

HISTORICAL_COLUMNS = [
    "historical_id",
    "origin",
    "strategy_family",
    "strategy_name",
    "status",
    "net_profit",
    "profit_factor",
    "after_cost_profit",
    "after_cost_pf",
    "trades",
    "max_drawdown_pct",
    "source_file",
    "manifest_file",
    "results_file",
    "evidence_notes",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_member(zf: zipfile.ZipFile, member: str | None) -> dict[str, Any]:
    if not member:
        return {}
    try:
        value = json.loads(zf.read(member).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _find_member(members: list[str], suffix: str) -> str | None:
    matches = [name for name in members if name.endswith(suffix)]
    return matches[0] if matches else None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ledger_rows(zf: zipfile.ZipFile, members: list[str]) -> list[dict[str, Any]]:
    ledger_member = _find_member(members, "experiment_ledger.jsonl")
    if not ledger_member:
        return []
    rows: list[dict[str, Any]] = []
    text = zf.read(ledger_member).decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _historical_id(row: dict[str, Any]) -> str | None:
    value = _first(row, "historical_id", "lab_id", "experiment_id", "id")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def import_historical_checkpoint(
    archive_path: str | Path,
    result_root: str | Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    archive_path = Path(archive_path)
    result_root = Path(result_root)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    result_root.mkdir(parents=True, exist_ok=True)

    imported: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = sorted(name for name in zf.namelist() if not name.endswith("/"))
        for ledger in _ledger_rows(zf, members):
            historical_id = _historical_id(ledger)
            if not historical_id:
                continue
            results_member = _find_member(members, f"{historical_id}_results.json")
            manifest_member = _find_member(members, f"{historical_id}_manifest.json")
            results = _read_json_member(zf, results_member)
            manifest = _read_json_member(zf, manifest_member)

            notes: dict[str, Any] = {}
            for key in ("build", "terminal_build", "symbol", "timeframe", "notes", "reason"):
                value = _first(manifest, key)
                if value is None:
                    value = _first(ledger, key)
                if value is not None:
                    notes[key] = value

            imported.append(
                {
                    "historical_id": historical_id,
                    "origin": "historical_mt5",
                    "strategy_family": _first(ledger, "strategy_family", "family"),
                    "strategy_name": _first(ledger, "strategy_name", "strategy", "name"),
                    "status": _first(ledger, "status", "decision"),
                    "net_profit": _to_float(_first(results, "net_profit", "profit", "net_pnl")),
                    "profit_factor": _to_float(_first(results, "profit_factor", "pf")),
                    "after_cost_profit": _to_float(_first(results, "after_cost_profit", "after_cost_pnl")),
                    "after_cost_pf": _to_float(_first(results, "after_cost_pf", "after_cost_profit_factor")),
                    "trades": _to_int(_first(results, "trades", "completed_trades", "total_trades")),
                    "max_drawdown_pct": _to_float(_first(results, "max_drawdown_pct", "drawdown_pct")),
                    "source_file": _first(manifest, "source_file", "report_file", "source"),
                    "manifest_file": manifest_member,
                    "results_file": results_member,
                    "evidence_notes": json.dumps(notes, sort_keys=True, separators=(",", ":")) if notes else None,
                }
            )

    if imported:
        frame = pl.DataFrame(imported).select(HISTORICAL_COLUMNS)
    else:
        frame = pl.DataFrame({column: [] for column in HISTORICAL_COLUMNS})
    frame.write_csv(result_root / "HISTORICAL_EVIDENCE.csv")

    manifest = {
        "source_archive": str(archive_path),
        "source_sha256": _sha256(archive_path),
        "members": members,
        "imported_rows": frame.height,
    }
    manifest_path = result_root / "HISTORICAL_IMPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return frame, manifest


def _normalize_strategy_name(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _values_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return True
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return str(left).strip().lower() == str(right).strip().lower()
    if not (math.isfinite(left_float) and math.isfinite(right_float)):
        return False
    return math.isclose(left_float, right_float, rel_tol=1e-10, abs_tol=1e-12)


def historical_matches(
    strategy_name: str,
    canonical_parameters: dict,
    evidence: pl.DataFrame,
) -> list[str]:
    if "historical_id" not in evidence.columns or "strategy_name" not in evidence.columns:
        return []
    target = _normalize_strategy_name(strategy_name)
    matches: set[str] = set()
    shared_parameter_columns = [key for key in canonical_parameters if key in evidence.columns]

    for row in evidence.iter_rows(named=True):
        if _normalize_strategy_name(row.get("strategy_name") or "") != target:
            continue
        if all(
            row.get(key) is None or _values_equal(row.get(key), canonical_parameters[key])
            for key in shared_parameter_columns
        ):
            historical_id = row.get("historical_id")
            if historical_id:
                matches.add(str(historical_id))
    return sorted(matches)


__all__ = [
    "HISTORICAL_COLUMNS",
    "historical_matches",
    "import_historical_checkpoint",
]
