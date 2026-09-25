from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


CANONICAL_COLUMNS = [
    "event_id",
    "timestamp_utc",
    "event_family",
    "event_name",
    "actual",
    "consensus",
    "previous",
    "unit",
    "directionality",
    "surprise_raw",
    "surprise_std",
    "source",
]

RAW_REQUIRED_COLUMNS = [
    "event_id",
    "timestamp_utc",
    "event_family",
    "event_name",
    "actual",
    "consensus",
    "previous",
    "unit",
    "source",
]

DIRECTIONALITY = {
    "CPI": 1,
    "CORE_CPI": 1,
    "NFP": 1,
    "UNEMPLOYMENT": -1,
    "PCE": 1,
    "CORE_PCE": 1,
}


class ValidationError(ValueError):
    pass


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValidationError("unsupported H16 config schema_version")
    return data


def _parse_utc_timestamp(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValidationError("timestamp_utc is missing")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"invalid timestamp_utc: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"timestamp_utc must include an explicit UTC offset: {value!r}")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValidationError(f"timestamp_utc must be UTC (+00:00/Z): {value!r}")
    return parsed.astimezone(timezone.utc)


def _parse_number(value: str, field: str) -> float:
    text = value.strip()
    if not text:
        raise ValidationError(f"{field} is missing")
    try:
        number = float(text.replace(",", ""))
    except ValueError as exc:
        raise ValidationError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite")
    return number


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return format(value, ".12g")


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _oos_start(config: dict) -> datetime:
    raw = config["final_oos"]["from"]
    return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)


def canonicalize_rows(
    rows: list[dict[str, str]],
    config: dict,
    *,
    unlock_oos: bool = False,
) -> list[dict[str, str]]:
    allowed = set(config["event_families"])
    window = int(config["normalization"]["rolling_prior_releases"])
    min_history = int(config["normalization"]["minimum_prior_releases"])
    ddof = int(config["normalization"]["sample_std_ddof"])
    if window < 2:
        raise ValidationError("rolling_prior_releases must be >= 2")
    if min_history < 2 or min_history > window:
        raise ValidationError(
            "minimum_prior_releases must be between 2 and rolling_prior_releases"
        )
    if ddof != 1:
        raise ValidationError("H16 v1 freezes sample_std_ddof=1")

    seen_ids: set[str] = set()
    parsed_rows: list[dict] = []
    oos_start = _oos_start(config)

    for line_no, row in enumerate(rows, start=2):
        event_id = row.get("event_id", "").strip()
        if not event_id:
            raise ValidationError(f"line {line_no}: event_id is missing")
        if event_id in seen_ids:
            raise ValidationError(f"line {line_no}: duplicate event_id {event_id!r}")
        seen_ids.add(event_id)

        ts = _parse_utc_timestamp(row.get("timestamp_utc", ""))
        if ts >= oos_start and not unlock_oos:
            raise ValidationError(
                f"line {line_no}: {event_id} is inside locked OOS beginning "
                f"{config['final_oos']['from']}; rerun only during the preregistered "
                "OOS evaluation with --unlock-oos"
            )

        family = row.get("event_family", "").strip().upper()
        if family not in allowed or family not in DIRECTIONALITY:
            raise ValidationError(
                f"line {line_no}: unsupported event_family {family!r}"
            )

        event_name = row.get("event_name", "").strip()
        if not event_name:
            raise ValidationError(f"line {line_no}: event_name is missing")

        actual = _parse_number(row.get("actual", ""), "actual")
        consensus = _parse_number(row.get("consensus", ""), "consensus")
        previous_text = row.get("previous", "").strip()
        previous = (
            _parse_number(previous_text, "previous") if previous_text else None
        )

        unit = row.get("unit", "").strip()
        if not unit:
            raise ValidationError(f"line {line_no}: unit is missing")

        source = row.get("source", "").strip()
        if not source:
            raise ValidationError(f"line {line_no}: source is missing")

        directionality = DIRECTIONALITY[family]
        surprise_raw = directionality * (actual - consensus)

        parsed_rows.append(
            {
                "event_id": event_id,
                "timestamp": ts,
                "event_family": family,
                "event_name": event_name,
                "actual": actual,
                "consensus": consensus,
                "previous": previous,
                "unit": unit,
                "directionality": directionality,
                "surprise_raw": surprise_raw,
                "source": source,
                "_line_no": line_no,
            }
        )

    parsed_rows.sort(key=lambda item: (item["timestamp"], item["event_id"]))

    seen_release_keys: set[tuple[str, datetime]] = set()
    history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
    output: list[dict[str, str]] = []

    for item in parsed_rows:
        key = (item["event_family"], item["timestamp"])
        if key in seen_release_keys:
            raise ValidationError(
                f"duplicate release for {item['event_family']} at "
                f"{_format_timestamp(item['timestamp'])}"
            )
        seen_release_keys.add(key)

        prior = history[item["event_family"]]
        surprise_std: float | None = None
        if len(prior) >= min_history:
            sigma = statistics.stdev(prior)
            if sigma > 0.0 and math.isfinite(sigma):
                surprise_std = item["surprise_raw"] / sigma

        output.append(
            {
                "event_id": item["event_id"],
                "timestamp_utc": _format_timestamp(item["timestamp"]),
                "event_family": item["event_family"],
                "event_name": item["event_name"],
                "actual": _format_number(item["actual"]),
                "consensus": _format_number(item["consensus"]),
                "previous": _format_number(item["previous"]),
                "unit": item["unit"],
                "directionality": str(item["directionality"]),
                "surprise_raw": _format_number(item["surprise_raw"]),
                "surprise_std": _format_number(surprise_std),
                "source": item["source"],
            }
        )

        # Causality: current release enters history only after its own score is computed.
        prior.append(item["surprise_raw"])

    return output


def load_raw_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError("input CSV has no header")
        missing = [
            name for name in RAW_REQUIRED_COLUMNS if name not in reader.fieldnames
        ]
        if missing:
            raise ValidationError(
                f"input CSV missing required columns: {missing}"
            )
        return list(reader)


def write_canonical_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CANONICAL_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate raw H16 US macro releases and produce the canonical "
            "causal surprise dataset."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/macro/h16_us_macro_events.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/h16_macro_surprise_v1_frozen.json"),
    )
    parser.add_argument(
        "--unlock-oos",
        action="store_true",
        help=(
            "Allow rows on/after the frozen OOS start. Use only for the "
            "preregistered OOS evaluation."
        ),
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    rows = load_raw_csv(args.input)
    canonical = canonicalize_rows(
        rows,
        config,
        unlock_oos=args.unlock_oos,
    )
    write_canonical_csv(args.output, canonical)

    eligible = sum(1 for row in canonical if row["surprise_std"])
    print(
        json.dumps(
            {
                "rows": len(canonical),
                "standardized_rows": eligible,
                "output": str(args.output),
                "oos_unlocked": bool(args.unlock_oos),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
