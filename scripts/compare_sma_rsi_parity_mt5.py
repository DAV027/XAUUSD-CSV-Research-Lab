from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_index(reference: str) -> int:
    letters = ""
    for ch in reference:
        if ch.isalpha():
            letters += ch
        else:
            break
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def _shared_strings(book: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in book.namelist():
        return []
    root = ET.fromstring(book.read(name))
    values: list[str] = []
    for si in root.findall("x:si", NS):
        parts = [node.text or "" for node in si.findall(".//x:t", NS)]
        values.append("".join(parts))
    return values


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//x:t", NS))
    value = cell.find("x:v", NS)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared[int(value.text)]
    return value.text


def _xlsx_rows(path: Path) -> list[dict[int, str]]:
    with zipfile.ZipFile(path) as book:
        shared = _shared_strings(book)
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in book.namelist():
            raise ValueError("expected first worksheet at xl/worksheets/sheet1.xml")
        root = ET.fromstring(book.read(sheet_name))

    rows: list[dict[int, str]] = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        values: dict[int, str] = {}
        for cell in row.findall("x:c", NS):
            ref = cell.attrib.get("r", "")
            values[_column_index(ref)] = _cell_value(cell, shared)
        rows.append(values)
    return rows


def _parse_mt5_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y.%m.%d %H:%M:%S")


def _m5_bucket(value: datetime) -> datetime:
    return value.replace(minute=(value.minute // 5) * 5, second=0, microsecond=0)


def _extract_mt5(path: Path):
    rows = _xlsx_rows(path)
    in_orders = False
    entries: list[dict] = []
    intervals: list[dict] = []
    current: dict | None = None

    for row in rows:
        first = row.get(0, "")
        if first == "Orders":
            in_orders = True
            continue
        if first == "Deals" and in_orders:
            break
        if not in_orders or first in ("", "Open Time"):
            continue

        comment = row.get(12, "")
        direction = row.get(3, "").lower()
        if not first or not comment:
            continue

        try:
            when = _parse_mt5_time(first)
        except ValueError:
            continue

        if comment in ("SMA_RSI_EXACT_BUY", "SMA_RSI_EXACT_SELL"):
            entry_direction = "BUY" if comment.endswith("BUY") else "SELL"
            entry = {
                "time": when,
                "bucket": _m5_bucket(when),
                "direction": entry_direction,
                "comment": comment,
            }
            entries.append(entry)
            if current is not None:
                current["exit_time"] = when
                current["exit_reason"] = "implicit_next_entry"
                intervals.append(current)
            current = {
                "entry_time": when,
                "entry_bucket": _m5_bucket(when),
                "direction": entry_direction,
            }
        elif current is not None and (
            comment.lower().startswith("sl ")
            or comment.lower().startswith("tp ")
            or comment.lower() == "end of test"
        ):
            current["exit_time"] = when
            current["exit_reason"] = comment
            intervals.append(current)
            current = None

    if current is not None:
        current["exit_time"] = datetime.max.replace(microsecond=0)
        current["exit_reason"] = "unclosed"
        intervals.append(current)

    return entries, intervals


def _parse_python_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


_GUARD_PATTERNS = (
    ("SKIPPED_SPREAD_GUARD", "No trade: spread="),
    ("SKIPPED_INDICATOR_DATA", "Indicator data unavailable."),
    ("SKIPPED_HTF_DATA", "HTF completed-bar data unavailable"),
    ("SKIPPED_VOLUME_GUARD", "Trade skipped: invalid/below-minimum volume."),
    ("SKIPPED_RISK_CALC", "OrderCalcProfit failed"),
    ("SKIPPED_ORDER_FAILED", "Order failed |"),
)
_SIM_TIME_RE = re.compile(r"(20\\d{2}\\.\\d{2}\\.\\d{2} \\d{2}:\\d{2}:\\d{2})")


def _read_log_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp1252"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\\x00" not in text[:1000]:
            return text.splitlines()
    return raw.decode("utf-8", errors="replace").splitlines()


def _extract_v21_run(lines: list[str]) -> list[str]:
    start_candidates = [
        i
        for i, line in enumerate(lines)
        if "testing of Experts\\V2_1_HTFTrend.ex5" in line and "started with inputs:" in line
    ]
    if not start_candidates:
        start_candidates = [
            i for i, line in enumerate(lines) if '"V2_1_HTFTrend.ex5" AVX2' in line
        ]
    if not start_candidates:
        raise ValueError("could not locate V2_1_HTFTrend test start in MT5 log")

    start = start_candidates[-1]
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if "Test passed in" in line:
            end = i + 1
            break
        if "final balance" in line:
            end = i + 1
    return lines[start:end]


def _extract_guard_buckets(path: Path) -> tuple[dict[datetime, tuple[str, str]], dict[str, int]]:
    lines = _extract_v21_run(_read_log_lines(path))
    by_bucket: dict[datetime, tuple[str, str]] = {}
    totals: dict[str, int] = {}

    for line in lines:
        matched_kind = None
        for kind, needle in _GUARD_PATTERNS:
            if needle in line:
                matched_kind = kind
                break
        if matched_kind is None:
            continue

        time_match = _SIM_TIME_RE.search(line)
        if time_match is None:
            continue
        when = _parse_mt5_time(time_match.group(1))
        bucket = _m5_bucket(when)
        totals[matched_kind] = totals.get(matched_kind, 0) + 1
        by_bucket.setdefault(bucket, (matched_kind, line.strip()))

    return by_bucket, totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SMA/RSI Python raw signals with an MT5 Strategy Tester report."
    )
    parser.add_argument("--python-signals", type=Path, required=True)
    parser.add_argument("--mt5-report", type=Path, required=True)
    parser.add_argument(
        "--mt5-log",
        type=Path,
        help="Optional MT5 tester/agent log used to explain flat-state Python signals by execution guard.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/sma_rsi_htf_v21/PARITY_CLASSIFICATION.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/sma_rsi_htf_v21/PARITY_SUMMARY.json"))
    parser.add_argument(
        "--python-offset-minutes",
        type=int,
        default=0,
        help="Optional wall-clock offset applied to Python entry_utc before matching MT5 report times.",
    )
    args = parser.parse_args()

    mt5_entries, intervals = _extract_mt5(args.mt5_report)
    mt5_keys = {(item["bucket"], item["direction"]) for item in mt5_entries}
    guard_buckets: dict[datetime, tuple[str, str]] = {}
    guard_totals: dict[str, int] = {}
    if args.mt5_log is not None:
        guard_buckets, guard_totals = _extract_guard_buckets(args.mt5_log)

    with args.python_signals.open("r", encoding="utf-8", newline="") as handle:
        python_rows = list(csv.DictReader(handle))

    classified: list[dict[str, str]] = []
    python_keys: set[tuple[datetime, str]] = set()

    for row in python_rows:
        direction = row["direction"].upper()
        when = _parse_python_time(row["entry_utc"]) + timedelta(minutes=args.python_offset_minutes)
        bucket = _m5_bucket(when)
        key = (bucket, direction)
        python_keys.add(key)

        if key in mt5_keys:
            classification = "EXECUTED_MATCH"
        else:
            open_interval = next(
                (
                    item
                    for item in intervals
                    if item["entry_time"] <= when < item["exit_time"]
                ),
                None,
            )
            if open_interval is not None:
                classification = "SKIPPED_POSITION_OPEN"
            else:
                guard = guard_buckets.get(bucket)
                if guard is not None:
                    classification = guard[0]
                else:
                    classification = "PYTHON_ONLY_WHILE_MT5_FLAT"

        output = dict(row)
        output["comparison_time"] = when.strftime("%Y-%m-%d %H:%M:%S")
        output["comparison_m5_bucket"] = bucket.strftime("%Y-%m-%d %H:%M:%S")
        output["classification"] = classification
        guard = guard_buckets.get(bucket)
        output["mt5_guard_log"] = guard[1] if guard is not None else ""
        classified.append(output)

    mt5_missing = [
        item
        for item in mt5_entries
        if (item["bucket"], item["direction"]) not in python_keys
    ]

    counts: dict[str, int] = {}
    for row in classified:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1

    summary = {
        "python_raw_signals": len(python_rows),
        "mt5_executed_entries": len(mt5_entries),
        "python_offset_minutes": args.python_offset_minutes,
        "executed_matches": counts.get("EXECUTED_MATCH", 0),
        "python_skipped_while_mt5_position_open": counts.get("SKIPPED_POSITION_OPEN", 0),
        "python_skipped_spread_guard": counts.get("SKIPPED_SPREAD_GUARD", 0),
        "python_skipped_indicator_data": counts.get("SKIPPED_INDICATOR_DATA", 0),
        "python_skipped_htf_data": counts.get("SKIPPED_HTF_DATA", 0),
        "python_skipped_volume_guard": counts.get("SKIPPED_VOLUME_GUARD", 0),
        "python_skipped_risk_calc": counts.get("SKIPPED_RISK_CALC", 0),
        "python_skipped_order_failed": counts.get("SKIPPED_ORDER_FAILED", 0),
        "python_only_while_mt5_flat": counts.get("PYTHON_ONLY_WHILE_MT5_FLAT", 0),
        "mt5_entries_missing_python_signal": len(mt5_missing),
        "mt5_log_guard_totals": guard_totals,
        "parity_pass": (
            counts.get("PYTHON_ONLY_WHILE_MT5_FLAT", 0) == 0
            and len(mt5_missing) == 0
        ),
        "notes": [
            "Matching is by M5 bucket and direction, so MT5 first-tick seconds do not create false mismatches.",
            "Python-only signals while an MT5 position was open are expected from InpOnePositionOnly behavior.",
            "When --mt5-log is supplied, flat-state Python signals are matched by M5 bucket to MT5 spread/data/risk/order guard messages.",
            "Only PYTHON_ONLY_WHILE_MT5_FLAT remains unexplained after tester-log guard matching.",
        ],
        "mt5_missing_examples": [
            {
                "time": item["time"].strftime("%Y-%m-%d %H:%M:%S"),
                "bucket": item["bucket"].strftime("%Y-%m-%d %H:%M:%S"),
                "direction": item["direction"],
            }
            for item in mt5_missing[:20]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if classified:
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(classified[0].keys()))
            writer.writeheader()
            writer.writerows(classified)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"classification_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
