from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RISK_PERCENT = 0.20
CONTRACT_SIZE = 100.0
VOLUME_MIN = 0.01
VOLUME_STEP = 0.01
VOLUME_MAX = 100.0
ATR_MULTIPLIER = 1.5

_SIM_TIME_RE = re.compile(r"(20\d{2}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})")
_EXACT_RE = re.compile(
    r"SMA_RSI_EXACT_(BUY|SELL)\s*\|\s*volume=([0-9.]+)\s*\|\s*ATR=([0-9.]+)\s*\|\s*SL=([0-9.]+)\s*\|\s*TP=([0-9.]+)",
    re.IGNORECASE,
)
_VOLUME_SKIP = "Trade skipped: invalid/below-minimum volume."


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
            values[_column_index(cell.attrib.get("r", ""))] = _cell_value(cell, shared)
        rows.append(values)
    return rows


def _parse_mt5_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y.%m.%d %H:%M:%S")


def _parse_python_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _m5_bucket(value: datetime) -> datetime:
    return value.replace(minute=(value.minute // 5) * 5, second=0, microsecond=0)


def _to_float(value: str) -> float:
    cleaned = str(value).replace(" ", "").replace(",", "")
    return float(cleaned)


def _read_log_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp1252"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text[:1000]:
            return text.splitlines()
    return raw.decode("utf-8", errors="replace").splitlines()


def _extract_v21_run(lines: list[str]) -> list[str]:
    starts = [
        i
        for i, line in enumerate(lines)
        if "testing of Experts\\V2_1_HTFTrend.ex5" in line and "started with inputs:" in line
    ]
    if not starts:
        starts = [i for i, line in enumerate(lines) if '"V2_1_HTFTrend.ex5" AVX2' in line]
    if not starts:
        raise ValueError("could not locate V2_1_HTFTrend test start in MT5 log")

    start = starts[-1]
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if "Test passed in" in lines[i]:
            end = i + 1
            break
        if "final balance" in lines[i]:
            end = i + 1
    return lines[start:end]


def _extract_log_events(path: Path) -> tuple[list[dict], list[dict]]:
    executed: list[dict] = []
    skipped: list[dict] = []

    for line in _extract_v21_run(_read_log_lines(path)):
        tm = _SIM_TIME_RE.search(line)
        if tm is None:
            continue
        when = _parse_mt5_time(tm.group(1))

        exact = _EXACT_RE.search(line)
        if exact is not None:
            executed.append(
                {
                    "time": when,
                    "bucket": _m5_bucket(when),
                    "direction": exact.group(1).upper(),
                    "volume": float(exact.group(2)),
                    "sl": float(exact.group(4)),
                    "tp": float(exact.group(5)),
                }
            )
            continue

        if _VOLUME_SKIP in line:
            skipped.append({"time": when, "bucket": _m5_bucket(when)})

    return executed, skipped


def _extract_deals(path: Path, initial_balance: float) -> tuple[list[dict], list[dict], float]:
    rows = _xlsx_rows(path)
    deals_start = None
    for i, row in enumerate(rows):
        if row.get(0, "") == "Deals":
            deals_start = i
            break
    if deals_start is None:
        raise ValueError("Deals section not found in MT5 report")

    header_index = None
    headers: dict[str, int] = {}
    for i in range(deals_start + 1, min(len(rows), deals_start + 8)):
        normalized = {str(value).strip().casefold(): col for col, value in rows[i].items()}
        if "time" in normalized and "direction" in normalized and "balance" in normalized:
            header_index = i
            headers = normalized
            break
    if header_index is None:
        raise ValueError("Deals header row not found or missing Time/Direction/Balance")

    required = ("time", "symbol", "type", "direction", "volume", "balance")
    missing = [name for name in required if name not in headers]
    if missing:
        raise ValueError(f"Deals section missing required columns: {missing}")

    current_balance = float(initial_balance)
    entry_deals: list[dict] = []
    balance_timeline: list[dict] = []

    for row in rows[header_index + 1 :]:
        first = row.get(headers["time"], "")
        if not first:
            continue
        try:
            when = _parse_mt5_time(first)
        except ValueError:
            continue

        direction_field = row.get(headers["direction"], "").strip().casefold()
        symbol = row.get(headers["symbol"], "").strip()
        deal_type = row.get(headers["type"], "").strip().casefold()
        volume_text = row.get(headers["volume"], "")
        balance_text = row.get(headers["balance"], "")

        balance_before = current_balance
        if balance_text:
            try:
                current_balance = _to_float(balance_text)
            except ValueError:
                pass

        balance_timeline.append({"time": when, "balance": current_balance})

        if (
            symbol == "XAUUSD.r"
            and direction_field == "in"
            and deal_type in ("buy", "sell")
        ):
            entry_deals.append(
                {
                    "time": when,
                    "bucket": _m5_bucket(when),
                    "direction": deal_type.upper(),
                    "volume": _to_float(volume_text),
                    "balance_before": balance_before,
                    "balance_after": current_balance,
                }
            )

    return entry_deals, balance_timeline, current_balance


def _floor_volume(raw_volume: float) -> float:
    normalized = math.floor(raw_volume / VOLUME_STEP + 1e-10) * VOLUME_STEP
    if normalized < VOLUME_MIN:
        return 0.0
    normalized = min(normalized, VOLUME_MAX)
    return round(normalized, 8)


def _expected_volume(equity: float, stop_distance: float) -> tuple[float, float, float]:
    risk_money = equity * RISK_PERCENT / 100.0
    one_lot_loss = stop_distance * CONTRACT_SIZE
    if equity <= 0.0 or risk_money <= 0.0 or one_lot_loss <= 0.0:
        return 0.0, risk_money, one_lot_loss
    raw = risk_money / one_lot_loss
    return _floor_volume(raw), risk_money, one_lot_loss


def _balance_at(time: datetime, timeline: list[dict], initial_balance: float) -> float:
    balance = float(initial_balance)
    for item in timeline:
        if item["time"] > time:
            break
        balance = float(item["balance"])
    return balance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen V2.1 current-equity risk sizing and broker volume normalization."
    )
    parser.add_argument("--python-signals", type=Path, required=True)
    parser.add_argument("--mt5-report", type=Path, required=True)
    parser.add_argument("--mt5-log", type=Path, required=True)
    parser.add_argument("--initial-balance", type=float, default=5000.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/VOLUME_PARITY.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/VOLUME_PARITY_SUMMARY.json"),
    )
    args = parser.parse_args()

    with args.python_signals.open("r", encoding="utf-8", newline="") as handle:
        python_rows = list(csv.DictReader(handle))
    python_by_bucket = {
        _m5_bucket(_parse_python_time(row["entry_utc"])): row
        for row in python_rows
    }

    executed, skipped = _extract_log_events(args.mt5_log)
    entry_deals, balance_timeline, final_balance = _extract_deals(
        args.mt5_report, args.initial_balance
    )

    if len(entry_deals) != len(executed):
        raise ValueError(
            f"executed entry count mismatch: log={len(executed)} report_deals={len(entry_deals)}"
        )

    rows: list[dict] = []
    executed_mismatches = 0
    entry_sequence_mismatches = 0

    for log_trade, deal in zip(executed, entry_deals):
        same_sequence = (
            log_trade["bucket"] == deal["bucket"]
            and log_trade["direction"] == deal["direction"]
        )
        if not same_sequence:
            entry_sequence_mismatches += 1

        stop_distance = abs(log_trade["tp"] - log_trade["sl"]) / 2.0
        expected, risk_money, one_lot_loss = _expected_volume(
            float(deal["balance_before"]), stop_distance
        )
        volume_match = abs(expected - float(log_trade["volume"])) < 1e-9
        if not volume_match:
            executed_mismatches += 1

        rows.append(
            {
                "event": "EXECUTED",
                "time": log_trade["time"].strftime("%Y-%m-%d %H:%M:%S"),
                "direction": log_trade["direction"],
                "equity_reference": deal["balance_before"],
                "stop_distance": stop_distance,
                "risk_money": risk_money,
                "one_lot_loss": one_lot_loss,
                "expected_volume": expected,
                "logged_volume": log_trade["volume"],
                "volume_match": volume_match,
                "sequence_match": same_sequence,
            }
        )

    skip_mismatches = 0
    skip_signal_missing = 0
    for event in skipped:
        python_row = python_by_bucket.get(event["bucket"])
        if python_row is None:
            skip_signal_missing += 1
            rows.append(
                {
                    "event": "VOLUME_SKIP",
                    "time": event["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "direction": "",
                    "equity_reference": "",
                    "stop_distance": "",
                    "risk_money": "",
                    "one_lot_loss": "",
                    "expected_volume": "",
                    "logged_volume": 0.0,
                    "volume_match": False,
                    "sequence_match": False,
                }
            )
            continue

        equity = _balance_at(event["time"], balance_timeline, args.initial_balance)
        atr = float(python_row["m5_atr14"])
        stop_distance = ATR_MULTIPLIER * atr
        expected, risk_money, one_lot_loss = _expected_volume(equity, stop_distance)
        valid_skip = expected == 0.0
        if not valid_skip:
            skip_mismatches += 1

        rows.append(
            {
                "event": "VOLUME_SKIP",
                "time": event["time"].strftime("%Y-%m-%d %H:%M:%S"),
                "direction": python_row["direction"].upper(),
                "equity_reference": equity,
                "stop_distance": stop_distance,
                "risk_money": risk_money,
                "one_lot_loss": one_lot_loss,
                "expected_volume": expected,
                "logged_volume": 0.0,
                "volume_match": valid_skip,
                "sequence_match": True,
            }
        )

    summary = {
        "mt5_executed_entries": len(executed),
        "report_entry_deals": len(entry_deals),
        "entry_sequence_mismatches": entry_sequence_mismatches,
        "executed_volume_mismatches": executed_mismatches,
        "mt5_volume_skip_events": len(skipped),
        "volume_skip_signal_missing": skip_signal_missing,
        "volume_skip_rule_mismatches": skip_mismatches,
        "initial_balance": args.initial_balance,
        "report_final_balance": final_balance,
        "volume_parity_pass": (
            len(executed) > 0
            and entry_sequence_mismatches == 0
            and executed_mismatches == 0
            and len(skipped) > 0
            and skip_signal_missing == 0
            and skip_mismatches == 0
        ),
        "scope": (
            "Reproduces V2.1 sizing as current realized balance * 0.20% divided by "
            "one-lot SL loss, then floors to FXIFY XAUUSD.r volume_step=0.01 and rejects "
            "values below volume_min=0.01. Because the EA is one-position-only, account "
            "equity at each flat-state entry equals realized balance. The one-lot loss uses "
            "the validated frozen SL distance and contract_size=100 USD per price unit per lot."
        ),
        "mismatch_examples": [
            row for row in rows if not bool(row["volume_match"]) or not bool(row["sequence_match"])
        ][:20],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"volume_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
