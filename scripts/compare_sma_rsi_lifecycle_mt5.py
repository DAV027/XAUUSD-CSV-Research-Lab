from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

_SIM_TIME_RE = re.compile(r"(20\d{2}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})")
_ENTRY_RE = re.compile(
    r"SMA_RSI_EXACT_(BUY|SELL)\s*\|\s*volume=([0-9.]+)\s*\|\s*ATR=([0-9.]+)\s*\|\s*SL=([0-9.]+)\s*\|\s*TP=([0-9.]+)",
    re.IGNORECASE,
)


@dataclass
class LogTrade:
    entry_time: datetime
    direction: str
    volume: float
    sl: float
    tp: float
    exit_time: datetime | None = None
    exit_reason: str | None = None


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


def _to_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return 0.0
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


def _extract_log_lifecycle(path: Path) -> list[LogTrade]:
    trades: list[LogTrade] = []
    current: LogTrade | None = None

    for line in _extract_v21_run(_read_log_lines(path)):
        tm = _SIM_TIME_RE.search(line)
        if tm is None:
            continue
        when = _parse_mt5_time(tm.group(1))

        entry = _ENTRY_RE.search(line)
        if entry is not None:
            if current is not None:
                raise ValueError(
                    f"new entry at {when} before previous log trade exited from {current.entry_time}"
                )
            current = LogTrade(
                entry_time=when,
                direction=entry.group(1).upper(),
                volume=float(entry.group(2)),
                sl=float(entry.group(4)),
                tp=float(entry.group(5)),
            )
            continue

        reason = None
        lower = line.casefold()
        if "stop loss triggered" in lower:
            reason = "SL"
        elif "take profit triggered" in lower:
            reason = "TP"
        elif "position closed due end of test" in lower:
            reason = "END_OF_TEST"

        if reason is not None and current is not None:
            current.exit_time = when
            current.exit_reason = reason
            trades.append(current)
            current = None

    if current is not None:
        raise ValueError(f"unclosed log trade from {current.entry_time}")

    return trades


def _find_deals_header(rows: list[dict[int, str]]) -> tuple[int, dict[str, int]]:
    deals_start = None
    for i, row in enumerate(rows):
        if row.get(0, "") == "Deals":
            deals_start = i
            break
    if deals_start is None:
        raise ValueError("Deals section not found")

    for i in range(deals_start + 1, min(len(rows), deals_start + 10)):
        normalized = {str(value).strip().casefold(): col for col, value in rows[i].items()}
        if "time" in normalized and "direction" in normalized and "balance" in normalized:
            return i, normalized
    raise ValueError("Deals header not found")


def _extract_report_lifecycle(path: Path) -> tuple[list[dict], float]:
    rows = _xlsx_rows(path)
    header_index, headers = _find_deals_header(rows)

    required = (
        "time",
        "symbol",
        "type",
        "direction",
        "volume",
        "price",
        "commission",
        "swap",
        "profit",
        "balance",
    )
    missing = [name for name in required if name not in headers]
    if missing:
        raise ValueError(f"Deals section missing required columns: {missing}")

    trades: list[dict] = []
    current: dict | None = None
    final_balance = 0.0

    for row in rows[header_index + 1 :]:
        raw_time = row.get(headers["time"], "")
        if not raw_time:
            continue
        try:
            when = _parse_mt5_time(raw_time)
        except ValueError:
            continue

        symbol = row.get(headers["symbol"], "").strip()
        if symbol != "XAUUSD.r":
            continue

        deal_type = row.get(headers["type"], "").strip().upper()
        direction_field = row.get(headers["direction"], "").strip().casefold()
        volume = _to_float(row.get(headers["volume"], ""))
        price = _to_float(row.get(headers["price"], ""))
        commission = _to_float(row.get(headers["commission"], ""))
        swap = _to_float(row.get(headers["swap"], ""))
        profit = _to_float(row.get(headers["profit"], ""))
        balance = _to_float(row.get(headers["balance"], ""))
        if balance:
            final_balance = balance

        comment = ""
        comment_col = headers.get("comment")
        if comment_col is not None:
            comment = row.get(comment_col, "").strip()

        if direction_field == "in" and deal_type in ("BUY", "SELL"):
            if current is not None:
                raise ValueError(
                    f"new report entry at {when} before previous trade exited from {current['entry_time']}"
                )
            current = {
                "entry_time": when,
                "entry_direction": deal_type,
                "entry_volume": volume,
                "entry_price": price,
                "entry_commission": commission,
                "entry_swap": swap,
                "entry_profit": profit,
                "entry_balance": balance,
                "entry_comment": comment,
            }
            continue

        if direction_field == "out" and deal_type in ("BUY", "SELL") and current is not None:
            current.update(
                {
                    "exit_time": when,
                    "exit_deal_type": deal_type,
                    "exit_volume": volume,
                    "exit_price": price,
                    "exit_commission": commission,
                    "exit_swap": swap,
                    "exit_profit": profit,
                    "exit_balance": balance,
                    "exit_comment": comment,
                    "reported_net_delta": (
                        current["entry_commission"]
                        + current["entry_swap"]
                        + current["entry_profit"]
                        + commission
                        + swap
                        + profit
                    ),
                }
            )
            trades.append(current)
            current = None

    if current is not None:
        raise ValueError(f"unclosed report trade from {current['entry_time']}")

    return trades, final_balance


def _classify_report_exit(comment: str) -> str:
    text = comment.strip().casefold()
    if text.startswith("sl "):
        return "SL"
    if text.startswith("tp "):
        return "TP"
    if text == "end of test":
        return "END_OF_TEST"
    return "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen V2.1 MT5 trade lifecycle and report P/L accounting."
    )
    parser.add_argument("--mt5-report", type=Path, required=True)
    parser.add_argument("--mt5-log", type=Path, required=True)
    parser.add_argument("--initial-balance", type=float, default=5000.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/LIFECYCLE_PARITY.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/LIFECYCLE_PARITY_SUMMARY.json"),
    )
    args = parser.parse_args()

    log_trades = _extract_log_lifecycle(args.mt5_log)
    report_trades, report_final_balance = _extract_report_lifecycle(args.mt5_report)

    rows: list[dict] = []
    count_mismatch = len(log_trades) != len(report_trades)
    sequence_mismatches = 0
    direction_mismatches = 0
    volume_mismatches = 0
    exit_reason_mismatches = 0
    balance_step_mismatches = 0

    running_balance = float(args.initial_balance)

    for i, (log_trade, report_trade) in enumerate(zip(log_trades, report_trades), start=1):
        sequence_match = (
            log_trade.entry_time == report_trade["entry_time"]
            and log_trade.exit_time == report_trade["exit_time"]
        )
        direction_match = log_trade.direction == report_trade["entry_direction"]
        volume_match = (
            abs(log_trade.volume - report_trade["entry_volume"]) < 1e-9
            and abs(report_trade["entry_volume"] - report_trade["exit_volume"]) < 1e-9
        )
        report_reason = _classify_report_exit(report_trade["exit_comment"])
        reason_match = log_trade.exit_reason == report_reason

        if not sequence_match:
            sequence_mismatches += 1
        if not direction_match:
            direction_mismatches += 1
        if not volume_match:
            volume_mismatches += 1
        if not reason_match:
            exit_reason_mismatches += 1

        expected_balance = running_balance + report_trade["reported_net_delta"]
        balance_match = abs(expected_balance - report_trade["exit_balance"]) <= 0.011
        if not balance_match:
            balance_step_mismatches += 1
        running_balance = report_trade["exit_balance"]

        rows.append(
            {
                "trade_index": i,
                "entry_time_log": log_trade.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "entry_time_report": report_trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time_log": log_trade.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time_report": report_trade["exit_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "direction_log": log_trade.direction,
                "direction_report": report_trade["entry_direction"],
                "volume_log": log_trade.volume,
                "volume_report": report_trade["entry_volume"],
                "exit_reason_log": log_trade.exit_reason,
                "exit_reason_report": report_reason,
                "entry_price_report": report_trade["entry_price"],
                "exit_price_report": report_trade["exit_price"],
                "commission_total": report_trade["entry_commission"] + report_trade["exit_commission"],
                "swap_total": report_trade["entry_swap"] + report_trade["exit_swap"],
                "profit_total": report_trade["entry_profit"] + report_trade["exit_profit"],
                "reported_net_delta": report_trade["reported_net_delta"],
                "exit_balance": report_trade["exit_balance"],
                "sequence_match": sequence_match,
                "direction_match": direction_match,
                "volume_match": volume_match,
                "exit_reason_match": reason_match,
                "balance_step_match": balance_match,
            }
        )

    exit_counts: dict[str, int] = {}
    for trade in log_trades:
        key = str(trade.exit_reason)
        exit_counts[key] = exit_counts.get(key, 0) + 1

    net_from_deals = sum(float(row["reported_net_delta"]) for row in rows)
    expected_final_balance = float(args.initial_balance) + net_from_deals
    final_balance_match = abs(expected_final_balance - report_final_balance) <= 0.011

    summary = {
        "log_trade_count": len(log_trades),
        "report_trade_count": len(report_trades),
        "trade_count_mismatch": count_mismatch,
        "entry_exit_sequence_mismatches": sequence_mismatches,
        "direction_mismatches": direction_mismatches,
        "volume_mismatches": volume_mismatches,
        "exit_reason_mismatches": exit_reason_mismatches,
        "balance_step_mismatches": balance_step_mismatches,
        "exit_reason_counts": exit_counts,
        "initial_balance": args.initial_balance,
        "net_from_report_deals": net_from_deals,
        "expected_final_balance": expected_final_balance,
        "report_final_balance": report_final_balance,
        "final_balance_match": final_balance_match,
        "lifecycle_accounting_parity_pass": (
            not count_mismatch
            and len(log_trades) > 0
            and sequence_mismatches == 0
            and direction_mismatches == 0
            and volume_mismatches == 0
            and exit_reason_mismatches == 0
            and balance_step_mismatches == 0
            and final_balance_match
        ),
        "scope": (
            "Golden-reference audit only: pairs the isolated V2.1 tester-log lifecycle with the native MT5 "
            "Deals section and reconciles commission/swap/profit through the final account balance. "
            "This does not yet claim that an M1 Python simulator can reproduce real-tick exit fills or 250 ms delay."
        ),
        "mismatch_examples": [
            row
            for row in rows
            if not (
                row["sequence_match"]
                and row["direction_match"]
                and row["volume_match"]
                and row["exit_reason_match"]
                and row["balance_step_match"]
            )
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
    print(f"lifecycle_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
