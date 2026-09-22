from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


_SIM_TIME_RE = re.compile(r"(20\d{2}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})")
_MARKET_RE = re.compile(
    r"market\s+(buy|sell)\s+([0-9.]+)\s+XAUUSD\.r\s+sl:\s*([0-9.]+)\s+tp:\s*([0-9.]+)\s+\(([0-9.]+)\s*/\s*([0-9.]+)\)",
    re.IGNORECASE,
)
_EXACT_RE = re.compile(
    r"SMA_RSI_EXACT_(BUY|SELL)\s*\|\s*volume=([0-9.]+)\s*\|\s*ATR=([0-9.]+)\s*\|\s*SL=([0-9.]+)\s*\|\s*TP=([0-9.]+)",
    re.IGNORECASE,
)

ATR_MULTIPLIER = 1.5
PRICE_DIGITS = 2


def _parse_mt5_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y.%m.%d %H:%M:%S")


def _parse_python_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _m5_bucket(value: datetime) -> datetime:
    return value.replace(minute=(value.minute // 5) * 5, second=0, microsecond=0)


def _round_digits(value: float, digits: int = PRICE_DIGITS) -> float:
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


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


def _extract_mt5_construction(path: Path) -> list[dict]:
    lines = _extract_v21_run(_read_log_lines(path))
    quotes: dict[tuple[datetime, str], dict] = {}
    trades: list[dict] = []

    for line in lines:
        tm = _SIM_TIME_RE.search(line)
        if tm is None:
            continue
        when = _parse_mt5_time(tm.group(1))

        market = _MARKET_RE.search(line)
        if market is not None:
            direction = market.group(1).upper()
            quotes[(when, direction)] = {
                "requested_volume": float(market.group(2)),
                "market_sl": float(market.group(3)),
                "market_tp": float(market.group(4)),
                "bid": float(market.group(5)),
                "ask": float(market.group(6)),
                "market_log": line.strip(),
            }
            continue

        exact = _EXACT_RE.search(line)
        if exact is None:
            continue

        direction = exact.group(1).upper()
        quote = quotes.get((when, direction))
        trades.append(
            {
                "time": when,
                "bucket": _m5_bucket(when),
                "direction": direction,
                "volume": float(exact.group(2)),
                "logged_atr": float(exact.group(3)),
                "logged_sl": float(exact.group(4)),
                "logged_tp": float(exact.group(5)),
                "quote": quote,
                "exact_log": line.strip(),
            }
        )

    return trades


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen V2.1 ATR/SL/TP trade construction against the MT5 tester log."
    )
    parser.add_argument("--python-signals", type=Path, required=True)
    parser.add_argument("--mt5-log", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/TRADE_CONSTRUCTION_PARITY.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/TRADE_CONSTRUCTION_SUMMARY.json"),
    )
    args = parser.parse_args()

    with args.python_signals.open("r", encoding="utf-8", newline="") as handle:
        python_rows = list(csv.DictReader(handle))

    python_by_key: dict[tuple[datetime, str], dict] = {}
    for row in python_rows:
        when = _parse_python_time(row["entry_utc"])
        key = (_m5_bucket(when), row["direction"].upper())
        python_by_key[key] = row

    mt5_trades = _extract_mt5_construction(args.mt5_log)
    rows: list[dict[str, str | int | float | bool]] = []

    for trade in mt5_trades:
        key = (trade["bucket"], trade["direction"])
        python_row = python_by_key.get(key)
        quote = trade["quote"]

        out: dict[str, str | int | float | bool] = {
            "mt5_time": trade["time"].strftime("%Y-%m-%d %H:%M:%S"),
            "m5_bucket": trade["bucket"].strftime("%Y-%m-%d %H:%M:%S"),
            "direction": trade["direction"],
            "volume": trade["volume"],
            "python_signal_found": python_row is not None,
            "quote_found": quote is not None,
            "logged_atr": trade["logged_atr"],
            "logged_sl": trade["logged_sl"],
            "logged_tp": trade["logged_tp"],
        }

        if python_row is None:
            out.update(
                {
                    "python_atr": "",
                    "python_atr_rounded": "",
                    "atr_match": False,
                    "entry_reference": "",
                    "expected_sl": "",
                    "expected_tp": "",
                    "sl_match": False,
                    "tp_match": False,
                }
            )
            rows.append(out)
            continue

        python_atr = float(python_row["m5_atr14"])
        python_atr_rounded = _round_digits(python_atr)
        atr_match = python_atr_rounded == trade["logged_atr"]

        out["python_atr"] = python_atr
        out["python_atr_rounded"] = python_atr_rounded
        out["atr_match"] = atr_match

        if quote is None:
            out.update(
                {
                    "entry_reference": "",
                    "expected_sl": "",
                    "expected_tp": "",
                    "sl_match": False,
                    "tp_match": False,
                }
            )
            rows.append(out)
            continue

        entry = quote["ask"] if trade["direction"] == "BUY" else quote["bid"]
        distance = python_atr * ATR_MULTIPLIER
        if trade["direction"] == "BUY":
            expected_sl = _round_digits(entry - distance)
            expected_tp = _round_digits(entry + distance)
        else:
            expected_sl = _round_digits(entry + distance)
            expected_tp = _round_digits(entry - distance)

        out["entry_reference"] = entry
        out["bid"] = quote["bid"]
        out["ask"] = quote["ask"]
        out["expected_sl"] = expected_sl
        out["expected_tp"] = expected_tp
        out["sl_match"] = expected_sl == trade["logged_sl"]
        out["tp_match"] = expected_tp == trade["logged_tp"]
        out["market_log_sl_match"] = quote["market_sl"] == trade["logged_sl"]
        out["market_log_tp_match"] = quote["market_tp"] == trade["logged_tp"]
        rows.append(out)

    executed = len(mt5_trades)
    python_missing = sum(not bool(row["python_signal_found"]) for row in rows)
    quote_missing = sum(not bool(row["quote_found"]) for row in rows)
    atr_mismatch = sum(not bool(row["atr_match"]) for row in rows)
    sl_mismatch = sum(not bool(row["sl_match"]) for row in rows)
    tp_mismatch = sum(not bool(row["tp_match"]) for row in rows)

    summary = {
        "mt5_executed_trade_constructions": executed,
        "python_signal_missing": python_missing,
        "mt5_quote_missing": quote_missing,
        "atr_mismatches": atr_mismatch,
        "sl_mismatches_without_stop_distance_adjustment": sl_mismatch,
        "tp_mismatches_without_stop_distance_adjustment": tp_mismatch,
        "construction_parity_pass": (
            executed > 0
            and python_missing == 0
            and quote_missing == 0
            and atr_mismatch == 0
            and sl_mismatch == 0
            and tp_mismatch == 0
        ),
        "scope": (
            "Validates M5 ATR14 and 1.5x ATR SL/TP construction against MT5 bid/ask quotes. "
            "Volume is reported but not independently validated here because V2.1 sizes from current equity "
            "through MT5 OrderCalcProfit. Exit/P&L parity is not claimed."
        ),
        "mismatch_examples": [
            {
                "mt5_time": row["mt5_time"],
                "direction": row["direction"],
                "atr_match": row["atr_match"],
                "sl_match": row["sl_match"],
                "tp_match": row["tp_match"],
            }
            for row in rows
            if not (
                bool(row["python_signal_found"])
                and bool(row["quote_found"])
                and bool(row["atr_match"])
                and bool(row["sl_match"])
                and bool(row["tp_match"])
            )
        ][:20],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"construction_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
