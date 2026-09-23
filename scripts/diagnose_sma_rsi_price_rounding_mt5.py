from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from compare_sma_rsi_trade_construction_mt5 import (
    _extract_mt5_construction,
    _m5_bucket,
    _parse_python_time,
)

ATR_MULTIPLIER = 1.5
POINT = 0.01


def _mt5_like_round(value: float, eps: float) -> float:
    # Candidate reproduction of NormalizeDouble(value, 2). The epsilon is
    # applied only to break binary floating-point ties near an exact half point.
    scaled = value * 100.0
    if scaled >= 0:
        return int(scaled + 0.5 + eps) / 100.0
    return int(scaled - 0.5 - eps) / 100.0


def _levels(entry: float, atr: float, direction: str, eps: float) -> tuple[float, float]:
    distance = atr * ATR_MULTIPLIER
    if direction == "BUY":
        return (
            _mt5_like_round(entry - distance, eps),
            _mt5_like_round(entry + distance, eps),
        )
    return (
        _mt5_like_round(entry + distance, eps),
        _mt5_like_round(entry - distance, eps),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen V2.1 exact SL/TP endpoint normalization against MT5 logs."
    )
    parser.add_argument("--python-signals", type=Path, required=True)
    parser.add_argument("--mt5-log", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/PRICE_ROUNDING_DIAGNOSTIC.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/PRICE_ROUNDING_DIAGNOSTIC.csv"),
    )
    args = parser.parse_args()

    with args.python_signals.open("r", encoding="utf-8", newline="") as handle:
        python_rows = list(csv.DictReader(handle))

    python_by_key: dict[tuple[datetime, str], dict] = {}
    for row in python_rows:
        when = _parse_python_time(row["entry_utc"])
        python_by_key[(_m5_bucket(when), row["direction"].upper())] = row

    mt5_trades = _extract_mt5_construction(args.mt5_log)
    epsilons = [0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7]
    mismatch_counts = {f"{eps:.0e}": 0 for eps in epsilons}
    rows: list[dict] = []

    for index, trade in enumerate(mt5_trades, start=1):
        py = python_by_key[(trade["bucket"], trade["direction"])]
        atr = float(py["m5_atr14"])
        logged_sl = float(trade["logged_sl"])
        logged_tp = float(trade["logged_tp"])
        midpoint = (logged_sl + logged_tp) / 2.0

        # Reconstruct the pre-send quote from the actual first-tick side when possible.
        # The midpoint is allowed to be shifted by half a point because both endpoints
        # are independently normalized.
        quote = trade["quote"]
        if quote is None:
            continue
        entry = float(quote["ask"] if trade["direction"] == "BUY" else quote["bid"])

        candidate_results = {}
        for eps in epsilons:
            sl, tp = _levels(entry, atr, trade["direction"], eps)
            match = abs(sl - logged_sl) < 5e-9 and abs(tp - logged_tp) < 5e-9
            candidate_results[f"{eps:.0e}"] = {
                "sl": sl,
                "tp": tp,
                "match": match,
            }
            if not match:
                mismatch_counts[f"{eps:.0e}"] += 1

        raw_distance = atr * ATR_MULTIPLIER
        raw_sl = entry - raw_distance if trade["direction"] == "BUY" else entry + raw_distance
        raw_tp = entry + raw_distance if trade["direction"] == "BUY" else entry - raw_distance

        if not candidate_results["0e+00"]["match"]:
            rows.append(
                {
                    "trade_index": index,
                    "time": trade["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "direction": trade["direction"],
                    "entry": entry,
                    "python_atr": atr,
                    "distance": raw_distance,
                    "raw_sl": raw_sl,
                    "raw_tp": raw_tp,
                    "logged_sl": logged_sl,
                    "logged_tp": logged_tp,
                    "logged_midpoint": midpoint,
                    "baseline_sl": candidate_results["0e+00"]["sl"],
                    "baseline_tp": candidate_results["0e+00"]["tp"],
                    **{
                        f"match_eps_{key}": value["match"]
                        for key, value in candidate_results.items()
                    },
                }
            )

    best_key = min(mismatch_counts, key=mismatch_counts.get)
    summary = {
        "mt5_trade_count": len(mt5_trades),
        "candidate_endpoint_mismatch_counts": mismatch_counts,
        "best_epsilon": best_key,
        "best_epsilon_mismatches": mismatch_counts[best_key],
        "baseline_zero_epsilon_mismatches": mismatch_counts["0e+00"],
        "diagnostic_only": True,
        "interpretation_rule": (
            "Do not patch standalone replay rounding unless one deterministic normalization "
            "rule reduces exact endpoint mismatches to zero or isolates a clearly attributable "
            "ATR floating-point case. Never special-case a trade index."
        ),
        "baseline_mismatch_examples": rows[:20],
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
    print(f"diagnostic_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
