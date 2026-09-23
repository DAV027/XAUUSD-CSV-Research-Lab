from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from compare_sma_rsi_lifecycle_mt5 import (
    _extract_log_lifecycle,
    _extract_report_lifecycle,
)


DELAY_MS = 250


def _utc_ms(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _m5_start_ms(value: datetime) -> int:
    floored = value.replace(
        minute=(value.minute // 5) * 5,
        second=0,
        microsecond=0,
    )
    return _utc_ms(floored)


def _build_partition_index(root: Path) -> list[tuple[Path, int, int]]:
    index: list[tuple[Path, int, int]] = []
    for path in sorted(root.glob("broker_date=*/ticks.parquet")):
        bounds = (
            pl.scan_parquet(path)
            .select(
                pl.col("time_msc").min().alias("first"),
                pl.col("time_msc").max().alias("last"),
            )
            .collect()
        )
        index.append((path, int(bounds["first"][0]), int(bounds["last"][0])))
    return index


def _load_window(
    start_ms: int,
    end_ms: int,
    partition_index: list[tuple[Path, int, int]],
    cache: dict[Path, pl.DataFrame],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for path, first, last in partition_index:
        if last < start_ms or first > end_ms:
            continue
        frame = cache.get(path)
        if frame is None:
            frame = (
                pl.read_parquet(path, columns=["time_msc", "bid", "ask"])
                .sort("time_msc", maintain_order=True)
            )
            cache[path] = frame
        piece = frame.filter(
            (pl.col("time_msc") >= start_ms) & (pl.col("time_msc") <= end_ms)
        )
        if piece.height:
            frames.append(piece)

    if not frames:
        return pl.DataFrame(
            schema={"time_msc": pl.Int64, "bid": pl.Float64, "ask": pl.Float64}
        )
    return pl.concat(frames, how="vertical_relaxed").sort("time_msc", maintain_order=True)


def _price_match(observed: float | None, expected: float, point: float) -> bool:
    if observed is None:
        return False
    return abs(float(observed) - float(expected)) <= point / 2.0 + 1e-12


def _side_price(row: dict | None, direction: str) -> float | None:
    if row is None:
        return None
    return float(row["ask"] if direction == "BUY" else row["bid"])


def _row_at(frame: pl.DataFrame, index: int) -> dict | None:
    if frame.height == 0 or index < 0 or index >= frame.height:
        return None
    return frame.row(index, named=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the frozen V2.1 Strategy Tester 250 ms entry-delay mapping "
            "against exported FXIFY real ticks."
        )
    )
    parser.add_argument("--mt5-report", type=Path, required=True)
    parser.add_argument("--mt5-log", type=Path, required=True)
    parser.add_argument("--ticks-root", type=Path, required=True)
    parser.add_argument("--delay-ms", type=int, default=DELAY_MS)
    parser.add_argument("--point", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/ENTRY_DELAY_DIAGNOSTIC.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/ENTRY_DELAY_DIAGNOSTIC_SUMMARY.json"),
    )
    args = parser.parse_args()

    if args.delay_ms < 0:
        raise ValueError("delay-ms must be nonnegative")

    metadata = json.loads(
        (args.ticks_root / "_metadata.json").read_text(encoding="utf-8")
    )
    log_trades = _extract_log_lifecycle(args.mt5_log)
    report_trades, report_final_balance = _extract_report_lifecycle(args.mt5_report)
    if len(log_trades) != len(report_trades):
        raise ValueError(
            f"trade count mismatch: log={len(log_trades)} report={len(report_trades)}"
        )

    partition_index = _build_partition_index(args.ticks_root)
    cache: dict[Path, pl.DataFrame] = {}
    rows: list[dict] = []

    request_quote_mismatches = 0
    request_tick_missing = 0
    prev_matches = 0
    next_matches = 0
    exact_target_matches = 0
    either_endpoint_matches = 0
    neither_matches = 0

    for trade_index, (log_trade, report_trade) in enumerate(
        zip(log_trades, report_trades), start=1
    ):
        direction = report_trade["entry_direction"]
        bucket_start_ms = _m5_start_ms(report_trade["entry_time"])
        report_second_start_ms = _utc_ms(report_trade["entry_time"])
        report_second_end_ms = report_second_start_ms + 999

        # The frozen EA evaluates once on the first tick of the new M5 bar.
        # That first raw tick is therefore the request-construction tick.
        window = _load_window(
            bucket_start_ms,
            max(report_second_end_ms, bucket_start_ms + 5000),
            partition_index,
            cache,
        )

        request = _row_at(window, 0)
        if request is None:
            request_tick_missing += 1
            rows.append(
                {
                    "trade_index": trade_index,
                    "direction": direction,
                    "request_tick_msc": "",
                    "request_side_price": "",
                    "constructed_entry_midpoint": (log_trade.sl + log_trade.tp) / 2.0,
                    "request_quote_match": False,
                    "delay_target_msc": "",
                    "prev_tick_msc": "",
                    "prev_side_price": "",
                    "prev_matches_report_fill": False,
                    "next_tick_msc": "",
                    "next_side_price": "",
                    "next_matches_report_fill": False,
                    "report_fill_price": report_trade["entry_price"],
                    "report_fill_second": report_trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            continue

        request_ms = int(request["time_msc"])
        request_side = _side_price(request, direction)
        midpoint = (float(log_trade.sl) + float(log_trade.tp)) / 2.0

        # SL/TP endpoints are rounded to 2 digits, so their midpoint can be
        # displaced by half a point from the true construction quote.
        request_quote_match = abs(float(request_side) - midpoint) <= args.point + 1e-12
        if not request_quote_match:
            request_quote_mismatches += 1

        target_ms = request_ms + args.delay_ms

        # Make sure the window extends beyond the delay endpoint.
        if window.height == 0 or int(window["time_msc"][-1]) < target_ms:
            window = _load_window(
                bucket_start_ms,
                max(report_second_end_ms, target_ms + 2000),
                partition_index,
                cache,
            )

        before = window.filter(pl.col("time_msc") <= target_ms)
        after = window.filter(pl.col("time_msc") >= target_ms)

        prev = _row_at(before, before.height - 1)
        nxt = _row_at(after, 0)

        prev_price = _side_price(prev, direction)
        next_price = _side_price(nxt, direction)
        fill_price = float(report_trade["entry_price"])

        prev_match = _price_match(prev_price, fill_price, args.point)
        next_match = _price_match(next_price, fill_price, args.point)

        if prev_match:
            prev_matches += 1
        if next_match:
            next_matches += 1
        if prev is not None and int(prev["time_msc"]) == target_ms and prev_match:
            exact_target_matches += 1
        if prev_match or next_match:
            either_endpoint_matches += 1
        else:
            neither_matches += 1

        rows.append(
            {
                "trade_index": trade_index,
                "direction": direction,
                "request_tick_msc": request_ms,
                "request_side_price": request_side,
                "constructed_entry_midpoint": midpoint,
                "request_quote_match": request_quote_match,
                "delay_target_msc": target_ms,
                "prev_tick_msc": int(prev["time_msc"]) if prev is not None else "",
                "prev_side_price": prev_price if prev_price is not None else "",
                "prev_matches_report_fill": prev_match,
                "next_tick_msc": int(nxt["time_msc"]) if nxt is not None else "",
                "next_side_price": next_price if next_price is not None else "",
                "next_matches_report_fill": next_match,
                "report_fill_price": fill_price,
                "report_fill_second": report_trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    summary = {
        "tick_metadata_total_rows": int(metadata.get("total_rows", 0)),
        "trade_count": len(report_trades),
        "delay_ms": args.delay_ms,
        "request_tick_missing": request_tick_missing,
        "request_quote_mismatches": request_quote_mismatches,
        "fill_match_using_last_tick_at_or_before_delay_endpoint": prev_matches,
        "fill_match_using_first_tick_at_or_after_delay_endpoint": next_matches,
        "exact_tick_at_delay_endpoint_and_fill_match": exact_target_matches,
        "fill_match_by_either_endpoint_rule": either_endpoint_matches,
        "fill_match_by_neither_endpoint_rule": neither_matches,
        "report_final_balance": report_final_balance,
        "diagnostic_only": True,
        "interpretation_rule": (
            "Do not promote an execution-delay rule until one deterministic rule explains "
            "the fills with no material unexplained remainder. The request tick is defined "
            "by frozen EA behavior as the first real tick of the new M5 bar. The constructed "
            "entry quote is independently checked against the SL/TP midpoint."
        ),
        "mismatch_examples": [
            row
            for row in rows
            if (
                not bool(row["request_quote_match"])
                or (
                    not bool(row["prev_matches_report_fill"])
                    and not bool(row["next_matches_report_fill"])
                )
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
    print(f"delay_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
