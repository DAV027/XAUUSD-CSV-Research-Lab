from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from compare_sma_rsi_lifecycle_mt5 import (
    _classify_report_exit,
    _extract_log_lifecycle,
    _extract_report_lifecycle,
)
from compare_sma_rsi_tick_exits_mt5 import _first_trigger, _find_entry_tick


DELAY_MS = 250
POINT = 0.01


def _utc_ms(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


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


def _side_price(row: dict | None, direction: str) -> float | None:
    if row is None:
        return None
    # Closing a BUY sells at bid; closing a SELL buys at ask.
    return float(row["bid"] if direction == "BUY" else row["ask"])


def _price_match(observed: float | None, expected: float) -> bool:
    if observed is None:
        return False
    return abs(float(observed) - float(expected)) <= POINT / 2.0 + 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether frozen V2.1 protective exits use the 250 ms tester delay rule."
    )
    parser.add_argument("--mt5-report", type=Path, required=True)
    parser.add_argument("--mt5-log", type=Path, required=True)
    parser.add_argument("--ticks-root", type=Path, required=True)
    parser.add_argument("--delay-ms", type=int, default=DELAY_MS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/EXIT_DELAY_DIAGNOSTIC.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/EXIT_DELAY_DIAGNOSTIC_SUMMARY.json"),
    )
    args = parser.parse_args()

    log_trades = _extract_log_lifecycle(args.mt5_log)
    report_trades, report_final_balance = _extract_report_lifecycle(args.mt5_report)
    if len(log_trades) != len(report_trades):
        raise ValueError(
            f"trade count mismatch: log={len(log_trades)} report={len(report_trades)}"
        )

    partition_index = _build_partition_index(args.ticks_root)
    cache: dict[Path, pl.DataFrame] = {}
    rows: list[dict] = []

    protective = 0
    trigger_missing = 0
    trigger_quote_matches = 0
    prev_delay_matches = 0
    next_delay_matches = 0
    either_delay_matches = 0
    neither_delay_matches = 0

    for index, (log_trade, report_trade) in enumerate(
        zip(log_trades, report_trades), start=1
    ):
        report_reason = _classify_report_exit(report_trade["exit_comment"])
        if report_reason == "END_OF_TEST":
            continue
        protective += 1

        entry_second_ms = _utc_ms(report_trade["entry_time"])
        entry_ticks = _load_window(
            entry_second_ms,
            entry_second_ms + 999,
            partition_index,
            cache,
        )
        entry_tick_ms, _ = _find_entry_tick(
            entry_ticks,
            report_trade["entry_direction"],
            float(report_trade["entry_price"]),
            POINT,
        )
        if entry_tick_ms is None:
            # Same validated fallback used in Stage 2d.
            entry_tick_ms = entry_second_ms + 999

        exit_second_end_ms = _utc_ms(report_trade["exit_time"]) + 999
        replay_ticks = _load_window(
            int(entry_tick_ms),
            exit_second_end_ms,
            partition_index,
            cache,
        )
        reason, trigger_ms, trigger_price = _first_trigger(
            replay_ticks,
            report_trade["entry_direction"],
            float(log_trade.sl),
            float(log_trade.tp),
            int(entry_tick_ms),
        )
        if trigger_ms is None:
            trigger_missing += 1
            rows.append(
                {
                    "trade_index": index,
                    "direction": report_trade["entry_direction"],
                    "report_reason": report_reason,
                    "trigger_reason": reason or "",
                    "trigger_msc": "",
                    "trigger_side_price": "",
                    "report_exit_price": report_trade["exit_price"],
                    "trigger_quote_matches_report_fill": False,
                    "delay_target_msc": "",
                    "prev_delay_tick_msc": "",
                    "prev_delay_side_price": "",
                    "prev_delay_matches_report_fill": False,
                    "next_delay_tick_msc": "",
                    "next_delay_side_price": "",
                    "next_delay_matches_report_fill": False,
                }
            )
            continue

        target_ms = int(trigger_ms) + args.delay_ms
        delay_window = _load_window(
            int(trigger_ms),
            target_ms + 2000,
            partition_index,
            cache,
        )
        before = delay_window.filter(pl.col("time_msc") <= target_ms)
        after = delay_window.filter(pl.col("time_msc") >= target_ms)

        prev = before.row(before.height - 1, named=True) if before.height else None
        nxt = after.row(0, named=True) if after.height else None

        report_exit_price = float(report_trade["exit_price"])
        trigger_match = _price_match(trigger_price, report_exit_price)
        prev_price = _side_price(prev, report_trade["entry_direction"])
        next_price = _side_price(nxt, report_trade["entry_direction"])
        prev_match = _price_match(prev_price, report_exit_price)
        next_match = _price_match(next_price, report_exit_price)

        trigger_quote_matches += int(trigger_match)
        prev_delay_matches += int(prev_match)
        next_delay_matches += int(next_match)
        either_delay_matches += int(prev_match or next_match)
        neither_delay_matches += int(not (prev_match or next_match))

        rows.append(
            {
                "trade_index": index,
                "direction": report_trade["entry_direction"],
                "report_reason": report_reason,
                "trigger_reason": reason or "",
                "trigger_msc": int(trigger_ms),
                "trigger_side_price": trigger_price,
                "report_exit_price": report_exit_price,
                "trigger_quote_matches_report_fill": trigger_match,
                "delay_target_msc": target_ms,
                "prev_delay_tick_msc": int(prev["time_msc"]) if prev is not None else "",
                "prev_delay_side_price": prev_price if prev_price is not None else "",
                "prev_delay_matches_report_fill": prev_match,
                "next_delay_tick_msc": int(nxt["time_msc"]) if nxt is not None else "",
                "next_delay_side_price": next_price if next_price is not None else "",
                "next_delay_matches_report_fill": next_match,
            }
        )

    summary = {
        "protective_exit_trades": protective,
        "delay_ms": args.delay_ms,
        "trigger_missing": trigger_missing,
        "fill_match_using_trigger_tick_quote": trigger_quote_matches,
        "fill_match_using_last_tick_at_or_before_trigger_plus_delay": prev_delay_matches,
        "fill_match_using_first_tick_at_or_after_trigger_plus_delay": next_delay_matches,
        "fill_match_by_either_delay_endpoint_rule": either_delay_matches,
        "fill_match_by_neither_delay_endpoint_rule": neither_delay_matches,
        "report_final_balance": report_final_balance,
        "diagnostic_only": True,
        "interpretation_rule": (
            "Promote delayed protective execution only if one deterministic rule explains "
            "all or effectively all MT5 exit fills without degrading the already-validated "
            "exit reason and exit-second parity."
        ),
        "mismatch_examples": [
            row
            for row in rows
            if not bool(row["trigger_quote_matches_report_fill"])
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
    print(f"diagnostic_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
