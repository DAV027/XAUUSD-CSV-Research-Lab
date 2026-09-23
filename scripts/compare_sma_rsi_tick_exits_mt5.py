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


def _utc_ms(value: datetime) -> int:
    # For this frozen FXIFY tester run, Strategy Tester/log timestamps align
    # directly with the raw MT5 tick epoch (UTC), not with the convenience
    # time_broker column added by the exporter.
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _tick_partition(root: Path, utc_ms: int) -> Path:
    day = datetime.fromtimestamp(utc_ms / 1000.0, tz=timezone.utc).date().isoformat()
    return root / f"broker_date={day}" / "ticks.parquet"


def _candidate_partitions(root: Path, start_ms: int, end_ms: int) -> list[Path]:
    # Broker-date partitions are shifted from UTC during DST, so select by
    # partition metadata rather than assuming a UTC date maps to one folder.
    candidates: list[Path] = []
    for path in sorted(root.glob("broker_date=*/ticks.parquet")):
        frame = pl.scan_parquet(path).select(
            pl.col("time_msc").min().alias("first"),
            pl.col("time_msc").max().alias("last"),
        ).collect()
        first = int(frame["first"][0])
        last = int(frame["last"][0])
        if last >= start_ms and first <= end_ms:
            candidates.append(path)
    return candidates


def _load_window(
    root: Path,
    start_ms: int,
    end_ms: int,
    cache: dict[Path, pl.DataFrame],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for path in _candidate_partitions(root, start_ms, end_ms):
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


def _find_entry_tick(
    ticks: pl.DataFrame,
    direction: str,
    entry_price: float,
    point: float,
) -> tuple[int | None, float | None]:
    if ticks.height == 0:
        return None, None

    side = "ask" if direction == "BUY" else "bid"
    exact = ticks.filter((pl.col(side) - entry_price).abs() <= point / 2.0 + 1e-12)
    if exact.height:
        return int(exact["time_msc"][0]), float(exact[side][0])

    # Keep the audit diagnostic rather than inventing an exact fill tick.
    return None, None


def _first_trigger(
    ticks: pl.DataFrame,
    direction: str,
    sl: float,
    tp: float,
    after_ms: int,
) -> tuple[str | None, int | None, float | None]:
    ticks = ticks.filter(pl.col("time_msc") > after_ms)
    if ticks.height == 0:
        return None, None, None

    if direction == "BUY":
        side = "bid"
        with_flags = ticks.with_columns(
            (pl.col("bid") <= sl).alias("sl_hit"),
            (pl.col("bid") >= tp).alias("tp_hit"),
        )
    else:
        side = "ask"
        with_flags = ticks.with_columns(
            (pl.col("ask") >= sl).alias("sl_hit"),
            (pl.col("ask") <= tp).alias("tp_hit"),
        )

    hits = with_flags.filter(pl.col("sl_hit") | pl.col("tp_hit"))
    if hits.height == 0:
        return None, None, None

    first = hits.row(0, named=True)
    sl_hit = bool(first["sl_hit"])
    tp_hit = bool(first["tp_hit"])
    if sl_hit and tp_hit:
        reason = "AMBIGUOUS"
    elif sl_hit:
        reason = "SL"
    else:
        reason = "TP"
    return reason, int(first["time_msc"]), float(first[side])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen V2.1 protective exits against exported FXIFY real ticks."
    )
    parser.add_argument("--mt5-report", type=Path, required=True)
    parser.add_argument("--mt5-log", type=Path, required=True)
    parser.add_argument("--ticks-root", type=Path, required=True)
    parser.add_argument("--point", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/TICK_EXIT_PARITY.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/TICK_EXIT_PARITY_SUMMARY.json"),
    )
    args = parser.parse_args()

    metadata_path = args.ticks_root / "_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    log_trades = _extract_log_lifecycle(args.mt5_log)
    report_trades, report_final_balance = _extract_report_lifecycle(args.mt5_report)
    if len(log_trades) != len(report_trades):
        raise ValueError(
            f"trade count mismatch before tick replay: log={len(log_trades)} report={len(report_trades)}"
        )

    cache: dict[Path, pl.DataFrame] = {}
    rows: list[dict] = []

    entry_tick_missing = 0
    entry_second_fallbacks = 0
    entry_second_level_touch_ambiguities = 0
    trigger_missing = 0
    exit_reason_mismatches = 0
    exit_second_mismatches = 0
    ambiguous_triggers = 0
    protective_trades = 0
    end_of_test_trades = 0

    for index, (log_trade, report_trade) in enumerate(
        zip(log_trades, report_trades), start=1
    ):
        report_reason = _classify_report_exit(report_trade["exit_comment"])
        entry_start_ms = _utc_ms(report_trade["entry_time"])
        entry_end_ms = entry_start_ms + 999
        entry_ticks = _load_window(
            args.ticks_root, entry_start_ms, entry_end_ms, cache
        )
        entry_tick_ms, entry_side_price = _find_entry_tick(
            entry_ticks,
            report_trade["entry_direction"],
            float(report_trade["entry_price"]),
            args.point,
        )

        row = {
            "trade_index": index,
            "direction": report_trade["entry_direction"],
            "entry_time_report": report_trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price_report": report_trade["entry_price"],
            "entry_tick_msc": entry_tick_ms if entry_tick_ms is not None else "",
            "entry_tick_side_price": entry_side_price if entry_side_price is not None else "",
            "sl": log_trade.sl,
            "tp": log_trade.tp,
            "exit_reason_report": report_reason,
            "exit_time_report": report_trade["exit_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "tick_trigger_reason": "",
            "tick_trigger_msc": "",
            "tick_trigger_side_price": "",
            "entry_tick_found": entry_tick_ms is not None,
            "trigger_found": False,
            "exit_reason_match": False,
            "exit_second_match": False,
        }

        if report_reason == "END_OF_TEST":
            end_of_test_trades += 1
            row["trigger_found"] = True
            row["exit_reason_match"] = True
            row["exit_second_match"] = True
            rows.append(row)
            continue

        protective_trades += 1

        replay_anchor_ms = entry_tick_ms
        if entry_tick_ms is None:
            entry_tick_missing += 1

            # An exact MT5 deal fill price need not appear as a raw quote tick.
            # We can still establish an exact protective-exit replay if no SL/TP
            # level was touched anywhere in the reported entry second. In that
            # case, beginning immediately after that second cannot skip an exit.
            entry_second_reason, _, _ = _first_trigger(
                entry_ticks,
                report_trade["entry_direction"],
                float(log_trade.sl),
                float(log_trade.tp),
                entry_start_ms - 1,
            )
            if entry_second_reason is not None:
                entry_second_level_touch_ambiguities += 1
                row["entry_second_level_touch"] = entry_second_reason
                rows.append(row)
                continue

            entry_second_fallbacks += 1
            row["entry_second_level_touch"] = ""
            replay_anchor_ms = entry_end_ms
        else:
            row["entry_second_level_touch"] = ""

        exit_end_ms = _utc_ms(report_trade["exit_time"]) + 999
        replay_ticks = _load_window(
            args.ticks_root, int(replay_anchor_ms), exit_end_ms, cache
        )
        reason, trigger_ms, trigger_price = _first_trigger(
            replay_ticks,
            report_trade["entry_direction"],
            float(log_trade.sl),
            float(log_trade.tp),
            int(replay_anchor_ms),
        )

        row["tick_trigger_reason"] = reason or ""
        row["tick_trigger_msc"] = trigger_ms if trigger_ms is not None else ""
        row["tick_trigger_side_price"] = trigger_price if trigger_price is not None else ""
        row["trigger_found"] = trigger_ms is not None

        if reason == "AMBIGUOUS":
            ambiguous_triggers += 1

        if trigger_ms is None:
            trigger_missing += 1
            rows.append(row)
            continue

        reason_match = reason == report_reason
        report_exit_second = _utc_ms(report_trade["exit_time"]) // 1000
        trigger_second = trigger_ms // 1000
        second_match = trigger_second == report_exit_second

        row["exit_reason_match"] = reason_match
        row["exit_second_match"] = second_match

        if not reason_match:
            exit_reason_mismatches += 1
        if not second_match:
            exit_second_mismatches += 1
        rows.append(row)

    summary = {
        "tick_metadata_total_rows": int(metadata.get("total_rows", 0)),
        "log_trade_count": len(log_trades),
        "report_trade_count": len(report_trades),
        "protective_exit_trades": protective_trades,
        "end_of_test_trades": end_of_test_trades,
        "entry_fill_tick_diagnostic_missing": entry_tick_missing,
        "entry_second_fallbacks": entry_second_fallbacks,
        "entry_second_level_touch_ambiguities": entry_second_level_touch_ambiguities,
        "trigger_missing": trigger_missing,
        "ambiguous_triggers": ambiguous_triggers,
        "exit_reason_mismatches": exit_reason_mismatches,
        "exit_second_mismatches": exit_second_mismatches,
        "report_final_balance": report_final_balance,
        "tick_exit_parity_pass": (
            len(log_trades) > 0
            and entry_second_level_touch_ambiguities == 0
            and trigger_missing == 0
            and ambiguous_triggers == 0
            and exit_reason_mismatches == 0
            and exit_second_mismatches == 0
        ),
        "scope": (
            "Replays protective exits against exported FXIFY COPY_TICKS_ALL data. "
            "BUY positions trigger on bid; SELL positions trigger on ask. Entry execution "
            "ticks are anchored by the actual MT5 report fill price when that price exists in the raw tick stream. "
            "If the deal fill price is not present as a raw quote, the validator first proves that neither SL nor TP "
            "was touched anywhere in the reported entry second, then begins replay after that second. "
            "Tester/log timestamps for this frozen run are interpreted as UTC because they align "
            "with the raw tick epoch; time_broker is display metadata only. This stage validates "
            "the first protective-level crossing, not the 250 ms market-order request delay itself."
        ),
        "mismatch_examples": [
            row
            for row in rows
            if not (
                bool(row["trigger_found"])
                and bool(row["exit_reason_match"])
                and bool(row["exit_second_match"])
            )
            and row["exit_reason_report"] != "END_OF_TEST"
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
    print(f"tick_exit_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
