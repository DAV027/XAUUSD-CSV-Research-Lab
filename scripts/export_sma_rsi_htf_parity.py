from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from xau_lab.runner.campaign import load_market_bundle
from xau_lab.strategies.sma_rsi_htf import build_sma_rsi_htf_state


def _parse_utc(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) == 10:
        text += "T00:00:00+00:00"
    elif text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp())


def _iso_utc(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export frozen SMA/RSI HTF V2.1 Python signals for MT5 parity comparison"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/XAUUSD_M1_FEATURES.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/PYTHON_SIGNAL_PARITY.csv"),
    )
    parser.add_argument("--from-utc")
    parser.add_argument("--to-exclusive-utc")
    parser.add_argument(
        "--timestamp-semantics",
        choices=("broker_local_to_utc", "legacy_wall_clock"),
        default="broker_local_to_utc",
        help="Use true UTC conversion for broker-local source timestamps by default.",
    )
    args = parser.parse_args()

    start_epoch = _parse_utc(args.from_utc)
    end_epoch = _parse_utc(args.to_exclusive_utc)
    if start_epoch is not None and end_epoch is not None and start_epoch >= end_epoch:
        raise ValueError("--from-utc must be earlier than --to-exclusive-utc")

    market = load_market_bundle(
        args.features,
        timestamp_semantics=args.timestamp_semantics,
    )
    ctx = market.strategy_context()
    state = build_sma_rsi_htf_state(ctx)

    m5_by_source = {
        int(source_index): i
        for i, source_index in enumerate(state.m5_source_last_index)
    }

    fieldnames = [
        "signal_row_index",
        "direction",
        "signal_m1_epoch",
        "signal_m1_utc",
        "entry_epoch",
        "entry_utc",
        "m5_close_epoch",
        "m5_close_utc",
        "m5_close",
        "m5_sma9",
        "m5_sma21",
        "m5_rsi14",
        "m5_atr14",
        "m15_close_epoch",
        "m15_close_utc",
        "m15_close",
        "m15_sma200",
        "timestamp_semantics",
    ]

    rows: list[dict[str, object]] = []
    for signal_index in np.flatnonzero(state.signal):
        signal_index = int(signal_index)
        entry_index = signal_index + 1
        if entry_index >= len(ctx):
            continue

        entry_epoch = int(ctx.time_epoch[entry_index])
        if start_epoch is not None and entry_epoch < start_epoch:
            continue
        if end_epoch is not None and entry_epoch >= end_epoch:
            continue

        m5_index = m5_by_source.get(signal_index)
        if m5_index is None:
            raise RuntimeError(f"signal row {signal_index} is not mapped to a completed M5 bar")

        m5_close_epoch = int(state.m5_close_time[m5_index])
        htf_index = int(
            np.searchsorted(state.m15_close_time, m5_close_epoch, side="right") - 1
        )
        if htf_index < 0:
            raise RuntimeError("signal has no completed M15 state")

        signal_epoch = int(ctx.time_epoch[signal_index])
        rows.append(
            {
                "signal_row_index": signal_index,
                "direction": "BUY" if int(state.signal[signal_index]) > 0 else "SELL",
                "signal_m1_epoch": signal_epoch,
                "signal_m1_utc": _iso_utc(signal_epoch),
                "entry_epoch": entry_epoch,
                "entry_utc": _iso_utc(entry_epoch),
                "m5_close_epoch": m5_close_epoch,
                "m5_close_utc": _iso_utc(m5_close_epoch),
                "m5_close": float(state.m5_close[m5_index]),
                "m5_sma9": float(state.m5_sma_fast[m5_index]),
                "m5_sma21": float(state.m5_sma_slow[m5_index]),
                "m5_rsi14": float(state.m5_rsi[m5_index]),
                "m5_atr14": float(state.m5_atr[m5_index]),
                "m15_close_epoch": int(state.m15_close_time[htf_index]),
                "m15_close_utc": _iso_utc(int(state.m15_close_time[htf_index])),
                "m15_close": float(state.m15_close[htf_index]),
                "m15_sma200": float(state.m15_sma200[htf_index]),
                "timestamp_semantics": args.timestamp_semantics,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.output)

    buys = sum(row["direction"] == "BUY" for row in rows)
    sells = sum(row["direction"] == "SELL" for row in rows)
    print(f"wrote {len(rows)} frozen V2.1 signals -> {args.output}")
    print(f"BUY={buys} SELL={sells}")
    print(f"TIMESTAMP_SEMANTICS={args.timestamp_semantics}")
    print("No P/L was calculated; this export is for signal/indicator parity only.")


if __name__ == "__main__":
    main()
