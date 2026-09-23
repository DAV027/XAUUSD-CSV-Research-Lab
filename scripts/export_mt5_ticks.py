from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import polars as pl


def _provider_error() -> tuple[int, str] | None:
    value = mt5.last_error()
    if not isinstance(value, tuple) or not value:
        return None
    try:
        return int(value[0]), str(value[1]) if len(value) > 1 else ""
    except (TypeError, ValueError):
        return None


def _ticks_to_frame(ticks, broker_zone: ZoneInfo) -> pl.DataFrame:
    names = getattr(getattr(ticks, "dtype", None), "names", None)
    if not names:
        raise RuntimeError("MT5 returned ticks without named fields")

    frame = pl.DataFrame({name: ticks[name] for name in names})
    if "time_msc" not in frame.columns:
        raise RuntimeError("MT5 tick payload is missing time_msc")

    return frame.with_columns(
        pl.from_epoch(pl.col("time_msc"), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .alias("time_utc"),
        pl.from_epoch(pl.col("time_msc"), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(str(broker_zone))
        .alias("time_broker"),
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export FXIFY XAUUSD.r real ticks as broker-date Parquet partitions."
    )
    parser.add_argument("--symbol", default="XAUUSD.r")
    parser.add_argument("--server-timezone", default="EET")
    parser.add_argument("--from-broker-date", type=date.fromisoformat, required=True)
    parser.add_argument("--to-exclusive-broker-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/fxify_xauusdr/data/ticks"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.to_exclusive_broker_date <= args.from_broker_date:
        raise ValueError("to-exclusive date must be after from date")

    broker_zone = ZoneInfo(args.server_timezone)

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {_provider_error()}")

    try:
        info = mt5.symbol_info(args.symbol)
        if info is None:
            raise RuntimeError(f"symbol_info returned no data for {args.symbol}")

        args.output_root.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        exported_days = 0
        empty_days = 0
        first_msc: int | None = None
        last_msc: int | None = None
        day = args.from_broker_date

        while day < args.to_exclusive_broker_date:
            next_day = day + timedelta(days=1)
            start_local = datetime.combine(day, time.min, tzinfo=broker_zone)
            end_local = datetime.combine(next_day, time.min, tzinfo=broker_zone)
            start_utc = start_local.astimezone(timezone.utc)
            end_utc = end_local.astimezone(timezone.utc)

            target = args.output_root / f"broker_date={day.isoformat()}" / "ticks.parquet"
            if target.exists() and not args.overwrite:
                existing = pl.scan_parquet(target).select(pl.len()).collect().item()
                print(f"skip existing {day}: rows={existing} -> {target}")
                total_rows += int(existing)
                exported_days += 1
                day = next_day
                continue

            ticks = mt5.copy_ticks_range(
                args.symbol,
                start_utc,
                end_utc,
                mt5.COPY_TICKS_ALL,
            )
            if ticks is None:
                raise RuntimeError(
                    f"copy_ticks_range failed for {day}: last_error={_provider_error()}"
                )

            if len(ticks) == 0:
                empty_days += 1
                print(f"{day}: no ticks")
                day = next_day
                continue

            frame = _ticks_to_frame(ticks, broker_zone).sort("time_msc", maintain_order=True)

            # The MT5 API range endpoints can be inclusive. Enforce the broker-date
            # half-open interval explicitly so adjacent partitions cannot overlap.
            frame = frame.filter(
                (pl.col("time_broker").dt.date() >= day)
                & (pl.col("time_broker").dt.date() < next_day)
            )

            if frame.height == 0:
                empty_days += 1
                print(f"{day}: no ticks after broker-date boundary filter")
                day = next_day
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            frame.write_parquet(tmp, compression="zstd")
            tmp.replace(target)

            day_first = int(frame["time_msc"][0])
            day_last = int(frame["time_msc"][-1])
            first_msc = day_first if first_msc is None else min(first_msc, day_first)
            last_msc = day_last if last_msc is None else max(last_msc, day_last)
            total_rows += frame.height
            exported_days += 1

            print(
                f"{day}: rows={frame.height} "
                f"first={frame['time_broker'][0]} last={frame['time_broker'][-1]} -> {target}"
            )
            day = next_day

        metadata = {
            "symbol": args.symbol,
            "server_timezone": args.server_timezone,
            "from_broker_date": args.from_broker_date.isoformat(),
            "to_exclusive_broker_date": args.to_exclusive_broker_date.isoformat(),
            "copy_ticks_mode": "COPY_TICKS_ALL",
            "partitioning": "broker_date",
            "format": "parquet",
            "compression": "zstd",
            "exported_days": exported_days,
            "empty_days": empty_days,
            "total_rows": total_rows,
            "first_time_msc": first_msc,
            "last_time_msc": last_msc,
            "digits": int(info.digits),
            "point": float(info.point),
            "contract_size": float(info.trade_contract_size),
            "volume_min": float(info.volume_min),
            "volume_max": float(info.volume_max),
            "volume_step": float(info.volume_step),
        }
        metadata_path = args.output_root / "_metadata.json"
        _atomic_json(metadata_path, metadata)

        print(json.dumps(metadata, indent=2))
        print(f"metadata={metadata_path}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
