from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from xau_lab.data.schema import DataPaths, SymbolMetadata, canonicalize_bar_frame


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _rates_to_frame(rates) -> pl.DataFrame:
    names = getattr(getattr(rates, "dtype", None), "names", None)
    if not names:
        raise RuntimeError("MT5 returned rates without named fields")
    return pl.DataFrame({name: rates[name] for name in names})


def _format_server_time(frame: pl.DataFrame, server_timezone: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.from_epoch(pl.col("time"), time_unit="s")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(server_timezone)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .alias("time")
    )


def export_all_m1(
    provider,
    symbol: str,
    paths: DataPaths,
    server_timezone: str,
    chunk_size: int = 100_000,
    overwrite: bool = False,
) -> SymbolMetadata:
    """Export all available MT5 M1 bars to the canonical immutable raw CSV."""
    if not symbol:
        raise ValueError("symbol must be non-empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    try:
        ZoneInfo(server_timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid IANA server timezone: {server_timezone!r}") from exc

    if paths.raw_csv.exists() and not overwrite:
        raise FileExistsError(
            f"raw export already exists: {paths.raw_csv}; pass overwrite=True explicitly to replace it"
        )

    initialized = False
    chunks: list[pl.DataFrame] = []
    try:
        if not provider.initialize():
            raise RuntimeError("MT5 provider initialization failed")
        initialized = True

        info = provider.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"MT5 symbol_info returned no metadata for {symbol}")

        start_pos = 0
        timeframe = provider.TIMEFRAME_M1
        while True:
            rates = provider.copy_rates_from_pos(symbol, timeframe, start_pos, chunk_size)
            if rates is None:
                raise RuntimeError(
                    f"MT5 copy_rates_from_pos failed for {symbol} at start_pos={start_pos}"
                )
            if len(rates) == 0:
                break
            chunks.append(_rates_to_frame(rates))
            start_pos += chunk_size

        if not chunks:
            raise RuntimeError(f"MT5 returned no M1 history for {symbol}")

        frame = pl.concat(chunks, how="vertical_relaxed")
        frame = frame.sort("time", maintain_order=True).unique(
            subset=["time"], keep="last", maintain_order=True
        )
        frame = _format_server_time(frame, server_timezone)
        frame = canonicalize_bar_frame(frame)

        paths.raw_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_tmp = paths.raw_csv.with_name(paths.raw_csv.name + ".tmp")
        frame.write_csv(raw_tmp)
        raw_tmp.replace(paths.raw_csv)

        metadata = SymbolMetadata(
            symbol=symbol,
            timeframe="M1",
            digits=int(info.digits),
            point=float(info.point),
            trade_contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            currency_profit=str(info.currency_profit),
            server_timezone=server_timezone,
            exported_rows=frame.height,
            first_time=str(frame["time"][0]),
            last_time=str(frame["time"][-1]),
        )
        _atomic_write_json(paths.metadata_json, metadata.to_dict())
        return metadata
    finally:
        if initialized:
            provider.shutdown()
