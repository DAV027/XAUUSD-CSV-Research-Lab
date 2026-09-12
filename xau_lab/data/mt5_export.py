from __future__ import annotations

import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from xau_lab.data.schema import DataPaths, SymbolMetadata, canonicalize_bar_frame


_SHORT_HISTORY_STABILITY_PROBES = 3
_SHORT_HISTORY_PROBE_DELAY_SECONDS = 1.0
_SHORT_HISTORY_NEXT_PROBE_COUNT = 1_000


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


def _rates_time_signature(rates) -> tuple[int, ...]:
    names = getattr(getattr(rates, "dtype", None), "names", None)
    if not names or "time" not in names:
        raise RuntimeError("MT5 returned rates without a named time field")
    return tuple(int(value) for value in rates["time"])


def _format_server_time(frame: pl.DataFrame, server_timezone: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.from_epoch(pl.col("time"), time_unit="s")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(server_timezone)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .alias("time")
    )


def _terminal_maxbars(provider) -> int | None:
    terminal_info = getattr(provider, "terminal_info", None)
    if not callable(terminal_info):
        return None
    info = terminal_info()
    if info is None:
        return None
    value = getattr(info, "maxbars", None)
    try:
        maxbars = int(value)
    except (TypeError, ValueError):
        return None
    return maxbars if maxbars > 0 else None


def _provider_last_error(provider) -> tuple[int, str] | None:
    last_error = getattr(provider, "last_error", None)
    if not callable(last_error):
        return None
    value = last_error()
    if not isinstance(value, tuple) or not value:
        return None
    try:
        code = int(value[0])
    except (TypeError, ValueError):
        return None
    message = str(value[1]) if len(value) > 1 else ""
    return code, message


def _confirm_short_history_boundary(
    provider,
    symbol: str,
    timeframe,
    start_pos: int,
    chunk_size: int,
    rates,
    maxbars: int | None,
) -> None:
    expected_signature = _rates_time_signature(rates)
    returned = len(rates)

    for _ in range(_SHORT_HISTORY_STABILITY_PROBES - 1):
        time.sleep(_SHORT_HISTORY_PROBE_DELAY_SECONDS)
        repeated = provider.copy_rates_from_pos(symbol, timeframe, start_pos, chunk_size)
        if repeated is None or len(repeated) == 0:
            error = _provider_last_error(provider)
            raise RuntimeError(
                f"MT5 history changed during short-chunk stability check for {symbol} at "
                f"start_pos={start_pos}; repeated probe returned no data; last_error={error}. "
                "Wait for history synchronization to finish, then re-export."
            )
        if _rates_time_signature(repeated) != expected_signature:
            raise RuntimeError(
                f"MT5 history changed during short-chunk stability check for {symbol} at "
                f"start_pos={start_pos}. Wait for history synchronization to finish, then re-export."
            )

    next_pos = start_pos + returned
    if maxbars is not None and next_pos >= maxbars:
        raise RuntimeError(
            f"MT5 reached terminal Max bars in chart ({maxbars}) for {symbol} at "
            f"start_pos={next_pos}; history may be capped. Increase the terminal "
            "'Max bars in chart' setting, reload history, and re-export before research."
        )

    next_rates = provider.copy_rates_from_pos(
        symbol,
        timeframe,
        next_pos,
        min(chunk_size, _SHORT_HISTORY_NEXT_PROBE_COUNT),
    )
    if next_rates is not None and len(next_rates) > 0:
        raise RuntimeError(
            f"MT5 history changed during short-chunk stability check for {symbol}: "
            f"additional bars appeared at start_pos={next_pos}. Wait for history synchronization "
            "to finish, then re-export."
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

        maxbars = _terminal_maxbars(provider)
        start_pos = 0
        timeframe = provider.TIMEFRAME_M1
        while True:
            rates = provider.copy_rates_from_pos(symbol, timeframe, start_pos, chunk_size)
            if rates is None:
                error = _provider_last_error(provider)
                error_detail = f"; last_error={error}" if error is not None else ""
                if maxbars is not None and start_pos >= maxbars:
                    raise RuntimeError(
                        f"MT5 reached terminal Max bars in chart ({maxbars}) for {symbol} at "
                        f"start_pos={start_pos}; history may be capped. Increase the terminal "
                        f"'Max bars in chart' setting, reload history, and re-export before research"
                        f"{error_detail}"
                    )
                if error is not None and error[0] == -4:
                    break
                raise RuntimeError(
                    f"MT5 copy_rates_from_pos failed for {symbol} at start_pos={start_pos}"
                    f"{error_detail}"
                )
            if len(rates) == 0:
                break

            returned = len(rates)
            if returned < chunk_size:
                _confirm_short_history_boundary(
                    provider,
                    symbol,
                    timeframe,
                    start_pos,
                    chunk_size,
                    rates,
                    maxbars,
                )
                chunks.append(_rates_to_frame(rates))
                break

            chunks.append(_rates_to_frame(rates))
            start_pos += returned

        if not chunks:
            raise RuntimeError(f"MT5 returned no M1 history for {symbol}")

        frame = pl.concat(chunks, how="vertical_relaxed")
        frame = frame.sort("time", maintain_order=True).unique(
            subset=["time"], keep="last", maintain_order=True
        )
        if maxbars is not None and frame.height >= maxbars - 1:
            raise RuntimeError(
                f"MT5 returned {frame.height} M1 bars while terminal Max bars in chart is {maxbars}; "
                "history may be capped. Increase the terminal 'Max bars in chart' setting, reload "
                "history, and re-export before research."
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
