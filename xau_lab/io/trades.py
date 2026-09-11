from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from xau_lab.backtest.models import Trade


TRADE_COLUMNS = [
    "experiment_id",
    "parameter_set_id",
    "fingerprint",
    "signal_time",
    "entry_time",
    "direction",
    "signal_price",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "lot_size",
    "planned_risk_usd",
    "risk_R",
    "gross_pnl",
    "spread_cost",
    "commission",
    "slippage_cost",
    "net_pnl",
    "pnl_R",
    "hold_minutes",
    "broker_date",
    "session_asia",
    "session_london",
    "session_new_york",
    "session_overlap",
]


def _iso_broker_time(epoch: int, broker_timezone: str) -> str:
    try:
        zone = ZoneInfo(broker_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"invalid broker timezone: {broker_timezone}") from exc
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).astimezone(zone).isoformat()


def trade_to_record(trade: Trade) -> dict[str, object]:
    signal_time = _iso_broker_time(trade.signal_time, trade.broker_timezone)
    entry_time = _iso_broker_time(trade.entry_time, trade.broker_timezone)
    exit_time = _iso_broker_time(trade.exit_time, trade.broker_timezone)
    return {
        "experiment_id": trade.experiment_id,
        "parameter_set_id": trade.parameter_set_id,
        "fingerprint": trade.fingerprint,
        "signal_time": signal_time,
        "entry_time": entry_time,
        "direction": trade.direction,
        "signal_price": trade.signal_price,
        "entry_price": trade.entry_price,
        "stop_price": trade.stop_price,
        "target_price": trade.target_price,
        "exit_time": exit_time,
        "exit_price": trade.exit_price,
        "exit_reason": trade.exit_reason,
        "lot_size": trade.lot_size,
        "planned_risk_usd": trade.planned_risk_usd,
        "risk_R": trade.risk_R,
        "gross_pnl": trade.gross_pnl,
        "spread_cost": trade.spread_cost,
        "commission": trade.commission,
        "slippage_cost": trade.slippage_cost,
        "net_pnl": trade.net_pnl,
        "pnl_R": trade.pnl_R,
        "hold_minutes": trade.hold_minutes,
        "broker_date": trade.broker_date,
        "session_asia": trade.session_asia,
        "session_london": trade.session_london,
        "session_new_york": trade.session_new_york,
        "session_overlap": trade.session_overlap,
    }


def trades_to_frame(trades: Iterable[Trade]):
    import polars as pl

    records = [trade_to_record(trade) for trade in trades]
    if not records:
        return pl.DataFrame({column: [] for column in TRADE_COLUMNS}).select(TRADE_COLUMNS)
    return pl.DataFrame(records).select(TRADE_COLUMNS)
