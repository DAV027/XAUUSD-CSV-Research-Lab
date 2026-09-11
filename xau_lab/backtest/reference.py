from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Sequence
from zoneinfo import ZoneInfo

from xau_lab.backtest.models import BacktestResult, CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec, Trade
from xau_lab.backtest.pricing import _round_price, buy_entry_price, fill_exit_price, sell_entry_price
from xau_lab.backtest.sizing import size_for_stop


@dataclass
class _Position:
    direction: int
    signal_index: int
    entry_index: int
    raw_entry_bid: float
    entry_price: float
    initial_stop: float
    active_stop: float
    target_price: float | None
    lot: float
    planned_risk_usd: float


def _broker_date(epoch: int, broker_timezone: str) -> str:
    zone = ZoneInfo(broker_timezone)
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).astimezone(zone).date().isoformat()


def _spread_cost_usd(
    direction: int,
    entry_spread: int,
    exit_spread: int,
    lot: float,
    symbol: SymbolSpec,
) -> float:
    spread_points = entry_spread if direction == 1 else exit_spread
    return float(spread_points) * symbol.point * symbol.contract_size * lot


def _slippage_cost_usd(cost: CostModel, lot: float, symbol: SymbolSpec) -> float:
    return 2.0 * cost.slippage_points_per_fill * symbol.point * symbol.contract_size * lot


def _short_ask(value: float, spread_points: int, symbol: SymbolSpec) -> float:
    return value + spread_points * symbol.point


def _stop_target_hits(
    position: _Position,
    bars: MarketBars,
    index: int,
    symbol: SymbolSpec,
) -> tuple[bool, bool]:
    if position.direction == 1:
        stop_hit = bars.low[index] <= position.active_stop
        target_hit = position.target_price is not None and bars.high[index] >= position.target_price
    else:
        ask_high = _short_ask(float(bars.high[index]), int(bars.spread[index]), symbol)
        ask_low = _short_ask(float(bars.low[index]), int(bars.spread[index]), symbol)
        stop_hit = ask_high >= position.active_stop
        target_hit = position.target_price is not None and ask_low <= position.target_price
    return bool(stop_hit), bool(target_hit)


def _stop_reason(position: _Position, target_hit: bool) -> str:
    if target_hit:
        return "stop_same_bar_ambiguous"
    if abs(position.active_stop - position.initial_stop) > 1e-12:
        return "trail"
    return "stop"


def _ratchet_trail(
    position: _Position,
    bars: MarketBars,
    index: int,
    symbol: SymbolSpec,
    exit_spec: ExitSpec,
) -> None:
    if exit_spec.atr_trail is None:
        return
    atr = float(bars.atr[index])
    if not isfinite(atr) or atr <= 0:
        return
    close = float(bars.close[index])
    if position.direction == 1:
        candidate = _round_price(close - exit_spec.atr_trail * atr, symbol)
        position.active_stop = max(position.active_stop, candidate)
    else:
        candidate = _round_price(close + exit_spec.atr_trail * atr, symbol)
        position.active_stop = min(position.active_stop, candidate)


def _raw_bid_for_level(
    level: float,
    direction: int,
    spread_points: int,
    symbol: SymbolSpec,
) -> float:
    if direction == 1:
        return float(level)
    return float(level) - spread_points * symbol.point


def _stop_raw_bid(position: _Position, bars: MarketBars, index: int, symbol: SymbolSpec) -> float:
    spread = int(bars.spread[index])
    level_bid = _raw_bid_for_level(position.active_stop, position.direction, spread, symbol)
    if position.direction == 1:
        return min(level_bid, float(bars.open[index]))
    return max(level_bid, float(bars.open[index]))


def _target_raw_bid(position: _Position, bars: MarketBars, index: int, symbol: SymbolSpec) -> float:
    assert position.target_price is not None
    return _raw_bid_for_level(position.target_price, position.direction, int(bars.spread[index]), symbol)


def _finalize_trade(
    position: _Position,
    bars: MarketBars,
    exit_index: int,
    raw_exit_bid: float,
    exit_reason: str,
    symbol: SymbolSpec,
    cost: CostModel,
) -> Trade:
    direction = position.direction
    lot = position.lot
    entry_index = position.entry_index
    exit_spread = int(bars.spread[exit_index])
    exit_price = fill_exit_price(raw_exit_bid, exit_spread, direction, symbol, cost)

    gross_pnl = direction * (raw_exit_bid - position.raw_entry_bid) * symbol.contract_size * lot
    spread_cost = _spread_cost_usd(
        direction,
        int(bars.spread[entry_index]),
        exit_spread,
        lot,
        symbol,
    )
    slippage_cost = _slippage_cost_usd(cost, lot, symbol)
    commission = cost.commission_round_trip_per_lot * lot
    net_pnl = gross_pnl - spread_cost - slippage_cost - commission
    pnl_r = net_pnl / position.planned_risk_usd if position.planned_risk_usd > 0 else None
    hold_minutes = (int(bars.time_epoch[exit_index]) - int(bars.time_epoch[entry_index])) / 60.0

    return Trade(
        experiment_id=None,
        parameter_set_id=None,
        fingerprint=None,
        signal_time=int(bars.time_epoch[position.signal_index]),
        entry_time=int(bars.time_epoch[entry_index]),
        direction=direction,
        signal_price=float(bars.close[position.signal_index]),
        entry_price=float(position.entry_price),
        stop_price=float(position.initial_stop),
        target_price=None if position.target_price is None else float(position.target_price),
        exit_time=int(bars.time_epoch[exit_index]),
        exit_price=float(exit_price),
        exit_reason=exit_reason,
        lot_size=float(lot),
        planned_risk_usd=float(position.planned_risk_usd),
        risk_R=1.0,
        gross_pnl=float(gross_pnl),
        spread_cost=float(spread_cost),
        commission=float(commission),
        slippage_cost=float(slippage_cost),
        net_pnl=float(net_pnl),
        pnl_R=None if pnl_r is None else float(pnl_r),
        hold_minutes=float(hold_minutes),
        broker_date=_broker_date(int(bars.time_epoch[entry_index]), bars.broker_timezone),
        broker_timezone=bars.broker_timezone,
        entry_index=entry_index,
        exit_index=exit_index,
    )


def _open_position(
    bars: MarketBars,
    index: int,
    signal_index: int,
    direction: int,
    symbol: SymbolSpec,
    cost: CostModel,
    risk: RiskModel,
    exit_spec: ExitSpec,
) -> _Position | None:
    atr = float(bars.atr[signal_index])
    if not isfinite(atr) or atr <= 0:
        return None

    raw_bid = float(bars.open[index])
    spread = int(bars.spread[index])
    if direction == 1:
        entry_price = buy_entry_price(raw_bid, spread, symbol, cost)
        stop_price = _round_price(entry_price - exit_spec.stop_atr * atr, symbol)
        stop_distance = entry_price - stop_price
        target_price = None if exit_spec.target_r is None else _round_price(
            entry_price + exit_spec.target_r * stop_distance, symbol
        )
    else:
        entry_price = sell_entry_price(raw_bid, spread, symbol, cost)
        stop_price = _round_price(entry_price + exit_spec.stop_atr * atr, symbol)
        stop_distance = stop_price - entry_price
        target_price = None if exit_spec.target_r is None else _round_price(
            entry_price - exit_spec.target_r * stop_distance, symbol
        )

    try:
        sized = size_for_stop(entry_price, stop_price, symbol, risk)
    except ValueError:
        return None
    if not sized.feasible:
        return None

    return _Position(
        direction=direction,
        signal_index=signal_index,
        entry_index=index,
        raw_entry_bid=raw_bid,
        entry_price=entry_price,
        initial_stop=stop_price,
        active_stop=stop_price,
        target_price=target_price,
        lot=sized.lot,
        planned_risk_usd=sized.risk_usd,
    )


def run_reference_backtest(
    bars: MarketBars,
    signals: Sequence[int],
    symbol: SymbolSpec,
    cost: CostModel,
    risk: RiskModel,
    exit_spec: ExitSpec,
) -> BacktestResult:
    signal_values = list(signals)
    if len(signal_values) != len(bars):
        raise ValueError("signals length must match MarketBars")
    if any(signal not in (-1, 0, 1) for signal in signal_values):
        raise ValueError("signals must contain only -1, 0, +1")

    trades: list[Trade] = []
    position: _Position | None = None
    risk_skip_count = 0

    for index in range(len(bars)):
        if position is not None:
            stop_hit, target_hit = _stop_target_hits(position, bars, index, symbol)
            if stop_hit:
                reason = _stop_reason(position, target_hit)
                raw_exit_bid = _stop_raw_bid(position, bars, index, symbol)
                trades.append(_finalize_trade(position, bars, index, raw_exit_bid, reason, symbol, cost))
                position = None
                continue
            if target_hit:
                raw_exit_bid = _target_raw_bid(position, bars, index, symbol)
                trades.append(_finalize_trade(position, bars, index, raw_exit_bid, "target", symbol, cost))
                position = None
                continue

            if exit_spec.time_exit_minutes is not None:
                elapsed_minutes = (
                    int(bars.time_epoch[index]) - int(bars.time_epoch[position.entry_index])
                ) / 60.0
                if elapsed_minutes >= exit_spec.time_exit_minutes:
                    trades.append(
                        _finalize_trade(
                            position,
                            bars,
                            index,
                            float(bars.close[index]),
                            "time",
                            symbol,
                            cost,
                        )
                    )
                    position = None
                    continue
            _ratchet_trail(position, bars, index, symbol, exit_spec)
            continue

        if index == 0:
            continue
        signal_index = index - 1
        direction = int(signal_values[signal_index])
        if direction == 0:
            continue

        candidate = _open_position(bars, index, signal_index, direction, symbol, cost, risk, exit_spec)
        if candidate is None:
            atr = float(bars.atr[signal_index])
            if isfinite(atr) and atr > 0:
                risk_skip_count += 1
            continue
        position = candidate

        stop_hit, target_hit = _stop_target_hits(position, bars, index, symbol)
        if stop_hit:
            reason = _stop_reason(position, target_hit)
            raw_exit_bid = _stop_raw_bid(position, bars, index, symbol)
            trades.append(_finalize_trade(position, bars, index, raw_exit_bid, reason, symbol, cost))
            position = None
            continue
        if target_hit:
            raw_exit_bid = _target_raw_bid(position, bars, index, symbol)
            trades.append(_finalize_trade(position, bars, index, raw_exit_bid, "target", symbol, cost))
            position = None
            continue

        _ratchet_trail(position, bars, index, symbol, exit_spec)

    if position is not None and len(bars) > 0:
        final_index = len(bars) - 1
        trades.append(
            _finalize_trade(
                position,
                bars,
                final_index,
                float(bars.close[final_index]),
                "end_of_data",
                symbol,
                cost,
            )
        )

    return BacktestResult(tuple(trades), risk_skip_count=risk_skip_count)
