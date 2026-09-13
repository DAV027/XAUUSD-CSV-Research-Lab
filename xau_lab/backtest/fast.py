from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numba import njit
from xau_lab.signals import valid_signals

from xau_lab.backtest.models import BacktestResult, CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.reference import _Position, _finalize_trade


_STOP = 1
_STOP_AMBIGUOUS = 2
_TARGET = 3
_TIME = 4
_END_OF_DATA = 5
_TRAIL = 6

_REASON = {
    _STOP: "stop",
    _STOP_AMBIGUOUS: "stop_same_bar_ambiguous",
    _TARGET: "target",
    _TIME: "time",
    _END_OF_DATA: "end_of_data",
    _TRAIL: "trail",
}


@njit(cache=True)
def _round_price_kernel(price: float, digits: int) -> float:
    scale = 10.0 ** digits
    scaled = price * scale
    if scaled >= 0.0:
        units = math.floor(scaled + 0.5 + 1e-9)
    else:
        units = math.ceil(scaled - 0.5 - 1e-9)
    return units / scale


@njit(cache=True)
def _floor_to_step(value: float, step: float) -> float:
    units = math.floor((value / step) + 1e-12)
    return round(units * step, 10)


@njit(cache=True)
def _size_for_stop_kernel(
    entry_price: float,
    stop_price: float,
    contract_size: float,
    volume_min: float,
    volume_step: float,
    volume_max: float,
    preferred_risk_usd: float,
    hard_risk_usd: float,
    max_lot: float,
) -> tuple[bool, float, float]:
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0.0:
        return False, 0.0, 0.0
    risk_per_lot = stop_distance * contract_size
    preferred_lot = preferred_risk_usd / risk_per_lot
    executable_cap = min(max_lot, volume_max)
    if executable_cap < volume_min:
        return False, 0.0, 0.0

    if preferred_lot < volume_min:
        min_risk = risk_per_lot * volume_min
        if min_risk > hard_risk_usd + 1e-12:
            return False, 0.0, 0.0
        lot = volume_min
    else:
        lot = _floor_to_step(min(preferred_lot, executable_cap), volume_step)
        if lot < volume_min:
            min_risk = risk_per_lot * volume_min
            if min_risk > hard_risk_usd + 1e-12 or volume_min > executable_cap:
                return False, 0.0, 0.0
            lot = volume_min

    lot = min(lot, executable_cap)
    lot = _floor_to_step(lot, volume_step)
    if lot < volume_min:
        return False, 0.0, 0.0
    actual_risk = risk_per_lot * lot
    if actual_risk > hard_risk_usd + 1e-12:
        return False, 0.0, 0.0
    return True, lot, actual_risk


@njit(cache=True)
def _kernel(
    time_epoch,
    open_,
    high,
    low,
    close,
    spread,
    atr,
    signals,
    point,
    digits,
    contract_size,
    volume_min,
    volume_step,
    volume_max,
    commission_round_trip_per_lot,
    slippage_points_per_fill,
    preferred_risk_usd,
    hard_risk_usd,
    max_lot,
    stop_atr,
    target_r,
    time_exit_minutes,
    atr_trail,
):
    n = len(time_epoch)
    # Each trade consumes a distinct nonzero signal before the final bar.
    capacity = np.count_nonzero(signals[:max(0, n - 1)])
    out_direction = np.zeros(capacity, dtype=np.int8)
    out_signal_index = np.full(capacity, -1, dtype=np.int64)
    out_entry_index = np.full(capacity, -1, dtype=np.int64)
    out_exit_index = np.full(capacity, -1, dtype=np.int64)
    out_raw_entry = np.zeros(capacity, dtype=np.float64)
    out_entry_price = np.zeros(capacity, dtype=np.float64)
    out_initial_stop = np.zeros(capacity, dtype=np.float64)
    out_target = np.full(capacity, np.nan, dtype=np.float64)
    out_lot = np.zeros(capacity, dtype=np.float64)
    out_planned_risk = np.zeros(capacity, dtype=np.float64)
    out_raw_exit = np.zeros(capacity, dtype=np.float64)
    out_reason = np.zeros(capacity, dtype=np.int8)

    trade_count = 0
    risk_skip_count = 0

    pos_open = False
    p_direction = 0
    p_signal_index = -1
    p_entry_index = -1
    p_raw_entry = 0.0
    p_entry_price = 0.0
    p_initial_stop = 0.0
    p_active_stop = 0.0
    p_target = np.nan
    p_has_target = False
    p_lot = 0.0
    p_planned_risk = 0.0

    for index in range(n):
        if pos_open:
            sp = int(spread[index])
            if p_direction == 1:
                stop_hit = low[index] <= p_active_stop
                target_hit = p_has_target and high[index] >= p_target
            else:
                ask_high = high[index] + sp * point
                ask_low = low[index] + sp * point
                stop_hit = ask_high >= p_active_stop
                target_hit = p_has_target and ask_low <= p_target

            if stop_hit:
                if p_direction == 1:
                    level_bid = p_active_stop
                    raw_exit = min(level_bid, open_[index])
                else:
                    level_bid = p_active_stop - sp * point
                    raw_exit = max(level_bid, open_[index])
                if target_hit:
                    reason = _STOP_AMBIGUOUS
                elif abs(p_active_stop - p_initial_stop) > 1e-12:
                    reason = _TRAIL
                else:
                    reason = _STOP

                k = trade_count
                out_direction[k] = p_direction
                out_signal_index[k] = p_signal_index
                out_entry_index[k] = p_entry_index
                out_exit_index[k] = index
                out_raw_entry[k] = p_raw_entry
                out_entry_price[k] = p_entry_price
                out_initial_stop[k] = p_initial_stop
                out_target[k] = p_target if p_has_target else np.nan
                out_lot[k] = p_lot
                out_planned_risk[k] = p_planned_risk
                out_raw_exit[k] = raw_exit
                out_reason[k] = reason
                trade_count += 1
                pos_open = False
                continue

            if target_hit:
                raw_exit = p_target if p_direction == 1 else p_target - sp * point
                k = trade_count
                out_direction[k] = p_direction
                out_signal_index[k] = p_signal_index
                out_entry_index[k] = p_entry_index
                out_exit_index[k] = index
                out_raw_entry[k] = p_raw_entry
                out_entry_price[k] = p_entry_price
                out_initial_stop[k] = p_initial_stop
                out_target[k] = p_target
                out_lot[k] = p_lot
                out_planned_risk[k] = p_planned_risk
                out_raw_exit[k] = raw_exit
                out_reason[k] = _TARGET
                trade_count += 1
                pos_open = False
                continue

            if time_exit_minutes >= 0.0:
                elapsed = (time_epoch[index] - time_epoch[p_entry_index]) / 60.0
                if elapsed >= time_exit_minutes:
                    k = trade_count
                    out_direction[k] = p_direction
                    out_signal_index[k] = p_signal_index
                    out_entry_index[k] = p_entry_index
                    out_exit_index[k] = index
                    out_raw_entry[k] = p_raw_entry
                    out_entry_price[k] = p_entry_price
                    out_initial_stop[k] = p_initial_stop
                    out_target[k] = p_target if p_has_target else np.nan
                    out_lot[k] = p_lot
                    out_planned_risk[k] = p_planned_risk
                    out_raw_exit[k] = close[index]
                    out_reason[k] = _TIME
                    trade_count += 1
                    pos_open = False
                    continue

            if atr_trail >= 0.0 and np.isfinite(atr[index]) and atr[index] > 0.0:
                if p_direction == 1:
                    candidate = _round_price_kernel(close[index] - atr_trail * atr[index], digits)
                    p_active_stop = max(p_active_stop, candidate)
                else:
                    candidate = _round_price_kernel(close[index] + atr_trail * atr[index], digits)
                    p_active_stop = min(p_active_stop, candidate)
            continue

        if index == 0:
            continue
        signal_index = index - 1
        direction = int(signals[signal_index])
        if direction == 0:
            continue
        signal_atr = atr[signal_index]
        if not np.isfinite(signal_atr) or signal_atr <= 0.0:
            continue

        raw_bid = open_[index]
        sp = int(spread[index])
        slip_price = slippage_points_per_fill * point
        if direction == 1:
            entry_price = _round_price_kernel(raw_bid + sp * point + slip_price, digits)
            stop_price = _round_price_kernel(entry_price - stop_atr * signal_atr, digits)
            stop_distance = entry_price - stop_price
            has_target = target_r >= 0.0
            target_price = (
                _round_price_kernel(entry_price + target_r * stop_distance, digits)
                if has_target
                else np.nan
            )
        else:
            entry_price = _round_price_kernel(raw_bid - slip_price, digits)
            stop_price = _round_price_kernel(entry_price + stop_atr * signal_atr, digits)
            stop_distance = stop_price - entry_price
            has_target = target_r >= 0.0
            target_price = (
                _round_price_kernel(entry_price - target_r * stop_distance, digits)
                if has_target
                else np.nan
            )

        feasible, lot, planned_risk = _size_for_stop_kernel(
            entry_price,
            stop_price,
            contract_size,
            volume_min,
            volume_step,
            volume_max,
            preferred_risk_usd,
            hard_risk_usd,
            max_lot,
        )
        if not feasible:
            risk_skip_count += 1
            continue

        pos_open = True
        p_direction = direction
        p_signal_index = signal_index
        p_entry_index = index
        p_raw_entry = raw_bid
        p_entry_price = entry_price
        p_initial_stop = stop_price
        p_active_stop = stop_price
        p_target = target_price
        p_has_target = has_target
        p_lot = lot
        p_planned_risk = planned_risk

        if direction == 1:
            stop_hit = low[index] <= p_active_stop
            target_hit = p_has_target and high[index] >= p_target
        else:
            ask_high = high[index] + sp * point
            ask_low = low[index] + sp * point
            stop_hit = ask_high >= p_active_stop
            target_hit = p_has_target and ask_low <= p_target

        if stop_hit:
            if direction == 1:
                raw_exit = min(p_active_stop, open_[index])
            else:
                level_bid = p_active_stop - sp * point
                raw_exit = max(level_bid, open_[index])
            reason = _STOP_AMBIGUOUS if target_hit else _STOP
            k = trade_count
            out_direction[k] = p_direction
            out_signal_index[k] = p_signal_index
            out_entry_index[k] = p_entry_index
            out_exit_index[k] = index
            out_raw_entry[k] = p_raw_entry
            out_entry_price[k] = p_entry_price
            out_initial_stop[k] = p_initial_stop
            out_target[k] = p_target if p_has_target else np.nan
            out_lot[k] = p_lot
            out_planned_risk[k] = p_planned_risk
            out_raw_exit[k] = raw_exit
            out_reason[k] = reason
            trade_count += 1
            pos_open = False
            continue

        if target_hit:
            raw_exit = p_target if direction == 1 else p_target - sp * point
            k = trade_count
            out_direction[k] = p_direction
            out_signal_index[k] = p_signal_index
            out_entry_index[k] = p_entry_index
            out_exit_index[k] = index
            out_raw_entry[k] = p_raw_entry
            out_entry_price[k] = p_entry_price
            out_initial_stop[k] = p_initial_stop
            out_target[k] = p_target
            out_lot[k] = p_lot
            out_planned_risk[k] = p_planned_risk
            out_raw_exit[k] = raw_exit
            out_reason[k] = _TARGET
            trade_count += 1
            pos_open = False
            continue

        if atr_trail >= 0.0 and np.isfinite(atr[index]) and atr[index] > 0.0:
            if direction == 1:
                candidate = _round_price_kernel(close[index] - atr_trail * atr[index], digits)
                p_active_stop = max(p_active_stop, candidate)
            else:
                candidate = _round_price_kernel(close[index] + atr_trail * atr[index], digits)
                p_active_stop = min(p_active_stop, candidate)

    if pos_open and n > 0:
        index = n - 1
        k = trade_count
        out_direction[k] = p_direction
        out_signal_index[k] = p_signal_index
        out_entry_index[k] = p_entry_index
        out_exit_index[k] = index
        out_raw_entry[k] = p_raw_entry
        out_entry_price[k] = p_entry_price
        out_initial_stop[k] = p_initial_stop
        out_target[k] = p_target if p_has_target else np.nan
        out_lot[k] = p_lot
        out_planned_risk[k] = p_planned_risk
        out_raw_exit[k] = close[index]
        out_reason[k] = _END_OF_DATA
        trade_count += 1

    return (
        trade_count,
        risk_skip_count,
        out_direction,
        out_signal_index,
        out_entry_index,
        out_exit_index,
        out_raw_entry,
        out_entry_price,
        out_initial_stop,
        out_target,
        out_lot,
        out_planned_risk,
        out_raw_exit,
        out_reason,
    )


def run_fast_backtest(
    bars: MarketBars,
    signals: Sequence[int],
    symbol: SymbolSpec,
    cost: CostModel,
    risk: RiskModel,
    exit_spec: ExitSpec,
) -> BacktestResult:
    signal_values = np.asarray(signals)
    if signal_values.ndim != 1 or len(signal_values) != len(bars):
        raise ValueError("signals length must match MarketBars")
    if not valid_signals(signal_values):
        raise ValueError("signals must contain only -1, 0, +1")
    signal_array = np.ascontiguousarray(np.asarray(signal_values, dtype=np.int8))

    target_r = -1.0 if exit_spec.target_r is None else float(exit_spec.target_r)
    time_exit = -1.0 if exit_spec.time_exit_minutes is None else float(exit_spec.time_exit_minutes)
    trail = -1.0 if exit_spec.atr_trail is None else float(exit_spec.atr_trail)

    packed = _kernel(
        bars.time_epoch,
        bars.open,
        bars.high,
        bars.low,
        bars.close,
        bars.spread,
        bars.atr,
        signal_array,
        float(symbol.point),
        int(symbol.digits),
        float(symbol.contract_size),
        float(symbol.volume_min),
        float(symbol.volume_step),
        float(symbol.volume_max),
        float(cost.commission_round_trip_per_lot),
        float(cost.slippage_points_per_fill),
        float(risk.preferred_risk_usd),
        float(risk.hard_risk_usd),
        float(risk.max_lot),
        float(exit_spec.stop_atr),
        target_r,
        time_exit,
        trail,
    )

    (
        trade_count,
        risk_skip_count,
        directions,
        signal_indices,
        entry_indices,
        exit_indices,
        raw_entries,
        entry_prices,
        initial_stops,
        targets,
        lots,
        planned_risks,
        raw_exits,
        reasons,
    ) = packed

    trades = []
    for i in range(int(trade_count)):
        target = None if np.isnan(targets[i]) else float(targets[i])
        position = _Position(
            direction=int(directions[i]),
            signal_index=int(signal_indices[i]),
            entry_index=int(entry_indices[i]),
            raw_entry_bid=float(raw_entries[i]),
            entry_price=float(entry_prices[i]),
            initial_stop=float(initial_stops[i]),
            active_stop=float(initial_stops[i]),
            target_price=target,
            lot=float(lots[i]),
            planned_risk_usd=float(planned_risks[i]),
        )
        trades.append(
            _finalize_trade(
                position,
                bars,
                int(exit_indices[i]),
                float(raw_exits[i]),
                _REASON[int(reasons[i])],
                symbol,
                cost,
            )
        )

    return BacktestResult(tuple(trades), risk_skip_count=int(risk_skip_count))
