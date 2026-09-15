from __future__ import annotations

import math

import numpy as np
from numba import njit

from xau_lab.backtest.models import CostModel, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.packed import PackedBacktestResult


@njit(cache=True)
def _trade_economics(
    i, direction, entry_index, exit_index, raw_entry, lot, planned_risk, raw_exit,
    spread, time_epoch, point, contract_size, commission_per_lot, slippage_points,
):
    """Mirror _finalize_trade's operations and evaluation order exactly."""
    d = int(direction[i])
    entry = int(entry_index[i])
    exit_ = int(exit_index[i])
    size = float(lot[i])
    gross = d * (float(raw_exit[i]) - float(raw_entry[i])) * contract_size * size
    spread_points = int(spread[entry]) if d == 1 else int(spread[exit_])
    spread_cost = float(spread_points) * point * contract_size * size
    slippage_cost = 2.0 * slippage_points * point * contract_size * size
    commission = commission_per_lot * size
    net = gross - spread_cost - slippage_cost - commission
    pnl_r = net / float(planned_risk[i]) if float(planned_risk[i]) > 0.0 else np.nan
    hold = (int(time_epoch[exit_]) - int(time_epoch[entry])) / 60.0
    return net, pnl_r, hold, commission, spread_cost, slippage_cost


@njit(cache=True)
def _sum_add(totals, index, value):
    """CPython 3.12 float sum, including its ordered compensation updates.

    See CPython v3.12.0 Python/bltinmodule.c, builtin_sum_impl.
    Calendar balances and equity deliberately do not use this helper.
    """
    total = totals[index, 0]
    updated = total + value
    if abs(total) >= abs(value):
        totals[index, 1] += (total - updated) + value
    else:
        totals[index, 1] += (value - updated) + total
    totals[index, 0] = updated


@njit(cache=True)
def _sum_result(totals, index):
    total = totals[index, 0]
    compensation = totals[index, 1]
    if compensation != 0.0 and math.isfinite(compensation):
        total += compensation
    return total


@njit(cache=True)
def _median_exact(values, count):
    if count <= 0:
        return np.nan
    ordered = np.sort(values[:count].copy())
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0


@njit(cache=True)
def _pf(gross_profit, gross_loss, count):
    if count == 0 or gross_loss == 0.0:
        return np.nan
    return gross_profit / abs(gross_loss)


@njit(cache=True)
def _summary_kernel(
    trade_count, direction, entry_index, exit_index, raw_entry, lot, planned_risk,
    raw_exit, spread, time_epoch, point, contract_size, commission_per_lot,
    slippage_points, starting_equity, day_id, month_id, year_id,
    day_group_count, month_group_count, year_group_count,
):
    hold_values = np.empty(trade_count, dtype=np.float64)
    daily = np.zeros(day_group_count, dtype=np.float64)
    monthly = np.zeros(month_group_count, dtype=np.float64)
    yearly = np.zeros(year_group_count, dtype=np.float64)
    daily_active = np.zeros(day_group_count, dtype=np.bool_)
    monthly_active = np.zeros(month_group_count, dtype=np.bool_)
    yearly_active = np.zeros(year_group_count, dtype=np.bool_)
    top5 = np.full(5, -np.inf, dtype=np.float64)
    # Constant-size (sum, compensation) pairs, never per-trade economics arrays.
    profit, loss, net_total, r_total = 0, 1, 2, 3
    commission_total, spread_total, slippage_total = 4, 5, 6
    long_profit, long_loss, short_profit, short_loss, top_total = 7, 8, 9, 10, 11
    totals = np.zeros((12, 2), dtype=np.float64)
    wins = losses = long_count = short_count = r_count = 0
    loss_streak = max_loss_streak = 0
    equity = peak = starting_equity
    max_dd_usd = max_dd_pct = 0.0

    for i in range(trade_count):
        net, pnl_r, hold, commission, spread_cost, slippage_cost = _trade_economics(
            i, direction, entry_index, exit_index, raw_entry, lot, planned_risk,
            raw_exit, spread, time_epoch, point, contract_size, commission_per_lot,
            slippage_points,
        )
        if net > 0.0:
            wins += 1
            _sum_add(totals, profit, net)
        elif net < 0.0:
            losses += 1
            _sum_add(totals, loss, net)
        if net < 0.0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0
        equity += net
        peak = max(peak, equity)
        drawdown = peak - equity
        drawdown_pct = drawdown / peak * 100.0 if peak > 0.0 else 0.0
        max_dd_usd = max(max_dd_usd, drawdown)
        max_dd_pct = max(max_dd_pct, drawdown_pct)
        _sum_add(totals, net_total, net)
        if planned_risk[i] > 0.0:
            _sum_add(totals, r_total, pnl_r)
            r_count += 1
        _sum_add(totals, commission_total, commission)
        _sum_add(totals, spread_total, spread_cost)
        _sum_add(totals, slippage_total, slippage_cost)
        if direction[i] == 1:
            long_count += 1
            if net > 0.0:
                _sum_add(totals, long_profit, net)
            elif net < 0.0:
                _sum_add(totals, long_loss, net)
        elif direction[i] == -1:
            short_count += 1
            if net > 0.0:
                _sum_add(totals, short_profit, net)
            elif net < 0.0:
                _sum_add(totals, short_loss, net)
        hold_values[i] = hold
        entry = entry_index[i]
        day, month, year = day_id[entry], month_id[entry], year_id[entry]
        daily[day] += net
        monthly[month] += net
        yearly[year] += net
        daily_active[day] = True
        monthly_active[month] = True
        yearly_active[year] = True
        if net > 0.0:
            for slot in range(5):
                if net > top5[slot]:
                    for shift in range(4, slot, -1):
                        top5[shift] = top5[shift - 1]
                    top5[slot] = net
                    break

    active_days = 0
    compact_daily = np.empty(day_group_count, dtype=np.float64)
    for group in range(day_group_count):
        if daily_active[group]:
            compact_daily[active_days] = daily[group]
            active_days += 1
    active_months = positive_months = 0
    worst_month = np.inf
    best_month = -np.inf
    for group in range(month_group_count):
        if monthly_active[group]:
            active_months += 1
            positive_months += int(monthly[group] > 0.0)
            worst_month = min(worst_month, monthly[group])
            best_month = max(best_month, monthly[group])
    active_years = positive_years = 0
    for group in range(year_group_count):
        if yearly_active[group]:
            active_years += 1
            positive_years += int(yearly[group] > 0.0)
    for slot in range(5):
        if math.isfinite(top5[slot]):
            _sum_add(totals, top_total, top5[slot])
    gross_profit = _sum_result(totals, profit)
    gross_loss = _sum_result(totals, loss)
    sum_net = _sum_result(totals, net_total)
    # Two tuple expressions avoid Python 3.12's large-tuple LIST_TO_TUPLE
    # bytecode, which Numba does not support; the returned tuple stays flat.
    return (
        trade_count, wins, losses, wins / trade_count if trade_count else np.nan,
        gross_profit, gross_loss, _pf(gross_profit, gross_loss, trade_count),
        sum_net, sum_net / trade_count if trade_count else np.nan,
        max_dd_usd, max_dd_pct, max_loss_streak,
        _sum_result(totals, r_total) / r_count if r_count else np.nan,
        sum_net / active_days if active_days else np.nan,
        _median_exact(compact_daily, active_days),
        positive_years / active_years if active_years else np.nan,
        positive_months / active_months if active_months else np.nan,
        active_months, worst_month if active_months else np.nan,
    ) + (
        _pf(_sum_result(totals, long_profit), _sum_result(totals, long_loss), long_count),
        _pf(_sum_result(totals, short_profit), _sum_result(totals, short_loss), short_count),
        long_count, short_count, trade_count / active_days if active_days else np.nan,
        _median_exact(hold_values, trade_count),
        _sum_result(totals, top_total) / gross_profit if gross_profit > 0.0 else np.nan,
        max(0.0, best_month) / gross_profit if active_months and gross_profit > 0.0 else np.nan,
        sum_net, _sum_result(totals, commission_total),
        _sum_result(totals, spread_total), _sum_result(totals, slippage_total),
    )


def _optional(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _calendar_array(values: np.ndarray, name: str, size: int) -> tuple[np.ndarray, int]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != size:
        raise ValueError(f"{name} must be one-dimensional with market bars length")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer group IDs")
    count = int(array.max()) + 1 if size else 0
    if size and (int(array.min()) < 0 or count > size):
        raise ValueError(f"{name} group IDs must be nonnegative and less than market bars length")
    return array, count


def summarize_packed_backtest(
    packed: PackedBacktestResult, bars: MarketBars, symbol: SymbolSpec,
    cost: CostModel, risk: RiskModel, broker_day_id: np.ndarray,
    broker_month_id: np.ndarray, broker_year_id: np.ndarray,
) -> dict[str, float | int | None]:
    """Summarize packed rows with the exact detailed CPython 3.12 semantics."""
    for value in (packed.trade_count, packed.risk_skip_count):
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("packed counts must be nonnegative integers")
    # Arrays can be mutated after construction; validate before unsafe compiled reads.
    packed.__post_init__()
    for name in ("entry_index", "exit_index"):
        indices = np.asarray(getattr(packed, name))
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"packed {name} must contain integer indices")
        used = indices[:packed.trade_count]
        if len(used) and (int(used.min()) < 0 or int(used.max()) >= len(bars)):
            raise ValueError(f"packed {name} index exceeds market bars bounds")
    day, day_count = _calendar_array(broker_day_id, "broker_day_id", len(bars))
    month, month_count = _calendar_array(broker_month_id, "broker_month_id", len(bars))
    year, year_count = _calendar_array(broker_year_id, "broker_year_id", len(bars))
    (
        completed, wins, losses, win_rate, gross_profit, gross_loss, profit_factor,
        after_cost_profit, expectancy_usd, max_dd_usd, max_dd_pct, max_loss_streak,
        expectancy_r, profit_per_day, median_day, positive_year, positive_month,
        active_months, worst_month, long_pf, short_pf, long_trades, short_trades,
        trades_per_day, median_hold, top5_fraction, best_month_fraction, net_profit,
        commission_cost, spread_cost, slippage_cost,
    ) = _summary_kernel(
        int(packed.trade_count), packed.direction, packed.entry_index, packed.exit_index,
        packed.raw_entry, packed.lot, packed.planned_risk, packed.raw_exit,
        bars.spread, bars.time_epoch, float(symbol.point), float(symbol.contract_size),
        float(cost.commission_round_trip_per_lot), float(cost.slippage_points_per_fill),
        float(risk.account_equity), day, month, year, day_count, month_count, year_count,
    )
    return {
        "completed_trades": int(completed), "wins": int(wins), "losses": int(losses),
        "win_rate": _optional(win_rate), "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss), "profit_factor": _optional(profit_factor),
        "after_cost_profit": float(after_cost_profit), "expectancy_usd": _optional(expectancy_usd),
        "max_drawdown_usd": float(max_dd_usd), "max_drawdown_pct": float(max_dd_pct),
        "max_loss_streak": int(max_loss_streak), "expectancy_R": _optional(expectancy_r),
        "profit_per_active_day": _optional(profit_per_day),
        "median_profit_per_active_day": _optional(median_day),
        "positive_year_fraction": _optional(positive_year),
        "positive_month_fraction": _optional(positive_month), "active_months": int(active_months),
        "worst_month": _optional(worst_month), "long_PF": _optional(long_pf),
        "short_PF": _optional(short_pf), "long_trades": int(long_trades),
        "short_trades": int(short_trades), "trades_per_active_day": _optional(trades_per_day),
        "median_hold_minutes": _optional(median_hold),
        "top_5_trade_profit_fraction": _optional(top5_fraction),
        "best_month_profit_fraction": _optional(best_month_fraction),
        "net_profit": float(net_profit), "commission_cost": float(commission_cost),
        "spread_cost": float(spread_cost), "slippage_cost": float(slippage_cost),
    }
