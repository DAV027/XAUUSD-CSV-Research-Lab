from __future__ import annotations

from collections import defaultdict
from datetime import date
from statistics import median
from typing import Iterable, Sequence

from xau_lab.backtest.models import Trade


def _profit_factor(values: Sequence[float]) -> float | None:
    if not values:
        return None
    gross_profit = sum(value for value in values if value > 0)
    gross_loss = sum(value for value in values if value < 0)
    if gross_loss == 0:
        return None
    return float(gross_profit / abs(gross_loss))


def _max_loss_streak(values: Sequence[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _drawdown(values: Sequence[float], starting_equity: float) -> tuple[float, float]:
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")
    equity = float(starting_equity)
    peak = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        drawdown = peak - equity
        drawdown_pct = (drawdown / peak * 100.0) if peak > 0 else 0.0
        max_dd_usd = max(max_dd_usd, drawdown)
        max_dd_pct = max(max_dd_pct, drawdown_pct)
    return float(max_dd_usd), float(max_dd_pct)


def summarize_pnl(pnl: Iterable[float], starting_equity: float) -> dict[str, float | int | None]:
    values = [float(value) for value in pnl]
    gross_profit = float(sum(value for value in values if value > 0))
    gross_loss = float(sum(value for value in values if value < 0))
    max_dd_usd, max_dd_pct = _drawdown(values, starting_equity)
    wins = sum(value > 0 for value in values)
    losses = sum(value < 0 for value in values)
    return {
        "completed_trades": len(values),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(values)) if values else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": _profit_factor(values),
        "after_cost_profit": float(sum(values)),
        "expectancy_usd": (float(sum(values)) / len(values)) if values else None,
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "max_loss_streak": _max_loss_streak(values),
    }


def summarize_trades(trades: Iterable[Trade], starting_equity: float) -> dict[str, float | int | None]:
    trade_list = list(trades)
    net = [float(trade.net_pnl) for trade in trade_list]
    out: dict[str, float | int | None] = dict(summarize_pnl(net, starting_equity))

    pnl_r_values = [float(trade.pnl_R) for trade in trade_list if trade.pnl_R is not None]
    out["expectancy_R"] = (sum(pnl_r_values) / len(pnl_r_values)) if pnl_r_values else None

    daily: dict[str, float] = defaultdict(float)
    monthly: dict[str, float] = defaultdict(float)
    yearly: dict[int, float] = defaultdict(float)
    for trade in trade_list:
        parsed = date.fromisoformat(trade.broker_date)
        daily[trade.broker_date] += float(trade.net_pnl)
        monthly[f"{parsed.year:04d}-{parsed.month:02d}"] += float(trade.net_pnl)
        yearly[parsed.year] += float(trade.net_pnl)

    active_days = len(daily)
    daily_values = list(daily.values())
    out["profit_per_active_day"] = (sum(net) / active_days) if active_days else None
    out["median_profit_per_active_day"] = float(median(daily_values)) if daily_values else None
    out["positive_year_fraction"] = (
        sum(value > 0 for value in yearly.values()) / len(yearly) if yearly else None
    )
    out["positive_month_fraction"] = (
        sum(value > 0 for value in monthly.values()) / len(monthly) if monthly else None
    )
    out["worst_month"] = float(min(monthly.values())) if monthly else None

    long_values = [float(trade.net_pnl) for trade in trade_list if trade.direction == 1]
    short_values = [float(trade.net_pnl) for trade in trade_list if trade.direction == -1]
    out["long_PF"] = _profit_factor(long_values)
    out["short_PF"] = _profit_factor(short_values)
    out["trades_per_active_day"] = (len(trade_list) / active_days) if active_days else None
    out["median_hold_minutes"] = (
        float(median([float(trade.hold_minutes) for trade in trade_list])) if trade_list else None
    )

    gross_profit = float(out["gross_profit"] or 0.0)
    positive_trades = sorted((value for value in net if value > 0), reverse=True)
    out["top_5_trade_profit_fraction"] = (
        sum(positive_trades[:5]) / gross_profit if gross_profit > 0 else None
    )
    out["best_month_profit_fraction"] = (
        max(0.0, max(monthly.values())) / gross_profit if monthly and gross_profit > 0 else None
    )

    out["net_profit"] = float(sum(net))
    out["commission_cost"] = float(sum(trade.commission for trade in trade_list))
    out["spread_cost"] = float(sum(trade.spread_cost for trade in trade_list))
    out["slippage_cost"] = float(sum(trade.slippage_cost for trade in trade_list))
    return out
