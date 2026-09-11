from xau_lab.backtest.models import Trade
from xau_lab.metrics.performance import summarize_pnl, summarize_trades


def test_profit_factor_drawdown_expectancy_and_streak():
    out = summarize_pnl([10.0, -5.0, -5.0, 20.0, -10.0], starting_equity=100.0)
    assert out["gross_profit"] == 30.0
    assert out["gross_loss"] == -20.0
    assert out["profit_factor"] == 1.5
    assert out["expectancy_usd"] == 2.0
    assert out["max_loss_streak"] == 2
    assert out["max_drawdown_usd"] == 10.0
    assert out["max_drawdown_pct"] == 10.0 / 110.0 * 100.0


def _trade(net_pnl, direction, broker_date, hold_minutes, pnl_r):
    return Trade(
        experiment_id="exp",
        parameter_set_id="params",
        fingerprint="fp",
        signal_time=1,
        entry_time=2,
        direction=direction,
        signal_price=100.0,
        entry_price=100.0,
        stop_price=99.0 if direction == 1 else 101.0,
        target_price=None,
        exit_time=3,
        exit_price=100.0,
        exit_reason="time",
        lot_size=0.01,
        planned_risk_usd=10.0,
        risk_R=1.0,
        gross_pnl=net_pnl,
        spread_cost=0.0,
        commission=0.0,
        slippage_cost=0.0,
        net_pnl=net_pnl,
        pnl_R=pnl_r,
        hold_minutes=hold_minutes,
        broker_date=broker_date,
    )


def test_trade_summary_covers_required_daily_temporal_side_and_concentration_metrics():
    trades = [
        _trade(10.0, 1, "2026-01-01", 10, 1.0),
        _trade(-5.0, -1, "2026-01-01", 20, -0.5),
        _trade(20.0, 1, "2026-02-01", 30, 2.0),
        _trade(-10.0, 1, "2026-02-01", 40, -1.0),
    ]
    out = summarize_trades(trades, starting_equity=100.0)
    assert out["completed_trades"] == 4
    assert out["wins"] == 2
    assert out["losses"] == 2
    assert out["win_rate"] == 0.5
    assert out["gross_profit"] == 30.0
    assert out["gross_loss"] == -15.0
    assert out["profit_factor"] == 2.0
    assert out["after_cost_profit"] == 15.0
    assert out["expectancy_usd"] == 3.75
    assert out["expectancy_R"] == 0.375
    assert out["max_drawdown_usd"] == 10.0
    assert out["max_drawdown_pct"] == 8.0
    assert out["max_loss_streak"] == 1
    assert out["profit_per_active_day"] == 7.5
    assert out["median_profit_per_active_day"] == 7.5
    assert out["positive_year_fraction"] == 1.0
    assert out["positive_month_fraction"] == 1.0
    assert out["worst_month"] == 5.0
    assert out["long_PF"] == 3.0
    assert out["short_PF"] == 0.0
    assert out["trades_per_active_day"] == 2.0
    assert out["median_hold_minutes"] == 25.0
    assert out["top_5_trade_profit_fraction"] == 1.0
    assert out["best_month_profit_fraction"] == 10.0 / 30.0


def test_undefined_profit_factor_and_empty_side_metrics_are_null_not_zero():
    out = summarize_trades([_trade(10.0, 1, "2026-01-01", 5, 1.0)], starting_equity=100.0)
    assert out["profit_factor"] is None
    assert out["long_PF"] is None
    assert out["short_PF"] is None
