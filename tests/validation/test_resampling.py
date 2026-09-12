from __future__ import annotations

from dataclasses import dataclass

from xau_lab.validation.resampling import (
    RESAMPLING_SEED,
    block_bootstrap_daily,
    bootstrap_days,
    shuffle_trade_order,
    top_trade_removal,
    trade_order_monte_carlo,
)


@dataclass(frozen=True)
class FakeTrade:
    broker_date: str
    net_pnl: float


def test_day_bootstrap_is_seeded_and_reproducible():
    pnl = [3.0, -1.0, 2.0, -2.0, 4.0]
    first = bootstrap_days(pnl, n=250, seed=RESAMPLING_SEED)
    second = bootstrap_days(pnl, n=250, seed=RESAMPLING_SEED)
    assert first == second
    assert first["seed"] == 9_216_000
    assert first["n"] == 250


def test_approved_block_bootstrap_aggregates_whole_broker_days_before_resampling():
    trades = [
        FakeTrade("2026-01-02", 3.0),
        FakeTrade("2026-01-02", -1.0),
        FakeTrade("2026-01-03", -1.0),
    ]
    first = block_bootstrap_daily(trades, n=250, seed=RESAMPLING_SEED)
    second = block_bootstrap_daily(trades, n=250, seed=RESAMPLING_SEED)
    assert first == second
    assert first["days_per_sample"] == 2
    assert first["source_daily_pnl"] == [2.0, -1.0]


def test_constant_day_path_has_exact_bootstrap_quantiles():
    out = bootstrap_days([1.0, 1.0, 1.0], n=100, seed=1)
    assert out["mean_daily_p5"] == 1.0
    assert out["mean_daily_p50"] == 1.0
    assert out["mean_daily_p95"] == 1.0
    assert out["total_pnl_p5"] == 3.0
    assert out["total_pnl_p50"] == 3.0
    assert out["total_pnl_p95"] == 3.0


def test_trade_order_shuffle_is_seeded_and_labeled_as_sequence_only():
    pnl = [5.0, -3.0, 2.0, -4.0, 6.0]
    first = trade_order_monte_carlo(pnl, n=250, seed=RESAMPLING_SEED)
    second = trade_order_monte_carlo(pnl, n=250, seed=RESAMPLING_SEED)
    assert first == second
    assert first == shuffle_trade_order(pnl, n=250, seed=RESAMPLING_SEED)
    assert first["label"] == "sequence_only_not_entry_edge_proof"
    assert first["total_pnl"] == sum(pnl)
    assert 0.0 <= first["max_drawdown_p5"] <= first["max_drawdown_p95"]


def test_top_trade_removal_removes_only_profitable_trades_and_recomputes_net_pf_and_sensitivity():
    pnl = [10.0, 8.0, 5.0, -4.0, -3.0, 2.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    out = top_trade_removal(pnl, counts=(1, 5, 10))
    assert out["baseline_net_profit"] == 21.0
    assert out["baseline_profit_factor"] == 29.0 / 8.0

    by_count = {row["requested_count"]: row for row in out["scenarios"]}
    assert by_count[1]["removed_count"] == 1
    assert by_count[1]["net_profit"] == 11.0
    assert by_count[1]["profit_factor"] == 19.0 / 8.0
    assert by_count[1]["net_profit_sensitivity_pct"] == (10.0 / 21.0) * 100.0
    assert by_count[1]["profit_factor_sensitivity_pct"] == ((29.0 / 8.0 - 19.0 / 8.0) / (29.0 / 8.0)) * 100.0
    assert not by_count[1]["turns_negative"]

    assert by_count[5]["removed_count"] == 5
    assert by_count[5]["net_profit"] == -5.0
    assert by_count[5]["profit_factor"] == 3.0 / 8.0
    assert by_count[5]["turns_negative"]

    # Only eight profitable trades exist, so requesting ten never removes losses.
    assert by_count[10]["removed_count"] == 8
    assert by_count[10]["net_profit"] == -8.0
    assert by_count[10]["profit_factor"] == 0.0
    assert by_count[10]["turns_negative"]


def test_invalid_resampling_inputs_fail_closed():
    for fn in (bootstrap_days, shuffle_trade_order):
        try:
            fn([], n=10, seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError("empty input should fail")
        try:
            fn([1.0], n=0, seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError("nonpositive resample count should fail")
