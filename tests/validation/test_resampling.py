from __future__ import annotations

from xau_lab.validation.resampling import (
    RESAMPLING_SEED,
    bootstrap_days,
    shuffle_trade_order,
    top_trade_removal,
)


def test_day_bootstrap_is_seeded_and_reproducible():
    pnl = [3.0, -1.0, 2.0, -2.0, 4.0]
    first = bootstrap_days(pnl, n=250, seed=RESAMPLING_SEED)
    second = bootstrap_days(pnl, n=250, seed=RESAMPLING_SEED)
    assert first == second
    assert first["seed"] == 9_216_000
    assert first["n"] == 250


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
    first = shuffle_trade_order(pnl, n=250, seed=RESAMPLING_SEED)
    second = shuffle_trade_order(pnl, n=250, seed=RESAMPLING_SEED)
    assert first == second
    assert first["label"] == "sequence_only_not_entry_edge_proof"
    assert first["total_pnl"] == sum(pnl)
    assert 0.0 <= first["max_drawdown_p5"] <= first["max_drawdown_p95"]


def test_top_trade_removal_reports_remaining_net_and_negative_flip():
    pnl = [10.0, 8.0, 5.0, -4.0, -3.0, 2.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    out = top_trade_removal(pnl, counts=(1, 5, 10))
    assert out["baseline_net_profit"] == 21.0
    by_count = {row["removed_count"]: row for row in out["scenarios"]}
    assert by_count[1]["net_profit"] == 11.0
    assert not by_count[1]["turns_negative"]
    assert by_count[5]["net_profit"] < 0.0
    assert by_count[5]["turns_negative"]
    assert by_count[10]["net_profit"] < 0.0


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
