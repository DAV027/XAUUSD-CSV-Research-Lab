import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.breakout import (
    bollinger_expansion,
    compression_breakout,
    nbar_breakout,
    nbar_failed_breakout,
    range_expansion,
)
from xau_lab.strategies.registry import get_strategy


def context(open_, high, low, close):
    close = np.asarray(close, dtype=float)
    n = len(close)
    return StrategyContext(
        open=np.asarray(open_, dtype=float),
        high=np.asarray(high, dtype=float),
        low=np.asarray(low, dtype=float),
        close=close,
        spread=np.zeros(n),
        atr14=np.ones(n),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={},
    )


def test_nbar_breakout_excludes_signal_bar_from_reference_high():
    ctx = context(
        [9.5, 9.6, 9.7, 10.0],
        [10.0, 10.0, 10.0, 11.5],
        [9.0, 9.1, 9.2, 9.8],
        [9.7, 9.8, 9.9, 11.0],
    )
    sig = nbar_breakout(ctx, {"lookback": 3})
    assert sig.tolist() == [0, 0, 0, 1]


def test_failed_breakout_reverses_after_sweep_and_reclaim():
    ctx = context(
        [9.5, 9.6, 9.7, 10.0],
        [10.0, 10.0, 10.0, 10.8],
        [9.0, 9.1, 9.2, 9.7],
        [9.7, 9.8, 9.9, 9.8],
    )
    sig = nbar_failed_breakout(ctx, {"lookback": 3})
    assert sig[-1] == -1


def test_range_expansion_uses_prior_median_and_candle_direction():
    ctx = context(
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        [10.5, 10.5, 10.5, 10.5, 10.5, 13.0],
        [9.5, 9.5, 9.5, 9.5, 9.5, 9.5],
        [10.1, 10.1, 10.1, 10.1, 10.1, 12.5],
    )
    sig = range_expansion(ctx, {"median_lookback": 5, "range_multiple": 2.0})
    assert sig[-1] == 1


def test_bollinger_expansion_and_compression_breakout_emit_on_completed_bar():
    close = np.array([10.0, 10.1, 9.9, 10.0, 10.05, 10.0, 10.0, 10.0, 10.0, 11.5])
    ctx = context(close - 0.05, close + 0.1, close - 0.1, close)
    bb = bollinger_expansion(
        ctx,
        {"lookback": 5, "std_multiplier": 1.0, "bandwidth_percentile": 0.5},
    )
    comp = compression_breakout(
        ctx,
        {"compression_lookback": 5, "compression_percentile": 0.4, "breakout_lookback": 3},
    )
    assert bb[-1] == 1
    assert comp[-1] == 1


def test_breakout_family_is_registered():
    for name in (
        "nbar_breakout",
        "nbar_failed_breakout",
        "range_expansion",
        "bollinger_expansion",
        "compression_breakout",
    ):
        assert get_strategy(name).family == "breakout"
