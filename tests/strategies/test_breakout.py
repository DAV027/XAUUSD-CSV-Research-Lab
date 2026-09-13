import numpy as np

from xau_lab.strategies import breakout as breakout_module
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


def _finite(values: np.ndarray) -> bool:
    return len(values) > 0 and np.isfinite(values).all()


def _compression_breakout_reference(ctx: StrategyContext, params: dict) -> np.ndarray:
    """Literal pre-optimization behavior used only as a semantic oracle."""
    compression_lookback = int(params["compression_lookback"])
    compression_percentile = float(params["compression_percentile"])
    breakout_lookback = int(params["breakout_lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    ranges = np.asarray(ctx.high - ctx.low, dtype=np.float64)
    warmup = max(compression_lookback, breakout_lookback)
    for i in range(warmup, len(ctx)):
        compressed = ranges[i - compression_lookback : i]
        history = ranges[:i]
        breakout_highs = ctx.high[i - breakout_lookback : i]
        breakout_lows = ctx.low[i - breakout_lookback : i]
        if not (
            _finite(compressed)
            and _finite(history)
            and _finite(breakout_highs)
            and _finite(breakout_lows)
        ):
            continue
        threshold = float(np.quantile(history, compression_percentile))
        if float(np.mean(compressed)) > threshold:
            continue
        close = float(ctx.close[i])
        if not np.isfinite(close):
            continue
        if close > float(np.max(breakout_highs)):
            out[i] = 1
        elif close < float(np.min(breakout_lows)):
            out[i] = -1
    return out


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


def test_expanding_linear_quantiles_match_numpy_default_linear_prefixes_exactly():
    values = np.array([3.0, 1.0, 1.0, 8.0, 5.0, 2.0, 13.0, 8.0], dtype=np.float64)
    q = 0.236588625
    expected = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(1, len(values)):
        expected[i] = np.quantile(values[:i], q)

    actual = breakout_module._expanding_linear_quantiles(values, q)

    assert np.allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_expanding_linear_quantiles_fail_closed_after_nonfinite_history():
    values = np.array([2.0, 1.0, np.nan, 4.0, 3.0], dtype=np.float64)
    actual = breakout_module._expanding_linear_quantiles(values, 0.4)

    assert np.isnan(actual[0])
    assert actual[1] == 2.0
    assert actual[2] == np.quantile(values[:2], 0.4)
    assert np.isnan(actual[3:]).all()


def test_compression_breakout_matches_preoptimization_reference_across_randomized_inputs():
    rng = np.random.default_rng(9_215_777)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.3, size=160))
    open_ = close - rng.normal(0.0, 0.1, size=160)
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.7, size=160)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.7, size=160)
    ctx = context(open_, high, low, close)

    for params in (
        {"compression_lookback": 10, "compression_percentile": 0.05, "breakout_lookback": 2},
        {"compression_lookback": 36, "compression_percentile": 0.236588625, "breakout_lookback": 4},
        {"compression_lookback": 80, "compression_percentile": 0.40, "breakout_lookback": 25},
    ):
        expected = _compression_breakout_reference(ctx, params)
        actual = compression_breakout(ctx, params)
        assert np.array_equal(actual, expected)


def test_compression_breakout_matches_reference_with_nonfinite_and_repeated_ranges():
    close = np.array(
        [10.0, 10.0, 10.1, 10.1, 10.2, 10.2, 10.3, 10.3, 10.4, 10.5, 10.6, 10.7],
        dtype=float,
    )
    open_ = close - 0.05
    high = close + np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2, 0.2, 0.2])
    low = close - np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2, 0.2, 0.2])
    high[7] = np.nan
    ctx = context(open_, high, low, close)
    params = {"compression_lookback": 4, "compression_percentile": 0.4, "breakout_lookback": 3}

    assert np.array_equal(
        compression_breakout(ctx, params),
        _compression_breakout_reference(ctx, params),
    )


def test_breakout_family_is_registered():
    for name in (
        "nbar_breakout",
        "nbar_failed_breakout",
        "range_expansion",
        "bollinger_expansion",
        "compression_breakout",
    ):
        assert get_strategy(name).family == "breakout"
