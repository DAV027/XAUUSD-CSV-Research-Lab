import numpy as np
import pytest
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.breakout import compression_breakout


def _finite(values):
    return len(values) > 0 and np.isfinite(values).all()


def reference_compression_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
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
        if not (_finite(compressed) and _finite(history) and _finite(breakout_highs) and _finite(breakout_lows)):
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



def make_context(n, seed=41, bad=None):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(size=n))
    widths = rng.choice([0.125, 0.25, 0.5, 1., 2., 4.], n)
    high = close + widths
    low = close - widths
    if bad is not None and n:
        high[n // 3] = bad
    return StrategyContext(close.copy(), high, low, close, np.zeros(n),
                           np.ones(n), np.arange(n, dtype=np.int64), {})


@pytest.mark.parametrize('n', [0, 1, 9, 36, 37, 257, 1024])
@pytest.mark.parametrize('q', [0., 0.05, 0.236588625, 0.4, 0.5, 1.])
@pytest.mark.parametrize('bad', [None, np.nan, np.inf, -np.inf])
def test_exact_legacy_signals(n, q, bad):
    ctx = make_context(n, bad=bad)
    params = dict(compression_lookback=36, compression_percentile=q, breakout_lookback=4)
    np.testing.assert_array_equal(compression_breakout(ctx, params),
                                  reference_compression_breakout(ctx, params))


def test_expanding_quantile_work_is_not_quadratic(monkeypatch):
    ctx = make_context(2000)
    original = np.quantile
    scanned = 0
    def measured(values, *args, **kwargs):
        nonlocal scanned
        scanned += len(values)
        return original(values, *args, **kwargs)
    monkeypatch.setattr(np, 'quantile', measured)
    actual = compression_breakout(ctx, dict(compression_lookback=36,
        compression_percentile=0.236588625, breakout_lookback=4))
    assert len(actual) == len(ctx)
    assert scanned <= 4 * len(ctx), 'expanding history is repeatedly rescanned'


def test_future_bars_do_not_change_prefix_signals():
    ctx = make_context(512)
    prefix = StrategyContext(*(getattr(ctx, name)[:256] for name in
        ('open', 'high', 'low', 'close', 'spread', 'atr14', 'time_epoch')), {})
    params = dict(compression_lookback=36, compression_percentile=0.236588625, breakout_lookback=4)
    np.testing.assert_array_equal(compression_breakout(ctx, params)[:256],
                                  compression_breakout(prefix, params))


def test_strategy_validation_uses_bounded_temporary_memory(monkeypatch):
    from xau_lab.strategies.base import StrategyDefinition
    ctx = make_context(10000)
    original = np.isin
    def bounded(values, *args, **kwargs):
        assert len(values) <= 4096, "full-signal validation temporary"
        return original(values, *args, **kwargs)
    monkeypatch.setattr(np, 'isin', bounded)
    definition = StrategyDefinition('test', 'test', lambda c, p: np.zeros(len(c)), {})
    np.testing.assert_array_equal(definition.generate(ctx, {}), np.zeros(len(ctx), dtype=np.int8))


@pytest.mark.parametrize('q', [0., .05, .236588625, .4, .5, 1.])
def test_quantile_values_match_numpy_exactly(q):
    from xau_lab.strategies.order_statistics import expanding_quantiles
    rng = np.random.default_rng(177)
    values = rng.normal(size=1001)
    actual = expanding_quantiles(values, q)
    expected = np.array([np.nan] + [np.quantile(values[:i], q) for i in range(1, len(values))])
    np.testing.assert_array_equal(actual, expected)


def test_finite_extreme_ranges_preserve_legacy_overflow_behavior():
    high = np.array([-1e308] * 20 + [1e308] * 60)
    low = np.zeros(80)
    close = np.full(80, 1.5e308)
    ctx = StrategyContext(close, high, low, close, np.zeros(80), np.ones(80), np.arange(80), {})
    params = dict(compression_lookback=10, compression_percentile=.49, breakout_lookback=4)
    with np.errstate(over='ignore', invalid='ignore'):
        np.testing.assert_array_equal(compression_breakout(ctx, params), reference_compression_breakout(ctx, params))

