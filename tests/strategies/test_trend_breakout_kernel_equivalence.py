from __future__ import annotations

import numpy as np
import pytest

from tests.strategies.reference_trend_breakout_signals import (
    reference_bollinger_expansion,
    reference_compression_breakout,
    reference_efficiency_trend,
    reference_ema_slope,
    reference_nbar_breakout,
    reference_nbar_failed_breakout,
    reference_range_expansion,
    reference_regression_slope,
    reference_roc_momentum,
    reference_sma_slope,
)
from tests.strategies.test_signal_kernel_equivalence import adversarial_context, random_context
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.breakout import (
    bollinger_expansion,
    compression_breakout,
    nbar_breakout,
    nbar_failed_breakout,
    range_expansion,
)
from xau_lab.strategies.trend import (
    efficiency_trend,
    ema_slope,
    regression_slope,
    roc_momentum,
    sma_slope,
)


def assert_signal_equal(reference, optimized, ctx, params):
    expected = reference(ctx, params)
    actual = optimized(ctx, params)
    assert expected.dtype == np.int8
    assert actual.dtype == np.int8
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "optimized,reference,params",
    [
        (sma_slope, reference_sma_slope, {"lookback": 37, "slope_horizon": 4, "threshold_atr": 0.17}),
        (ema_slope, reference_ema_slope, {"lookback": 41, "slope_horizon": 7, "threshold_atr": 0.21}),
        (regression_slope, reference_regression_slope, {"lookback": 53, "threshold_atr": 0.13}),
        (roc_momentum, reference_roc_momentum, {"lookback": 19, "threshold_pct": 0.37}),
        (efficiency_trend, reference_efficiency_trend, {"lookback": 31, "er_threshold": 0.43}),
        (nbar_breakout, reference_nbar_breakout, {"lookback": 29}),
        (nbar_failed_breakout, reference_nbar_failed_breakout, {"lookback": 23}),
        (range_expansion, reference_range_expansion, {"median_lookback": 27, "range_multiple": 1.71}),
        (
            bollinger_expansion,
            reference_bollinger_expansion,
            {"lookback": 31, "std_multiplier": 1.83, "bandwidth_percentile": 0.731},
        ),
        (
            compression_breakout,
            reference_compression_breakout,
            {"compression_lookback": 37, "compression_percentile": 0.236588625, "breakout_lookback": 17},
        ),
    ],
)
def test_trend_breakout_signal_equivalence(optimized, reference, params):
    for market in (random_context(seed=44081), adversarial_context()):
        assert_signal_equal(reference, optimized, market, params)


@pytest.mark.parametrize(
    "optimized,reference,params",
    [
        (sma_slope, reference_sma_slope, {"lookback": 5, "slope_horizon": 1, "threshold_atr": 0.0}),
        (sma_slope, reference_sma_slope, {"lookback": 100, "slope_horizon": 10, "threshold_atr": 0.5}),
        (ema_slope, reference_ema_slope, {"lookback": 5, "slope_horizon": 1, "threshold_atr": 0.0}),
        (ema_slope, reference_ema_slope, {"lookback": 100, "slope_horizon": 10, "threshold_atr": 0.5}),
        (regression_slope, reference_regression_slope, {"lookback": 5, "threshold_atr": 0.0}),
        (regression_slope, reference_regression_slope, {"lookback": 120, "threshold_atr": 0.5}),
        (roc_momentum, reference_roc_momentum, {"lookback": 2, "threshold_pct": 0.0}),
        (roc_momentum, reference_roc_momentum, {"lookback": 60, "threshold_pct": 1.5}),
        (efficiency_trend, reference_efficiency_trend, {"lookback": 5, "er_threshold": 0.1}),
        (efficiency_trend, reference_efficiency_trend, {"lookback": 80, "er_threshold": 0.8}),
        (nbar_breakout, reference_nbar_breakout, {"lookback": 2}),
        (nbar_breakout, reference_nbar_breakout, {"lookback": 100}),
        (nbar_failed_breakout, reference_nbar_failed_breakout, {"lookback": 2}),
        (nbar_failed_breakout, reference_nbar_failed_breakout, {"lookback": 100}),
        (range_expansion, reference_range_expansion, {"median_lookback": 5, "range_multiple": 1.1}),
        (range_expansion, reference_range_expansion, {"median_lookback": 100, "range_multiple": 3.0}),
        (
            bollinger_expansion,
            reference_bollinger_expansion,
            {"lookback": 10, "std_multiplier": 1.0, "bandwidth_percentile": 0.5},
        ),
        (
            bollinger_expansion,
            reference_bollinger_expansion,
            {"lookback": 80, "std_multiplier": 3.0, "bandwidth_percentile": 0.95},
        ),
        (
            compression_breakout,
            reference_compression_breakout,
            {"compression_lookback": 10, "compression_percentile": 0.05, "breakout_lookback": 2},
        ),
        (
            compression_breakout,
            reference_compression_breakout,
            {"compression_lookback": 100, "compression_percentile": 0.40, "breakout_lookback": 50},
        ),
    ],
)
def test_trend_breakout_parameter_boundaries_match_reference(optimized, reference, params):
    assert_signal_equal(reference, optimized, random_context(seed=19117), params)


def _simple_context(close, high=None, low=None, atr=None) -> StrategyContext:
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if high is None:
        high = close.copy()
    if low is None:
        low = close.copy()
    if atr is None:
        atr = np.ones(n, dtype=np.float64)
    return StrategyContext(
        open=close.copy(),
        high=np.asarray(high, dtype=np.float64),
        low=np.asarray(low, dtype=np.float64),
        close=close,
        spread=np.zeros(n, dtype=np.float64),
        atr14=np.asarray(atr, dtype=np.float64),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={},
    )


def test_exact_roc_threshold_equality_is_flat():
    ctx = _simple_context([100.0, 101.0])
    exact = (101.0 / 100.0 - 1.0) * 100.0
    assert roc_momentum(ctx, {"lookback": 1, "threshold_pct": exact})[-1] == 0


def test_nbar_breakout_exact_level_equality_is_not_breakout():
    ctx = _simple_context(
        [9.0, 10.0, 10.0],
        high=[10.0, 10.0, 10.0],
        low=[8.0, 8.0, 8.0],
    )
    assert nbar_breakout(ctx, {"lookback": 2})[-1] == 0


def test_failed_breakout_requires_strict_reclaim():
    ctx = _simple_context(
        [9.0, 9.0, 10.0],
        high=[10.0, 10.0, 11.0],
        low=[8.0, 8.0, 8.0],
    )
    assert nbar_failed_breakout(ctx, {"lookback": 2})[-1] == 0


def test_efficiency_threshold_equality_is_inclusive():
    ctx = _simple_context([1.0, 2.0, 3.0])
    assert efficiency_trend(ctx, {"lookback": 2, "er_threshold": 1.0})[-1] == 1
