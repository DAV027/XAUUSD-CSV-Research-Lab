from __future__ import annotations

import numpy as np
import pytest

from tests.strategies.reference_mean_stat_signals import (
    reference_bollinger_reversion,
    reference_cci_extreme,
    reference_range_position_reversal,
    reference_return_continuation,
    reference_return_reversal,
    reference_rolling_autocorr_direction,
    reference_rsi_extreme,
    reference_standardized_return_signal,
    reference_stochastic_extreme,
    reference_williams_r_extreme,
    reference_zscore_reversion,
)
from tests.strategies.test_signal_kernel_equivalence import adversarial_context, random_context
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.mean_reversion import (
    bollinger_reversion,
    cci_extreme,
    rsi_extreme,
    stochastic_extreme,
    williams_r_extreme,
    zscore_reversion,
)
from xau_lab.strategies.statistical import (
    range_position_reversal,
    return_continuation,
    return_reversal,
    rolling_autocorr_direction,
    standardized_return_signal,
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
        (zscore_reversion, reference_zscore_reversion, {"lookback": 37, "z_threshold": 1.73}),
        (bollinger_reversion, reference_bollinger_reversion, {"lookback": 41, "std_multiplier": 1.91}),
        (rsi_extreme, reference_rsi_extreme, {"lookback": 17, "lower": 31.0}),
        (stochastic_extreme, reference_stochastic_extreme, {"lookback": 19, "lower": 23.0}),
        (cci_extreme, reference_cci_extreme, {"lookback": 21, "threshold": 137.0}),
        (williams_r_extreme, reference_williams_r_extreme, {"lookback": 23, "edge_band": 17.0}),
        (return_continuation, reference_return_continuation, {"lookback": 13, "threshold_pct": 0.37}),
        (return_reversal, reference_return_reversal, {"lookback": 13, "threshold_pct": 0.37}),
        (
            rolling_autocorr_direction,
            reference_rolling_autocorr_direction,
            {"lookback": 31, "min_abs_autocorr": 0.19},
        ),
        (range_position_reversal, reference_range_position_reversal, {"lookback": 29, "edge_fraction": 0.21}),
        (
            standardized_return_signal,
            reference_standardized_return_signal,
            {"lookback": 27, "z_threshold": 1.43},
        ),
    ],
)
def test_mean_reversion_and_statistical_signal_equivalence(optimized, reference, params):
    for market in (random_context(seed=2109), adversarial_context()):
        assert_signal_equal(reference, optimized, market, params)


@pytest.mark.parametrize(
    "optimized,reference,params",
    [
        (zscore_reversion, reference_zscore_reversion, {"lookback": 10, "z_threshold": 1.0}),
        (zscore_reversion, reference_zscore_reversion, {"lookback": 200, "z_threshold": 3.5}),
        (bollinger_reversion, reference_bollinger_reversion, {"lookback": 10, "std_multiplier": 1.0}),
        (bollinger_reversion, reference_bollinger_reversion, {"lookback": 100, "std_multiplier": 3.5}),
        (rsi_extreme, reference_rsi_extreme, {"lookback": 5, "lower": 10.0}),
        (rsi_extreme, reference_rsi_extreme, {"lookback": 40, "lower": 40.0}),
        (stochastic_extreme, reference_stochastic_extreme, {"lookback": 5, "lower": 5.0}),
        (stochastic_extreme, reference_stochastic_extreme, {"lookback": 40, "lower": 35.0}),
        (cci_extreme, reference_cci_extreme, {"lookback": 5, "threshold": 50.0}),
        (cci_extreme, reference_cci_extreme, {"lookback": 60, "threshold": 250.0}),
        (williams_r_extreme, reference_williams_r_extreme, {"lookback": 5, "edge_band": 5.0}),
        (williams_r_extreme, reference_williams_r_extreme, {"lookback": 40, "edge_band": 35.0}),
        (return_continuation, reference_return_continuation, {"lookback": 1, "threshold_pct": 0.0}),
        (return_reversal, reference_return_reversal, {"lookback": 60, "threshold_pct": 1.5}),
        (
            rolling_autocorr_direction,
            reference_rolling_autocorr_direction,
            {"lookback": 5, "min_abs_autocorr": 0.0},
        ),
        (
            rolling_autocorr_direction,
            reference_rolling_autocorr_direction,
            {"lookback": 100, "min_abs_autocorr": 0.8},
        ),
        (range_position_reversal, reference_range_position_reversal, {"lookback": 5, "edge_fraction": 0.05}),
        (range_position_reversal, reference_range_position_reversal, {"lookback": 100, "edge_fraction": 0.40}),
        (
            standardized_return_signal,
            reference_standardized_return_signal,
            {"lookback": 5, "z_threshold": 0.5},
        ),
        (
            standardized_return_signal,
            reference_standardized_return_signal,
            {"lookback": 100, "z_threshold": 3.5},
        ),
    ],
)
def test_mean_stat_parameter_boundaries_match_reference(optimized, reference, params):
    assert_signal_equal(reference, optimized, random_context(seed=8119), params)


def _context_from_ohlc(open_, high, low, close) -> StrategyContext:
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    return StrategyContext(
        open=np.asarray(open_, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        low=np.asarray(low, dtype=np.float64),
        close=close,
        spread=np.zeros(n, dtype=np.float64),
        atr14=np.ones(n, dtype=np.float64),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={},
    )


def test_inclusive_zscore_and_bollinger_equality_boundaries():
    ctx = _context_from_ohlc([0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0])
    assert zscore_reversion(ctx, {"lookback": 2, "z_threshold": 1.0})[-1] == -1
    assert bollinger_reversion(ctx, {"lookback": 2, "std_multiplier": 1.0})[-1] == -1


def test_rsi_inclusive_lower_and_upper_boundaries():
    lower_ctx = _context_from_ohlc([100.0, 102.0, 99.0], [100.0, 102.0, 99.0], [100.0, 102.0, 99.0], [100.0, 102.0, 99.0])
    upper_ctx = _context_from_ohlc([100.0, 103.0, 101.0], [100.0, 103.0, 101.0], [100.0, 103.0, 101.0], [100.0, 103.0, 101.0])
    assert rsi_extreme(lower_ctx, {"lookback": 2, "lower": 40.0})[-1] == 1
    assert rsi_extreme(upper_ctx, {"lookback": 2, "lower": 40.0})[-1] == -1


def test_stochastic_and_williams_inclusive_edge_boundaries():
    low_ctx = _context_from_ohlc([5.0, 2.0], [10.0, 10.0], [0.0, 0.0], [5.0, 2.0])
    high_ctx = _context_from_ohlc([5.0, 8.0], [10.0, 10.0], [0.0, 0.0], [5.0, 8.0])
    assert stochastic_extreme(low_ctx, {"lookback": 2, "lower": 20.0})[-1] == 1
    assert stochastic_extreme(high_ctx, {"lookback": 2, "lower": 20.0})[-1] == -1
    assert williams_r_extreme(low_ctx, {"lookback": 2, "edge_band": 20.0})[-1] == 1
    assert williams_r_extreme(high_ctx, {"lookback": 2, "edge_band": 20.0})[-1] == -1


def test_cci_inclusive_threshold_boundary():
    ctx = _context_from_ohlc([0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0])
    threshold = 1.0 / 0.015
    assert cci_extreme(ctx, {"lookback": 2, "threshold": threshold})[-1] == -1


def test_return_continuation_strict_threshold_equality_stays_flat():
    ctx = _context_from_ohlc([100.0, 101.0], [100.0, 101.0], [100.0, 101.0], [100.0, 101.0])
    exact = (101.0 / 100.0 - 1.0) * 100.0
    assert return_continuation(ctx, {"lookback": 1, "threshold_pct": exact})[-1] == 0
    assert return_reversal(ctx, {"lookback": 1, "threshold_pct": exact})[-1] == 0


def test_range_position_inclusive_edge_fraction_boundaries():
    low_ctx = _context_from_ohlc([5.0, 2.0], [10.0, 10.0], [0.0, 0.0], [5.0, 2.0])
    high_ctx = _context_from_ohlc([5.0, 8.0], [10.0, 10.0], [0.0, 0.0], [5.0, 8.0])
    assert range_position_reversal(low_ctx, {"lookback": 2, "edge_fraction": 0.2})[-1] == 1
    assert range_position_reversal(high_ctx, {"lookback": 2, "edge_fraction": 0.2})[-1] == -1


def test_autocorr_minimum_is_inclusive_when_exactly_one():
    close = [1.0, 2.0, 6.0, 24.0, 120.0]
    ctx = _context_from_ohlc(close, close, close, close)
    signal = rolling_autocorr_direction(ctx, {"lookback": 4, "min_abs_autocorr": 1.0})[-1]
    assert signal == 1


def test_standardized_return_inclusive_z_boundary():
    close = [100.0, 90.0, 99.0, 108.9]
    ctx = _context_from_ohlc(close, close, close, close)
    expected = reference_standardized_return_signal(ctx, {"lookback": 2, "z_threshold": 1.0})
    actual = standardized_return_signal(ctx, {"lookback": 2, "z_threshold": 1.0})
    assert np.array_equal(actual, expected)
    assert actual[-1] in (-1, 0, 1)
