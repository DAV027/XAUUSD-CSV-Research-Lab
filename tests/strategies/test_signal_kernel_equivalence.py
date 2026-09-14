from __future__ import annotations

import numpy as np
import pytest

from tests.strategies.reference_signals import (
    reference_atr_percentile_regime,
    reference_volatility_contraction_reversion,
    reference_volatility_expansion_direction,
)
from xau_lab.strategies._kernels import (
    _body_direction,
    _linear_quantile_sorted_window,
    _window_all_finite,
    _window_mean_std,
    _window_min_max,
)
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.volatility import (
    atr_percentile_regime,
    volatility_contraction_reversion,
    volatility_expansion_direction,
)


def random_context(seed: int = 9215000, n: int = 4096) -> StrategyContext:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.8, n))
    open_ = close + rng.normal(0.0, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0.01, 1.2, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 1.2, n)
    atr = rng.uniform(0.1, 5.0, n)
    flags = np.zeros(n, dtype=bool)
    for start in range(100, n, 500):
        flags[start : min(start + 180, n)] = True
    return StrategyContext(
        open=np.ascontiguousarray(open_),
        high=np.ascontiguousarray(high),
        low=np.ascontiguousarray(low),
        close=np.ascontiguousarray(close),
        spread=np.zeros(n, dtype=float),
        atr14=np.ascontiguousarray(atr),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={
            "session_london": flags.copy(),
            "session_asia": flags.copy(),
            "session_new_york": flags.copy(),
        },
    )


def adversarial_context() -> StrategyContext:
    ctx = random_context(n=1024)
    open_ = ctx.open.copy()
    high = ctx.high.copy()
    low = ctx.low.copy()
    close = ctx.close.copy()
    atr = ctx.atr14.copy()
    close[220] = np.nan
    high[350] = np.inf
    low[351] = -np.inf
    atr[500] = np.nan
    close[700:710] = 2100.0
    open_[700:710] = 2100.0
    high[700:710] = 2100.0
    low[700:710] = 2100.0
    return StrategyContext(
        open=open_,
        high=high,
        low=low,
        close=close,
        spread=np.zeros(len(close)),
        atr14=atr,
        time_epoch=ctx.time_epoch.copy(),
        features={name: values.copy() for name, values in ctx.features.items()},
    )


def assert_signal_equal(reference, optimized, ctx, params):
    expected = reference(ctx, params)
    actual = optimized(ctx, params)
    assert expected.dtype == np.int8
    assert actual.dtype == np.int8
    assert np.array_equal(actual, expected)


def test_linear_quantile_window_matches_numpy_default_linear():
    values = np.array([4.0, 1.0, 9.0, 2.0, 2.0, 8.0], dtype=np.float64)
    scratch = np.empty(5, dtype=np.float64)
    for q in (0.05, 0.236588625, 0.4, 0.5, 0.95):
        got = _linear_quantile_sorted_window(values, 1, 6, q, scratch)
        expected = float(np.quantile(values[1:6], q))
        assert got == expected


def test_quantile_helper_rejects_invalid_or_nonfinite_windows():
    values = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
    scratch = np.empty(4, dtype=np.float64)
    assert np.isnan(_linear_quantile_sorted_window(values, 0, 4, 0.5, scratch))
    assert np.isnan(_linear_quantile_sorted_window(values, 1, 1, 0.5, scratch))
    assert np.isnan(_linear_quantile_sorted_window(values, 0, 2, -0.1, scratch))
    assert np.isnan(_linear_quantile_sorted_window(values, 0, 2, 1.1, scratch))


def test_window_helpers_fail_closed_on_nonfinite_values():
    values = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
    assert not _window_all_finite(values, 0, 4)
    valid, mean, std = _window_mean_std(values, 0, 4)
    assert not valid
    assert np.isnan(mean)
    assert np.isnan(std)
    valid, low, high = _window_min_max(values, 0, 4)
    assert not valid
    assert np.isnan(low)
    assert np.isnan(high)


def test_window_helpers_match_numpy_on_finite_window():
    values = np.array([8.0, 1.0, 4.0, 7.0, 3.0], dtype=np.float64)
    assert _window_all_finite(values, 1, 5)
    valid, mean, std = _window_mean_std(values, 1, 5)
    assert valid
    assert mean == float(np.mean(values[1:5]))
    assert std == float(np.std(values[1:5], ddof=0))
    valid, low, high = _window_min_max(values, 1, 5)
    assert valid
    assert low == float(np.min(values[1:5]))
    assert high == float(np.max(values[1:5]))


def test_body_direction_matches_baseline_boundaries():
    assert _body_direction(1.0, 2.0) == np.int8(1)
    assert _body_direction(2.0, 1.0) == np.int8(-1)
    assert _body_direction(1.0, 1.0) == np.int8(0)
    assert _body_direction(np.nan, 1.0) == np.int8(0)
    assert _body_direction(1.0, np.inf) == np.int8(0)


@pytest.mark.parametrize(
    "optimized,reference,params",
    [
        (
            atr_percentile_regime,
            reference_atr_percentile_regime,
            {"lookback": 37, "percentile": 0.731},
        ),
        (
            volatility_expansion_direction,
            reference_volatility_expansion_direction,
            {"lookback": 41, "range_multiple": 1.83},
        ),
        (
            volatility_contraction_reversion,
            reference_volatility_contraction_reversion,
            {"lookback": 53, "percentile": 0.236588625},
        ),
    ],
)
def test_volatility_kernel_signal_equivalence(optimized, reference, params):
    for market in (random_context(), adversarial_context()):
        assert_signal_equal(reference, optimized, market, params)


@pytest.mark.parametrize(
    "optimized,reference,params",
    [
        (atr_percentile_regime, reference_atr_percentile_regime, {"lookback": 10, "percentile": 0.5}),
        (atr_percentile_regime, reference_atr_percentile_regime, {"lookback": 200, "percentile": 0.95}),
        (
            volatility_expansion_direction,
            reference_volatility_expansion_direction,
            {"lookback": 5, "range_multiple": 1.1},
        ),
        (
            volatility_expansion_direction,
            reference_volatility_expansion_direction,
            {"lookback": 100, "range_multiple": 3.0},
        ),
        (
            volatility_contraction_reversion,
            reference_volatility_contraction_reversion,
            {"lookback": 5, "percentile": 0.05},
        ),
        (
            volatility_contraction_reversion,
            reference_volatility_contraction_reversion,
            {"lookback": 100, "percentile": 0.40},
        ),
    ],
)
def test_volatility_parameter_boundaries_match_reference(optimized, reference, params):
    assert_signal_equal(reference, optimized, random_context(seed=73119), params)
