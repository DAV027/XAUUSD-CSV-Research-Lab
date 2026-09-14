from __future__ import annotations

import numpy as np
import pytest

from tests.strategies.reference_price_session_signals import (
    reference_engulfing,
    reference_inside_bar_break,
    reference_level_sweep_reclaim,
    reference_opening_range_breakout,
    reference_outside_bar,
    reference_prior_session_high_low_break,
    reference_rejection_candle,
    reference_session_open_momentum,
)
from tests.strategies.test_signal_kernel_equivalence import adversarial_context, random_context
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.price_action import (
    engulfing,
    inside_bar_break,
    level_sweep_reclaim,
    outside_bar,
    rejection_candle,
)
from xau_lab.strategies.session import (
    opening_range_breakout,
    prior_session_high_low_break,
    session_open_momentum,
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
        (engulfing, reference_engulfing, {"min_body_atr": 0.37}),
        (rejection_candle, reference_rejection_candle, {"wick_body_ratio": 2.73}),
        (inside_bar_break, reference_inside_bar_break, {"break_buffer_atr": 0.19}),
        (outside_bar, reference_outside_bar, {"min_range_atr": 0.83}),
        (level_sweep_reclaim, reference_level_sweep_reclaim, {"lookback": 29}),
        (session_open_momentum, reference_session_open_momentum, {"session": "london", "threshold_atr": 0.27}),
        (prior_session_high_low_break, reference_prior_session_high_low_break, {"session": "london"}),
        (opening_range_breakout, reference_opening_range_breakout, {"session": "london", "opening_range_bars": 7}),
    ],
)
def test_price_action_session_signal_equivalence(optimized, reference, params):
    for market in (random_context(seed=731811), adversarial_context()):
        assert_signal_equal(reference, optimized, market, params)


@pytest.mark.parametrize(
    "optimized,reference,params",
    [
        (engulfing, reference_engulfing, {"min_body_atr": 0.0}),
        (engulfing, reference_engulfing, {"min_body_atr": 1.0}),
        (rejection_candle, reference_rejection_candle, {"wick_body_ratio": 1.5}),
        (rejection_candle, reference_rejection_candle, {"wick_body_ratio": 5.0}),
        (inside_bar_break, reference_inside_bar_break, {"break_buffer_atr": 0.0}),
        (inside_bar_break, reference_inside_bar_break, {"break_buffer_atr": 0.5}),
        (outside_bar, reference_outside_bar, {"min_range_atr": 0.0}),
        (outside_bar, reference_outside_bar, {"min_range_atr": 2.0}),
        (level_sweep_reclaim, reference_level_sweep_reclaim, {"lookback": 2}),
        (level_sweep_reclaim, reference_level_sweep_reclaim, {"lookback": 100}),
        (session_open_momentum, reference_session_open_momentum, {"session": "asia", "threshold_atr": 0.0}),
        (session_open_momentum, reference_session_open_momentum, {"session": "new_york", "threshold_atr": 1.0}),
        (opening_range_breakout, reference_opening_range_breakout, {"session": "asia", "opening_range_bars": 2}),
        (opening_range_breakout, reference_opening_range_breakout, {"session": "new_york", "opening_range_bars": 30}),
    ],
)
def test_price_session_parameter_boundaries_match_reference(optimized, reference, params):
    assert_signal_equal(reference, optimized, random_context(seed=31201), params)


def _context(open_, high, low, close, atr=None, flags=None) -> StrategyContext:
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if atr is None:
        atr = np.ones(n, dtype=np.float64)
    features = {} if flags is None else {"session_london": np.asarray(flags, dtype=bool)}
    return StrategyContext(
        open=np.asarray(open_, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        low=np.asarray(low, dtype=np.float64),
        close=close,
        spread=np.zeros(n, dtype=np.float64),
        atr14=np.asarray(atr, dtype=np.float64),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features=features,
    )


def test_rejection_equal_wicks_stays_flat():
    market = _context([10.0], [11.0], [9.5], [10.5])
    assert rejection_candle(market, {"wick_body_ratio": 1.0})[-1] == 0


def test_inside_bar_exact_mother_boundaries_are_allowed_but_break_is_strict():
    market = _context(
        [10.0, 10.0, 10.0],
        [12.0, 12.0, 12.0],
        [8.0, 8.0, 8.0],
        [10.0, 10.0, 12.0],
    )
    assert inside_bar_break(market, {"break_buffer_atr": 0.0})[-1] == 0


def test_outside_bar_requires_strict_outside_levels():
    market = _context([10.0, 10.0], [11.0, 11.0], [9.0, 8.0], [10.0, 10.5])
    assert outside_bar(market, {"min_range_atr": 0.0})[-1] == 0


def test_level_sweep_reclaim_requires_strict_close_reclaim():
    market = _context(
        [10.0, 10.0, 9.0],
        [11.0, 11.0, 10.0],
        [9.0, 9.0, 8.0],
        [10.0, 10.0, 9.0],
    )
    assert level_sweep_reclaim(market, {"lookback": 2})[-1] == 0


def test_opening_range_invalid_bar_still_counts_toward_opening_bars():
    market = _context(
        [100.0, 100.0, 100.0, 100.0],
        [101.0, np.nan, 101.5, 103.0],
        [99.0, 99.0, 99.5, 101.0],
        [100.0, 100.0, 101.0, 102.0],
        flags=[True, True, True, True],
    )
    expected = reference_opening_range_breakout(market, {"session": "london", "opening_range_bars": 2})
    actual = opening_range_breakout(market, {"session": "london", "opening_range_bars": 2})
    assert np.array_equal(actual, expected)
    assert actual[2] == 0
    assert actual[3] == 1


@pytest.mark.parametrize(
    "fn,params",
    [
        (session_open_momentum, {"session": "london", "threshold_atr": 0.0}),
        (prior_session_high_low_break, {"session": "london"}),
        (opening_range_breakout, {"session": "london", "opening_range_bars": 2}),
    ],
)
def test_session_public_wrapper_preserves_missing_feature_keyerror(fn, params):
    market = _context([100.0], [101.0], [99.0], [100.0])
    with pytest.raises(KeyError, match="session_london"):
        fn(market, params)
