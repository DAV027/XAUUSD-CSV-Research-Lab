import numpy as np
import pytest

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
from xau_lab.strategies.statistical import (
    range_position_reversal,
    return_continuation,
    return_reversal,
    rolling_autocorr_direction,
    standardized_return_signal,
)
from xau_lab.strategies.volatility import (
    atr_percentile_regime,
    volatility_contraction_reversion,
    volatility_expansion_direction,
)
from xau_lab.strategies.registry import get_strategy


def ctx(open_, high, low, close, *, atr=None, features=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    return StrategyContext(
        open=np.asarray(open_, dtype=float),
        high=np.asarray(high, dtype=float),
        low=np.asarray(low, dtype=float),
        close=close,
        spread=np.zeros(n),
        atr14=np.ones(n) if atr is None else np.asarray(atr, dtype=float),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={} if features is None else features,
    )


def test_engulfing_golden_bullish_signal():
    market = ctx([10.0, 8.8], [10.2, 10.4], [8.8, 8.6], [9.0, 10.2])
    assert engulfing(market, {})[-1] == 1


def test_rejection_candle_golden_lower_wick_signal():
    market = ctx([10.0], [10.3], [8.0], [10.2])
    assert rejection_candle(market, {"wick_body_ratio": 3.0})[-1] == 1


def test_inside_bar_break_uses_completed_three_bar_pattern():
    market = ctx(
        [10.0, 10.2, 10.4],
        [12.0, 11.0, 12.5],
        [8.0, 9.0, 10.0],
        [10.5, 10.3, 12.2],
    )
    assert inside_bar_break(market, {})[-1] == 1


def test_outside_bar_uses_body_direction():
    market = ctx([10.0, 9.5], [11.0, 12.0], [9.0, 8.0], [10.2, 11.5])
    assert outside_bar(market, {})[-1] == 1


def test_level_sweep_reclaim_uses_prior_levels_only():
    market = ctx(
        [10.0, 10.0, 10.0, 9.5],
        [11.0, 11.2, 11.1, 10.5],
        [9.0, 9.1, 9.2, 8.5],
        [10.0, 10.1, 10.2, 9.6],
    )
    assert level_sweep_reclaim(market, {"lookback": 3})[-1] == 1


def session_market(flags):
    flags = np.asarray(flags, dtype=bool)
    close = np.array([100.0, 101.0, 100.5, 100.5, 103.0, 104.0, 106.0])[: len(flags)]
    open_ = close - 0.4
    high = close + 0.5
    low = close - 0.5
    return ctx(open_, high, low, close, features={"session_london": flags})


@pytest.mark.parametrize(
    "fn,params",
    [
        (session_open_momentum, {"session": "london", "threshold_atr": 0.0}),
        (prior_session_high_low_break, {"session": "london"}),
        (opening_range_breakout, {"session": "london", "opening_range_bars": 2}),
    ],
)
def test_session_strategies_fail_closed_without_explicit_session_feature(fn, params):
    market = ctx([100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
    with pytest.raises(KeyError, match="session_london"):
        fn(market, params)


def test_session_open_momentum_signals_on_false_to_true_transition():
    market = session_market([False, False, True, True])
    assert session_open_momentum(market, {"session": "london", "threshold_atr": 0.0})[2] == 1


def test_prior_session_high_low_break_uses_prior_completed_session_block():
    market = session_market([True, True, False, False, True])
    assert prior_session_high_low_break(market, {"session": "london"})[-1] == 1


def test_opening_range_breakout_uses_first_session_bars_only():
    market = session_market([False, True, True, True, True])
    market = ctx(
        [100.0, 100.0, 100.2, 100.5, 102.5],
        [100.5, 101.0, 101.2, 101.0, 103.5],
        [99.5, 99.0, 99.2, 99.8, 102.0],
        [100.0, 100.2, 100.4, 100.8, 103.0],
        features={"session_london": np.array([False, True, True, True, True])},
    )
    sig = opening_range_breakout(market, {"session": "london", "opening_range_bars": 2})
    assert sig[-1] == 1


def test_atr_percentile_regime_filters_then_uses_candle_direction():
    market = ctx(
        [100, 100, 100, 100, 100, 100],
        [101, 101, 101, 101, 101, 104],
        [99, 99, 99, 99, 99, 99],
        [100, 100, 100, 100, 100, 103],
        atr=[1, 1, 1, 1, 1, 3],
    )
    assert atr_percentile_regime(market, {"lookback": 5, "percentile": 0.8})[-1] == 1


def test_volatility_expansion_direction_uses_prior_range_baseline():
    market = ctx(
        [100, 100, 100, 100, 100, 100],
        [101, 101, 101, 101, 101, 104],
        [99, 99, 99, 99, 99, 99],
        [100, 100, 100, 100, 100, 103],
    )
    assert volatility_expansion_direction(market, {"lookback": 5, "range_multiple": 2.0})[-1] == 1


def test_volatility_contraction_reversion_fades_current_move():
    market = ctx(
        [100, 100, 100, 100, 100, 100.0],
        [102, 102, 102, 102, 102, 100.4],
        [98, 98, 98, 98, 98, 99.9],
        [100, 100, 100, 100, 100, 100.3],
    )
    sig = volatility_contraction_reversion(market, {"lookback": 5, "percentile": 0.25})
    assert sig[-1] == -1


def statistical_market():
    close = np.array([100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0])
    return ctx(close - 0.2, close + 0.5, close - 0.5, close)


def test_return_continuation_and_reversal_are_opposite_controls():
    market = statistical_market()
    cont = return_continuation(market, {"lookback": 2, "threshold_pct": 0.0})
    rev = return_reversal(market, {"lookback": 2, "threshold_pct": 0.0})
    assert cont[-1] == 1
    assert rev[-1] == -1


def test_rolling_autocorr_direction_continues_positive_autocorrelation():
    market = statistical_market()
    sig = rolling_autocorr_direction(market, {"lookback": 5, "min_abs_autocorr": 0.0})
    assert sig[-1] == 1


def test_range_position_reversal_fades_top_of_prior_range():
    market = statistical_market()
    sig = range_position_reversal(market, {"lookback": 5, "edge_fraction": 0.2})
    assert sig[-1] == -1


def test_standardized_return_signal_detects_large_positive_return():
    close = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 105.0])
    market = ctx(close, close + 0.2, close - 0.2, close)
    sig = standardized_return_signal(market, {"lookback": 4, "z_threshold": 1.0})
    assert sig[-1] == 1


def test_misc_strategy_registry_contains_all_families():
    expected = {
        "engulfing": "price_action",
        "rejection_candle": "price_action",
        "inside_bar_break": "price_action",
        "outside_bar": "price_action",
        "level_sweep_reclaim": "price_action",
        "session_open_momentum": "session",
        "prior_session_high_low_break": "session",
        "opening_range_breakout": "session",
        "atr_percentile_regime": "volatility",
        "volatility_expansion_direction": "volatility",
        "volatility_contraction_reversion": "volatility",
        "return_continuation": "statistical",
        "return_reversal": "statistical",
        "rolling_autocorr_direction": "statistical",
        "range_position_reversal": "statistical",
        "standardized_return_signal": "statistical",
    }
    for name, family in expected.items():
        assert get_strategy(name).family == family
