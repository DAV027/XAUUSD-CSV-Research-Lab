import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.edge_b import trend_pullback_recovery
from xau_lab.strategies.registry import get_strategy


def _context(close, atr=1.0):
    close = np.asarray(close, dtype=float)
    n = len(close)
    atr_values = (
        np.full(n, float(atr), dtype=float)
        if np.isscalar(atr)
        else np.asarray(atr, dtype=float)
    )
    return StrategyContext(
        open=close.copy(),
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        spread=np.zeros(n, dtype=float),
        atr14=atr_values,
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={},
    )


def _params():
    return {
        "trend_lookback": 8,
        "trend_threshold_atr": 2.0,
        "pullback_lookback": 3,
        "pullback_threshold_atr": 1.0,
    }


def test_long_signal_occurs_only_on_first_recovery_turn():
    close = [100, 102, 104, 106, 108, 110, 112, 114, 113, 112, 111, 112, 113]
    signal = trend_pullback_recovery(_context(close), _params())

    assert np.all(signal[:11] == 0)
    assert signal[11] == 1
    assert signal[12] == 0


def test_short_signal_is_mirror_image():
    close = [114, 112, 110, 108, 106, 104, 102, 100, 101, 102, 103, 102, 101]
    signal = trend_pullback_recovery(_context(close), _params())

    assert np.all(signal[:11] == 0)
    assert signal[11] == -1
    assert signal[12] == 0


def test_continuing_pullback_without_recovery_does_not_signal():
    close = [100, 102, 104, 106, 108, 110, 112, 114, 113, 112, 111, 110]
    signal = trend_pullback_recovery(_context(close), _params())
    assert np.all(signal == 0)


def test_invalid_lagged_atr_blocks_recovery_signal():
    close = [100, 102, 104, 106, 108, 110, 112, 114, 113, 112, 111, 112]
    atr = np.ones(len(close), dtype=float)
    atr[10] = np.nan
    signal = trend_pullback_recovery(_context(close, atr=atr), _params())
    assert signal[11] == 0


def test_edge_b_strategy_is_registered_with_frozen_domains():
    definition = get_strategy("trend_pullback_recovery")
    assert definition.family == "edge_b_trend_pullback"
    assert dict(definition.parameter_domain) == {
        "trend_lookback": (60, 240),
        "trend_threshold_atr": (1.0, 4.0),
        "pullback_lookback": (3, 20),
        "pullback_threshold_atr": (0.25, 1.5),
    }
