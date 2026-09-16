import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.edge_b_v2 import trend_pullback_recovery_v2
from xau_lab.strategies.registry import get_strategy


def _context(close, atr=1.0):
    close = np.asarray(close, dtype=float)
    n = len(close)
    atr_values = np.full(n, float(atr), dtype=float)
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
        "recovery_threshold_atr": 0.5,
    }


def test_v2_emits_only_one_signal_inside_same_pullback_episode():
    close = [100, 102, 104, 106, 108, 110, 112, 114, 116, 115, 114, 113, 114, 113, 114]
    signal = trend_pullback_recovery_v2(_context(close), _params())

    assert signal[12] == 1
    assert signal[14] == 0
    assert int(np.sum(signal == 1)) == 1


def test_v2_requires_recovery_move_to_clear_atr_threshold():
    close = [100, 102, 104, 106, 108, 110, 112, 114, 116, 115, 114, 113, 113.2]
    signal = trend_pullback_recovery_v2(_context(close), _params())
    assert signal[12] == 0


def test_v2_short_side_is_symmetric():
    close = [116, 114, 112, 110, 108, 106, 104, 102, 100, 101, 102, 103, 102]
    signal = trend_pullback_recovery_v2(_context(close), _params())
    assert signal[12] == -1


def test_v2_strategy_registry_exposes_cost_aware_domains():
    definition = get_strategy("trend_pullback_recovery_v2")
    assert definition.family == "edge_b_trend_pullback_v2"
    assert dict(definition.parameter_domain) == {
        "trend_lookback": (120, 480),
        "trend_threshold_atr": (4.0, 12.0),
        "pullback_lookback": (5, 30),
        "pullback_threshold_atr": (1.0, 4.0),
        "recovery_threshold_atr": (0.25, 1.5),
    }
