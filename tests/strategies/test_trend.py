import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.registry import get_strategy
from xau_lab.strategies.trend import (
    efficiency_trend,
    ema_slope,
    regression_slope,
    roc_momentum,
    sma_slope,
)


def context_from_close(close):
    close = np.asarray(close, dtype=float)
    n = len(close)
    return StrategyContext(
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        spread=np.zeros(n),
        atr14=np.ones(n),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={},
    )


def test_regression_slope_follows_monotonic_series_after_warmup():
    up = context_from_close(np.arange(1.0, 80.0))
    down = context_from_close(np.arange(80.0, 1.0, -1.0))
    params = {"lookback": 20, "threshold_atr": 0.0}
    up_sig = regression_slope(up, params)
    down_sig = regression_slope(down, params)
    assert np.all(up_sig[:19] == 0)
    assert np.all(down_sig[:19] == 0)
    assert np.all(up_sig[19:] == 1)
    assert np.all(down_sig[19:] == -1)


def test_sma_and_ema_slope_require_lookback_plus_horizon():
    ctx = context_from_close(np.arange(1.0, 100.0))
    params = {"lookback": 10, "slope_horizon": 3, "threshold_atr": 0.0}
    for fn in (sma_slope, ema_slope):
        sig = fn(ctx, params)
        assert np.all(sig[:12] == 0)
        assert np.all(sig[12:] == 1)


def test_roc_momentum_and_efficiency_trend_follow_net_move():
    up = context_from_close(np.linspace(100.0, 130.0, 80))
    down = context_from_close(np.linspace(130.0, 100.0, 80))
    roc_params = {"lookback": 5, "threshold_pct": 0.0}
    er_params = {"lookback": 5, "er_threshold": 0.1}
    assert np.all(roc_momentum(up, roc_params)[5:] == 1)
    assert np.all(roc_momentum(down, roc_params)[5:] == -1)
    assert np.all(efficiency_trend(up, er_params)[5:] == 1)
    assert np.all(efficiency_trend(down, er_params)[5:] == -1)


def test_nan_required_values_never_emit_signal():
    close = np.arange(1.0, 50.0)
    close[30] = np.nan
    ctx = context_from_close(close)
    sig = regression_slope(ctx, {"lookback": 10, "threshold_atr": 0.0})
    assert np.all(sig[30:40] == 0)


def test_trend_strategies_are_registered_with_expected_domains():
    expected = {
        "sma_slope",
        "ema_slope",
        "regression_slope",
        "roc_momentum",
        "efficiency_trend",
    }
    for name in expected:
        definition = get_strategy(name)
        assert definition.family == "trend_momentum"
        assert definition.parameter_domain
