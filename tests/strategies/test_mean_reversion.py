import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.mean_reversion import (
    bollinger_reversion,
    cci_extreme,
    rsi_extreme,
    stochastic_extreme,
    williams_r_extreme,
    zscore_reversion,
)
from xau_lab.strategies.registry import get_strategy


def context_from_close(close):
    close = np.asarray(close, dtype=float)
    n = len(close)
    return StrategyContext(
        open=close.copy(),
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        spread=np.zeros(n),
        atr14=np.ones(n),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={},
    )


def test_zscore_and_bollinger_revert_completed_bar_overextension():
    close = np.full(25, 100.0)
    close[19] = 110.0
    ctx = context_from_close(close)
    z = zscore_reversion(ctx, {"lookback": 10, "z_threshold": 2.0})
    bb = bollinger_reversion(ctx, {"lookback": 10, "std_multiplier": 2.0})
    assert z[19] == -1
    assert bb[19] == -1
    assert z[18] == 0 and bb[18] == 0
    assert z[20] == 0 and bb[20] == 0


def test_rsi_extreme_reverts_strong_one_way_move():
    close = np.concatenate([np.full(10, 100.0), np.arange(101.0, 116.0)])
    ctx = context_from_close(close)
    sig = rsi_extreme(ctx, {"lookback": 5, "lower": 30.0})
    assert sig[-1] == -1


def test_stochastic_and_williams_r_revert_range_extremes():
    close = np.linspace(100.0, 110.0, 20)
    ctx = context_from_close(close)
    stoch = stochastic_extreme(ctx, {"lookback": 5, "lower": 20.0})
    wr = williams_r_extreme(ctx, {"lookback": 5, "edge_band": 20.0})
    assert stoch[-1] == -1
    assert wr[-1] == -1


def test_cci_extreme_reverts_large_typical_price_move():
    close = np.full(20, 100.0)
    close[-1] = 108.0
    ctx = context_from_close(close)
    sig = cci_extreme(ctx, {"lookback": 10, "threshold": 100.0})
    assert sig[-1] == -1


def test_mean_reversion_strategies_are_registered():
    for name in (
        "zscore_reversion",
        "bollinger_reversion",
        "rsi_extreme",
        "stochastic_extreme",
        "cci_extreme",
        "williams_r_extreme",
    ):
        assert get_strategy(name).family == "mean_reversion"
