import numpy as np
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.registry import get_strategy, list_strategies
import xau_lab.strategies.all  # register

def ctx_from_close(close):
    c=np.asarray(close,float); n=len(c)
    return StrategyContext(
        open=c.copy(),high=c+0.2,low=c-0.2,close=c,spread=np.zeros(n),atr14=np.ones(n),
        time_epoch=np.arange(n,dtype=np.int64)*60,features={}
    )

def test_registry_contains_independent_families():
    names=set(list_strategies())
    for name in [
        'regression_slope','roc_momentum','nbar_breakout','nbar_failed_breakout',
        'zscore_reversion','rsi_extreme','engulfing','return_continuation',
        'return_reversal','bollinger_expansion','compression_breakout',
        'session_open_momentum','opening_range_breakout',
        'wma_slope','hma_slope','kama_slope','dema_slope','tema_slope',
        'macd_momentum','adx_dmi_trend','supertrend_direction','ichimoku_cloud',
        'keltner_reversion','choppiness_trend','previous_day_high_low_break',
    ]:
        assert name in names

def test_regression_slope_monotonic():
    c=ctx_from_close(np.arange(1.,90.))
    s=get_strategy('regression_slope').signal(c,{'lookback':20,'threshold_atr':0.0})
    assert np.all(s[25:]==1)

def test_breakout_excludes_signal_bar_from_lookback():
    c=ctx_from_close([10,10,10,10,12])
    c.high=np.array([10.2,10.2,10.2,10.2,12.2])
    s=get_strategy('nbar_breakout').signal(c,{'lookback':3})
    assert s[-1]==1


def test_macd_momentum_is_long_in_persistent_rise():
    c=ctx_from_close(np.linspace(100.0,180.0,300))
    s=get_strategy('macd_momentum').signal(c,{'fast':8,'slow':21,'signal':5,'min_hist_atr':0.0})
    assert np.count_nonzero(s[-80:]==1) > 60


def test_keltner_reversion_fades_large_upper_deviation():
    close=np.full(80,100.0); close[-1]=110.0
    c=ctx_from_close(close)
    c.high[-1]=110.2; c.low[-1]=109.8
    s=get_strategy('keltner_reversion').signal(c,{'lookback':20,'atr_mult':1.5})
    assert s[-1] == -1
