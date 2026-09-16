import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.edge_c import compression_breakout_retest
from xau_lab.strategies.registry import get_strategy


def _context(open_, high, low, close, atr=1.0):
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    if np.isscalar(atr):
        atr14 = np.full(n, float(atr), dtype=float)
    else:
        atr14 = np.asarray(atr, dtype=float)
    return StrategyContext(
        open=open_,
        high=high,
        low=low,
        close=close,
        spread=np.zeros(n, dtype=float),
        atr14=atr14,
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={},
    )


def _params(**overrides):
    values = {
        "compression_lookback": 3,
        "compression_atr_ratio": 0.50,
        "breakout_buffer_atr": 0.25,
        "breakout_body_atr": 0.25,
        "retest_window": 3,
        "retest_tolerance_atr": 0.10,
        "confirmation_atr": 0.10,
    }
    values.update(overrides)
    return values


def _long_episode():
    # Bars 0-2 form a 0.45 ATR compression. Bar 3 is deliberately large:
    # if the current bar leaked into the compression range this setup would fail.
    return _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.30, 100.45],
        high=[100.20, 100.25, 100.20, 100.70, 100.55, 100.65],
        low=[99.80, 99.85, 99.82, 100.05, 100.20, 100.35],
        close=[100.05, 100.10, 100.08, 100.60, 100.50, 100.60],
    )


def test_compression_uses_only_prior_bars_and_valid_retest_confirms_once():
    signals = compression_breakout_retest(_long_episode(), _params())

    assert np.flatnonzero(signals).tolist() == [4]
    assert signals[4] == 1
    assert int(np.count_nonzero(signals)) == 1


def test_wick_or_tiny_body_breakout_does_not_qualify():
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.56, 100.30, 100.35],
        high=[100.20, 100.25, 100.20, 101.10, 100.55, 100.60],
        low=[99.80, 99.85, 99.82, 100.50, 100.20, 100.25],
        close=[100.05, 100.10, 100.08, 100.60, 100.50, 100.55],
    )
    signals = compression_breakout_retest(ctx, _params(breakout_body_atr=0.25))
    assert int(np.count_nonzero(signals)) == 0


def test_short_breakout_retest_confirmation_is_symmetric():
    ctx = _context(
        open_=[100.00, 99.95, 99.98, 99.90, 99.70, 99.55],
        high=[100.20, 100.15, 100.18, 99.95, 99.80, 99.65],
        low=[99.80, 99.75, 99.80, 99.30, 99.45, 99.35],
        close=[99.95, 99.90, 99.92, 99.40, 99.50, 99.40],
    )
    signals = compression_breakout_retest(ctx, _params())

    assert np.flatnonzero(signals).tolist() == [4]
    assert signals[4] == -1


def test_close_through_retest_tolerance_invalidates_episode():
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.20, 100.45],
        high=[100.20, 100.25, 100.20, 100.70, 100.30, 100.65],
        low=[99.80, 99.85, 99.82, 100.05, 100.00, 100.35],
        close=[100.05, 100.10, 100.08, 100.60, 100.10, 100.60],
    )
    signals = compression_breakout_retest(ctx, _params())
    assert int(np.count_nonzero(signals)) == 0


def test_no_retest_before_timeout_produces_no_signal():
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.60, 100.62, 100.64, 100.30],
        high=[100.20, 100.25, 100.20, 100.70, 100.75, 100.78, 100.80, 100.55],
        low=[99.80, 99.85, 99.82, 100.05, 100.50, 100.52, 100.54, 100.20],
        close=[100.05, 100.10, 100.08, 100.60, 100.65, 100.67, 100.69, 100.50],
    )
    signals = compression_breakout_retest(ctx, _params(retest_window=2))
    assert int(np.count_nonzero(signals)) == 0


def test_breakout_bar_itself_cannot_count_as_retest():
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.60],
        high=[100.20, 100.25, 100.20, 100.80, 100.75],
        low=[99.80, 99.85, 99.82, 100.20, 100.50],
        close=[100.05, 100.10, 100.08, 100.70, 100.65],
    )
    signals = compression_breakout_retest(ctx, _params())
    assert int(np.count_nonzero(signals)) == 0


def test_fresh_compression_can_rearm_only_after_noncompressed_bar():
    ctx = _context(
        open_=[
            100.00, 100.05, 100.02, 100.10, 100.30,
            101.00, 102.00, 102.05, 102.02, 102.10, 102.30,
        ],
        high=[
            100.20, 100.25, 100.20, 100.70, 100.55,
            102.00, 102.20, 102.25, 102.20, 102.70, 102.55,
        ],
        low=[
            99.80, 99.85, 99.82, 100.05, 100.20,
            100.20, 101.80, 101.85, 101.82, 102.05, 102.20,
        ],
        close=[
            100.05, 100.10, 100.08, 100.60, 100.50,
            101.50, 102.05, 102.10, 102.08, 102.60, 102.50,
        ],
    )
    signals = compression_breakout_retest(ctx, _params())

    assert np.flatnonzero(signals).tolist() == [4, 10]


def test_nan_atr_blocks_arming_and_progression_safely():
    atr = [1.0, 1.0, np.nan, 1.0, 1.0, 1.0]
    signals = compression_breakout_retest(_long_episode().__class__(
        open=_long_episode().open,
        high=_long_episode().high,
        low=_long_episode().low,
        close=_long_episode().close,
        spread=_long_episode().spread,
        atr14=np.asarray(atr, dtype=float),
        time_epoch=_long_episode().time_epoch,
        features={},
    ), _params())
    assert int(np.count_nonzero(signals)) == 0


def test_edge_c_registry_exposes_exact_frozen_domain():
    definition = get_strategy("compression_breakout_retest")
    assert definition.family == "edge_c_breakout_retest"
    assert dict(definition.parameter_domain) == {
        "compression_lookback": (20, 120),
        "compression_atr_ratio": (0.35, 0.80),
        "breakout_buffer_atr": (0.25, 1.50),
        "breakout_body_atr": (0.25, 1.50),
        "retest_window": (2, 20),
        "retest_tolerance_atr": (0.10, 0.75),
        "confirmation_atr": (0.10, 1.00),
    }
