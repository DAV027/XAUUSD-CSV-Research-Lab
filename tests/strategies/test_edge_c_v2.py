import math

import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.edge_c_v2 import compression_breakout_retest_v2
from xau_lab.strategies.registry import get_strategy


def _context(open_, high, low, close, atr=1.0):
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    atr14 = np.full(n, float(atr), dtype=float) if np.isscalar(atr) else np.asarray(atr, dtype=float)
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


def _long_episode(atr=1.0):
    return _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.30, 100.45],
        high=[100.20, 100.25, 100.20, 100.70, 100.55, 100.65],
        low=[99.80, 99.85, 99.82, 100.05, 100.20, 100.35],
        close=[100.05, 100.10, 100.08, 100.60, 100.50, 100.60],
        atr=atr,
    )


def test_v2_uses_atr_times_sqrt_lookback_and_inclusive_threshold():
    # Prior four-bar range is exactly 1.0. With ATR=1 and L=4 the v2 score
    # is exactly 1 / sqrt(4) == 0.50, so the inclusive boundary must arm.
    ctx = _context(
        open_=[100.0, 100.1, 99.9, 100.0, 100.2, 100.55],
        high=[100.4, 100.5, 100.3, 100.45, 101.10, 100.75],
        low=[99.5, 99.6, 99.7, 99.55, 100.15, 100.45],
        close=[100.1, 100.0, 100.0, 100.1, 101.00, 100.65],
    )
    signals = compression_breakout_retest_v2(
        ctx,
        _params(compression_lookback=4, compression_atr_ratio=0.50),
    )
    assert np.flatnonzero(signals).tolist() == [5]
    assert signals[5] == 1


def test_v2_can_activate_range_that_v1_normalization_would_reject():
    # Width=0.90, ATR=1, L=4. v1 width/ATR=0.90 > 0.50; v2
    # width/(ATR*sqrt(L))=0.45 <= 0.50.
    width = 0.90
    assert width / 1.0 > 0.50
    assert width / (1.0 * math.sqrt(4.0)) <= 0.50
    ctx = _context(
        open_=[100.0, 100.1, 99.9, 100.0, 100.2, 100.55],
        high=[100.35, 100.45, 100.30, 100.40, 101.05, 100.70],
        low=[99.55, 99.60, 99.65, 99.60, 100.10, 100.40],
        close=[100.1, 100.0, 100.0, 100.1, 100.95, 100.60],
    )
    signals = compression_breakout_retest_v2(
        ctx,
        _params(compression_lookback=4, compression_atr_ratio=0.50),
    )
    assert np.flatnonzero(signals).tolist() == [5]


def test_v2_compression_window_is_strict_prior_and_current_breakout_bar_does_not_pollute_it():
    signals = compression_breakout_retest_v2(_long_episode(), _params())
    assert np.flatnonzero(signals).tolist() == [4]
    assert signals[4] == 1


def test_v2_breakout_bar_cannot_self_retest_but_later_bar_may_retest_and_confirm_same_bar():
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.42],
        high=[100.20, 100.25, 100.20, 100.80, 100.70],
        low=[99.80, 99.85, 99.82, 100.15, 100.20],
        close=[100.05, 100.10, 100.08, 100.70, 100.62],
    )
    signals = compression_breakout_retest_v2(ctx, _params())
    assert np.flatnonzero(signals).tolist() == [4]


def test_v2_short_breakout_retest_confirmation_is_symmetric():
    ctx = _context(
        open_=[100.00, 99.95, 99.98, 99.90, 99.70, 99.55],
        high=[100.20, 100.15, 100.18, 99.95, 99.80, 99.65],
        low=[99.80, 99.75, 99.80, 99.30, 99.45, 99.35],
        close=[99.95, 99.90, 99.92, 99.40, 99.50, 99.40],
    )
    signals = compression_breakout_retest_v2(ctx, _params())
    assert np.flatnonzero(signals).tolist() == [4]
    assert signals[4] == -1


def test_v2_close_through_retest_tolerance_invalidates_episode():
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.20, 100.45],
        high=[100.20, 100.25, 100.20, 100.70, 100.30, 100.65],
        low=[99.80, 99.85, 99.82, 100.05, 100.00, 100.35],
        close=[100.05, 100.10, 100.08, 100.60, 100.10, 100.60],
    )
    signals = compression_breakout_retest_v2(ctx, _params())
    assert int(np.count_nonzero(signals)) == 0


def test_v2_no_retest_before_timeout_produces_no_signal():
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.60, 100.62, 100.64, 100.30],
        high=[100.20, 100.25, 100.20, 100.70, 100.75, 100.78, 100.80, 100.55],
        low=[99.80, 99.85, 99.82, 100.05, 100.50, 100.52, 100.54, 100.20],
        close=[100.05, 100.10, 100.08, 100.60, 100.65, 100.67, 100.69, 100.50],
    )
    signals = compression_breakout_retest_v2(ctx, _params(retest_window=2))
    assert int(np.count_nonzero(signals)) == 0


def test_v2_episode_atr_is_frozen_at_arm_time():
    # Arm with ATR=1. Later ATR collapses; confirmation still uses frozen 1.0,
    # so close=100.30 is below broken_level(100.25)+0.10 and must not signal.
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.22, 100.25],
        high=[100.20, 100.25, 100.20, 100.70, 100.35, 100.35],
        low=[99.80, 99.85, 99.82, 100.05, 100.20, 100.20],
        close=[100.05, 100.10, 100.08, 100.60, 100.30, 100.30],
        atr=[1.0, 1.0, 1.0, 0.05, 0.05, 0.05],
    )
    signals = compression_breakout_retest_v2(ctx, _params())
    assert int(np.count_nonzero(signals)) == 0


def test_v2_fresh_compression_rearms_only_after_noncompressed_bar():
    ctx = _context(
        open_=[100.00, 100.05, 100.02, 100.10, 100.30, 101.00, 102.00, 102.05, 102.02, 102.10, 102.30],
        high=[100.20, 100.25, 100.20, 100.70, 100.55, 102.00, 102.20, 102.25, 102.20, 102.70, 102.55],
        low=[99.80, 99.85, 99.82, 100.05, 100.20, 100.20, 101.80, 101.85, 101.82, 102.05, 102.20],
        close=[100.05, 100.10, 100.08, 100.60, 100.50, 101.50, 102.05, 102.10, 102.08, 102.60, 102.50],
    )
    signals = compression_breakout_retest_v2(ctx, _params())
    assert np.flatnonzero(signals).tolist() == [4, 10]


def test_v2_nan_or_invalid_atr_blocks_arming_safely():
    base = _long_episode()
    for bad in (np.nan, 0.0, -1.0):
        atr = np.ones(len(base.close), dtype=float)
        atr[2] = bad
        ctx = StrategyContext(
            open=base.open,
            high=base.high,
            low=base.low,
            close=base.close,
            spread=base.spread,
            atr14=atr,
            time_epoch=base.time_epoch,
            features={},
        )
        signals = compression_breakout_retest_v2(ctx, _params())
        assert int(np.count_nonzero(signals)) == 0


def test_v2_nan_in_prior_range_blocks_compression():
    ctx = _long_episode()
    high = ctx.high.copy()
    high[1] = np.nan
    broken = StrategyContext(
        open=ctx.open,
        high=high,
        low=ctx.low,
        close=ctx.close,
        spread=ctx.spread,
        atr14=ctx.atr14,
        time_epoch=ctx.time_epoch,
        features={},
    )
    assert int(np.count_nonzero(compression_breakout_retest_v2(broken, _params()))) == 0


def test_edge_c_v2_registry_exposes_exact_frozen_domain():
    definition = get_strategy("compression_breakout_retest_v2")
    assert definition.family == "edge_c_breakout_retest_v2"
    assert dict(definition.parameter_domain) == {
        "compression_lookback": (20, 120),
        "compression_atr_ratio": (0.50, 1.10),
        "breakout_buffer_atr": (0.25, 1.50),
        "breakout_body_atr": (0.25, 1.50),
        "retest_window": (2, 20),
        "retest_tolerance_atr": (0.10, 0.75),
        "confirmation_atr": (0.10, 1.00),
    }
