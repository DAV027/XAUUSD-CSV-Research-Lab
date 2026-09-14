from __future__ import annotations

from tests.strategies.test_signal_kernel_equivalence import random_context
from xau_lab.strategies.breakout import _compression_breakout_signal_kernel, compression_breakout
from xau_lab.strategies.trend import _regression_slope_kernel, regression_slope


def test_regression_slope_uses_numba_dispatcher():
    market = random_context(n=512)
    regression_slope(market, {"lookback": 25, "threshold_atr": 0.2})
    assert _regression_slope_kernel.signatures


def test_compression_breakout_uses_numba_dispatcher():
    market = random_context(n=512)
    compression_breakout(
        market,
        {"compression_lookback": 25, "compression_percentile": 0.25, "breakout_lookback": 10},
    )
    assert _compression_breakout_signal_kernel.signatures
