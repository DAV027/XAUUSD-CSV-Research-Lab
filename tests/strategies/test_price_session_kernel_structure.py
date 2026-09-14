from __future__ import annotations

from tests.strategies.test_signal_kernel_equivalence import random_context
from xau_lab.strategies.price_action import _rejection_candle_kernel, rejection_candle
from xau_lab.strategies.session import _opening_range_breakout_kernel, opening_range_breakout


def test_rejection_candle_uses_numba_dispatcher():
    market = random_context(n=512)
    rejection_candle(market, {"wick_body_ratio": 2.0})
    assert _rejection_candle_kernel.signatures


def test_opening_range_breakout_uses_numba_dispatcher():
    market = random_context(n=512)
    opening_range_breakout(market, {"session": "london", "opening_range_bars": 7})
    assert _opening_range_breakout_kernel.signatures
