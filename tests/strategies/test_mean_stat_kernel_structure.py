from __future__ import annotations

from tests.strategies.test_signal_kernel_equivalence import random_context
from xau_lab.strategies.mean_reversion import _zscore_reversion_kernel, zscore_reversion
from xau_lab.strategies.statistical import (
    _standardized_return_signal_kernel,
    standardized_return_signal,
)


def test_zscore_reversion_uses_numba_dispatcher():
    market = random_context(n=512)
    zscore_reversion(market, {"lookback": 25, "z_threshold": 1.5})
    assert _zscore_reversion_kernel.signatures


def test_standardized_return_signal_uses_numba_dispatcher():
    market = random_context(n=512)
    standardized_return_signal(market, {"lookback": 25, "z_threshold": 1.5})
    assert _standardized_return_signal_kernel.signatures
