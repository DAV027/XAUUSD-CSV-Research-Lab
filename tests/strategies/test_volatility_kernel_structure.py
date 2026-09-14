from __future__ import annotations

from tests.strategies.test_signal_kernel_equivalence import random_context
from xau_lab.strategies.volatility import (
    _volatility_contraction_reversion_kernel,
    volatility_contraction_reversion,
)


def test_volatility_contraction_uses_numba_dispatcher():
    market = random_context(n=512)
    volatility_contraction_reversion(
        market,
        {"lookback": 25, "percentile": 0.25},
    )
    assert _volatility_contraction_reversion_kernel.signatures
