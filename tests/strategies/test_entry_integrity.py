import numpy as np
import pytest

import xau_lab.strategies.base as base_module
from xau_lab.strategies.base import StrategyContext, StrategyDefinition


def _context(n: int, *, entry_allowed=None) -> StrategyContext:
    features = {} if entry_allowed is None else {"entry_allowed": np.asarray(entry_allowed, dtype=bool)}
    return StrategyContext(
        open=np.ones(n),
        high=np.ones(n),
        low=np.ones(n),
        close=np.ones(n),
        spread=np.zeros(n),
        atr14=np.ones(n),
        time_epoch=np.arange(n, dtype=np.int64),
        features=features,
    )


def test_strategy_generation_zeroes_entries_where_integrity_guard_disallows_trading():
    def always_long(ctx, params):
        return np.ones(len(ctx), dtype=np.int8)

    ctx = _context(4, entry_allowed=[True, False, False, True])

    signal = StrategyDefinition("test", "entry_integrity_guard", always_long, {}).generate(ctx, {})

    assert signal.tolist() == [1, 0, 0, 1]


def test_strategy_generation_validates_domain_without_numpy_isin(
    monkeypatch: pytest.MonkeyPatch,
):
    def valid_signal(ctx, params):
        return np.array([1, 0, -1, 0], dtype=np.int8)

    def forbidden_isin(*args, **kwargs):
        raise AssertionError("np.isin must not allocate a full-size validation temporary")

    monkeypatch.setattr(base_module.np, "isin", forbidden_isin)

    signal = StrategyDefinition("test", "no_isin_validation", valid_signal, {}).generate(
        _context(4), {}
    )

    assert signal.tolist() == [1, 0, -1, 0]


@pytest.mark.parametrize("invalid", [0.5, 1.5, np.nan, np.inf, -np.inf])
def test_strategy_generation_rejects_non_domain_values_before_int8_coercion(invalid):
    def invalid_signal(ctx, params):
        return np.array([1.0, 0.0, invalid, -1.0], dtype=np.float64)

    definition = StrategyDefinition("test", "reject_invalid_signal", invalid_signal, {})

    with pytest.raises(ValueError, match="strategy signals must contain only -1, 0, 1"):
        definition.generate(_context(4), {})


def test_strategy_generation_accepts_integral_float_domain_values():
    def float_signal(ctx, params):
        return np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64)

    signal = StrategyDefinition("test", "integral_float_signal", float_signal, {}).generate(
        _context(4), {}
    )

    assert signal.dtype == np.int8
    assert signal.tolist() == [1, 0, -1, 0]
