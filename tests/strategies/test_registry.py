import numpy as np
import pytest

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import get_strategy, list_strategies, register_strategy


def dummy(ctx, params):
    return np.zeros(len(ctx.close), dtype=np.int8)


def test_registry_round_trip_and_signal_contract():
    definition = StrategyDefinition("test_family", "dummy_registry_roundtrip", dummy, {"lookback": (2, 10)})
    register_strategy(definition)
    found = get_strategy("dummy_registry_roundtrip")
    assert found.family == "test_family"
    ctx = StrategyContext(
        open=np.ones(4),
        high=np.ones(4),
        low=np.ones(4),
        close=np.ones(4),
        spread=np.zeros(4),
        atr14=np.ones(4),
        time_epoch=np.arange(4, dtype=np.int64),
        features={},
    )
    signal = found.generate(ctx, {})
    assert signal.dtype == np.int8
    assert len(signal) == len(ctx.close)
    assert set(np.unique(signal)).issubset({-1, 0, 1})


def test_duplicate_strategy_name_is_rejected():
    name = "dummy_duplicate_guard"
    register_strategy(StrategyDefinition("test_family", name, dummy, {}))
    with pytest.raises(ValueError, match="duplicate strategy"):
        register_strategy(StrategyDefinition("other_family", name, dummy, {}))


def test_signal_contract_rejects_invalid_values_and_wrong_length():
    def wrong_length(ctx, params):
        return np.zeros(len(ctx.close) - 1, dtype=np.int8)

    def invalid_values(ctx, params):
        return np.full(len(ctx.close), 2, dtype=np.int8)

    ctx = StrategyContext(
        open=np.ones(3),
        high=np.ones(3),
        low=np.ones(3),
        close=np.ones(3),
        spread=np.zeros(3),
        atr14=np.ones(3),
        time_epoch=np.arange(3, dtype=np.int64),
        features={},
    )

    with pytest.raises(ValueError, match="length"):
        StrategyDefinition("test", "wrong_length", wrong_length, {}).generate(ctx, {})
    with pytest.raises(ValueError, match="-1, 0, 1"):
        StrategyDefinition("test", "invalid_values", invalid_values, {}).generate(ctx, {})


def test_list_strategies_is_sorted():
    register_strategy(StrategyDefinition("test", "zz_registry_sorted", dummy, {}))
    register_strategy(StrategyDefinition("test", "aa_registry_sorted", dummy, {}))
    names = [definition.name for definition in list_strategies()]
    assert names == sorted(names)
