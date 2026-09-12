import numpy as np

from xau_lab.strategies.base import StrategyContext, StrategyDefinition


def test_strategy_generation_zeroes_entries_where_integrity_guard_disallows_trading():
    def always_long(ctx, params):
        return np.ones(len(ctx), dtype=np.int8)

    ctx = StrategyContext(
        open=np.ones(4),
        high=np.ones(4),
        low=np.ones(4),
        close=np.ones(4),
        spread=np.zeros(4),
        atr14=np.ones(4),
        time_epoch=np.arange(4, dtype=np.int64),
        features={"entry_allowed": np.array([True, False, False, True])},
    )

    signal = StrategyDefinition("test", "entry_integrity_guard", always_long, {}).generate(ctx, {})

    assert signal.tolist() == [1, 0, 0, 1]
