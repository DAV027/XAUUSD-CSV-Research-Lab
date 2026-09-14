from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedBacktestResult:
    trade_count: int
    risk_skip_count: int
    direction: np.ndarray
    signal_index: np.ndarray
    entry_index: np.ndarray
    exit_index: np.ndarray
    raw_entry: np.ndarray
    entry_price: np.ndarray
    initial_stop: np.ndarray
    target: np.ndarray
    lot: np.ndarray
    planned_risk: np.ndarray
    raw_exit: np.ndarray
    reason: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.trade_count)
        if count < 0 or int(self.risk_skip_count) < 0:
            raise ValueError("packed counts must be nonnegative")
        arrays = (
            self.direction,
            self.signal_index,
            self.entry_index,
            self.exit_index,
            self.raw_entry,
            self.entry_price,
            self.initial_stop,
            self.target,
            self.lot,
            self.planned_risk,
            self.raw_exit,
            self.reason,
        )
        if any(np.asarray(values).ndim != 1 for values in arrays):
            raise ValueError("packed trade arrays must be one-dimensional")
        if any(len(values) < count for values in arrays):
            raise ValueError("packed trade array shorter than trade_count")
