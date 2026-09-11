from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Mapping
import numpy as np

@dataclass
class StrategyContext:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread: np.ndarray
    atr14: np.ndarray
    time_epoch: np.ndarray
    features: Mapping[str,np.ndarray]=field(default_factory=dict)
    def __len__(self): return len(self.close)

@dataclass(frozen=True)
class StrategyDefinition:
    family: str
    name: str
    signal: Callable[[StrategyContext,dict],np.ndarray]
    parameter_domains: dict
