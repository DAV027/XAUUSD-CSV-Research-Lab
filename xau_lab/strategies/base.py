from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from xau_lab.signals import signal_domain_is_valid, zero_disallowed_signals

SignalFunction = Callable[["StrategyContext", dict], np.ndarray]


@dataclass(frozen=True)
class StrategyContext:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread: np.ndarray
    atr14: np.ndarray
    time_epoch: np.ndarray
    features: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        arrays = {
            "open": (self.open, np.float64),
            "high": (self.high, np.float64),
            "low": (self.low, np.float64),
            "close": (self.close, np.float64),
            "spread": (self.spread, np.float64),
            "atr14": (self.atr14, np.float64),
            "time_epoch": (self.time_epoch, np.int64),
        }
        normalized: dict[str, np.ndarray] = {}
        lengths: set[int] = set()
        for name, (value, dtype) in arrays.items():
            array = np.ascontiguousarray(value, dtype=dtype)
            array.setflags(write=False)
            normalized[name] = array
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError("StrategyContext arrays must have equal length")
        n = next(iter(lengths), 0)
        feature_map: dict[str, np.ndarray] = {}
        for name, value in self.features.items():
            array = np.ascontiguousarray(value)
            if len(array) != n:
                raise ValueError(f"feature {name!r} length does not match context")
            array.setflags(write=False)
            feature_map[str(name)] = array
        for name, array in normalized.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "features", MappingProxyType(feature_map))

    def __len__(self) -> int:
        return len(self.close)

    def feature(self, name: str) -> np.ndarray:
        try:
            return self.features[name]
        except KeyError as exc:
            raise KeyError(f"required feature {name!r} is unavailable") from exc


@dataclass(frozen=True)
class StrategyDefinition:
    family: str
    name: str
    signal: SignalFunction
    parameter_domain: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.family or not self.name:
            raise ValueError("strategy family and name are required")
        if not callable(self.signal):
            raise TypeError("signal must be callable")
        object.__setattr__(self, "parameter_domain", MappingProxyType(dict(self.parameter_domain)))

    def generate(self, ctx: StrategyContext, params: dict) -> np.ndarray:
        signal = np.asarray(self.signal(ctx, dict(params)))
        if signal.ndim != 1 or len(signal) != len(ctx):
            raise ValueError("strategy signal length must equal context length")
        if not signal_domain_is_valid(signal):
            raise ValueError("strategy signals must contain only -1, 0, 1")

        filtered = np.ascontiguousarray(signal, dtype=np.int8)
        entry_allowed = ctx.features.get("entry_allowed")
        if entry_allowed is not None:
            allowed = np.asarray(entry_allowed, dtype=np.bool_)
            if len(allowed) != len(filtered):
                raise ValueError("entry_allowed feature length must equal context length")
            # The signal may alias strategy-owned/read-only storage. Only allocate
            # a private writable copy when the integrity mask must mutate it.
            filtered = filtered.copy()
            zero_disallowed_signals(filtered, allowed)
        return filtered
