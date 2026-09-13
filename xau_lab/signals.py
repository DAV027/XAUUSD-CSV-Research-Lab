from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _signal_domain_is_valid(values: np.ndarray) -> bool:
    """Validate {-1, 0, 1} without allocating a full-size temporary array."""
    for index in range(len(values)):
        value = values[index]
        if value != -1 and value != 0 and value != 1:
            return False
    return True


@njit(cache=True)
def _zero_disallowed_signals(values: np.ndarray, allowed: np.ndarray) -> None:
    for index in range(len(values)):
        if not allowed[index]:
            values[index] = 0


def signal_domain_is_valid(values: np.ndarray) -> bool:
    """Return whether a one-dimensional numeric array contains only -1, 0, or 1."""
    array = np.asarray(values)
    if array.ndim != 1:
        return False
    if array.dtype.kind not in "biuf":
        return False
    return bool(_signal_domain_is_valid(array))


def normalize_signal_array(values: object) -> np.ndarray:
    """Normalize a validated numeric signal vector to contiguous int8."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("signals must be one-dimensional")
    if not signal_domain_is_valid(array):
        raise ValueError("signals must contain only -1, 0, +1")
    return np.ascontiguousarray(array, dtype=np.int8)


def zero_disallowed_signals(values: np.ndarray, allowed: np.ndarray) -> None:
    """Mask a private writable signal array in place without a boolean temporary."""
    if values.ndim != 1 or allowed.ndim != 1 or len(values) != len(allowed):
        raise ValueError("signal and allowed arrays must be one-dimensional and equal length")
    _zero_disallowed_signals(values, allowed)


__all__ = [
    "normalize_signal_array",
    "signal_domain_is_valid",
    "zero_disallowed_signals",
]
