"""Exact expanding linear quantiles; only previously inserted observations count."""
import numpy as np
from numba import njit


@njit(cache=True)
def _select(tree, rank):
    index = 0
    step = 1
    while step < len(tree):
        step <<= 1
    while step:
        candidate = index + step
        if candidate < len(tree) and tree[candidate] <= rank:
            rank -= tree[candidate]
            index = candidate
        step >>= 1
    return index


@njit(cache=True)
def _prefix_length(values):
    for i in range(len(values)):
        if not np.isfinite(values[i]):
            return i
    return len(values)


@njit(cache=True)
def _quantiles(values, coordinates, q, valid):
    tree = np.zeros(len(coordinates) + 1, dtype=np.int64)
    out = np.full(len(values), np.nan)
    for i in range(min(valid + 1, len(values))):
        if i:
            position = (i - 1) * q
            lower = int(np.floor(position))
            upper = min(lower + 1, i - 1)
            weight = position - lower
            a = coordinates[_select(tree, lower)]
            b = coordinates[_select(tree, upper)]
            # Match NumPy's linear interpolation, including its >= .5 branch.
            difference = b - a
            if weight >= 0.5:
                out[i] = b - difference * (1.0 - weight)
            else:
                out[i] = a + difference * weight
        if i < valid:
            index = np.searchsorted(coordinates, values[i]) + 1
            while index < len(tree):
                tree[index] += 1
                index += index & -index
    return out


def expanding_quantiles(values, q):
    if not 0 <= q <= 1:
        raise ValueError("Quantiles must be in the range [0, 1]")
    valid = _prefix_length(values)
    # Future values supply rank coordinates only, never counts or thresholds.
    coordinates = np.unique(values[:valid])
    return _quantiles(values, coordinates, q, valid)
