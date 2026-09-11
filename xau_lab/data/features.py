from __future__ import annotations

import numpy as np
import polars as pl

from xau_lab.data.schema import canonicalize_bar_frame
from xau_lab.data.sessions import SessionConfig, add_session_features


def _shift_one(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size > 1:
        out[1:] = values[:-1]
    return out


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or values.size < window:
        return out
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(safe)))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    window_sums = sums[window:] - sums[:-window]
    window_counts = counts[window:] - counts[:-window]
    valid = window_counts == window
    target = out[window - 1 :]
    target[valid] = window_sums[valid]
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    summed = _rolling_sum(values, window)
    return summed / float(window)


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or values.size < window:
        return out
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(safe)))
    sums_sq = np.concatenate(([0.0], np.cumsum(safe * safe)))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    win_sum = sums[window:] - sums[:-window]
    win_sq = sums_sq[window:] - sums_sq[:-window]
    win_count = counts[window:] - counts[:-window]
    valid = win_count == window
    mean = win_sum / float(window)
    variance = np.maximum(win_sq / float(window) - mean * mean, 0.0)
    target = out[window - 1 :]
    target[valid] = np.sqrt(variance[valid])
    return out


def _rolling_extreme(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or values.size < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    if kind == "max":
        result = np.max(windows, axis=1)
    elif kind == "min":
        result = np.min(windows, axis=1)
    else:
        raise ValueError(f"unsupported rolling extreme kind: {kind}")
    out[window - 1 :] = result
    return out


def _ema(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or values.size < window:
        return out
    seed = values[:window]
    if not np.all(np.isfinite(seed)):
        return out
    alpha = 2.0 / (window + 1.0)
    out[window - 1] = float(np.mean(seed))
    for i in range(window, values.size):
        if not np.isfinite(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _wilder(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or values.size < window:
        return out
    seed = values[:window]
    if not np.all(np.isfinite(seed)):
        return out
    out[window - 1] = float(np.mean(seed))
    for i in range(window, values.size):
        if not np.isfinite(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = ((window - 1.0) * out[i - 1] + values[i]) / window
    return out


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    out = np.full(close.shape, np.nan, dtype=np.float64)
    if window <= 0 or close.size <= window:
        return out

    diff = np.diff(close, prepend=np.nan)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    gains[0] = np.nan
    losses[0] = np.nan

    avg_gain = float(np.mean(gains[1 : window + 1]))
    avg_loss = float(np.mean(losses[1 : window + 1]))

    def value(gain: float, loss: float) -> float:
        if loss == 0.0:
            return 50.0 if gain == 0.0 else 100.0
        rs = gain / loss
        return 100.0 - 100.0 / (1.0 + rs)

    out[window] = value(avg_gain, avg_loss)
    for i in range(window + 1, close.size):
        avg_gain = ((window - 1.0) * avg_gain + gains[i]) / window
        avg_loss = ((window - 1.0) * avg_loss + losses[i]) / window
        out[i] = value(avg_gain, avg_loss)
    return out


def _regression_slope(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 1 or values.size < window:
        return out
    x = np.arange(window, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    slopes = windows @ centered / denominator
    out[window - 1 :] = slopes
    return out


def _efficiency_ratio(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or values.size <= window:
        return out
    diffs = np.abs(np.diff(values, prepend=np.nan))
    path = _rolling_sum(diffs, window)
    direction = np.full(values.shape, np.nan, dtype=np.float64)
    direction[window:] = np.abs(values[window:] - values[:-window])
    valid = np.isfinite(path) & (path > 0.0) & np.isfinite(direction)
    out[valid] = direction[valid] / path[valid]
    zero_path = np.isfinite(path) & (path == 0.0) & np.isfinite(direction)
    out[zero_path] = 0.0
    return out


def build_shared_features(
    frame: pl.DataFrame,
    session_config: SessionConfig | None = None,
) -> pl.DataFrame:
    canonical = canonicalize_bar_frame(frame)
    output = add_session_features(canonical, session_config) if session_config is not None else canonical

    open_ = canonical.get_column("open").to_numpy().astype(np.float64, copy=False)
    high = canonical.get_column("high").to_numpy().astype(np.float64, copy=False)
    low = canonical.get_column("low").to_numpy().astype(np.float64, copy=False)
    close = canonical.get_column("close").to_numpy().astype(np.float64, copy=False)

    features: dict[str, np.ndarray] = {}

    return_1 = np.full(close.shape, np.nan, dtype=np.float64)
    log_return_1 = np.full(close.shape, np.nan, dtype=np.float64)
    if close.size > 1:
        return_1[1:] = close[1:] / close[:-1] - 1.0
        log_return_1[1:] = np.log(close[1:] / close[:-1])
    features["return_1"] = return_1
    features["log_return_1"] = log_return_1
    features["range"] = high - low
    features["body"] = np.abs(close - open_)
    features["upper_wick"] = high - np.maximum(open_, close)
    features["lower_wick"] = np.minimum(open_, close) - low

    for window in (5, 9, 21, 50, 100, 200):
        features[f"sma_{window}"] = _rolling_mean(close, window)

    for window in (9, 21, 50, 100, 200):
        features[f"ema_{window}"] = _ema(close, window)

    previous_close = np.roll(close, 1)
    if close.size:
        previous_close[0] = close[0]
    true_range = np.maximum.reduce(
        [high - low, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    for window in (7, 14, 28):
        features[f"atr_{window}"] = _wilder(true_range, window)

    for window in (7, 14, 21):
        features[f"rsi_{window}"] = _rsi(close, window)

    boll_mid = _rolling_mean(close, 20)
    boll_std = _rolling_std(close, 20)
    boll_upper = boll_mid + 2.0 * boll_std
    boll_lower = boll_mid - 2.0 * boll_std
    boll_bandwidth = np.full(close.shape, np.nan, dtype=np.float64)
    boll_zscore = np.full(close.shape, np.nan, dtype=np.float64)
    valid_mid = np.isfinite(boll_mid) & (boll_mid != 0.0) & np.isfinite(boll_std)
    boll_bandwidth[valid_mid] = (boll_upper[valid_mid] - boll_lower[valid_mid]) / boll_mid[valid_mid]
    valid_std = np.isfinite(boll_std) & (boll_std > 0.0) & np.isfinite(boll_mid)
    boll_zscore[valid_std] = (close[valid_std] - boll_mid[valid_std]) / boll_std[valid_std]
    features["boll_mid_20_2"] = boll_mid
    features["boll_upper_20_2"] = boll_upper
    features["boll_lower_20_2"] = boll_lower
    features["boll_bandwidth_20_2"] = boll_bandwidth
    features["boll_zscore_20_2"] = boll_zscore

    for window in (5, 10, 20, 50):
        rolling_high = _rolling_extreme(high, window, "max")
        rolling_low = _rolling_extreme(low, window, "min")
        position = np.full(close.shape, np.nan, dtype=np.float64)
        width = rolling_high - rolling_low
        valid = np.isfinite(width) & (width > 0.0)
        position[valid] = (close[valid] - rolling_low[valid]) / width[valid]
        features[f"rolling_high_{window}"] = rolling_high
        features[f"rolling_low_{window}"] = rolling_low
        features[f"range_position_{window}"] = position

    for window in (10, 20, 50):
        features[f"realized_vol_{window}"] = _rolling_std(log_return_1, window)
        features[f"regression_slope_{window}"] = _regression_slope(close, window)
        features[f"efficiency_ratio_{window}"] = _efficiency_ratio(close, window)

    lagged_series = [
        pl.Series(f"{name}_lag1", _shift_one(values), dtype=pl.Float64)
        for name, values in features.items()
    ]
    return output.with_columns(lagged_series)
