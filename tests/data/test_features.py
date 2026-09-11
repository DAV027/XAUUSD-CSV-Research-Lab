import math

import polars as pl

from xau_lab.data.features import build_shared_features


def test_rolling_features_are_shifted_one_completed_bar():
    frame = pl.DataFrame({
        "time": [f"2026-01-05 10:0{i}:00" for i in range(6)],
        "open": [10, 11, 12, 13, 14, 15],
        "high": [11, 12, 13, 14, 15, 101],
        "low": [9, 10, 11, 12, 13, 14],
        "close": [10, 11, 12, 13, 14, 100],
        "tick_volume": [1] * 6,
        "spread": [1] * 6,
        "real_volume": [0] * 6,
    })

    out = build_shared_features(frame)

    # The huge final close must not alter the feature available on that same bar.
    assert out["sma_5_lag1"][5] == 12.0
    assert out["rolling_high_5_lag1"][5] == 15.0


def test_signal_facing_shared_features_are_named_lag1():
    frame = pl.DataFrame({
        "time": [f"2026-01-05 10:{i:02d}:00" for i in range(60)],
        "open": [2000.0 + i for i in range(60)],
        "high": [2001.0 + i for i in range(60)],
        "low": [1999.0 + i for i in range(60)],
        "close": [2000.5 + i for i in range(60)],
        "tick_volume": [10] * 60,
        "spread": [35] * 60,
        "real_volume": [0] * 60,
    })

    out = build_shared_features(frame)
    raw = {"time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"}
    feature_columns = [name for name in out.columns if name not in raw]

    assert feature_columns
    assert all(name.endswith("_lag1") for name in feature_columns)
    assert math.isfinite(out["return_1_lag1"][2])
