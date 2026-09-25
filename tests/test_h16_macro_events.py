import json
import runpy
import statistics
from pathlib import Path

import pytest


MODULE = runpy.run_path("scripts/validate_h16_macro_events.py")
ValidationError = MODULE["ValidationError"]
canonicalize_rows = MODULE["canonicalize_rows"]


def _config():
    return json.loads(
        Path("config/h16_macro_surprise_v1_frozen.json").read_text(
            encoding="utf-8"
        )
    )


def _row(i, family="CPI", actual="3.2", consensus="3.0", ts=None):
    month = i + 1
    if ts is None:
        ts = f"2024-{month:02d}-01T13:30:00Z"
    return {
        "event_id": f"e{i}",
        "timestamp_utc": ts,
        "event_family": family,
        "event_name": family,
        "actual": actual,
        "consensus": consensus,
        "previous": "3.0",
        "unit": "%",
        "source": "fixture",
    }


def test_unemployment_sign_is_inverted():
    rows = [
        _row(
            i,
            family="UNEMPLOYMENT",
            actual=str(4.0 + i * 0.1),
            consensus="4.0",
        )
        for i in range(9)
    ]
    out = canonicalize_rows(rows, _config())
    assert float(out[-1]["surprise_raw"]) < 0


def test_standardization_uses_only_prior_same_family_releases():
    surprises = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 1.0]
    rows = [
        _row(i, actual=str(3.0 + surprise), consensus="3.0")
        for i, surprise in enumerate(surprises)
    ]
    out = canonicalize_rows(rows, _config())
    assert all(row["surprise_std"] == "" for row in out[:8])
    expected = surprises[8] / statistics.stdev(surprises[:8])
    assert float(out[8]["surprise_std"]) == pytest.approx(expected)


def test_oos_is_locked_by_default():
    rows = [_row(0, ts="2025-09-01T12:30:00Z")]
    with pytest.raises(ValidationError, match="locked OOS"):
        canonicalize_rows(rows, _config())


def test_oos_requires_explicit_unlock():
    rows = [_row(0, ts="2025-09-01T12:30:00Z")]
    out = canonicalize_rows(rows, _config(), unlock_oos=True)
    assert len(out) == 1


def test_timestamp_must_be_explicit_utc():
    rows = [_row(0, ts="2024-01-01T13:30:00")]
    with pytest.raises(ValidationError, match="explicit UTC offset"):
        canonicalize_rows(rows, _config())


def test_duplicate_family_timestamp_is_rejected():
    a = _row(0, ts="2024-01-01T13:30:00Z")
    b = _row(1, ts="2024-01-01T13:30:00Z")
    with pytest.raises(ValidationError, match="duplicate release"):
        canonicalize_rows([a, b], _config())


def test_duplicate_event_id_is_rejected():
    a = _row(0)
    b = _row(1)
    b["event_id"] = a["event_id"]
    with pytest.raises(ValidationError, match="duplicate event_id"):
        canonicalize_rows([a, b], _config())
