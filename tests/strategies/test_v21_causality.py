"""Causal signal/indicator checks using generated prices, never held-out data."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from tests.strategies.test_sma_rsi_htf import (
    _bearish_fixture, _bullish_fixture, _context_from_m5_closes,
)
from xau_lab.data.features import _rsi
from xau_lab.strategies.sma_rsi_htf import (
    _aggregate_complete_minutes, build_sma_rsi_htf_state,
)


@pytest.mark.parametrize("prices", [_bullish_fixture, _bearish_fixture])
def test_future_ohlc_cannot_change_confirmed_signal_or_execution_atr(prices):
    ctx = _context_from_m5_closes(prices())
    original = build_sma_rsi_htf_state(ctx)
    entry = int(np.flatnonzero(original.signal)[0]) + 1
    changed = {}
    for field in ("open", "high", "low", "close"):
        values = getattr(ctx, field).copy()
        values[entry:] += 1_000_000
        changed[field] = values
    perturbed = build_sma_rsi_htf_state(replace(ctx, **changed))
    np.testing.assert_array_equal(perturbed.signal[:entry], original.signal[:entry])
    np.testing.assert_array_equal(perturbed.execution_atr[:entry], original.execution_atr[:entry])


@pytest.mark.parametrize("prices", [_bullish_fixture, _bearish_fixture])
def test_prefix_with_next_bucket_timestamp_reproduces_confirmed_state(prices):
    ctx = _context_from_m5_closes(prices())
    full = build_sma_rsi_htf_state(ctx)
    entry = int(np.flatnonzero(full.signal)[0]) + 1
    prefix = replace(ctx, **{
        name: getattr(ctx, name)[:entry+1]
        for name in ("open", "high", "low", "close", "spread", "atr14", "time_epoch")
    })
    short = build_sma_rsi_htf_state(prefix)
    np.testing.assert_array_equal(short.signal[:entry], full.signal[:entry])
    np.testing.assert_array_equal(short.execution_atr[:entry], full.execution_atr[:entry])


def test_sparse_bucket_ohlc_and_exact_m15_close_boundary():
    ctx = _context_from_m5_closes(np.array([10., 20., 30., 40.]))
    keep = np.array([0, 2, 4, 5, 14, 15, 16])
    ctx = replace(ctx, **{
        name: getattr(ctx, name)[keep]
        for name in ("open", "high", "low", "close", "spread", "atr14", "time_epoch")
    })
    m5 = _aggregate_complete_minutes(ctx, 5)
    m15 = _aggregate_complete_minutes(ctx, 15)
    assert m5.close_time.tolist() == [300, 600, 900, 1200]
    assert m5.source_last_index.tolist() == [2, 3, 4, 6]
    assert m5.close.tolist() == [10., 20., 30., 40.]
    assert m15.close_time.tolist() == [900, 1800]
    assert m15.open.tolist() == [10., 40.]
    assert m15.close.tolist() == [30., 40.]
    np.testing.assert_allclose(m15.high, [30.05, 40.05])
    np.testing.assert_allclose(m15.low, [9.95, 39.95])
    # The next M15 candle is unavailable at the exact previous close boundary.
    assert np.searchsorted(m15.close_time, 900, side="right") - 1 == 0


@pytest.mark.parametrize("close,expected", [
    ([100.] * 20, 50.), (list(range(100, 120)), 100.), (list(range(120, 100, -1)), 0.),
])
def test_rsi_seed_and_zero_gain_loss_contract(close, expected):
    rsi = _rsi(np.array(close, dtype=float), 14)
    assert np.isnan(rsi[:14]).all()
    np.testing.assert_array_equal(rsi[14:], np.full(len(close)-14, expected))


def test_rsi_mixed_seed_and_recursive_update():
    rsi = _rsi(np.array([100., 101., 102., 101., 103.]), 3)
    assert rsi[3] == pytest.approx(200 / 3)
    assert rsi[4] == pytest.approx(250 / 3)


def test_runbook_documents_frozen_sparse_replay_and_reserved_windows():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/sma_rsi_htf_v21_runbook.md").read_text(encoding="utf-8")
    config = json.loads((root / "config/sma_rsi_htf_v21_frozen.json").read_text())
    assert "nonempty M5/M15" in text
    assert "scripts/replay_sma_rsi_v21_standalone.py" in text
    assert "No strategy optimization" in text
    assert "250 ms" in text and "80 points" in text
    assert config["development_period"]["to_exclusive"] == "2026-08-31"
    assert "August 31 exclusive" in text
    assert config["prospective_validation"]["from"] in text
    assert config["final_oos"]["to_exclusive"] in text
