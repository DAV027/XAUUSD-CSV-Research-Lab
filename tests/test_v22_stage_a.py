"""Synthetic Stage A diagnostics; no candidate rules or market data required."""
import importlib.util
from pathlib import Path
import sys
import subprocess

import numpy as np
import polars as pl
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage = load("analyze_sma_rsi_v22_stage_a")
diagnostic = load("summarize_sma_rsi_v22_stage_a")


def test_metrics_accounting_and_chronological_losses():
    result = stage.metrics([{"net": x} for x in [10, -4, -6, 0, 3, -2]])
    assert result["net"] == 1
    assert result["wins"] == 2 and result["losses"] == 3
    assert result["n"] == 6
    assert result["pf"] == pytest.approx(13 / 12)
    assert result["expectancy"] == pytest.approx(1 / 6)
    assert result["max_losing_streak"] == 2
    assert result["net_ex_top5"] == -12
    assert result["realized_balance_dd_pct"] == pytest.approx(10 / 5010 * 100)


def test_top_five_removal_retains_sixth_winner():
    assert stage.metrics([{"net": x} for x in [1, 2, 3, 4, 5, 6, -10]])["net_ex_top5"] == -9


def test_empty_metrics_do_not_invent_rates():
    result = stage.metrics([])
    assert result["n"] == result["net"] == 0
    assert result["pf"] is result["win_rate"] is result["expectancy"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_outcome_rejected(bad):
    with pytest.raises(ValueError):
        stage.metrics([{"net": bad}])


def test_m1_cutoff_excludes_boundary_and_future_without_modifying_source(tmp_path):
    path = tmp_path / "synthetic.parquet"
    pl.DataFrame({"time": ["2026-08-30 23:59:00", "2026-08-31 00:00:00", "2027-01-01 00:00:00"],
                  **{k: [1., 999., 9999.] for k in ["open", "high", "low", "close"]}}).write_parquet(path)
    original = path.read_bytes()
    frame, plan = stage.load_permitted_m1(path)
    assert frame.height == 1 and frame["close"].to_list() == [1.]
    assert "SELECTION" in plan
    assert path.read_bytes() == original


def test_no_permitted_m1_rejected(tmp_path):
    path = tmp_path / "empty.parquet"
    pl.DataFrame(schema={"time": pl.String, **{k: pl.Float64 for k in ["open", "high", "low", "close"]}}).write_parquet(path)
    with pytest.raises(ValueError, match="no permitted"):
        stage.load_permitted_m1(path)


def test_percentile_uses_prior_window_and_is_prefix_invariant():
    values = np.array([1000., 1., 4., 6., 5., -9999.])
    assert stage.prior_percentile(values, 4, 3) == pytest.approx(2 / 3)
    assert stage.prior_percentile(values[:5], 4, 3) == stage.prior_percentile(values, 4, 3)


@pytest.mark.parametrize("values,index,window", [([1, 2], 1, 3), ([1, 2], 1, 0), ([1, 2], 2, 1), ([np.nan, 2], 1, 1)])
def test_invalid_percentile_rejected(values, index, window):
    with pytest.raises(ValueError):
        stage.prior_percentile(values, index, window)


@pytest.mark.parametrize("subdir", ["results/sma_rsi_htf_v21", "datasets/fxify_xauusdr/data"])
def test_output_cannot_overwrite_frozen_inputs(tmp_path, subdir):
    with pytest.raises(ValueError, match="overwrite frozen"):
        stage.analyze(tmp_path, tmp_path / subdir)


def test_bootstrap_constant_and_repeatable():
    result = diagnostic.block_bootstrap([2.] * 10, 3, n=20)
    assert result["total_pnl_2_5_50_97_5"] == [20., 20., 20.]
    assert result["fraction_resamples_positive"] == 1
    values = [1, -2, 5, 0, -1]
    assert diagnostic.block_bootstrap(values, 2) == diagnostic.block_bootstrap(values, 2)


@pytest.mark.parametrize("values,block,n", [([], 1, 10), ([1], 2, 10), ([1], 0, 10), ([1], 1, 0), ([np.inf], 1, 10)])
def test_invalid_bootstrap_rejected(values, block, n):
    with pytest.raises(ValueError):
        diagnostic.block_bootstrap(values, block, n=n)


def test_evidence_gates_remain_active_under_optimized_python():
    result = subprocess.run([sys.executable, "-O", "-c",
                             "from analyze_sma_rsi_v22_stage_a import require; require(False, 'evidence gate')"],
                            cwd=SCRIPTS, capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError: evidence gate" in result.stderr
