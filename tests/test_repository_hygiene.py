"""Repository safeguards; all paths below are synthetic, never broker data."""
import os
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_real_tick_and_generated_data_paths_are_ignored():
    paths = [
        "datasets/fxify_xauusdr/data/ticks/broker_date=2026-06-01/ticks.parquet",
        "custom/broker_date=2026-06-01/ticks.parquet",
        "custom/broker_date=2026-06-01/ticks.parquet.tmp",
        "data/features/XAUUSD_M1_FEATURES.parquet",
        "data/raw/XAUUSD_M1_RAW.csv",
        "results/sma_rsi_htf_v21/STANDALONE_REPLAY.csv",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        input=("\0".join(paths) + "\0").encode(), capture_output=True, cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.decode().rstrip("\0").split("\0")) == set(paths)
    source = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        input=b"config/sma_rsi_htf_v21_frozen.json\0tests/test_repository_hygiene.py\0",
        capture_output=True, cwd=ROOT, check=False,
    )
    assert source.returncode == 1
    assert not source.stdout


def test_local_test_helpers_win_over_an_installed_tests_package(tmp_path):
    competing = tmp_path / "tests"
    competing.mkdir()
    (competing / "__init__.py").write_text(
        "raise RuntimeError('unrelated installed tests package')\n", encoding="utf-8"
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", "import tests.strategies.reference_signals"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_windows_timezone_data_dependency_is_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert 'tzdata; sys_platform == "win32"' in project["project"]["dependencies"]
