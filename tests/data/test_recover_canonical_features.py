from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from scripts.recover_canonical_features import promote_if_sha_matches, select_research_prefix


def test_recovery_script_runs_as_direct_cli_entrypoint() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "recover_canonical_features.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Recover the frozen canonical XAUUSD feature artifact" in completed.stdout


def test_select_research_prefix_filters_before_taking_exact_row_count() -> None:
    frame = pl.DataFrame(
        {
            "time": [
                "2016-12-30 23:59:00",
                "2017-01-02 00:00:00",
                "2017-01-02 00:01:00",
                "2017-01-02 00:02:00",
                "2017-01-02 00:03:00",
            ],
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.1, 2.1, 3.1, 4.1, 5.1],
            "low": [0.9, 1.9, 2.9, 3.9, 4.9],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "tick_volume": [1, 1, 1, 1, 1],
            "spread": [1, 1, 1, 1, 1],
            "real_volume": [0, 0, 0, 0, 0],
        }
    )

    recovered = select_research_prefix(frame, expected_rows=3)

    assert recovered.height == 3
    assert recovered.get_column("time").to_list() == [
        "2017-01-02 00:00:00",
        "2017-01-02 00:01:00",
        "2017-01-02 00:02:00",
    ]


def test_select_research_prefix_rejects_insufficient_history() -> None:
    frame = pl.DataFrame(
        {
            "time": ["2017-01-02 00:00:00", "2017-01-02 00:01:00"],
            "open": [2.0, 3.0],
            "high": [2.1, 3.1],
            "low": [1.9, 2.9],
            "close": [2.0, 3.0],
            "tick_volume": [1, 1],
            "spread": [1, 1],
            "real_volume": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="expected at least 3 research rows"):
        select_research_prefix(frame, expected_rows=3)


def test_promote_if_sha_matches_refuses_to_overwrite_on_hash_mismatch(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.parquet"
    canonical = tmp_path / "canonical.parquet"
    candidate.write_bytes(b"candidate")
    canonical.write_bytes(b"existing-canonical")

    wrong_hash = sha256(b"something-else").hexdigest()

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        promote_if_sha_matches(candidate, canonical, wrong_hash)

    assert canonical.read_bytes() == b"existing-canonical"
    assert candidate.read_bytes() == b"candidate"


def test_promote_if_sha_matches_atomically_replaces_target_on_exact_hash(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.parquet"
    canonical = tmp_path / "canonical.parquet"
    payload = b"exact-canonical-candidate"
    candidate.write_bytes(payload)
    canonical.write_bytes(b"old")

    expected_hash = sha256(payload).hexdigest()

    observed_hash = promote_if_sha_matches(candidate, canonical, expected_hash)

    assert observed_hash == expected_hash
    assert canonical.read_bytes() == payload
    assert not candidate.exists()
