from pathlib import Path
import polars as pl
from xau_lab.data.schema import RAW_COLUMNS, DataPaths, canonicalize_bar_frame


def test_raw_columns_are_exact_and_ordered():
    assert RAW_COLUMNS == [
        "time", "open", "high", "low", "close",
        "tick_volume", "spread", "real_volume",
    ]


def test_data_paths_match_spec(tmp_path: Path):
    paths = DataPaths(tmp_path)
    assert paths.raw_csv == tmp_path / "data/raw/XAUUSD_M1_RAW.csv"
    assert paths.clean_csv == tmp_path / "data/clean/XAUUSD_M1_CLEAN.csv"
    assert paths.features_parquet == tmp_path / "data/features/XAUUSD_M1_FEATURES.parquet"
    assert paths.quality_csv == tmp_path / "results/DATA_QUALITY_REPORT.csv"
    assert paths.metadata_json == tmp_path / "results/DATA_METADATA.json"


def test_canonicalize_reorders_and_types_columns():
    frame = pl.DataFrame({
        "close": [2001.0], "time": ["2026-01-02 12:00:00"],
        "open": [2000.0], "high": [2002.0], "low": [1999.0],
        "spread": [35], "tick_volume": [50], "real_volume": [0],
    })
    out = canonicalize_bar_frame(frame)
    assert out.columns == RAW_COLUMNS
    assert out.schema["open"] == pl.Float64
    assert out.schema["spread"] == pl.Int64
