from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

RAW_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]
CLEAN_COLUMNS = RAW_COLUMNS.copy()


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def raw_csv(self) -> Path:
        return self.root / "data/raw/XAUUSD_M1_RAW.csv"

    @property
    def clean_csv(self) -> Path:
        return self.root / "data/clean/XAUUSD_M1_CLEAN.csv"

    @property
    def features_parquet(self) -> Path:
        return self.root / "data/features/XAUUSD_M1_FEATURES.parquet"

    @property
    def quality_csv(self) -> Path:
        return self.root / "results/DATA_QUALITY_REPORT.csv"

    @property
    def metadata_json(self) -> Path:
        return self.root / "results/DATA_METADATA.json"


@dataclass(frozen=True)
class SymbolMetadata:
    symbol: str
    timeframe: str
    digits: int
    point: float
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    currency_profit: str
    server_timezone: str | None
    exported_rows: int
    first_time: str
    last_time: str

    def to_dict(self) -> dict:
        return asdict(self)


def canonicalize_bar_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(RAW_COLUMNS).with_columns(
        pl.col("time").cast(pl.String),
        pl.col(["open", "high", "low", "close"]).cast(pl.Float64),
        pl.col(["tick_volume", "spread", "real_volume"]).cast(pl.Int64),
    )
