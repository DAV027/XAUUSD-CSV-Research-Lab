from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np


@dataclass(frozen=True)
class MarketBars:
    time_epoch: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread: np.ndarray
    atr: np.ndarray
    broker_timezone: str = "UTC"

    def __post_init__(self) -> None:
        arrays = {
            "time_epoch": np.asarray(self.time_epoch, dtype=np.int64),
            "open": np.asarray(self.open, dtype=np.float64),
            "high": np.asarray(self.high, dtype=np.float64),
            "low": np.asarray(self.low, dtype=np.float64),
            "close": np.asarray(self.close, dtype=np.float64),
            "spread": np.asarray(self.spread, dtype=np.int64),
            "atr": np.asarray(self.atr, dtype=np.float64),
        }
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("MarketBars arrays must have equal length")
        try:
            ZoneInfo(self.broker_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"invalid broker timezone: {self.broker_timezone}") from exc
        for name, value in arrays.items():
            value = np.ascontiguousarray(value)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return len(self.time_epoch)


@dataclass(frozen=True)
class SymbolSpec:
    point: float
    digits: int
    contract_size: float
    volume_min: float
    volume_step: float
    volume_max: float = 100.0

    def __post_init__(self) -> None:
        if self.point <= 0:
            raise ValueError("point must be positive")
        if self.digits < 0:
            raise ValueError("digits must be nonnegative")
        if self.contract_size <= 0:
            raise ValueError("contract_size must be positive")
        if self.volume_min <= 0 or self.volume_step <= 0 or self.volume_max <= 0:
            raise ValueError("volume limits must be positive")
        if self.volume_min > self.volume_max:
            raise ValueError("volume_min cannot exceed volume_max")


@dataclass(frozen=True)
class CostModel:
    commission_round_trip_per_lot: float = 6.0
    slippage_points_per_fill: float = 5.0

    def __post_init__(self) -> None:
        if self.commission_round_trip_per_lot < 0:
            raise ValueError("commission cannot be negative")
        if self.slippage_points_per_fill < 0:
            raise ValueError("slippage cannot be negative")


@dataclass(frozen=True)
class RiskModel:
    account_equity: float = 5000.0
    preferred_risk_usd: float = 2.0
    hard_risk_usd: float = 5.0
    max_lot: float = 0.10

    def __post_init__(self) -> None:
        if self.account_equity <= 0:
            raise ValueError("account_equity must be positive")
        if self.preferred_risk_usd <= 0 or self.hard_risk_usd <= 0:
            raise ValueError("risk limits must be positive")
        if self.preferred_risk_usd > self.hard_risk_usd:
            raise ValueError("preferred_risk_usd cannot exceed hard_risk_usd")
        if self.max_lot <= 0:
            raise ValueError("max_lot must be positive")


@dataclass(frozen=True)
class ExitSpec:
    stop_atr: float
    target_r: float | None = None
    time_exit_minutes: int | None = None
    atr_trail: float | None = None

    def __post_init__(self) -> None:
        if self.stop_atr <= 0:
            raise ValueError("stop_atr must be positive")
        if self.target_r is not None and self.target_r <= 0:
            raise ValueError("target_r must be positive when configured")
        if self.time_exit_minutes is not None and self.time_exit_minutes <= 0:
            raise ValueError("time_exit_minutes must be positive when configured")
        if self.atr_trail is not None and self.atr_trail <= 0:
            raise ValueError("atr_trail must be positive when configured")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str = ""
    parameter_set_id: str = ""
    fingerprint: str = ""
    family: str = ""
    strategy_name: str = ""
    parameters: tuple[tuple[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Trade:
    experiment_id: str | None
    parameter_set_id: str | None
    fingerprint: str | None
    signal_time: int
    entry_time: int
    direction: int
    signal_price: float
    entry_price: float
    stop_price: float
    target_price: float | None
    exit_time: int
    exit_price: float
    exit_reason: str
    lot_size: float
    planned_risk_usd: float
    risk_R: float
    gross_pnl: float
    spread_cost: float
    commission: float
    slippage_cost: float
    net_pnl: float
    pnl_R: float | None
    hold_minutes: float
    broker_date: str
    session_asia: bool | None = None
    session_london: bool | None = None
    session_new_york: bool | None = None
    session_overlap: bool | None = None
    entry_index: int = -1
    exit_index: int = -1
    risk_infeasible: bool = False
    broker_timezone: str = "UTC"

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if self.lot_size < 0:
            raise ValueError("lot_size cannot be negative")


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[Trade, ...]
    risk_skip_count: int = 0

    def __post_init__(self) -> None:
        if self.risk_skip_count < 0:
            raise ValueError("risk_skip_count cannot be negative")

    @property
    def net_pnl(self) -> float:
        return float(sum(trade.net_pnl for trade in self.trades))
