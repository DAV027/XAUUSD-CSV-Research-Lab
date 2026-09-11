from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SymbolSpec:
    point: float
    digits: int
    contract_size: float
    volume_min: float
    volume_step: float
    volume_max: float = 100.0
    def __post_init__(self):
        vals=(self.point,self.contract_size,self.volume_min,self.volume_step,self.volume_max)
        if any(v<=0 for v in vals): raise ValueError("symbol numeric fields must be positive")
        if self.digits < 0: raise ValueError("digits must be nonnegative")

@dataclass(frozen=True)
class CostModel:
    commission_round_trip_per_lot: float = 6.0
    slippage_points_per_fill: float = 5.0
    def __post_init__(self):
        if self.commission_round_trip_per_lot < 0 or self.slippage_points_per_fill < 0:
            raise ValueError("costs cannot be negative")

@dataclass(frozen=True)
class RiskModel:
    account_equity: float = 5000.0
    preferred_risk_usd: float = 2.0
    hard_risk_usd: float = 5.0
    max_lot: float = 0.10
    def __post_init__(self):
        if min(self.account_equity,self.preferred_risk_usd,self.hard_risk_usd,self.max_lot)<=0:
            raise ValueError("risk fields must be positive")
        if self.preferred_risk_usd > self.hard_risk_usd:
            raise ValueError("preferred risk cannot exceed hard cap")

@dataclass(frozen=True)
class ExitSpec:
    stop_atr: float
    target_r: float | None = None
    time_exit_minutes: int | None = None
    atr_trail: float | None = None
    def __post_init__(self):
        if self.stop_atr <= 0: raise ValueError("stop_atr must be > 0")
        if self.target_r is None and self.time_exit_minutes is None and self.atr_trail is None:
            raise ValueError("at least one exit beyond stop is required")
        if self.target_r is not None and self.target_r <= 0: raise ValueError("target_r must be >0")
        if self.time_exit_minutes is not None and self.time_exit_minutes <= 0: raise ValueError("time exit must be >0")
        if self.atr_trail is not None and self.atr_trail <= 0: raise ValueError("atr trail must be >0")

@dataclass(frozen=True)
class MarketBars:
    time_epoch: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread: np.ndarray
    atr: np.ndarray
    def __post_init__(self):
        n=len(self.open)
        if n == 0: raise ValueError("bars cannot be empty")
        if any(len(x)!=n for x in (self.time_epoch,self.high,self.low,self.close,self.spread,self.atr)):
            raise ValueError("bar arrays must have equal length")

@dataclass(frozen=True)
class SizingResult:
    feasible: bool
    lot: float
    stop_risk_usd: float
    preferred_lot: float

@dataclass(frozen=True)
class Trade:
    entry_index: int
    exit_index: int
    entry_time: int
    exit_time: int
    direction: int
    signal_price: float
    entry_price: float
    stop_price: float
    target_price: float | None
    exit_price: float
    exit_reason: str
    lot_size: float
    initial_risk_usd: float
    gross_pnl: float
    spread_cost: float
    commission: float
    slippage_cost: float
    net_pnl: float
    pnl_R: float | None
    hold_minutes: float
    risk_infeasible: bool = False

@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[Trade,...]
    risk_skip_count: int = 0

@dataclass(frozen=True)
class ExperimentSpec:
    direction_mode: str
    exit_spec: ExitSpec
