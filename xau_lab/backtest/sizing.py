from __future__ import annotations
import math
from .models import SymbolSpec,RiskModel,SizingResult

def _floor_step(x: float, step: float) -> float:
    return math.floor((x+1e-12)/step)*step

def size_for_stop(entry_price: float, stop_price: float, symbol: SymbolSpec, risk: RiskModel) -> SizingResult:
    per_lot=abs(entry_price-stop_price)*symbol.contract_size
    if per_lot <= 0: return SizingResult(False,0.0,0.0,0.0)
    preferred=risk.preferred_risk_usd/per_lot
    lot=min(risk.max_lot,symbol.volume_max,_floor_step(preferred,symbol.volume_step))
    if lot < symbol.volume_min-1e-12:
        minrisk=symbol.volume_min*per_lot
        if minrisk <= risk.hard_risk_usd+1e-12:
            return SizingResult(True,symbol.volume_min,minrisk,preferred)
        return SizingResult(False,0.0,minrisk,preferred)
    lot=max(symbol.volume_min,lot)
    actual=lot*per_lot
    if actual>risk.hard_risk_usd+1e-9:
        return SizingResult(False,0.0,actual,preferred)
    return SizingResult(True,round(lot,8),actual,preferred)
