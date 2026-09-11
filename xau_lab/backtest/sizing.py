from __future__ import annotations

import math
from dataclasses import dataclass

from xau_lab.backtest.models import RiskModel, SymbolSpec


@dataclass(frozen=True)
class SizedPosition:
    feasible: bool
    lot: float
    risk_usd: float
    risk_usd_per_lot: float


def _floor_to_step(value: float, step: float) -> float:
    units = math.floor((value / step) + 1e-12)
    return round(units * step, 10)


def size_for_stop(
    entry_price: float,
    stop_price: float,
    symbol: SymbolSpec,
    risk: RiskModel,
) -> SizedPosition:
    stop_distance = abs(float(entry_price) - float(stop_price))
    if stop_distance <= 0:
        raise ValueError("entry and stop prices must differ")

    risk_usd_per_lot = stop_distance * symbol.contract_size
    preferred_lot = risk.preferred_risk_usd / risk_usd_per_lot
    executable_cap = min(risk.max_lot, symbol.volume_max)

    if executable_cap < symbol.volume_min:
        return SizedPosition(False, 0.0, 0.0, risk_usd_per_lot)

    if preferred_lot < symbol.volume_min:
        min_risk = risk_usd_per_lot * symbol.volume_min
        if min_risk > risk.hard_risk_usd + 1e-12:
            return SizedPosition(False, 0.0, 0.0, risk_usd_per_lot)
        lot = symbol.volume_min
    else:
        lot = _floor_to_step(min(preferred_lot, executable_cap), symbol.volume_step)
        if lot < symbol.volume_min:
            min_risk = risk_usd_per_lot * symbol.volume_min
            if min_risk > risk.hard_risk_usd + 1e-12 or symbol.volume_min > executable_cap:
                return SizedPosition(False, 0.0, 0.0, risk_usd_per_lot)
            lot = symbol.volume_min

    lot = min(lot, executable_cap)
    lot = _floor_to_step(lot, symbol.volume_step)
    if lot < symbol.volume_min:
        return SizedPosition(False, 0.0, 0.0, risk_usd_per_lot)

    actual_risk = risk_usd_per_lot * lot
    if actual_risk > risk.hard_risk_usd + 1e-12:
        return SizedPosition(False, 0.0, 0.0, risk_usd_per_lot)

    return SizedPosition(True, lot, actual_risk, risk_usd_per_lot)
