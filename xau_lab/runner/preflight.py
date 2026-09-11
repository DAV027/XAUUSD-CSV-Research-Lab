from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from xau_lab.experiments.spec import CompleteExperiment
from xau_lab.runner.single import MarketBundle

class CampaignPreflightError(ValueError):
    """Raised when campaign inputs are structurally unsafe to execute."""

@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    market_rows: int
    catalog_rows: int
    session_features_required: bool

_REQUIRED_SESSION_FEATURES = {
    "session_asia","session_london","session_new_york","session_overlap","broker_date","hour","minute",
}

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignPreflightError(message)

def validate_campaign_inputs(market: MarketBundle,experiments: Sequence[CompleteExperiment],*,expected_catalog_size: int | None = None) -> PreflightReport:
    bars = market.bars
    n = len(bars.open)
    _require(n > 1, "market must contain at least two rows")
    arrays = (bars.open, bars.high, bars.low, bars.close)
    _require(all(np.all(np.isfinite(a)) and np.all(a > 0.0) for a in arrays),"market requires positive finite OHLC values")
    _require(np.all(bars.high >= np.maximum(bars.open, bars.close)) and np.all(bars.low <= np.minimum(bars.open, bars.close)) and np.all(bars.high >= bars.low),"market contains invalid OHLC relationships")
    _require(np.all(np.isfinite(bars.spread)) and np.all(bars.spread >= 0.0),"market requires non-negative finite spread values")
    _require(np.all(np.diff(bars.time_epoch) > 0),"market timestamps must be strictly increasing")
    _require(len(market.context) == n,"strategy context row count must match market bars")
    for name, values in market.context.features.items():
        _require(len(values) == n, f"feature {name!r} row count must match market bars")
    catalog_rows = len(experiments)
    _require(catalog_rows > 0, "catalog cannot be empty")
    if expected_catalog_size is not None:
        _require(catalog_rows == expected_catalog_size,f"catalog has {catalog_rows} rows; expected {expected_catalog_size}")
    experiment_ids = [e.experiment_id for e in experiments]
    fingerprints = [e.fingerprint for e in experiments]
    _require(len(set(experiment_ids)) == catalog_rows,"catalog contains duplicate experiment_id values")
    _require(len(set(fingerprints)) == catalog_rows,"catalog contains duplicate fingerprint values")
    session_required = any(e.strategy_family == "session" for e in experiments)
    if session_required:
        missing = sorted(_REQUIRED_SESSION_FEATURES.difference(market.context.features))
        _require(not missing,"catalog contains session strategies but required session features are missing: " + ", ".join(missing))
    return PreflightReport(ok=True,market_rows=n,catalog_rows=catalog_rows,session_features_required=session_required)
