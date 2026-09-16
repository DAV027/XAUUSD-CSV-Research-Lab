from __future__ import annotations

import json
from math import floor

import numpy as np
from scipy.stats import qmc

from xau_lab.experiments.domains import (
    DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT,
    DISCOVERY_SLIPPAGE_POINTS_PER_FILL,
)
from xau_lab.experiments.spec import CompleteExperiment, canonical_fingerprint, canonical_json

EDGE_B_V2_DEFAULT_BUDGET = 10_000
EDGE_B_V2_DEFAULT_SEED = 9_216_100
EDGE_B_V2_SAMPLER_VERSION = "edge_b_v2"
EDGE_B_V2_FAMILY = "edge_b_trend_pullback_v2"
EDGE_B_V2_STRATEGY = "trend_pullback_recovery_v2"

EDGE_B_V2_STOP_ATR_VALUES = (1.5, 2.0, 3.0, 4.0)
EDGE_B_V2_EXIT_TYPES = ("target_r", "atr_trail")
EDGE_B_V2_TARGET_R_VALUES = (1.5, 2.0, 3.0, 4.0)
EDGE_B_V2_ATR_TRAIL_VALUES = (1.0, 1.5, 2.0, 3.0)


def _sample_int(low: int, high: int, unit: float) -> int:
    width = high - low + 1
    return int(min(high, low + floor(float(unit) * width)))


def _sample_float(low: float, high: float, unit: float) -> float:
    return float(low + float(unit) * (high - low))


def _strategy_seed(seed: int, index: int) -> int:
    modulus = 2_147_483_647
    value = (int(seed) * 1_000_003 + index + 1) % modulus
    return value if value != 0 else modulus - 1


def _sample_exit(rng: np.random.Generator):
    exit_type = EDGE_B_V2_EXIT_TYPES[int(rng.integers(0, len(EDGE_B_V2_EXIT_TYPES)))]
    target_r = None
    atr_trail = None
    if exit_type == "target_r":
        target_r = float(
            EDGE_B_V2_TARGET_R_VALUES[int(rng.integers(0, len(EDGE_B_V2_TARGET_R_VALUES)))]
        )
    else:
        atr_trail = float(
            EDGE_B_V2_ATR_TRAIL_VALUES[int(rng.integers(0, len(EDGE_B_V2_ATR_TRAIL_VALUES)))]
        )
    return exit_type, target_r, atr_trail


def generate_edge_b_v2_catalog(
    total_budget: int = EDGE_B_V2_DEFAULT_BUDGET,
    seed: int = EDGE_B_V2_DEFAULT_SEED,
) -> list[CompleteExperiment]:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")

    lhs = qmc.LatinHypercube(d=5, seed=int(seed)).random(n=total_budget)
    rng = np.random.default_rng(int(seed) ^ 0x0E0B2027)
    catalog: list[CompleteExperiment] = []
    fingerprints: set[str] = set()

    for index in range(total_budget):
        unit = lhs[index]
        params = {
            "trend_lookback": _sample_int(120, 480, unit[0]),
            "trend_threshold_atr": _sample_float(4.0, 12.0, unit[1]),
            "pullback_lookback": _sample_int(5, 30, unit[2]),
            "pullback_threshold_atr": _sample_float(1.0, 4.0, unit[3]),
            "recovery_threshold_atr": _sample_float(0.25, 1.5, unit[4]),
        }
        params = json.loads(canonical_json(params))
        stop_atr = float(
            EDGE_B_V2_STOP_ATR_VALUES[int(rng.integers(0, len(EDGE_B_V2_STOP_ATR_VALUES)))]
        )
        exit_type, target_r, atr_trail = _sample_exit(rng)
        strategy_seed = _strategy_seed(seed, index)
        fingerprint = canonical_fingerprint(
            strategy_name=EDGE_B_V2_STRATEGY,
            canonical_parameters=params,
            direction_mode="combined",
            stop_atr=stop_atr,
            exit_type=exit_type,
            target_r=target_r,
            time_exit_minutes=None,
            atr_trail=atr_trail,
            commission_round_trip_per_lot=DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT,
            slippage_points_per_fill=DISCOVERY_SLIPPAGE_POINTS_PER_FILL,
            strategy_seed=strategy_seed,
            sampler_version=EDGE_B_V2_SAMPLER_VERSION,
        )
        if fingerprint in fingerprints:
            raise RuntimeError("duplicate Edge B v2 complete experiment fingerprint generated")
        fingerprints.add(fingerprint)
        catalog.append(
            CompleteExperiment(
                experiment_id="EXP" + fingerprint[:12].upper(),
                fingerprint=fingerprint,
                allocation_bucket=EDGE_B_V2_FAMILY,
                family=EDGE_B_V2_FAMILY,
                strategy_name=EDGE_B_V2_STRATEGY,
                parameters=params,
                canonical_parameters_json=canonical_json(params),
                direction_mode="combined",
                stop_atr=stop_atr,
                exit_type=exit_type,
                target_r=target_r,
                time_exit_minutes=None,
                atr_trail=atr_trail,
                commission_round_trip_per_lot=DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT,
                slippage_points_per_fill=DISCOVERY_SLIPPAGE_POINTS_PER_FILL,
                strategy_seed=strategy_seed,
                family_seed=int(seed),
                sampler_version=EDGE_B_V2_SAMPLER_VERSION,
            )
        )

    return catalog


__all__ = [
    "EDGE_B_V2_DEFAULT_BUDGET",
    "EDGE_B_V2_DEFAULT_SEED",
    "EDGE_B_V2_FAMILY",
    "EDGE_B_V2_SAMPLER_VERSION",
    "EDGE_B_V2_STRATEGY",
    "generate_edge_b_v2_catalog",
]
