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

EDGE_D_DEFAULT_BUDGET = 10_000
EDGE_D_DEFAULT_SEED = 9_216_400
EDGE_D_SAMPLER_VERSION = "edge_d_v1"
EDGE_D_FAMILY = "edge_d_session_sweep_reclaim"
EDGE_D_STRATEGY = "prior_session_sweep_reclaim"

EDGE_D_TRANSITIONS = ("asia_to_london", "london_pre_ny_to_new_york")
EDGE_D_STOP_ATR_VALUES = (0.75, 1.0, 1.5, 2.0)
EDGE_D_EXIT_TYPES = ("target_r", "time")
EDGE_D_TARGET_R_VALUES = (1.0, 1.5, 2.0, 3.0)
EDGE_D_TIME_EXIT_MINUTES = (30, 60, 120, 240)


def _sample_int(low: int, high: int, unit: float) -> int:
    width = high - low + 1
    return int(min(high, low + floor(float(unit) * width)))


def _sample_float(low: float, high: float, unit: float) -> float:
    return float(low + float(unit) * (high - low))


def _sample_choice(values: tuple, unit: float):
    index = min(len(values) - 1, int(floor(float(unit) * len(values))))
    return values[index]


def _strategy_seed(seed: int, index: int) -> int:
    modulus = 2_147_483_647
    value = (int(seed) * 1_000_003 + index + 1) % modulus
    return value if value != 0 else modulus - 1


def _sample_exit(rng: np.random.Generator):
    exit_type = EDGE_D_EXIT_TYPES[int(rng.integers(0, len(EDGE_D_EXIT_TYPES)))]
    target_r = None
    time_exit_minutes = None
    if exit_type == "target_r":
        target_r = float(
            EDGE_D_TARGET_R_VALUES[int(rng.integers(0, len(EDGE_D_TARGET_R_VALUES)))]
        )
    else:
        time_exit_minutes = int(
            EDGE_D_TIME_EXIT_MINUTES[int(rng.integers(0, len(EDGE_D_TIME_EXIT_MINUTES)))]
        )
    return exit_type, target_r, time_exit_minutes


def generate_edge_d_catalog(
    total_budget: int = EDGE_D_DEFAULT_BUDGET,
    seed: int = EDGE_D_DEFAULT_SEED,
) -> list[CompleteExperiment]:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")

    lhs = qmc.LatinHypercube(d=5, seed=int(seed)).random(n=total_budget)
    exit_rng = np.random.default_rng(int(seed) ^ 0x0ED40001)
    catalog: list[CompleteExperiment] = []
    fingerprints: set[str] = set()

    for index in range(total_budget):
        unit = lhs[index]
        params = {
            "transition": _sample_choice(EDGE_D_TRANSITIONS, unit[0]),
            "entry_window_bars": _sample_int(30, 180, unit[1]),
            "sweep_atr": _sample_float(0.10, 1.00, unit[2]),
            "max_reclaim_bars": _sample_int(0, 10, unit[3]),
            "reclaim_depth_atr": _sample_float(0.00, 0.50, unit[4]),
        }
        params = json.loads(canonical_json(params))
        stop_atr = float(
            EDGE_D_STOP_ATR_VALUES[
                int(exit_rng.integers(0, len(EDGE_D_STOP_ATR_VALUES)))
            ]
        )
        exit_type, target_r, time_exit_minutes = _sample_exit(exit_rng)
        strategy_seed = _strategy_seed(seed, index)
        fingerprint = canonical_fingerprint(
            strategy_name=EDGE_D_STRATEGY,
            canonical_parameters=params,
            direction_mode="combined",
            stop_atr=stop_atr,
            exit_type=exit_type,
            target_r=target_r,
            time_exit_minutes=time_exit_minutes,
            atr_trail=None,
            commission_round_trip_per_lot=DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT,
            slippage_points_per_fill=DISCOVERY_SLIPPAGE_POINTS_PER_FILL,
            strategy_seed=strategy_seed,
            sampler_version=EDGE_D_SAMPLER_VERSION,
        )
        if fingerprint in fingerprints:
            raise RuntimeError("duplicate Edge D complete experiment fingerprint generated")
        fingerprints.add(fingerprint)
        catalog.append(
            CompleteExperiment(
                experiment_id="EXP" + fingerprint[:12].upper(),
                fingerprint=fingerprint,
                allocation_bucket=EDGE_D_FAMILY,
                family=EDGE_D_FAMILY,
                strategy_name=EDGE_D_STRATEGY,
                parameters=params,
                canonical_parameters_json=canonical_json(params),
                direction_mode="combined",
                stop_atr=stop_atr,
                exit_type=exit_type,
                target_r=target_r,
                time_exit_minutes=time_exit_minutes,
                atr_trail=None,
                commission_round_trip_per_lot=DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT,
                slippage_points_per_fill=DISCOVERY_SLIPPAGE_POINTS_PER_FILL,
                strategy_seed=strategy_seed,
                family_seed=int(seed),
                sampler_version=EDGE_D_SAMPLER_VERSION,
            )
        )

    return catalog


__all__ = [
    "EDGE_D_DEFAULT_BUDGET",
    "EDGE_D_DEFAULT_SEED",
    "EDGE_D_FAMILY",
    "EDGE_D_SAMPLER_VERSION",
    "EDGE_D_STRATEGY",
    "generate_edge_d_catalog",
]
