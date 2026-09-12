from __future__ import annotations

import json
from math import floor
from typing import Iterable

import numpy as np
from scipy.stats import qmc

# Import approved strategy modules for deterministic registry population.
from xau_lab.strategies import breakout as _breakout  # noqa: F401
from xau_lab.strategies import mean_reversion as _mean_reversion  # noqa: F401
from xau_lab.strategies import price_action as _price_action  # noqa: F401
from xau_lab.strategies import session as _session  # noqa: F401
from xau_lab.strategies import statistical as _statistical  # noqa: F401
from xau_lab.strategies import trend as _trend  # noqa: F401
from xau_lab.strategies import volatility as _volatility  # noqa: F401
from xau_lab.strategies.base import StrategyDefinition
from xau_lab.strategies.registry import list_strategies

from xau_lab.experiments.domains import (
    ATR_TRAIL_VALUES,
    DIRECTION_MODES,
    DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT,
    DISCOVERY_SLIPPAGE_POINTS_PER_FILL,
    EXIT_TYPES,
    FAMILY_BUDGET_WEIGHTS,
    SAMPLER_VERSION,
    STOP_ATR_VALUES,
    TARGET_R_VALUES,
    TIME_EXIT_MINUTES,
)
from xau_lab.experiments.spec import CompleteExperiment, canonical_fingerprint, canonical_json

APPROVED_FAMILIES = tuple(key for key in FAMILY_BUDGET_WEIGHTS if key != "exit_execution")


def _allocate_budget(total_budget: int) -> dict[str, int]:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    total_weight = sum(FAMILY_BUDGET_WEIGHTS.values())
    raw = {key: total_budget * weight / total_weight for key, weight in FAMILY_BUDGET_WEIGHTS.items()}
    counts = {key: floor(value) for key, value in raw.items()}
    remaining = total_budget - sum(counts.values())
    ranked = sorted(
        FAMILY_BUDGET_WEIGHTS,
        key=lambda key: (-(raw[key] - counts[key]), list(FAMILY_BUDGET_WEIGHTS).index(key)),
    )
    for key in ranked[:remaining]:
        counts[key] += 1
    return counts


def _definitions_for_family(family: str) -> tuple[StrategyDefinition, ...]:
    definitions = tuple(definition for definition in list_strategies() if definition.family == family)
    if not definitions:
        raise RuntimeError(f"no registered strategies for approved family {family!r}")
    return definitions


def _all_approved_definitions() -> tuple[StrategyDefinition, ...]:
    allowed = set(APPROVED_FAMILIES)
    definitions = tuple(definition for definition in list_strategies() if definition.family in allowed)
    if not definitions:
        raise RuntimeError("no approved strategies are registered")
    return definitions


def _numeric_domain_keys(definition: StrategyDefinition) -> list[str]:
    keys: list[str] = []
    for key, domain in definition.parameter_domain.items():
        if (
            isinstance(domain, tuple)
            and len(domain) == 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in domain)
        ):
            keys.append(key)
    return keys


def _sample_parameter(domain: object, unit: float, rng: np.random.Generator):
    if isinstance(domain, tuple) and len(domain) == 2 and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in domain
    ):
        low, high = domain
        if isinstance(low, int) and isinstance(high, int):
            width = high - low + 1
            return int(min(high, low + floor(unit * width)))
        return float(low + unit * (high - low))
    if isinstance(domain, (tuple, list)):
        if not domain:
            raise ValueError("categorical parameter domain cannot be empty")
        return domain[int(rng.integers(0, len(domain)))]
    raise TypeError(f"unsupported parameter domain: {domain!r}")


def _sample_parameters(
    definition: StrategyDefinition,
    lhs_row: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    numeric_index = 0
    params: dict = {}
    numeric_keys = set(_numeric_domain_keys(definition))
    for key in sorted(definition.parameter_domain):
        domain = definition.parameter_domain[key]
        if key in numeric_keys:
            unit = float(lhs_row[numeric_index])
            numeric_index += 1
        else:
            unit = 0.5
        params[key] = _sample_parameter(domain, unit, rng)
    return json.loads(canonical_json(params))


def _family_seed(seed: int, bucket_index: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(bucket_index), 0x584155]).generate_state(1, dtype=np.uint32)[0])


def _strategy_seed(seed: int, global_index: int) -> int:
    modulus = 2_147_483_647
    value = (int(seed) * 1_000_003 + global_index + 1) % modulus
    return value if value != 0 else modulus - 1


def _sample_exit(rng: np.random.Generator, *, focused: bool, row_index: int):
    if focused:
        exit_type = EXIT_TYPES[row_index % len(EXIT_TYPES)]
    else:
        exit_type = EXIT_TYPES[int(rng.integers(0, len(EXIT_TYPES)))]
    target_r = None
    time_exit = None
    atr_trail = None
    if exit_type in {"target_r", "target_time"}:
        target_r = float(TARGET_R_VALUES[int(rng.integers(0, len(TARGET_R_VALUES)))])
    if exit_type in {"time", "target_time"}:
        time_exit = int(TIME_EXIT_MINUTES[int(rng.integers(0, len(TIME_EXIT_MINUTES)))])
    if exit_type == "atr_trail":
        atr_trail = float(ATR_TRAIL_VALUES[int(rng.integers(0, len(ATR_TRAIL_VALUES)))])
    return exit_type, target_r, time_exit, atr_trail


def generate_catalog(total_budget: int = 50_000, seed: int = 9_215_000) -> list[CompleteExperiment]:
    counts = _allocate_budget(total_budget)
    catalog: list[CompleteExperiment] = []
    fingerprints: set[str] = set()
    global_index = 0

    for bucket_index, (bucket, count) in enumerate(counts.items()):
        if count == 0:
            continue
        definitions = _all_approved_definitions() if bucket == "exit_execution" else _definitions_for_family(bucket)
        max_dimensions = max((len(_numeric_domain_keys(definition)) for definition in definitions), default=0)
        family_seed = _family_seed(seed, bucket_index)
        lhs = qmc.LatinHypercube(d=max(1, max_dimensions), seed=family_seed).random(n=count)
        rng = np.random.default_rng(family_seed ^ 0xA5A5A5A5)
        order = np.arange(len(definitions), dtype=np.int64)
        rng.shuffle(order)

        for row_index in range(count):
            definition = definitions[int(order[row_index % len(order)])]
            params = _sample_parameters(definition, lhs[row_index], rng)
            direction_mode = DIRECTION_MODES[int(rng.integers(0, len(DIRECTION_MODES)))]
            stop_atr = float(STOP_ATR_VALUES[int(rng.integers(0, len(STOP_ATR_VALUES)))])
            exit_type, target_r, time_exit, atr_trail = _sample_exit(
                rng,
                focused=bucket == "exit_execution",
                row_index=row_index,
            )
            strategy_seed = _strategy_seed(seed, global_index)
            fingerprint = canonical_fingerprint(
                strategy_name=definition.name,
                canonical_parameters=params,
                direction_mode=direction_mode,
                stop_atr=stop_atr,
                exit_type=exit_type,
                target_r=target_r,
                time_exit_minutes=time_exit,
                atr_trail=atr_trail,
                commission_round_trip_per_lot=DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT,
                slippage_points_per_fill=DISCOVERY_SLIPPAGE_POINTS_PER_FILL,
                strategy_seed=strategy_seed,
                sampler_version=SAMPLER_VERSION,
            )
            if fingerprint in fingerprints:
                raise RuntimeError("duplicate complete experiment fingerprint generated")
            fingerprints.add(fingerprint)
            catalog.append(
                CompleteExperiment(
                    experiment_id="EXP" + fingerprint[:12].upper(),
                    fingerprint=fingerprint,
                    allocation_bucket=bucket,
                    family=definition.family,
                    strategy_name=definition.name,
                    parameters=params,
                    canonical_parameters_json=canonical_json(params),
                    direction_mode=direction_mode,
                    stop_atr=stop_atr,
                    exit_type=exit_type,
                    target_r=target_r,
                    time_exit_minutes=time_exit,
                    atr_trail=atr_trail,
                    commission_round_trip_per_lot=DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT,
                    slippage_points_per_fill=DISCOVERY_SLIPPAGE_POINTS_PER_FILL,
                    strategy_seed=strategy_seed,
                    family_seed=family_seed,
                    sampler_version=SAMPLER_VERSION,
                )
            )
            global_index += 1

    if len(catalog) != total_budget:
        raise RuntimeError(f"catalog length {len(catalog)} does not match requested budget {total_budget}")
    return catalog


__all__ = ["FAMILY_BUDGET_WEIGHTS", "generate_catalog"]
