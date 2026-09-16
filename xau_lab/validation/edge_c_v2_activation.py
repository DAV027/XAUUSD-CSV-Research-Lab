from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

ACTIVATION_SAMPLE_SIZE = 256
ACTIVATION_MIN_10_COUNT = 205
ACTIVATION_MIN_300_COUNT = 26
ACTIVATION_MAX_END_OF_DATA_FRACTION = 0.05


@dataclass(frozen=True)
class ActivationDecision:
    passed: bool
    reasons: tuple[str, ...]
    sample_size: int
    zero_trade_count: int
    at_least_10_count: int
    at_least_300_count: int
    min_completed_trades: int | None
    median_completed_trades: float | None
    p90_completed_trades: float | None
    max_completed_trades: int | None
    total_risk_skips: int
    total_completed_trades: int
    total_end_of_data_exits: int
    end_of_data_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "sample_size": self.sample_size,
            "zero_trade_count": self.zero_trade_count,
            "zero_trade_fraction": (
                self.zero_trade_count / self.sample_size if self.sample_size else None
            ),
            "at_least_10_count": self.at_least_10_count,
            "at_least_10_fraction": (
                self.at_least_10_count / self.sample_size if self.sample_size else None
            ),
            "at_least_300_count": self.at_least_300_count,
            "at_least_300_fraction": (
                self.at_least_300_count / self.sample_size if self.sample_size else None
            ),
            "min_completed_trades": self.min_completed_trades,
            "median_completed_trades": self.median_completed_trades,
            "p90_completed_trades": self.p90_completed_trades,
            "max_completed_trades": self.max_completed_trades,
            "total_risk_skips": self.total_risk_skips,
            "total_completed_trades": self.total_completed_trades,
            "total_end_of_data_exits": self.total_end_of_data_exits,
            "end_of_data_fraction": self.end_of_data_fraction,
        }


def spread_sample_indices(
    budget: int,
    sample_size: int = ACTIVATION_SAMPLE_SIZE,
) -> tuple[int, ...]:
    budget = int(budget)
    sample_size = int(sample_size)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > budget:
        raise ValueError("sample_size cannot exceed budget")

    values = np.rint(np.linspace(0, budget - 1, sample_size)).astype(np.int64)
    indices = tuple(int(value) for value in values)
    if len(indices) != len(set(indices)):
        raise RuntimeError("spread sample produced duplicate catalog indices")
    return indices


def _empty_decision(reason: str, sample_size: int) -> ActivationDecision:
    return ActivationDecision(
        passed=False,
        reasons=(reason,),
        sample_size=int(sample_size),
        zero_trade_count=0,
        at_least_10_count=0,
        at_least_300_count=0,
        min_completed_trades=None,
        median_completed_trades=None,
        p90_completed_trades=None,
        max_completed_trades=None,
        total_risk_skips=0,
        total_completed_trades=0,
        total_end_of_data_exits=0,
        end_of_data_fraction=None,
    )


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or float(value) != float(int(value)):
            return None
        parsed = int(value)
    else:
        return None
    return parsed if parsed >= 0 else None


def evaluate_activation(
    rows: Sequence[Mapping[str, object]],
) -> ActivationDecision:
    sample_size = len(rows)
    if sample_size != ACTIVATION_SAMPLE_SIZE:
        return _empty_decision("sample_size_not_256", sample_size)

    completed_values: list[int] = []
    total_risk_skips = 0
    total_end_of_data = 0
    seen_ids: set[str] = set()

    for row in rows:
        experiment_id = row.get("experiment_id")
        completed = _nonnegative_int(row.get("completed_trades"))
        risk_skips = _nonnegative_int(row.get("risk_skip_count"))
        end_of_data = _nonnegative_int(row.get("end_of_data_exits"))
        if (
            not isinstance(experiment_id, str)
            or not experiment_id
            or experiment_id in seen_ids
            or completed is None
            or risk_skips is None
            or end_of_data is None
            or end_of_data > completed
        ):
            return _empty_decision("malformed_structural_row", sample_size)
        seen_ids.add(experiment_id)
        completed_values.append(completed)
        total_risk_skips += risk_skips
        total_end_of_data += end_of_data

    completed_array = np.asarray(completed_values, dtype=np.int64)
    total_completed = int(completed_array.sum())
    zero_count = int(np.count_nonzero(completed_array == 0))
    at_least_10 = int(np.count_nonzero(completed_array >= 10))
    at_least_300 = int(np.count_nonzero(completed_array >= 300))
    end_fraction = (
        float(total_end_of_data / total_completed) if total_completed > 0 else None
    )

    reasons: list[str] = []
    if total_completed == 0:
        reasons.append("no_completed_trades")
    if at_least_10 < ACTIVATION_MIN_10_COUNT:
        reasons.append("configs_with_10_trades_below_205")
    if at_least_300 < ACTIVATION_MIN_300_COUNT:
        reasons.append("configs_with_300_trades_below_26")
    if end_fraction is not None and end_fraction > ACTIVATION_MAX_END_OF_DATA_FRACTION:
        reasons.append("end_of_data_fraction_above_5_pct")

    return ActivationDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        sample_size=sample_size,
        zero_trade_count=zero_count,
        at_least_10_count=at_least_10,
        at_least_300_count=at_least_300,
        min_completed_trades=int(completed_array.min()),
        median_completed_trades=float(np.median(completed_array)),
        p90_completed_trades=float(np.percentile(completed_array, 90)),
        max_completed_trades=int(completed_array.max()),
        total_risk_skips=int(total_risk_skips),
        total_completed_trades=total_completed,
        total_end_of_data_exits=int(total_end_of_data),
        end_of_data_fraction=end_fraction,
    )


__all__ = [
    "ACTIVATION_MAX_END_OF_DATA_FRACTION",
    "ACTIVATION_MIN_10_COUNT",
    "ACTIVATION_MIN_300_COUNT",
    "ACTIVATION_SAMPLE_SIZE",
    "ActivationDecision",
    "evaluate_activation",
    "spread_sample_indices",
]
