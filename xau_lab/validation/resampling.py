from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

RESAMPLING_SEED = 9_216_000
DEFAULT_RESAMPLES = 5_000
SEQUENCE_ONLY_LABEL = "sequence_only_not_entry_edge_proof"


def _values(items: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(items), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("resampling input must be a nonempty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("resampling input must contain only finite values")
    return values


def _validate_n(n: int) -> int:
    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer")
    return int(n)


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    q = np.quantile(values, [0.05, 0.50, 0.95])
    return float(q[0]), float(q[1]), float(q[2])


def bootstrap_days(
    daily_pnl: Iterable[float],
    n: int = DEFAULT_RESAMPLES,
    seed: int = RESAMPLING_SEED,
) -> dict[str, float | int]:
    values = _values(daily_pnl)
    n = _validate_n(n)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(n, len(values)))
    samples = values[indices]
    means = samples.mean(axis=1)
    totals = samples.sum(axis=1)
    mean_p5, mean_p50, mean_p95 = _quantiles(means)
    total_p5, total_p50, total_p95 = _quantiles(totals)
    return {
        "seed": int(seed),
        "n": n,
        "days_per_sample": len(values),
        "mean_daily_p5": mean_p5,
        "mean_daily_p50": mean_p50,
        "mean_daily_p95": mean_p95,
        "total_pnl_p5": total_p5,
        "total_pnl_p50": total_p50,
        "total_pnl_p95": total_p95,
    }


def _max_drawdown(sequence: np.ndarray) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in sequence:
        equity += float(value)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return float(max_drawdown)


def shuffle_trade_order(
    trade_pnl: Iterable[float],
    n: int = DEFAULT_RESAMPLES,
    seed: int = RESAMPLING_SEED,
) -> dict[str, float | int | str]:
    values = _values(trade_pnl)
    n = _validate_n(n)
    rng = np.random.default_rng(int(seed))
    drawdowns = np.empty(n, dtype=np.float64)
    for i in range(n):
        drawdowns[i] = _max_drawdown(rng.permutation(values))
    p5, p50, p95 = _quantiles(drawdowns)
    return {
        "seed": int(seed),
        "n": n,
        "trade_count": len(values),
        "total_pnl": float(values.sum()),
        "max_drawdown_p5": p5,
        "max_drawdown_p50": p50,
        "max_drawdown_p95": p95,
        "label": SEQUENCE_ONLY_LABEL,
    }


def top_trade_removal(
    trade_pnl: Iterable[float],
    counts: Sequence[int] = (1, 5, 10),
) -> dict[str, object]:
    values = _values(trade_pnl)
    baseline = float(values.sum())
    ordered = np.sort(values)[::-1]
    scenarios: list[dict[str, object]] = []
    for requested in counts:
        if int(requested) != requested or requested <= 0:
            raise ValueError("removal counts must be positive integers")
        removed = min(int(requested), len(ordered))
        remaining = float(baseline - ordered[:removed].sum())
        scenarios.append(
            {
                "requested_count": int(requested),
                "removed_count": removed,
                "net_profit": remaining,
                "turns_negative": bool(baseline >= 0.0 and remaining < 0.0),
            }
        )
    return {
        "baseline_net_profit": baseline,
        "trade_count": len(values),
        "scenarios": scenarios,
    }


__all__ = [
    "DEFAULT_RESAMPLES",
    "RESAMPLING_SEED",
    "SEQUENCE_ONLY_LABEL",
    "bootstrap_days",
    "shuffle_trade_order",
    "top_trade_removal",
]
