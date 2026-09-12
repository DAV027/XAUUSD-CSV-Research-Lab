from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
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
) -> dict[str, object]:
    """Bootstrap an already aggregated daily P/L path.

    `block_bootstrap_daily` is the canonical ledger-facing interface. This
    lower-level helper remains available for callers that already hold daily
    aggregates.
    """
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
        "source_daily_pnl": [float(value) for value in values],
        "mean_daily_p5": mean_p5,
        "mean_daily_p50": mean_p50,
        "mean_daily_p95": mean_p95,
        "total_pnl_p5": total_p5,
        "total_pnl_p50": total_p50,
        "total_pnl_p95": total_p95,
    }


def _trade_value(trade: object, field: str) -> object:
    if isinstance(trade, Mapping):
        if field not in trade:
            raise ValueError(f"trade is missing required field {field!r}")
        return trade[field]
    if not hasattr(trade, field):
        raise ValueError(f"trade is missing required field {field!r}")
    return getattr(trade, field)


def block_bootstrap_daily(
    trades: Iterable[object],
    n: int = DEFAULT_RESAMPLES,
    seed: int = RESAMPLING_SEED,
) -> dict[str, object]:
    """Aggregate net P/L by broker date, then resample whole days.

    Whole-day blocks preserve dependence among trades that happened on the same
    broker date; individual trades are never resampled independently here.
    """
    daily: dict[str, float] = defaultdict(float)
    count = 0
    for trade in trades:
        broker_date = str(_trade_value(trade, "broker_date"))
        net_pnl = float(_trade_value(trade, "net_pnl"))
        if not np.isfinite(net_pnl):
            raise ValueError("trade net_pnl must be finite")
        daily[broker_date] += net_pnl
        count += 1
    if count == 0:
        raise ValueError("resampling input must contain at least one trade")
    ordered_daily = [daily[key] for key in sorted(daily)]
    result = bootstrap_days(ordered_daily, n=n, seed=seed)
    result["trade_count"] = count
    result["source_broker_dates"] = sorted(daily)
    return result


def _max_drawdown(sequence: np.ndarray) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in sequence:
        equity += float(value)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return float(max_drawdown)


def trade_order_monte_carlo(
    trade_pnl: Iterable[float],
    n: int = DEFAULT_RESAMPLES,
    seed: int = RESAMPLING_SEED,
) -> dict[str, float | int | str]:
    """Shuffle trade order only to estimate drawdown sequence sensitivity."""
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


def shuffle_trade_order(
    trade_pnl: Iterable[float],
    n: int = DEFAULT_RESAMPLES,
    seed: int = RESAMPLING_SEED,
) -> dict[str, float | int | str]:
    """Backward-compatible alias for `trade_order_monte_carlo`."""
    return trade_order_monte_carlo(trade_pnl, n=n, seed=seed)


def _profit_factor(values: np.ndarray) -> float | None:
    gross_profit = float(values[values > 0.0].sum())
    gross_loss = float(values[values < 0.0].sum())
    if gross_loss == 0.0:
        return None
    return float(gross_profit / abs(gross_loss))


def _relative_sensitivity_pct(baseline: float | None, stressed: float | None) -> float | None:
    if baseline is None or stressed is None or baseline == 0.0:
        return None
    return float((baseline - stressed) / abs(baseline) * 100.0)


def top_trade_removal(
    trade_pnl: Iterable[float],
    counts: Sequence[int] = (1, 5, 10),
) -> dict[str, object]:
    """Remove only the largest profitable trades and recompute net/PF evidence."""
    values = _values(trade_pnl)
    baseline_net = float(values.sum())
    baseline_pf = _profit_factor(values)
    positive_indices = [int(i) for i in np.where(values > 0.0)[0]]
    positive_indices.sort(key=lambda i: (-float(values[i]), i))

    scenarios: list[dict[str, object]] = []
    for requested in counts:
        if int(requested) != requested or requested <= 0:
            raise ValueError("removal counts must be positive integers")
        requested = int(requested)
        removed_indices = positive_indices[:requested]
        keep = np.ones(len(values), dtype=bool)
        if removed_indices:
            keep[np.asarray(removed_indices, dtype=np.int64)] = False
        remaining_values = values[keep]
        remaining_net = float(remaining_values.sum())
        remaining_pf = _profit_factor(remaining_values)
        scenarios.append(
            {
                "requested_count": requested,
                "removed_count": len(removed_indices),
                "removed_profit": float(values[removed_indices].sum()) if removed_indices else 0.0,
                "net_profit": remaining_net,
                "profit_factor": remaining_pf,
                "net_profit_sensitivity_pct": _relative_sensitivity_pct(
                    baseline_net, remaining_net
                ),
                "profit_factor_sensitivity_pct": _relative_sensitivity_pct(
                    baseline_pf, remaining_pf
                ),
                "turns_negative": bool(baseline_net >= 0.0 and remaining_net < 0.0),
            }
        )
    return {
        "baseline_net_profit": baseline_net,
        "baseline_profit_factor": baseline_pf,
        "trade_count": len(values),
        "profitable_trade_count": len(positive_indices),
        "scenarios": scenarios,
    }


__all__ = [
    "DEFAULT_RESAMPLES",
    "RESAMPLING_SEED",
    "SEQUENCE_ONLY_LABEL",
    "block_bootstrap_daily",
    "bootstrap_days",
    "shuffle_trade_order",
    "top_trade_removal",
    "trade_order_monte_carlo",
]
