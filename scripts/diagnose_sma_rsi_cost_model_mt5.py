from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from compare_sma_rsi_lifecycle_mt5 import _extract_report_lifecycle


def _rounded(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose frozen V2.1 commission and swap patterns from the native MT5 Deals ledger."
    )
    parser.add_argument("--mt5-report", type=Path, required=True)
    parser.add_argument("--initial-balance", type=float, default=5000.0)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/COST_MODEL_DIAGNOSTIC.json"),
    )
    args = parser.parse_args()

    trades, final_balance = _extract_report_lifecycle(args.mt5_report)

    entry_commission = sum(float(t["entry_commission"]) for t in trades)
    exit_commission = sum(float(t["exit_commission"]) for t in trades)
    total_commission = entry_commission + exit_commission
    total_swap = sum(float(t["entry_swap"]) + float(t["exit_swap"]) for t in trades)
    total_price_profit = sum(float(t["entry_profit"]) + float(t["exit_profit"]) for t in trades)
    total_net = sum(float(t["reported_net_delta"]) for t in trades)

    commission_rates: dict[str, Counter] = {
        "entry": Counter(),
        "exit": Counter(),
    }
    swap_rates_by_direction: dict[str, Counter] = defaultdict(Counter)
    swap_examples: list[dict] = []

    nonzero_entry_commission = 0
    nonzero_exit_commission = 0
    nonzero_swap_trades = 0

    for index, trade in enumerate(trades, start=1):
        volume = float(trade["entry_volume"])
        if volume <= 0:
            continue

        entry_c = float(trade["entry_commission"])
        exit_c = float(trade["exit_commission"])
        swap = float(trade["entry_swap"]) + float(trade["exit_swap"])

        if entry_c != 0.0:
            nonzero_entry_commission += 1
            commission_rates["entry"][_rounded(entry_c / volume)] += 1
        if exit_c != 0.0:
            nonzero_exit_commission += 1
            commission_rates["exit"][_rounded(exit_c / volume)] += 1

        if swap != 0.0:
            nonzero_swap_trades += 1
            direction = str(trade["entry_direction"])
            per_lot = _rounded(swap / volume)
            swap_rates_by_direction[direction][per_lot] += 1
            if len(swap_examples) < 40:
                duration_hours = (
                    trade["exit_time"] - trade["entry_time"]
                ).total_seconds() / 3600.0
                swap_examples.append(
                    {
                        "trade_index": index,
                        "direction": direction,
                        "volume": volume,
                        "entry_time": trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_time": trade["exit_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_hours": duration_hours,
                        "swap": swap,
                        "swap_per_lot": per_lot,
                    }
                )

    summary = {
        "trade_count": len(trades),
        "initial_balance": args.initial_balance,
        "report_final_balance": final_balance,
        "price_profit_total": total_price_profit,
        "entry_commission_total": entry_commission,
        "exit_commission_total": exit_commission,
        "commission_total": total_commission,
        "swap_total": total_swap,
        "net_total": total_net,
        "expected_final_balance": args.initial_balance + total_net,
        "nonzero_entry_commission_deals": nonzero_entry_commission,
        "nonzero_exit_commission_deals": nonzero_exit_commission,
        "commission_per_lot_rates": {
            side: [
                {"rate": rate, "count": count}
                for rate, count in sorted(counter.items())
            ]
            for side, counter in commission_rates.items()
        },
        "nonzero_swap_trades": nonzero_swap_trades,
        "swap_per_lot_rates_by_direction": {
            direction: [
                {"rate": rate, "count": count}
                for rate, count in sorted(counter.items())
            ]
            for direction, counter in sorted(swap_rates_by_direction.items())
        },
        "swap_examples": swap_examples,
        "diagnostic_only": True,
        "interpretation_rule": (
            "Commission may be promoted to a deterministic standalone replay rule only if the "
            "per-lot schedule is stable. Swap requires a separately justified rollover rule; "
            "do not infer a universal broker swap model from a few realized examples."
        ),
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
