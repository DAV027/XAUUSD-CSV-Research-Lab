from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import polars as pl

from compare_sma_rsi_lifecycle_mt5 import (
    _classify_report_exit,
    _extract_report_lifecycle,
)


POINT = 0.01
DIGITS = 2
CONTRACT_SIZE = 100.0
RISK_PERCENT = 0.20
VOLUME_MIN = 0.01
VOLUME_MAX = 100.0
VOLUME_STEP = 0.01
MAX_SPREAD_POINTS = 80
ATR_MULTIPLIER = 1.5
DELAY_MS = 250
COMMISSION_PER_LOT_ENTRY = -6.0
SWAP_LONG_POINTS = -57.849
SWAP_SHORT_POINTS = 36.963
SWAP_MODE = "POINTS"
TRIPLE_SWAP_WEEKDAY = 2  # Python weekday(): Wednesday
WEEKDAY_MULTIPLIERS = {0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 0, 6: 0}


def _parse_python_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _utc_ms(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _m5_bucket(value: datetime) -> datetime:
    return value.replace(minute=(value.minute // 5) * 5, second=0, microsecond=0)


def _round_price(value: float) -> float:
    quantum = Decimal(1).scaleb(-DIGITS)
    return float(Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def _round_money(value: float) -> float:
    return float(Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _floor_volume(raw: float) -> float:
    normalized = math.floor(raw / VOLUME_STEP + 1e-10) * VOLUME_STEP
    if normalized < VOLUME_MIN:
        return 0.0
    return round(min(normalized, VOLUME_MAX), 8)


class TickStore:
    def __init__(self, root: Path, max_cached_partitions: int = 6):
        self.root = root
        self.max_cached_partitions = max_cached_partitions
        self.cache: OrderedDict[Path, pl.DataFrame] = OrderedDict()
        self.index: list[tuple[Path, int, int]] = []
        for path in sorted(root.glob("broker_date=*/ticks.parquet")):
            bounds = (
                pl.scan_parquet(path)
                .select(
                    pl.col("time_msc").min().alias("first"),
                    pl.col("time_msc").max().alias("last"),
                )
                .collect()
            )
            self.index.append((path, int(bounds["first"][0]), int(bounds["last"][0])))

    def _frame(self, path: Path) -> pl.DataFrame:
        if path in self.cache:
            frame = self.cache.pop(path)
            self.cache[path] = frame
            return frame
        frame = (
            pl.read_parquet(path, columns=["time_msc", "bid", "ask"])
            .sort("time_msc", maintain_order=True)
        )
        self.cache[path] = frame
        while len(self.cache) > self.max_cached_partitions:
            self.cache.popitem(last=False)
        return frame

    def window(self, start_ms: int, end_ms: int) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for path, first, last in self.index:
            if last < start_ms or first > end_ms:
                continue
            piece = self._frame(path).filter(
                (pl.col("time_msc") >= start_ms) & (pl.col("time_msc") <= end_ms)
            )
            if piece.height:
                frames.append(piece)
        if not frames:
            return pl.DataFrame(
                schema={"time_msc": pl.Int64, "bid": pl.Float64, "ask": pl.Float64}
            )
        return pl.concat(frames, how="vertical_relaxed").sort(
            "time_msc", maintain_order=True
        )

    def first_tick_from(self, start_ms: int, end_ms: int) -> dict | None:
        frame = self.window(start_ms, end_ms)
        return frame.row(0, named=True) if frame.height else None

    def last_tick_at_or_before(self, start_ms: int, end_ms: int) -> dict | None:
        frame = self.window(start_ms, end_ms)
        return frame.row(frame.height - 1, named=True) if frame.height else None

    def last_available_tick(self) -> dict:
        for path, _, _ in reversed(self.index):
            frame = self._frame(path)
            if frame.height:
                return frame.row(frame.height - 1, named=True)
        raise RuntimeError("tick store is empty")


def _first_exit_tick(
    store: TickStore,
    direction: str,
    sl: float,
    tp: float,
    after_ms: int,
    end_ms: int,
) -> tuple[str | None, dict | None]:
    for path, first, last in store.index:
        if last <= after_ms or first > end_ms:
            continue
        start = max(after_ms + 1, first)
        stop = min(end_ms, last)
        frame = store.window(start, stop)
        if frame.height == 0:
            continue

        if direction == "BUY":
            hits = frame.with_columns(
                (pl.col("bid") <= sl).alias("sl_hit"),
                (pl.col("bid") >= tp).alias("tp_hit"),
            ).filter(pl.col("sl_hit") | pl.col("tp_hit"))
        else:
            hits = frame.with_columns(
                (pl.col("ask") >= sl).alias("sl_hit"),
                (pl.col("ask") <= tp).alias("tp_hit"),
            ).filter(pl.col("sl_hit") | pl.col("tp_hit"))

        if hits.height:
            row = hits.row(0, named=True)
            sl_hit = bool(row["sl_hit"])
            tp_hit = bool(row["tp_hit"])
            if sl_hit and tp_hit:
                return "AMBIGUOUS", row
            return ("SL" if sl_hit else "TP"), row
    return None, None


def _swap_for_trade(direction: str, volume: float, entry: datetime, exit_: datetime) -> float:
    if exit_ <= entry:
        return 0.0

    # MT5 ledger evidence shows rollover at the UTC/log midnight boundary for
    # this frozen tester run. Charge based on the weekday being rolled FROM.
    boundary = datetime.combine((entry + timedelta(days=1)).date(), datetime.min.time())
    total = 0.0
    rate = SWAP_LONG_POINTS if direction == "BUY" else SWAP_SHORT_POINTS

    while boundary <= exit_:
        rolled_from = boundary - timedelta(days=1)
        multiplier = WEEKDAY_MULTIPLIERS[rolled_from.weekday()]
        if multiplier:
            # POINTS mode conversion:
            # swap_points * point * (tick_value / tick_size) * volume.
            # Here point=tick_size=.01 and tick_value=$1, so the conversion
            # factor is 1 USD per point per lot.
            raw = rate * volume * multiplier
            total += _round_money(raw)
        boundary += timedelta(days=1)

    return _round_money(total)


def _price_profit(direction: str, volume: float, entry: float, exit_: float) -> float:
    delta = (exit_ - entry) if direction == "BUY" else (entry - exit_)
    return _round_money(delta * CONTRACT_SIZE * volume)


def _expected_volume(balance: float, request_entry: float, sl: float) -> float:
    risk_money = balance * RISK_PERCENT / 100.0
    one_lot_loss = abs(request_entry - sl) * CONTRACT_SIZE
    if risk_money <= 0.0 or one_lot_loss <= 0.0:
        return 0.0
    return _floor_volume(risk_money / one_lot_loss)


def _exit_price(reason: str, direction: str, sl: float, tp: float, tick: dict) -> float:
    # In the MT5 real-tick tester, the protective level is the trigger condition;
    # the actual deal fill uses the market side on the triggering tick. This
    # preserves gap/slippage through SL/TP instead of forcing the nominal level.
    return float(tick["bid"] if direction == "BUY" else tick["ask"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone frozen V2.1 real-tick replay with MT5 comparison."
    )
    parser.add_argument("--python-signals", type=Path, required=True)
    parser.add_argument("--ticks-root", type=Path, required=True)
    parser.add_argument("--mt5-report", type=Path, required=True)
    parser.add_argument("--initial-balance", type=float, default=5000.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/STANDALONE_REPLAY.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/sma_rsi_htf_v21/STANDALONE_REPLAY_SUMMARY.json"),
    )
    args = parser.parse_args()

    with args.python_signals.open("r", encoding="utf-8", newline="") as handle:
        signals = list(csv.DictReader(handle))
    signals.sort(key=lambda row: _parse_python_time(row["entry_utc"]))

    report_trades, report_final_balance = _extract_report_lifecycle(args.mt5_report)
    store = TickStore(args.ticks_root)
    final_tick = store.last_available_tick()
    replay_end_ms = int(final_tick["time_msc"])

    balance = float(args.initial_balance)
    open_until_ms = -1
    replay_trades: list[dict] = []
    skip_counts = {"SPREAD": 0, "POSITION": 0, "VOLUME": 0, "NO_TICK": 0, "AMBIGUOUS": 0}

    for signal in signals:
        direction = signal["direction"].upper()
        entry_time = _parse_python_time(signal["entry_utc"])
        bucket = _m5_bucket(entry_time)
        bucket_start_ms = _utc_ms(bucket)
        bucket_end_ms = bucket_start_ms + 5 * 60 * 1000 - 1

        request_tick = store.first_tick_from(bucket_start_ms, bucket_end_ms)
        if request_tick is None:
            skip_counts["NO_TICK"] += 1
            continue

        request_ms = int(request_tick["time_msc"])
        spread_points = (
            (float(request_tick["ask"]) - float(request_tick["bid"])) / POINT
        )
        if spread_points > MAX_SPREAD_POINTS + 1e-9:
            skip_counts["SPREAD"] += 1
            continue

        if request_ms < open_until_ms:
            skip_counts["POSITION"] += 1
            continue

        atr = float(signal["m5_atr14"])
        request_entry = float(
            request_tick["ask"] if direction == "BUY" else request_tick["bid"]
        )
        distance = atr * ATR_MULTIPLIER
        if direction == "BUY":
            sl = _round_price(request_entry - distance)
            tp = _round_price(request_entry + distance)
        else:
            sl = _round_price(request_entry + distance)
            tp = _round_price(request_entry - distance)

        volume = _expected_volume(balance, request_entry, sl)
        if volume <= 0.0:
            skip_counts["VOLUME"] += 1
            continue

        fill_tick = store.last_tick_at_or_before(request_ms, request_ms + DELAY_MS)
        if fill_tick is None:
            skip_counts["NO_TICK"] += 1
            continue
        fill_ms = int(fill_tick["time_msc"])
        fill_price = float(
            fill_tick["ask"] if direction == "BUY" else fill_tick["bid"]
        )

        reason, exit_tick = _first_exit_tick(
            store, direction, sl, tp, fill_ms, replay_end_ms
        )
        if reason == "AMBIGUOUS":
            skip_counts["AMBIGUOUS"] += 1
            raise RuntimeError(
                f"ambiguous simultaneous SL/TP trigger for signal at {bucket}"
            )

        if reason is None or exit_tick is None:
            reason = "END_OF_TEST"
            exit_tick = final_tick

        exit_ms = int(exit_tick["time_msc"])
        exit_time = datetime.fromtimestamp(exit_ms / 1000.0, tz=timezone.utc).replace(
            tzinfo=None
        )
        exit_price = _exit_price(reason, direction, sl, tp, exit_tick)

        commission = _round_money(COMMISSION_PER_LOT_ENTRY * volume)
        swap = _swap_for_trade(direction, volume, datetime.fromtimestamp(fill_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None), exit_time)
        price_profit = _price_profit(direction, volume, fill_price, exit_price)
        net = _round_money(commission + swap + price_profit)
        balance = _round_money(balance + net)
        open_until_ms = exit_ms

        replay_trades.append(
            {
                "trade_index": len(replay_trades) + 1,
                "signal_bucket": bucket.strftime("%Y-%m-%d %H:%M:%S"),
                "direction": direction,
                "request_msc": request_ms,
                "request_entry": request_entry,
                "spread_points": spread_points,
                "atr": atr,
                "sl": sl,
                "tp": tp,
                "volume": volume,
                "fill_msc": fill_ms,
                "fill_price": fill_price,
                "exit_reason": reason,
                "exit_msc": exit_ms,
                "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_price": exit_price,
                "price_profit": price_profit,
                "commission": commission,
                "swap": swap,
                "net": net,
                "balance": balance,
            }
        )

    compared = min(len(replay_trades), len(report_trades))
    sequence_mismatches = 0
    volume_mismatches = 0
    fill_price_mismatches = 0
    exit_reason_mismatches = 0
    exit_second_mismatches = 0
    exit_price_mismatches = 0
    price_profit_mismatches = 0
    commission_mismatches = 0
    swap_mismatches = 0
    net_mismatches = 0
    balance_mismatches = 0
    mismatch_examples: list[dict] = []

    for i in range(compared):
        replay = replay_trades[i]
        report = report_trades[i]
        report_reason = _classify_report_exit(report["exit_comment"])

        sequence_match = (
            replay["direction"] == report["entry_direction"]
            and replay["signal_bucket"] == _m5_bucket(report["entry_time"]).strftime("%Y-%m-%d %H:%M:%S")
        )
        volume_match = abs(float(replay["volume"]) - float(report["entry_volume"])) < 1e-9
        fill_match = abs(float(replay["fill_price"]) - float(report["entry_price"])) <= POINT / 2 + 1e-12
        reason_match = replay["exit_reason"] == report_reason
        exit_second_match = replay["exit_time"] == report["exit_time"].strftime("%Y-%m-%d %H:%M:%S")
        exit_price_match = abs(float(replay["exit_price"]) - float(report["exit_price"])) <= POINT / 2 + 1e-12
        price_profit_match = abs(float(replay["price_profit"]) - (float(report["entry_profit"]) + float(report["exit_profit"]))) <= 0.011
        commission_match = abs(float(replay["commission"]) - (float(report["entry_commission"]) + float(report["exit_commission"]))) <= 0.011
        swap_match = abs(float(replay["swap"]) - (float(report["entry_swap"]) + float(report["exit_swap"]))) <= 0.011
        report_net = float(report["reported_net_delta"])
        net_match = abs(float(replay["net"]) - report_net) <= 0.011
        balance_match = abs(float(replay["balance"]) - float(report["exit_balance"])) <= 0.011

        checks = {
            "sequence": sequence_match,
            "volume": volume_match,
            "fill_price": fill_match,
            "exit_reason": reason_match,
            "exit_second": exit_second_match,
            "exit_price": exit_price_match,
            "price_profit": price_profit_match,
            "commission": commission_match,
            "swap": swap_match,
            "net": net_match,
            "balance": balance_match,
        }
        sequence_mismatches += int(not sequence_match)
        volume_mismatches += int(not volume_match)
        fill_price_mismatches += int(not fill_match)
        exit_reason_mismatches += int(not reason_match)
        exit_second_mismatches += int(not exit_second_match)
        exit_price_mismatches += int(not exit_price_match)
        price_profit_mismatches += int(not price_profit_match)
        commission_mismatches += int(not commission_match)
        swap_mismatches += int(not swap_match)
        net_mismatches += int(not net_match)
        balance_mismatches += int(not balance_match)

        if not all(checks.values()) and len(mismatch_examples) < 20:
            mismatch_examples.append(
                {
                    "trade_index": i + 1,
                    "replay": replay,
                    "report": {
                        "entry_time": report["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "direction": report["entry_direction"],
                        "volume": report["entry_volume"],
                        "entry_price": report["entry_price"],
                        "exit_time": report["exit_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_reason": report_reason,
                        "exit_price": report["exit_price"],
                        "price_profit": float(report["entry_profit"]) + float(report["exit_profit"]),
                        "commission": float(report["entry_commission"]) + float(report["exit_commission"]),
                        "swap": float(report["entry_swap"]) + float(report["exit_swap"]),
                        "net": report_net,
                        "balance": report["exit_balance"],
                    },
                    "checks": checks,
                }
            )

    trade_count_mismatch = len(replay_trades) != len(report_trades)
    final_balance_match = abs(balance - report_final_balance) <= 0.011

    summary = {
        "python_raw_signals": len(signals),
        "standalone_executed_trades": len(replay_trades),
        "mt5_report_trades": len(report_trades),
        "trade_count_mismatch": trade_count_mismatch,
        "skip_counts": skip_counts,
        "sequence_mismatches": sequence_mismatches,
        "volume_mismatches": volume_mismatches,
        "fill_price_mismatches": fill_price_mismatches,
        "exit_reason_mismatches": exit_reason_mismatches,
        "exit_second_mismatches": exit_second_mismatches,
        "exit_price_mismatches": exit_price_mismatches,
        "price_profit_mismatches": price_profit_mismatches,
        "commission_mismatches": commission_mismatches,
        "swap_mismatches": swap_mismatches,
        "net_mismatches": net_mismatches,
        "balance_mismatches": balance_mismatches,
        "standalone_final_balance": balance,
        "mt5_report_final_balance": report_final_balance,
        "final_balance_match": final_balance_match,
        "standalone_replay_parity_pass": (
            not trade_count_mismatch
            and sequence_mismatches == 0
            and volume_mismatches == 0
            and fill_price_mismatches == 0
            and exit_reason_mismatches == 0
            and exit_second_mismatches == 0
            and exit_price_mismatches == 0
            and price_profit_mismatches == 0
            and commission_mismatches == 0
            and swap_mismatches == 0
            and net_mismatches == 0
            and balance_mismatches == 0
            and final_balance_match
        ),
        "scope": (
            "Standalone frozen V2.1 replay. Trade decisions use the exported Python raw signals, "
            "FXIFY COPY_TICKS_ALL bid/ask data, frozen spread/position/risk/SL/TP rules, empirically "
            "validated 250 ms entry-delay rule, real-tick protective exits, -6 USD/lot entry commission, "
            "and FXIFY points-mode swap with Wednesday triple rollover. The MT5 report is used only after "
            "the replay as a parity comparator."
        ),
        "mismatch_examples": mismatch_examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if replay_trades:
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(replay_trades[0].keys()))
            writer.writeheader()
            writer.writerows(replay_trades)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"replay_csv={args.output}")
    print(f"summary_json={args.summary}")


if __name__ == "__main__":
    main()
