"""Read-only Stage A analysis. No candidate strategy is backtested here."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from compare_sma_rsi_lifecycle_mt5 import _extract_report_lifecycle
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.sma_rsi_htf import _aggregate_complete_minutes, build_sma_rsi_htf_state

CUTOFF = datetime(2026, 8, 31)


def require(condition, message):
    """Evidence gates remain active under Python optimization mode."""
    if not condition:
        raise ValueError(message)


def load_permitted_m1(path):
    query = (pl.scan_parquet(path)
             .filter(pl.col("time") < "2026-08-31 00:00:00")
             .select("time", "open", "high", "low", "close"))
    frame = query.collect()
    if frame.height == 0 or frame["time"].max() >= "2026-08-31 00:00:00":
        raise ValueError("no permitted M1 history or cutoff violation")
    return frame, query.explain()


def prior_percentile(values, index, window=288):
    if window < 1 or index < 0 or index >= len(values):
        raise ValueError("invalid percentile window or index")
    if index < window:
        raise ValueError("insufficient trailing history")
    prior = np.asarray(values[index-window:index], dtype=float)
    if not np.isfinite(prior).all() or not np.isfinite(values[index]):
        raise ValueError("nonfinite percentile input")
    return float(np.mean(prior <= values[index]))


def metrics(rows):
    values = [float(r["net"]) for r in rows]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite trade outcome")
    wins = [x for x in values if x > 0]
    losses = [x for x in values if x < 0]
    balance = peak = 5000.
    dd = 0.
    streak = longest = 0
    for v in values:
        balance += v
        peak = max(peak, balance)
        dd = max(dd, (peak-balance)/peak*100)
        streak = streak+1 if v < 0 else 0
        longest = max(longest, streak)
    return dict(n=len(values), wins=len(wins), losses=len(losses), net=round(sum(values), 2),
                expectancy=float(np.mean(values)) if values else None,
                pf=sum(wins)/-sum(losses) if losses else None,
                win_rate=len(wins)/len(values) if values else None,
                average_winner=float(np.mean(wins)) if wins else None,
                average_loser=float(np.mean(losses)) if losses else None,
                max_losing_streak=longest, realized_balance_dd_pct=dd,
                net_ex_top5=round(sum(values)-sum(sorted(wins, reverse=True)[:5]), 2))


def analyze(root: Path, output: Path):
    SOURCE = root / "results/sma_rsi_htf_v21"
    DATA = root / "datasets/fxify_xauusdr"
    OUT = output.resolve()
    if OUT.is_relative_to(SOURCE.resolve()) or OUT.is_relative_to(DATA.resolve()):
        raise ValueError("output must not overwrite frozen source evidence or datasets")
    OUT.mkdir(parents=True, exist_ok=True)
    # Inspect only declared development evidence. Validate times before attribution.
    report, final_balance = _extract_report_lifecycle(SOURCE / "ReportTester-719702901.xlsx")
    require(len(report) == 364, 'Stage A evidence check failed: len(report) == 364')
    require(all(datetime(2026, 6, 1) <= r["entry_time"] <= r["exit_time"] < CUTOFF for r in report), 'Stage A evidence check failed: all(datetime(2026, 6, 1) <= r["entry_time"] <= r["exit_time"] < CUTOFF for r in report)')
    with (SOURCE / "FXIFY_PYTHON_SIGNAL_PARITY.csv").open(newline="", encoding="utf-8") as f:
        signals = list(csv.DictReader(f))
    require(len(signals) == 450, 'Stage A evidence check failed: len(signals) == 450')
    require(all(datetime.fromisoformat(r["entry_utc"]).replace(tzinfo=None) < CUTOFF for r in signals), 'Stage A evidence check failed: all(datetime.fromisoformat(r["entry_utc"]).replace(tzinfo=None) < CUTOFF for r in signals)')
    by_signal = {(int(r["entry_epoch"])//300, r["direction"]): r for r in signals}
    require(len(by_signal) == 450, 'Stage A evidence check failed: len(by_signal) == 450')
    with (SOURCE / "STANDALONE_REPLAY.csv").open(newline="", encoding="utf-8") as f:
        replay = list(csv.DictReader(f))
    require(len(replay) == len(report), 'Stage A evidence check failed: len(replay) == len(report)')
    require(all(datetime.fromisoformat(r["exit_time"]) < CUTOFF for r in replay), 'Stage A evidence check failed: all(datetime.fromisoformat(r["exit_time"]) < CUTOFF for r in replay)')

    feature_path = DATA / "data/features/XAUUSD_M1_FEATURES.parquet"
    frame, query_plan = load_permitted_m1(feature_path)
    (OUT / "input_query.txt").write_text(query_plan, encoding="utf-8")
    epochs = frame.select(pl.col("time").str.to_datetime(format="%Y-%m-%d %H:%M:%S")
                         .dt.replace_time_zone("EET", ambiguous="raise", non_existent="raise")
                         .dt.convert_time_zone("UTC").dt.epoch("s"))["time"].to_numpy()
    require(np.all(np.diff(epochs) > 0), 'Stage A evidence check failed: np.all(np.diff(epochs) > 0)')
    ctx = StrategyContext(**{k: frame[k].to_numpy() for k in ("open", "high", "low", "close")},
                          spread=np.zeros(len(epochs)), atr14=np.zeros(len(epochs)),
                          time_epoch=epochs, features={})
    print(f"Permitted M1 rows={len(ctx)}, latest={frame['time'].max()}", flush=True)
    state = build_sma_rsi_htf_state(ctx)
    bars = _aggregate_complete_minutes(ctx, 5)
    m5_index = {int(t): i for i, t in enumerate(state.m5_close_time)}
    rows = []
    ranges = bars.high-bars.low
    for number, (trade, actual) in enumerate(zip(report, replay), 1):
        entry = trade["entry_time"]
        epoch = int(entry.replace(tzinfo=timezone.utc).timestamp())
        direction = trade["entry_direction"]
        sign = 1 if direction == "BUY" else -1
        sig = by_signal[(epoch//300, direction)]
        i = m5_index[int(sig["m5_close_epoch"])]
        atr = float(state.m5_atr[i])
        require(i >= 288 and atr > 0, 'Stage A evidence check failed: i >= 288 and atr > 0')
        require(actual["direction"] == direction, 'Stage A evidence check failed: actual["direction"] == direction')
        require(int(actual["request_msc"])//300_000 == epoch//300, 'Stage A evidence check failed: int(actual["request_msc"])//300_000 == epoch//300')
        require(float(actual["volume"]) == trade["entry_volume"], 'Stage A evidence check failed: float(actual["volume"]) == trade["entry_volume"]')
        require(abs(float(sig["m5_atr14"])-atr) < 1e-7, 'Stage A evidence check failed: abs(float(sig["m5_atr14"])-atr) < 1e-7')
        require(abs(float(sig["m5_sma21"])-state.m5_sma_slow[i]) < 1e-7, 'Stage A evidence check failed: abs(float(sig["m5_sma21"])-state.m5_sma_slow[i]) < 1e-7')
        require(int(sig["m15_close_epoch"]) <= int(sig["m5_close_epoch"]) <= epoch, 'Stage A evidence check failed: int(sig["m15_close_epoch"]) <= int(sig["m5_close_epoch"]) <= epoch')
        require(state.signal[int(state.m5_source_last_index[i])] == sign, 'Stage A evidence check failed: state.signal[int(state.m5_source_last_index[i])] == sign')
        require(abs(float(actual["net"])-trade["reported_net_delta"]) <= .011, 'Stage A evidence check failed: abs(float(actual["net"])-trade["reported_net_delta"]) <= .011')
        close = float(bars.close[i])
        previous = bars.close[i-20:i+1]
        path = float(np.sum(np.abs(np.diff(previous))))
        high20 = float(np.max(bars.high[i-20:i]))
        low20 = float(np.min(bars.low[i-20:i]))
        width = float(ranges[i])
        prior_atr = state.m5_atr[i-288:i]
        require(np.isfinite(prior_atr).all(), 'Stage A evidence check failed: np.isfinite(prior_atr).all()')
        london = entry.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/London"))
        ny = entry.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York"))
        lon_open = 8 <= london.hour < 17
        ny_open = 8 <= ny.hour < 17
        session = "overlap" if lon_open and ny_open else "london" if lon_open else "new_york" if ny_open else "other"
        slope = sign * float(state.m5_sma_slow[i]-state.m5_sma_slow[i-3])/atr
        breakout = (sign == 1 and close > high20) or (sign == -1 and close < low20)
        pullback = (bars.low[i] <= state.m5_sma_slow[i] < close if sign == 1 else
                    bars.high[i] >= state.m5_sma_slow[i] > close)
        net = float(trade["reported_net_delta"])
        rows.append(dict(
            trade_index=number, entry_time=entry.isoformat(), direction=direction,
            month=entry.strftime("%Y-%m"), day=entry.date().isoformat(),
            net=net, price_profit=trade["entry_profit"]+trade["exit_profit"],
            commission=trade["entry_commission"]+trade["exit_commission"],
            swap=trade["entry_swap"]+trade["exit_swap"], volume=trade["entry_volume"],
            winner=net > 0, trend_efficiency=abs(float(previous[-1]-previous[0]))/path if path else 0.,
            sma_separation_atr=sign*float(state.m5_sma_fast[i]-state.m5_sma_slow[i])/atr,
            sma_slope_aligned_atr=slope,
            htf_distance_atr=sign*(float(sig["m15_close"])-float(sig["m15_sma200"]))/atr,
            rsi=float(state.m5_rsi[i]), rsi_momentum_aligned=sign*float(state.m5_rsi[i]-state.m5_rsi[i-3]),
            atr=atr, atr_percentile=prior_percentile(state.m5_atr, i),
            spread_points=float(actual["spread_points"]), spread_atr=float(actual["spread_points"])*.01/atr,
            utc_hour=entry.hour, broker_hour=entry.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("EET")).hour,
            weekday=entry.weekday(), session=session,
            distance_high_atr=(high20-close)/atr, distance_low_atr=(close-low20)/atr,
            candle_body_range=abs(close-float(bars.open[i]))/width if width else 0.,
            candle_body_aligned=sign*(close-float(bars.open[i]))/width if width else 0.,
            upper_wick_fraction=(float(bars.high[i])-max(close,float(bars.open[i])))/width if width else 0.,
            lower_wick_fraction=(min(close,float(bars.open[i]))-float(bars.low[i]))/width if width else 0.,
            breakout=bool(breakout), pullback=bool(pullback),
            volatility_expansion=float(np.mean(ranges[i-4:i+1])/np.mean(ranges[i-24:i-4])),
            slope_htf_alignment=slope > 0,
            m5_m15_alignment=bool(sign*(state.m5_sma_fast[i]-state.m5_sma_slow[i])>0 and sign*(float(sig["m15_close"])-float(sig["m15_sma200"]))>0),
            previous_inside=bool(bars.high[i-1] <= bars.high[i-2] and bars.low[i-1] >= bars.low[i-2]),
            signal_inside=bool(bars.high[i] <= bars.high[i-1] and bars.low[i] >= bars.low[i-1]),
            previous_body_aligned=sign*float(bars.close[i-1]-bars.open[i-1])/atr,
            holding_minutes=(trade["exit_time"]-entry).total_seconds()/60,
        ))
    require(round(sum(r["net"] for r in rows),2) == 80.25, 'Stage A evidence check failed: round(sum(r["net"] for r in rows),2) == 80.25')
    require(final_balance == 5080.25, 'Stage A evidence check failed: final_balance == 5080.25')
    table = pl.DataFrame(rows)
    table.write_csv(OUT / "ENTRY_FEATURES.csv")
    summary = dict(overall=metrics(rows), monthly={m:metrics([r for r in rows if r["month"]==m]) for m in sorted({r["month"] for r in rows})},
                   sides={s:metrics([r for r in rows if r["direction"]==s]) for s in ("BUY","SELL")},
                   costs={k:round(sum(r[k] for r in rows),2) for k in ("price_profit","commission","swap")})
    numeric = ["trend_efficiency","sma_separation_atr","sma_slope_aligned_atr","htf_distance_atr","rsi",
               "rsi_momentum_aligned","atr","atr_percentile","spread_points","spread_atr","distance_high_atr",
               "distance_low_atr","candle_body_range","candle_body_aligned","upper_wick_fraction",
               "lower_wick_fraction","volatility_expansion","previous_body_aligned"]
    summary["winner_loser_features"] = {key:{label:dict(n=len(group),median=float(np.median([r[key] for r in group])),
        mean=float(np.mean([r[key] for r in group]))) for label,group in (
            ("win",[r for r in rows if r["net"]>0]),("loss",[r for r in rows if r["net"]<0]))} for key in numeric}
    groups=[]
    for field in ("direction","month","session","weekday","utc_hour","broker_hour","breakout","pullback","slope_htf_alignment","m5_m15_alignment","previous_inside","signal_inside"):
        for value in sorted({r[field] for r in rows},key=str):
            part=[r for r in rows if r[field]==value]
            groups.append(dict(feature=field,group=str(value),scope="ALL",**metrics(part)))
            for scope in ("BUY","SELL","2026-06","2026-07","2026-08"):
                sub=[r for r in part if r["direction"]==scope or r["month"]==scope]
                groups.append(dict(feature=field,group=str(value),scope=scope,**metrics(sub)))
    # June-only broad quantiles are fixed for descriptive July/August bins.
    thresholds={}
    for key in numeric:
        cuts=np.quantile([r[key] for r in rows if r["month"]=="2026-06"],[1/3,2/3])
        thresholds[key]=cuts.tolist()
        for label in range(3):
            part=[r for r in rows if int(np.searchsorted(cuts,r[key],side="right"))==label]
            for scope in ("ALL","BUY","SELL","2026-06","2026-07","2026-08"):
                sub=part if scope=="ALL" else [r for r in part if r["direction"]==scope or r["month"]==scope]
                groups.append(dict(feature=key,group=("low","middle","high")[label],scope=scope,**metrics(sub)))
    pl.DataFrame(groups).write_csv(OUT / "FEATURE_DECOMPOSITION.csv")
    summary["june_only_tertile_cuts"] = thresholds
    summary["input_provenance"] = dict(
        repository_base="3d63a68218a921b50c5e1406eab355c58c026a88",
        artifact_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in (
            "ReportTester-719702901.xlsx","FXIFY_PYTHON_SIGNAL_PARITY.csv","STANDALONE_REPLAY.csv")},
        feature_projection_sha256=hashlib.sha256(b"".join(np.ascontiguousarray(getattr(ctx,k)).tobytes() for k in
            ("time_epoch","open","high","low","close"))).hexdigest(),
        permitted_m1_rows=len(ctx), first_broker_time=frame["time"].min(), last_broker_time=frame["time"].max(),
        cutoff="2026-08-31 00:00 exclusive", signal_clock="UTC", feature_source_clock="EET",
        outcome_source="MT5 reported net delta; existing standalone provides request-time spread only",
        stage="observational_attribution_not_candidate_backtest")
    (OUT / "STAGE_A_SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Read-only V2.2 Stage A attribution; no candidate backtest.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.source_root, args.output)
    print(json.dumps({k:summary[k] for k in ("overall","monthly","sides","costs")},indent=2))


if __name__ == "__main__":
    main()
