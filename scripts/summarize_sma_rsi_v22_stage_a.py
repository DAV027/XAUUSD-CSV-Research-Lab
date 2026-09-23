import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from analyze_sma_rsi_v22_stage_a import metrics, require


def block_bootstrap(values, block, seed=222026, n=5000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 1 <= block <= len(values) or n < 1:
        raise ValueError("bootstrap requires finite daily values and valid block/replicate counts")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values)-block+1, size=(n, int(np.ceil(len(values)/block))))
    indexes = (starts[:, :, None]+np.arange(block)).reshape(n, -1)[:, :len(values)]
    totals = values[indexes].sum(axis=1)
    return dict(block_days=block, seed=seed, replicates=n,
                total_pnl_2_5_50_97_5=np.quantile(totals, [.025,.5,.975]).tolist(),
                fraction_resamples_positive=float(np.mean(totals>0)))


def summarize(output: Path, dataset_root: Path):
    OUT = output
    DATA = dataset_root
    rows=pl.read_csv(OUT/"ENTRY_FEATURES.csv").to_dicts()
    day=date(2026,6,1)
    days=[]
    while day < date(2026,8,31):
        if day.weekday()<5:
            days.append(day.isoformat())
        day+=timedelta(days=1)
    daily=[sum(r["net"] for r in rows if r["day"]==d) for d in days]
    require(abs(sum(daily)-sum(r["net"] for r in rows)) < 1e-7, "weekday calendar omitted trade outcomes")
    bootstrap=[block_bootstrap(daily,b) for b in (3,5,7)]
    cutoff_ms=int(datetime(2026,8,31,tzinfo=timezone.utc).timestamp()*1000)
    partition_bounds=[]
    for path in sorted((DATA/"data/ticks").glob("broker_date=*/ticks.parquet")):
        partition_day=path.parent.name.removeprefix("broker_date=")
        require("2026-06-01" <= partition_day < "2026-08-31", 'Stage A evidence check failed: "2026-06-01" <= partition_day < "2026-08-31"')
        meta=pq.read_metadata(path)
        col=meta.schema.names.index("time_msc")
        minimum=[]
        maximum=[]
        for i in range(meta.num_row_groups):
            stats=meta.row_group(i).column(col).statistics
            require(stats and stats.has_min_max, 'Stage A evidence check failed: stats and stats.has_min_max')
            minimum.append(int(stats.min)); maximum.append(int(stats.max))
        require(max(maximum) < cutoff_ms, 'Stage A evidence check failed: max(maximum) < cutoff_ms')
        partition_bounds.append(dict(path=str(path), rows=meta.num_rows,
                                     min_time_msc=min(minimum),max_time_msc=max(maximum)))
    require(sum(x["rows"] for x in partition_bounds)==21607135, 'Stage A evidence check failed: sum(x["rows"] for x in partition_bounds)==21607135')
    # Fixed-ledger arithmetic stress; NOT a new path-dependent execution replay.
    scenarios=[("reference",0.,0.),("commission_plus_25pct",.25,0.),
               ("commission_plus_50pct",.50,0.),("plus_1_point_each_fill",0.,1.),
               ("plus_2_points_each_fill",0.,2.),("commission_25pct_plus_1_point_each_fill",.25,1.)]
    costs=[]
    for label,commission,points in scenarios:
        stressed=[dict(r,net=r["net"]-abs(r["commission"])*commission-2*points*.01*100*r["volume"]) for r in rows]
        for scope in ("ALL","BUY","SELL","2026-06","2026-07","2026-08"):
            subset=stressed if scope=="ALL" else [r for r in stressed if r["direction"]==scope or r["month"]==scope]
            costs.append(dict(scenario=label,scope=scope,**metrics(subset)))
    pl.DataFrame(costs).write_csv(OUT/"FIXED_LEDGER_COST_SENSITIVITY.csv")
    diagnostic=dict(bootstrap=bootstrap,calendar="UTC Monday-Friday, all 65 dates including zero-trade dates; not a certified exchange calendar",
                    daily_pnl=dict(zip(days,daily)),partition_bounds=partition_bounds,
                    tick_returns_read=False, tick_metadata_only=True,
                    sum_volume=sum(r["volume"] for r in rows),
                    top5_winners=[{k:r[k] for k in ("trade_index","entry_time","direction","net")} for r in sorted(rows,key=lambda r:r["net"],reverse=True)[:5]],
                    holding_time_by_outcome={outcome:dict(n=len(group), median_minutes=float(np.median([r["holding_minutes"] for r in group]))) for outcome,group in
                        (("win",[r for r in rows if r["net"]>0]),("loss",[r for r in rows if r["net"]<0]))})
    (OUT/"ROBUSTNESS_DIAGNOSTICS.json").write_text(json.dumps(diagnostic,indent=2,allow_nan=False),encoding="utf-8")
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description="Stage A fixed-ledger diagnostics, not execution stress.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.output, args.dataset_root)
    print(json.dumps(result["bootstrap"], indent=2))


if __name__ == "__main__":
    main()
