# XAUUSD CSV Research Lab

A deterministic research engine for broad XAUUSD M1 strategy discovery. It is designed to reject weak ideas quickly, account for trading costs from the start, preserve experiment identity, resume after interruption, and promote only candidates that survive chronological/stress validation.

## Research rules locked in

- XAUUSD only for Phase 1.
- All available M1 history from MT5.
- One open position maximum per experiment.
- Signal on completed bar `t`; earliest entry on `t+1` open.
- Recorded spread + $6/lot round-trip commission + 5 points/fill discovery slippage.
- Same-bar SL+TP ambiguity => SL first.
- Short SL/TP triggers are evaluated on the ask side using recorded spread.
- Preferred risk $2; normal hard risk ceiling $5; infeasible minimum-lot trades are skipped.
- Stage-1 promotion: after-cost PF >= 1.10, DD <= 5%, >=300 completed trades, positive expectancy, stability/concentration gates.
- Expanding and rolling 12m -> 3m walk-forward for survivors.
- Real tick MT5 validation only after CSV survivors exist.
- The prospective final holdout begins only after a strategy is frozen.

## Windows setup

Use Python 3.12. From PowerShell in this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

Or manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,mt5]"
python scripts\verify_environment.py
```

## 1. Prepare MetaTrader 5

Open the MT5 terminal that contains the XAUUSD history you want to research and log in. Ensure XAUUSD is visible in Market Watch and maximize the amount of chart/history data available in MT5.

The exporter requires an **explicit IANA timezone** for the timestamps written to the research CSV. Do not guess this value. The example session configuration uses `Etc/UTC` only as an example.

## 2. Export all available XAUUSD M1 bars

```powershell
python scripts\export_mt5_m1.py --symbol XAUUSD --server-timezone "Etc/UTC"
```

Add `--overwrite` only when you intentionally want to replace the immutable raw export.

Output:

`data/raw/XAUUSD_M1_RAW.csv`

and symbol metadata:

`results/DATA_METADATA.json`

## 3. Validate data

```powershell
python scripts\validate_data.py
```

Outputs:

- `data/clean/XAUUSD_M1_CLEAN.csv`
- `results/DATA_QUALITY_REPORT.csv`

The validator reports missing minutes and weekend closures but never fabricates bars.

## 4. Configure sessions

Copy:

```powershell
Copy-Item config\session_config.example.json config\session_config.json
```

Edit `config/session_config.json` to match the timezone used by the exported `time` column. Session experiments must not run against guessed timezone labels.

## 5. Build anti-lookahead features

```powershell
python scripts\build_features.py --session-config config\session_config.json
```

Output:

`data/features/XAUUSD_M1_FEATURES.parquet`

Signal-facing shared rolling features are shifted by one completed bar.

## 6. Experiment catalog

The supplied catalog is generated with seed `9215000`. It can be regenerated exactly:

```powershell
python scripts\create_catalog.py --budget 50000 --seed 9215000
```

Output:

`results/EXPERIMENT_CATALOG.csv`

`50,000` means 50,000 complete experiments, including entry parameters, direction mode, stop and exit model. It is not 50,000 entries multiplied by exits.

## 7. Mandatory real-data smoke run

```powershell
python scripts\run_smoke.py --count 100 --workers 5
```

Review:

- `results/MASTER_RESULTS.csv`
- `results/ERRORS.csv` if it exists
- `results/CHECKPOINT.json`
- `results/THROUGHPUT_BENCHMARK.json`

If `estimated_50000_hours` is above 24 hours, stop and profile before running all 50,000 experiments. Do not shorten history or weaken validation just to make the ETA look better.

## 8. Full campaign

```powershell
python scripts\run_campaign.py --workers 5
```

The parent process alone writes `MASTER_RESULTS.csv`. If the run stops, rerun the command; completed experiment IDs are reconstructed from the master CSV and are not repeated.

## 9. Stage-1 promotion

```powershell
python scripts\promote_candidates.py
```

Outputs:

- `results/REJECTED.csv`
- `results/SURVIVORS.csv`

A run with zero survivors is a valid research result. Do not weaken the gate merely to force a winner.

## 10. Robustness campaign

```powershell
python scripts\run_robustness.py
```

For survivors this runs both walk-forward schemes, parameter neighborhoods, spread/commission/slippage stress, block bootstrap, trade-order drawdown Monte Carlo, and top-trade-removal sensitivity.

Outputs include:

- `results/FOLD_RESULTS.csv`
- `results/STRESS_RESULTS.csv`
- `results/ROBUSTNESS_SUMMARY.json`
- `results/TOP_CANDIDATES.csv`
- `results/trade_logs/*.csv`

## 11. MT5 finalist handoff

Only robust candidates move to MQL5/MT5 real-tick verification. Finalist packages require:

- Every tick based on real ticks
- delays 0 / 100 / 250 / 500 / 1000 ms
- exact frozen parameters
- broker cost/risk audit
- forward/demo validation
- explicit approval before any real-money trading

The research engine itself is not intended to run on MetaTrader VPS. The final surviving strategy is converted to a self-contained MQL5 EA for VPS deployment.

## Existing Work/MT5 evidence

`outputs.zip` can be imported separately with:

```powershell
python scripts\import_historical_checkpoint.py C:\path\to\outputs.zip
```

Historical Work/MT5 rows remain evidence only; they are not inserted into `MASTER_RESULTS.csv` as if they were generated by this CSV engine.
