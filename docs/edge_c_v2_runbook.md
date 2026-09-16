# Edge C v2 Activation-First Runbook

Edge C v2 is a separate discovery hypothesis. It preserves Edge C v1 as historical evidence and tests volatility-scaled compression using `ATR * sqrt(compression_lookback)`.

The full 10,000-experiment campaign is **not** the first step. Edge C v2 must pass its frozen structural activation diagnostic first.

## Research identity

```text
family:        edge_c_breakout_retest_v2
strategy:      compression_breakout_retest_v2
sampler:       edge_c_v2
budget:        10000
seed:          9216300
workers:       2
result root:   edge_c_v2/results/
```

Do not alter these values under the v2 identity.

## 1. Update the repository after PR merge

From the canonical local repository:

```powershell
cd E:\XAUUSD-CSV-Research-Lab
git checkout main
git pull
```

Do not delete or reset canonical `data\` or `results\`.

## 2. Create the frozen Edge C v2 catalog

```powershell
python -m scripts.create_edge_c_v2_catalog `
  --budget 10000 `
  --seed 9216300
```

Expected output:

```text
edge_c_v2\results\EXPERIMENT_CATALOG.csv
```

The catalog is deterministic. Do not regenerate it with a different seed or modify rows after observing diagnostic results.

## 3. Run the mandatory 256-experiment activation diagnostic

```powershell
python -m scripts.run_edge_c_v2_activation
```

This uses 256 deterministic indices spread across the entire frozen 10,000-row catalog. It runs detailed backtests only to count structural activation quantities:

- completed trades;
- risk skips;
- zero-trade configurations;
- end-of-data exits.

The activation decision does **not** use net profit, profit factor, expectancy, drawdown, win rate, or returns.

Output:

```text
edge_c_v2\results\ACTIVATION_REPORT.json
```

### Frozen activation PASS criteria

All three conditions must pass:

```text
>= 205 / 256 sampled experiments have >= 10 completed trades
>= 26 / 256 sampled experiments have >= 300 completed trades
end-of-data exits <= 5% of all completed diagnostic trades
```

Total completed diagnostic trades must also be greater than zero.

### If activation reports FAIL

**Stop Edge C v2.**

Do not run the 10,000-experiment campaign. Preserve `ACTIVATION_REPORT.json` as the negative research result. Do not loosen the activation thresholds, alter v2 domains, select a different 256-row subset, or inspect diagnostic economics to redesign v2.

Any structural change after an activation failure is a new hypothesis such as Edge C v3.

### If activation reports PASS

PASS means only that Edge C v2 generates enough trades to be economically testable. It is **not evidence of profitability**.

The campaign wrapper independently verifies that the PASS report matches the current catalog SHA256, current feature SHA256, frozen v2 identity, frozen seed, and 10,000-row budget. A stale or copied PASS file cannot authorize another catalog or dataset.

## 4. Smoke 20 experiments after activation PASS

Only after activation PASS:

```powershell
python -m scripts.run_edge_c_v2_campaign `
  --workers 2 `
  --limit 20
```

Inspect that the command completes normally. Do not delete `edge_c_v2\results` after the smoke run.

## 5. Resume the same campaign to all 10,000

```powershell
python -m scripts.run_edge_c_v2_campaign `
  --workers 2 `
  --allow-slow
```

The same catalog, feature file, activation report, result root, seed, and manifest remain authoritative.

Expected campaign result:

```text
edge_c_v2\results\MASTER_RESULTS.csv
```

Zero profitable experiments or zero promotion survivors is a valid negative result. Do not weaken thresholds to manufacture survivors.

## 6. Run the unchanged robustness/promotion pipeline

Only after the full 10,000 campaign completes:

```powershell
python -m scripts.promote_candidates `
  --master edge_c_v2\results\MASTER_RESULTS.csv `
  --catalog edge_c_v2\results\EXPERIMENT_CATALOG.csv `
  --features data\features\XAUUSD_M1_FEATURES.parquet `
  --result-root edge_c_v2\results `
  --run-manifest edge_c_v2\results\RUN_MANIFEST.json
```

This is the repository's existing robustness process. Edge C v2 does not redefine its Stage-1 or robustness thresholds.

## 7. Apply the frozen Edge C v2 viability selector

```powershell
python -m scripts.select_edge_c_v2_candidates `
  --input edge_c_v2\results\TOP_CANDIDATES.csv `
  --output edge_c_v2\results\EDGE_C_V2_SHORTLIST_CANDIDATES.csv `
  --max-candidates 6
```

The v2 viability selector requires:

```text
net_profit > 0
profit_factor >= 1.10
expectancy_usd > 0 after costs
max_drawdown_pct <= 5.0
finite final_score
```

Trade frequency is neither capped nor rewarded.

## 8. Portfolio/business diagnostics come last

Only robust after-cost candidates may be evaluated for:

- average net P/L per active trading day;
- median active-day P/L;
- positive-day fraction;
- worst active day;
- trading-day coverage;
- overlap/correlation with frozen Edge A;
- risk scaling required to approach approximately `$50/day` at portfolio level;
- projected portfolio drawdown.

The `$50/day` objective is not a parameter-selection target and cannot be used to retune Edge C v2.

## Research-integrity summary

- Edge A remains frozen and its prospective OOS remains untouched.
- Edge B v1/v2 remain rejected historical research.
- Edge C v1 remains an immutable activation-failed/inconclusive hypothesis.
- Edge C v2 uses canonical discovery data as an explicitly new discovery hypothesis.
- The 256 activation subset may determine testability only, never profitability.
- A parameter/domain change after this freeze creates a new hypothesis.
- Discovery results alone do not authorize live trading.
