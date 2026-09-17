# Edge D v1 Activation-First Runbook

Edge D v1 is a separate XAUUSD M1 discovery hypothesis based on prior-session liquidity sweep and reclaim reversals.

The full 10,000-experiment campaign is **prohibited** until the frozen structural activation diagnostic passes. Activation determines only whether the hypothesis generates enough valid completed trades to justify economic testing; it does not evaluate profitability.

## Research identity

```text
family:        edge_d_session_sweep_reclaim
strategy:      prior_session_sweep_reclaim
sampler:       edge_d_v1
budget:        10000
seed:          9216400
workers:       2
result root:   edge_d/results/
direction:     combined
```

Do not alter these values under the Edge D v1 identity.

## 1. Update the repository after PR merge

From the canonical local repository:

```powershell
cd E:\XAUUSD-CSV-Research-Lab
git checkout main
git pull
```

Do not delete or reset canonical `data\`, historical `results\`, or prior Edge A/B/C research artifacts.

## 2. Create the frozen Edge D catalog

```powershell
python -m scripts.create_edge_d_catalog `
  --budget 10000 `
  --seed 9216400
```

Expected output:

```text
edge_d\results\EXPERIMENT_CATALOG.csv
```

The catalog is deterministic. Do not regenerate it with a different seed, edit rows, change strategy domains, or alter exit domains after observing any diagnostic or campaign results.

## 3. Run the mandatory 256-experiment activation diagnostic

```powershell
python -m scripts.run_edge_d_activation
```

The diagnostic evaluates exactly 256 deterministic catalog-spread rows from the frozen 10,000-row catalog and may use only structural metrics:

- completed trades;
- risk skips;
- zero-trade configurations;
- end-of-data exits.

The activation decision must not use net profit, profit factor, expectancy, win rate, drawdown, active-day P/L, or any other economic metric.

Expected output:

```text
edge_d\results\ACTIVATION_REPORT.json
```

### Frozen activation PASS criteria

All conditions must pass:

```text
>= 205 / 256 sampled experiments have >= 50 completed trades
>= 64 / 256 sampled experiments have >= 300 completed trades
end-of-data exits <= 5% of all completed diagnostic trades
total completed diagnostic trades > 0
```

### If activation reports FAIL

**Stop Edge D v1.**

Do not run the full campaign. Preserve `ACTIVATION_REPORT.json` as the negative research result.

Do not lower thresholds, choose another 256-row sample, change the seed, widen/narrow parameter domains, inspect diagnostic profitability as a redesign signal, or retune around a near-miss. Any structural redesign after an activation failure is a new hypothesis/version, not Edge D v1.

### If activation reports PASS

PASS means only that Edge D v1 is structurally active enough to test economically. It is **not evidence of profitability or robustness**.

The guarded campaign runner independently verifies that the persisted PASS report matches:

- the frozen Edge D identity;
- seed `9216400`;
- budget `10000`;
- the current experiment-catalog SHA256;
- the current feature-data SHA256.

A stale, copied, or mismatched PASS report cannot authorize a campaign.

## 4. Run the required 20-experiment smoke after activation PASS

Only after activation PASS:

```powershell
python -m scripts.run_edge_d_campaign `
  --workers 2 `
  --limit 20
```

The smoke run is for execution/provenance validation only. Do not use smoke profitability, win rate, PF, or P/L as a hidden selection gate.

Do not delete `edge_d\results` after the smoke run. The full campaign must resume in the same result directory under the same manifest and frozen inputs.

## 5. Resume the same campaign to all 10,000 experiments

```powershell
python -m scripts.run_edge_d_campaign `
  --workers 2 `
  --allow-slow
```

The same catalog, feature file, activation report, result root, seed, and run manifest remain authoritative.

Expected campaign output includes:

```text
edge_d\results\MASTER_RESULTS.csv
edge_d\results\RUN_MANIFEST.json
```

Do not create a second campaign directory to bypass prior results or manifest checks.

## 6. Run the unchanged Stage-1 robustness/promotion pipeline

Only after the full 10,000-experiment campaign completes:

```powershell
python -m scripts.promote_candidates `
  --master edge_d\results\MASTER_RESULTS.csv `
  --catalog edge_d\results\EXPERIMENT_CATALOG.csv `
  --features data\features\XAUUSD_M1_FEATURES.parquet `
  --result-root edge_d\results `
  --run-manifest edge_d\results\RUN_MANIFEST.json
```

Edge D receives no special relaxation. Existing Stage-1 requirements remain authoritative, including:

```text
profit_factor >= 1.10
max_drawdown_pct <= 5.0
completed trades >= 300
expectancy > 0 after costs
majority profitable years when available
existing concentration controls
research-integrity checks
```

### If Stage-1 produces zero survivors

**Stop Edge D v1.**

Zero survivors is a valid negative research result. Do not weaken robustness, concentration, trade-count, drawdown, PF, or expectancy requirements to manufacture candidates.

## 7. Apply the frozen Edge D viability selector

Only Stage-1 robust candidates may enter this selector:

```powershell
python -m scripts.select_edge_d_candidates `
  --input edge_d\results\TOP_CANDIDATES.csv `
  --output edge_d\results\EDGE_D_SHORTLIST_CANDIDATES.csv `
  --max-candidates 6
```

The selector requires:

```text
net_profit > 0
profit_factor >= 1.10
expectancy_usd > 0
max_drawdown_pct <= 5.0
finite final_score
```

Trade frequency is neither capped nor rewarded beyond the Stage-1 minimum sample-size requirement. At most six candidates are retained, ordered deterministically by existing robustness score with experiment ID as the stable tie-breaker.

Zero selected candidates is valid and ends Edge D v1 without retuning.

## 8. Portfolio/business review comes only after a frozen robust shortlist

Only after robust Edge D candidates exist may research compare:

- average net P/L per active day;
- median active-day P/L;
- positive-day fraction;
- worst active day;
- trading-day coverage;
- active-day overlap with Edge A;
- daily P/L correlation with Edge A;
- risk multiplier required to approach the portfolio `$50/day` objective;
- projected drawdown under that multiplier.

The `$50/day` objective is a later portfolio/business question. It must not alter Edge D strategy parameters, sampler domains, activation thresholds, Stage-1 rules, or viability gates.

## 9. OOS discipline

Do not choose, consume, or tune against a prospective Edge D OOS period before the in-sample robustness process has produced and frozen a shortlist.

If a shortlist exists, Edge D receives its own prospective OOS configuration. Do not reuse Edge A's prospective holdout as a tuning source for Edge D.

## Preservation and stop rules

- Activation FAIL stops Edge D v1.
- A completed 10,000-experiment campaign with zero Stage-1 survivors stops Edge D v1.
- Zero viability-selector survivors stops Edge D v1.
- Near-misses do not justify threshold relaxation or parameter retuning.
- Do not modify Edge A, Edge B, Edge C, canonical `data/`, historical result artifacts, or original sampler behavior as part of Edge D v1.
- Do not replace the frozen seed, catalog, feature data, activation report, or run manifest after seeing results.
- Do not use activation economics or smoke-run economics as hidden optimization gates.
- Discovery results alone do not authorize live trading or position-size increases.
