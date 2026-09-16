# Edge B v2 Daily-Profit Research Runbook

Edge B v2 is a new discovery hypothesis created after Edge B v1 was rejected. It does not overwrite v1, Edge A, canonical data, or any prospective OOS record.

## Objective

Research objective on the repository's existing risk model:

- account equity: $5,000
- preferred planned risk: $2 per trade
- hard planned risk ceiling: $5 per trade
- max lot: 0.10
- commission and slippage remain unchanged

Trade frequency is unrestricted.

After the unchanged robustness promotion pipeline, the v2 daily-profit selector requires:

- `profit_per_active_day >= 50.0`
- `median_profit_per_active_day > 0.0`

This is an acceptance threshold, not a sampler parameter and not a guarantee that every day earns $50.

## Structural changes from v1

v1 produced excessive low-value entries whose gross edge was overwhelmed by transaction costs. v2 therefore changes the hypothesis rather than loosening acceptance criteria:

- stronger trend displacement: 4-12 ATR
- deeper pullback: 1-4 ATR
- explicit recovery-bar displacement: 0.25-1.5 ATR
- one entry per continuous pullback episode
- wider stop space: 1.5, 2.0, 3.0, 4.0 ATR
- payoff-oriented exits only: target-R or ATR trailing
- no trade-frequency minimum or maximum

Strategy: `trend_pullback_recovery_v2`

Sampler: `edge_b_v2`

Default budget: 10,000

Default seed: 9,216,100

## 1. Update repository

```powershell
cd E:\XAUUSD-CSV-Research-Lab
git checkout main
git pull
```

## 2. Create v2 catalog

```powershell
python -m scripts.create_edge_b_v2_catalog `
  --budget 10000 `
  --seed 9216100
```

Default output:

```text
edge_b_v2/results/EXPERIMENT_CATALOG.csv
```

## 3. Smoke test

```powershell
python -m scripts.run_edge_b_v2_campaign `
  --workers 2 `
  --limit 20
```

Do not delete the v2 result directory after a successful smoke run.

## 4. Full discovery campaign

```powershell
python -m scripts.run_edge_b_v2_campaign `
  --workers 2 `
  --allow-slow
```

This resumes the same manifest/catalog and completes the remaining experiments.

## 5. Run unchanged robustness promotion

```powershell
python -m scripts.promote_candidates `
  --master edge_b_v2/results/MASTER_RESULTS.csv `
  --catalog edge_b_v2/results/EXPERIMENT_CATALOG.csv `
  --features data/features/XAUUSD_M1_FEATURES.parquet `
  --result-root edge_b_v2/results
```

Do not weaken PF, drawdown, expectancy, trade-count, concentration, or year-consistency rules to make v2 pass.

## 6. Apply the daily-profit objective

```powershell
python -m scripts.select_edge_b_v2_candidates
```

Default output:

```text
edge_b_v2/results/EDGE_B_V2_SHORTLIST_CANDIDATES.csv
```

No trade-frequency constraint is applied.

If no robust row meets the daily-profit threshold, v2 is rejected. Do not lower the $50 target merely to create a shortlist unless the research objective itself is deliberately revised before a new version.

## 7. Freeze before prospective OOS

Only after a v2 shortlist is reviewed and frozen should a new Edge B v2 prospective holdout boundary be created. v2 cannot inherit Edge A's holdout boundary or retroactively treat discovery data as OOS.

## Isolation invariants

Do not modify or replace:

```text
config/oos_shortlist_v1.json
config/oos_holdout_v1.json
config/post_selection_diagnostic_v1.json
data/
results/
edge_b/results/
diagnostic_results/
diagnostic_results_utc/
oos_results/
```

All v2 discovery artifacts belong under:

```text
edge_b_v2/
```
