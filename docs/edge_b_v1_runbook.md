# Edge B v1 Research Runbook

Edge B is a separate XAUUSD trend/pullback/recovery research track. It does not modify or inherit the frozen Edge A shortlist or Edge A prospective OOS record.

## Research hypothesis

Edge B tests whether an established directional move followed by a measurable counter-trend pullback and first local recovery turn has positive continuation expectancy after the repository's existing commission, slippage, risk, and exit models.

Strategy: `trend_pullback_recovery`

Sampler version: `edge_b_v1`

Default catalog budget: 10,000

Default seed: 9,216,000

Direction mode: combined only

## 1. Update the repository

```powershell
cd E:\XAUUSD-CSV-Research-Lab
git checkout main
git pull
```

Do not delete or replace canonical `data/`, `results/`, diagnostic, or OOS artifacts.

## 2. Create the Edge B catalog

```powershell
python -m scripts.create_edge_b_catalog `
  --budget 10000 `
  --seed 9216000
```

Default output:

```text
edge_b/results/EXPERIMENT_CATALOG.csv
```

This is intentionally separate from `results/EXPERIMENT_CATALOG.csv`.

## 3. Run the Edge B discovery campaign

```powershell
python -m scripts.run_edge_b_campaign `
  --workers 2
```

The campaign reuses the existing runner/backtester and canonical feature artifact. Its result/state paths are isolated below `edge_b/`.

Expected research outputs include:

```text
edge_b/results/MASTER_RESULTS.csv
edge_b/results/RUN_MANIFEST.json
edge_b/state/CHECKPOINT.json
```

The campaign is resumable. Rerunning the same command resumes the same manifest/catalog instead of creating a new hypothesis.

## 4. Run the existing robustness promotion pipeline

```powershell
python -m scripts.promote_candidates `
  --master edge_b/results/MASTER_RESULTS.csv `
  --catalog edge_b/results/EXPERIMENT_CATALOG.csv `
  --features data/features/XAUUSD_M1_FEATURES.parquet `
  --result-root edge_b/results
```

Use the repository's existing promotion rules unchanged. Do not weaken robustness criteria because Edge B is a new family.

The frequency selector consumes only the resulting robust candidate table:

```text
edge_b/results/TOP_CANDIDATES.csv
```

## 5. Apply the frozen Edge B frequency constraint

```powershell
python -m scripts.select_edge_b_candidates
```

Default eligibility is 3 to 10 completed trades per 7 calendar days, inclusive. Frequency is a constraint, not a score or optimization target.

Default output:

```text
edge_b/results/EDGE_B_SHORTLIST_CANDIDATES.csv
```

Rules:
- At most six candidates are retained.
- Ordering is existing `final_score` descending, then `experiment_id` ascending.
- If fewer than three robust candidates meet the frequency range, keep the smaller set and report insufficient qualifying candidates.
- Never lower robustness or frequency thresholds simply to fill a shortlist.

## 6. Freeze before any Edge B prospective OOS

Everything inspected during strategy construction, catalog search, campaign analysis, robustness promotion, frequency selection, and shortlist review is discovery data for Edge B.

Edge B receives a new prospective-OOS start only after its shortlist is frozen. It does not retroactively inherit Edge A's 2026-09-17 prospective boundary.

Once finalists are frozen, create a separate Edge B shortlist/holdout config and separate Edge B OOS results. Do not mix those rows with Edge A `OOS_RESULTS.csv`.

## 7. Independence analysis after finalists exist

Before making any combined Edge A + Edge B portfolio claim, compare the frozen edges using at least:
- trade-day overlap;
- daily P/L correlation;
- simultaneous drawdown behavior;
- exposure concentration during volatile regimes.

A profitable Edge B is not automatically independent merely because its entry rule has a different name.

## Frozen invariants

This research track must not modify:

```text
config/oos_shortlist_v1.json
config/oos_holdout_v1.json
config/post_selection_diagnostic_v1.json
data/
results/EXPERIMENT_CATALOG.csv
results/MASTER_RESULTS.csv
diagnostic_results/
diagnostic_results_utc/
oos_results/
```

The original sampler version `v1` and its 50,000-experiment discovery campaign remain historical, reproducible artifacts.
