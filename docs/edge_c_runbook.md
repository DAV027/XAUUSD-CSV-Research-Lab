# Edge C v1 Research Runbook

Edge C v1 is an isolated XAUUSD M1 compression-breakout-retest research campaign.

It does **not** modify or replace Edge A. Edge A remains frozen for its own prospective OOS validation. Edge B v1/v2 remain rejected historical research tracks.

## Frozen research identity

- Family: `edge_c_breakout_retest`
- Strategy: `compression_breakout_retest`
- Sampler: `edge_c_v1`
- Budget: 10,000 experiments
- Seed: 9,216,200
- Default workers: 2
- Result root: `edge_c/results/`
- Direction: combined
- Trade-frequency cap: none

The strategy-discovery objective is robust positive expectancy **after** the repository's modeled commission and slippage. The later `$50/day` objective is a portfolio/business diagnostic only; it is not a sampler parameter and must not be used to retune Edge C v1.

## 1. Update the local repository after PR merge

From the repository root:

```powershell
cd E:\XAUUSD-CSV-Research-Lab
git checkout main
git pull
```

Do not delete canonical `data/`, original `results/`, Edge A OOS files, or prior Edge B artifacts.

## 2. Create the frozen 10,000-experiment Edge C catalog

```powershell
python -m scripts.create_edge_c_catalog `
  --budget 10000 `
  --seed 9216200
```

Expected destination:

```text
edge_c\results\EXPERIMENT_CATALOG.csv
```

Catalog generation is deterministic for the same code, budget, and seed. Do not edit the catalog manually after creation.

## 3. Run a 20-experiment smoke campaign

```powershell
python -m scripts.run_edge_c_campaign `
  --workers 2 `
  --limit 20
```

This creates/validates the Edge C run manifest and exercises the real campaign path using the canonical feature parquet.

Inspect `edge_c\results` for the campaign outputs. If the smoke run succeeds, continue the **same** campaign directory.

**Do not delete `edge_c\results` between the smoke run and the full campaign.** The runner is designed to resume from existing completed results.

## 4. Resume to all 10,000 experiments

```powershell
python -m scripts.run_edge_c_campaign `
  --workers 2 `
  --allow-slow
```

The final discovery master file is expected at:

```text
edge_c\results\MASTER_RESULTS.csv
```

A completed campaign may legitimately contain zero profitable or zero robust configurations. That is a valid negative research result.

## 5. Run the unchanged robustness-promotion pipeline

Only after the full campaign is complete, run promotion against the isolated Edge C files:

```powershell
python -m scripts.promote_candidates `
  --master edge_c\results\MASTER_RESULTS.csv `
  --catalog edge_c\results\EXPERIMENT_CATALOG.csv `
  --features data\features\XAUUSD_M1_FEATURES.parquet `
  --result-root edge_c\results
```

The existing promotion pipeline remains authoritative for sample size/activity, stability, concentration, walk-forward, stress, and resampling requirements. Edge C does not weaken those requirements.

Promotion outputs are written under `edge_c\results`, including `TOP_CANDIDATES.csv` when robust candidates exist.

`ROBUST_CANDIDATE` is a research label only. It is not live-trading approval.

## 6. Apply the frozen Edge C economic viability gate

After promotion:

```powershell
python -m scripts.select_edge_c_candidates
```

Frozen minimum economic requirements are:

- net profit > 0
- profit factor >= 1.10
- expectancy per trade > 0 after modeled costs
- max drawdown <= 5%
- finite robustness score

The selector imposes **no minimum or maximum trade frequency**.

Default shortlist output:

```text
edge_c\results\EDGE_C_SHORTLIST_CANDIDATES.csv
```

At most six candidates are retained for portfolio/coverage review.

If the selector reports:

```text
EDGE_C_STATUS=NO_VIABLE_CANDIDATES_VALID_NEGATIVE_RESULT
```

preserve Edge C v1 unchanged and classify the hypothesis as rejected. Do not lower PF, expectancy, drawdown, robustness, breakout, retest, or confirmation thresholds merely to manufacture a survivor.

## 7. Portfolio-coverage analysis comes after robustness

Only robust, economically viable Edge C candidates should be compared with frozen Edge A candidates.

For each candidate, evaluate:

- net P/L per active day
- median active-day P/L
- positive active-day fraction
- completed trades and trades per active day
- active-day coverage
- max drawdown and expectancy
- active-day overlap with Edge A
- daily P/L correlation with Edge A when daily trade-ledger series are available

The goal is useful independence and coverage, not simply adding another highly correlated strategy.

## 8. `$50/day` scaling is diagnostic, not discovery tuning

For a robust candidate, a simple linear estimate may be computed:

```text
risk_multiplier ~= 50 / historical_average_active_day_profit
scaled_DD ~= historical_DD * risk_multiplier
```

This is only a rough scaling diagnostic. It is **not** a backtest, does not account for nonlinear execution/liquidity/risk effects, and does not prove that $50 will be earned every day.

Do not increase discovery risk or alter strategy parameters simply because the current conservative risk model does not reach the dollar target.

## 9. OOS rule

If Edge C produces a shortlist worth freezing, freeze those exact candidates first and assign Edge C its **own future prospective OOS start after that freeze**.

Do not retroactively reuse Edge A's 2026-09-17 prospective start as Edge C OOS. Any data observed while developing/selecting Edge C is selection data for Edge C.

## Research interpretation

Edge C v1 succeeds at discovery only if at least one candidate demonstrates credible positive after-cost expectancy and survives the unchanged robustness process. It does not need to produce `$50/day` at discovery sizing.

Zero survivors is a complete and useful research result. In that case, preserve v1 and require a structurally new hypothesis/version before another search rather than loosening frozen thresholds.
