# Edge B Trend/Pullback Recovery Design

## Goal

Add a research-only Edge B hypothesis for XAUUSD that is structurally independent from the frozen Edge A return-reversal cluster. Edge B searches for continuation after a counter-trend pullback and first local recovery turn. It must not modify Edge A candidates, OOS plans/results, canonical discovery hashes, or the behavior of the original sampler version `v1`.

## Hypothesis

When XAUUSD has an established directional move, then experiences a measurable counter-trend pullback, the first local recovery bar back in the original trend direction may have positive continuation expectancy after realistic costs.

## Strategy

Name: `trend_pullback_recovery`

Family: `edge_b_trend_pullback`

Parameters:
- `trend_lookback`: integer 60-240 M1 bars.
- `trend_threshold_atr`: float 1.0-4.0.
- `pullback_lookback`: integer 3-20 M1 bars.
- `pullback_threshold_atr`: float 0.25-1.50.

At bar `i`, using the already lagged ATR value:
1. Establish trend using data ending at `i-1`: compare `close[i-1]` with `close[i-1-trend_lookback]`.
2. Require the absolute trend move to be at least `trend_threshold_atr * atr14[i-1]`.
3. Measure the pullback ending at `i-1`: compare `close[i-1]` with `close[i-1-pullback_lookback]`.
4. For an uptrend, require the pullback to be negative by at least `pullback_threshold_atr * atr14[i-1]`; for a downtrend require the opposite.
5. Require the immediately previous bar itself to still move counter-trend (`close[i-1] < close[i-2]` for a long setup, opposite for short).
6. Require the current bar to turn back in the trend direction (`close[i] > close[i-1]` for long, opposite for short).
7. Emit exactly that recovery-turn signal. Do not emit a continuous trend state.

The strategy emits both long and short signals; Edge B experiments use `direction_mode="combined"` only to reduce unnecessary search dimensions.

## Edge B catalog

Create a dedicated deterministic generator, separate from `generate_catalog()`:
- default budget: 10,000 complete experiments.
- default seed: `9_216_000`.
- sampler version: `edge_b_v1`.
- strategy family/bucket: `edge_b_trend_pullback`.
- parameters sampled with Latin Hypercube sampling across the four strategy domains.
- stop, exit type, target/time/trailing values reuse the existing discovery domains.
- commission remains `$6.0` round trip per lot.
- slippage remains `5.0` points per fill.
- direction is always `combined`.

Registering Edge B must not change any experiment produced by the original `generate_catalog()` version `v1`. The new family is not part of `FAMILY_BUDGET_WEIGHTS`, so the original sampler must continue to see only its frozen approved families.

## Campaign execution

Reuse the existing campaign engine and backtester. Add an Edge B CLI wrapper with isolated defaults:
- catalog: `edge_b/results/EXPERIMENT_CATALOG.csv`
- result root: `edge_b/results`
- checkpoint/state: inherited as `edge_b/state`
- features: canonical `data/features/XAUUSD_M1_FEATURES.parquet`
- workers remain user-configurable; local production run should use 2 workers unless explicitly changed.

No new risk engine, pricing engine, or backtester is introduced.

## Promotion and frequency constraint

Run the existing promotion/robustness machinery against Edge B outputs. Frequency is a constraint, not an optimization objective.

After robust promotion, an Edge B candidate is frequency-eligible when its completed-trade rate over the full discovery span is between 3 and 10 trades per 7 calendar days, inclusive. Do not fill the shortlist with candidates outside this range merely to reach a target count.

If more than six robust candidates are frequency-eligible, retain at most six ordered by existing `final_score` descending with `experiment_id` as deterministic tie-breaker. If fewer than three qualify, keep the smaller set and report that the hypothesis has insufficient qualifying candidates rather than relaxing thresholds.

The selector writes a research-only shortlist candidate table. It does not create a prospective OOS freeze automatically.

## Edge A independence

Edge A files/configs/results are immutable inputs for this work. Edge B must not modify:
- `config/oos_shortlist_v1.json`
- `config/oos_holdout_v1.json`
- `config/post_selection_diagnostic_v1.json`
- canonical `data/` artifacts
- original `results/EXPERIMENT_CATALOG.csv`
- original `results/MASTER_RESULTS.csv`
- Edge A diagnostic/OOS result files.

After Edge B finalists exist, independence analysis should compare trade-day overlap and daily P/L correlation with Edge A before any combined portfolio claim. That analysis is downstream of the first Edge B campaign and is not allowed to alter either edge's frozen parameters.

## OOS rule

Edge B receives its own prospective OOS start only after its shortlist is frozen. It does not inherit Edge A's September 17 boundary retroactively. Any data inspected during Edge B development/selection is discovery data for Edge B.

## Verification

Required tests:
- long recovery signal on a synthetic uptrend/pullback/recovery sequence.
- short recovery signal on the mirror sequence.
- no signal before full warmup.
- no signal while pullback continues without recovery.
- no repeated continuous signals after the first recovery turn.
- strategy registration/domain test.
- deterministic 10k-capable catalog generation with only Edge B experiments and `combined` direction.
- original `generate_catalog()` output remains unchanged when Edge B is imported/registered.
- Edge B CLI defaults are isolated from `results/` and Edge A configs.
- frequency selector enforces 3-10 trades/week, deterministic max-six ordering, and never force-fills below three.
