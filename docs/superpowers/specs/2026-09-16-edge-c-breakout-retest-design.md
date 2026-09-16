# Edge C Breakout-Retest v1 Design

## Purpose

Edge C is a new, independent XAUUSD M1 research family intended to improve opportunity coverage when the frozen Edge A return-reversal family is inactive. It must not modify, retune, or reuse Edge A selection logic, and it must not extend the rejected Edge B trend/pullback family.

The hypothesis is that a period of genuine price compression followed by a decisive volatility expansion can have continuation expectancy when price subsequently retests the broken structure, holds outside the old range, and resumes in the breakout direction.

Edge C is a strategy-discovery track. Its first objective is positive after-cost expectancy with robust behavior. The later $50/day target is evaluated only after robustness and is not a strategy parameter, sampler objective, or guarantee.

## Evidence motivating the design

The repository already contains generic breakout and volatility strategies: n-bar breakout, failed breakout, range expansion, Bollinger expansion, compression breakout, ATR-percentile regime, volatility-expansion direction, and volatility-contraction reversion. These direct triggers were already represented in the original 50,000-experiment discovery campaign and did not produce a robust breakout/volatility survivor.

A weak near-miss existed in the original campaign: a Bollinger-expansion long configuration achieved PF about 1.124 and small positive net profit, but it had only 85 completed trades and negative median active-day P/L. Therefore Edge C must not be another direct breakout trigger. It must add structural state: compression, breakout qualification, retest/hold, confirmation, one signal, and rearm only after a fresh compression regime.

## Isolation

Edge C is independent from all existing research tracks.

- Family: `edge_c_breakout_retest`
- Strategy: `compression_breakout_retest`
- Sampler version: `edge_c_v1`
- Default budget: 10,000 experiments
- Default seed: 9,216,200
- Default workers: 2
- Result root: `edge_c/results/`

Edge C must not change:

- `config/oos_shortlist_v1.json`
- `config/oos_holdout_v1.json`
- Edge A candidate order or parameters
- canonical `data/`
- original `results/`
- `edge_b/` or `edge_b_v2/` code/results
- original sampler-v1 experiment allocation or hashes
- prospective Edge A OOS semantics or files

Registering the new strategy must not make it eligible for the original sampler-v1 family budgets.

## Strategy state machine

The strategy is stateful and processes M1 bars causally. It may emit at most one signal per compression-breakout structure.

### State 1: Seek compression

A candidate compression window ends at bar `i-1`; the current bar is never used to define the range that it may break.

For `compression_lookback = L`:

- `range_high = max(high[i-L:i])`
- `range_low = min(low[i-L:i])`
- `range_width = range_high - range_low`
- `atr_ref = atr14[i-1]`

Compression exists when all inputs are finite, `atr_ref > 0`, and:

`range_width / atr_ref <= compression_atr_ratio`

Parameter domain:

- `compression_lookback`: integer 20–120
- `compression_atr_ratio`: float 0.35–0.80

When compression is found, the strategy stores the range high, range low, and `atr_ref` in a direction-neutral armed state. The range and `atr_ref` remain fixed for the entire episode so breakout, retest, invalidation, and confirmation thresholds cannot drift as later bars change volatility.

### State 2: Require decisive breakout

For an armed compression episode, a long breakout occurs when the current close exceeds:

`range_high + breakout_buffer_atr * atr_ref`

A short breakout occurs when the current close is below:

`range_low - breakout_buffer_atr * atr_ref`

The breakout candle must also have a directional body large enough to reject wick-only breaks:

`abs(close[i] - open[i]) >= breakout_body_atr * atr_ref`

and the body direction must match the breakout direction.

Parameter domains:

- `breakout_buffer_atr`: float 0.25–1.50
- `breakout_body_atr`: float 0.25–1.50

After a valid breakout, the episode enters retest state and stores the broken level (`range_high` for long, `range_low` for short), the breakout direction, the fixed episode `atr_ref`, and the breakout bar index.

### State 3: Retest / hold

A breakout has a finite retest window of `retest_window` bars after the breakout bar. The first eligible retest bar is `breakout_index + 1`; the final eligible retest bar is `breakout_index + retest_window`.

Parameter domain:

- `retest_window`: integer 2–20
- `retest_tolerance_atr`: float 0.10–0.75

For a long breakout, price may revisit the broken high, but any close below:

`broken_level - retest_tolerance_atr * atr_ref`

invalidates the episode.

For a short breakout, any close above:

`broken_level + retest_tolerance_atr * atr_ref`

invalidates the episode.

A retest is considered observed when price reaches or crosses the broken level intrabar:

- long: `low <= broken_level + retest_tolerance_atr * atr_ref`
- short: `high >= broken_level - retest_tolerance_atr * atr_ref`

The strategy does not emit a signal merely because a retest occurred.

If no retest occurs by the end of `breakout_index + retest_window`, the episode expires with no signal.

### State 4: Confirmation and single entry

After a valid retest has occurred, continuation is confirmed when price closes sufficiently beyond the broken level in the breakout direction:

- long: `close >= broken_level + confirmation_atr * atr_ref`
- short: `close <= broken_level - confirmation_atr * atr_ref`

Parameter domain:

- `confirmation_atr`: float 0.10–1.00

The confirmation bar must close in the breakout direction relative to its open.

A retest and confirmation may occur on the same M1 bar if that bar touches the retest zone intrabar, never closes through the invalidation boundary, and then closes beyond the confirmation threshold in the breakout direction. This is causal because the signal is emitted only at that bar's close.

On the first confirming bar, emit exactly one signal (`+1` long, `-1` short), then permanently lock that episode.

No second signal is allowed from the same stored compression range, even if price retests again.

### State 5: Rearm

After signal, invalidation, or expiry, the strategy enters a locked/rearm-required state. A new episode may arm only after the rolling compression predicate first becomes false on at least one later bar and subsequently becomes true again on a later bar. Consecutive overlapping windows from the same quiet regime therefore cannot produce repeated armed episodes.

The rolling compression predicate used for rearming is the same State-1 definition applied causally to the current candidate window and its lagged ATR reference. This explicit rearm rule is part of the hypothesis and is not a tunable parameter.

## Causality and data integrity

- No future bar may contribute to a current signal.
- Compression range uses bars strictly before the breakout candidate bar.
- The episode `atr_ref` is `atr14[i-1]` at the bar where that compression episode arms and remains fixed for the episode.
- Invalid/NaN price or ATR inputs cannot arm, advance, confirm, or signal an episode.
- Existing centralized `entry_allowed` / data-integrity masking remains authoritative.
- Direction mode is `combined` for Edge C v1; long-only/short-only variants are not separately sampled in v1.

## Frozen strategy parameter domain

Edge C v1 samples exactly seven strategy parameters:

- `compression_lookback`: integer 20–120
- `compression_atr_ratio`: float 0.35–0.80
- `breakout_buffer_atr`: float 0.25–1.50
- `breakout_body_atr`: float 0.25–1.50
- `retest_window`: integer 2–20
- `retest_tolerance_atr`: float 0.10–0.75
- `confirmation_atr`: float 0.10–1.00

No RSI, MACD, ADX, session filter, news filter, or additional confirmation parameter is part of Edge C v1.

## Exit search space

Edge C searches only exits capable of capturing movement large enough to survive XAUUSD costs.

Stops:

- `stop_atr`: 1.0, 1.5, 2.0, 3.0

Target-R exits:

- `target_r`: 1.5, 2.0, 3.0, 4.0

ATR-trail exits:

- `atr_trail`: 1.0, 1.5, 2.0

No fixed short time-exit family is included in Edge C v1. The purpose is to test continuation after confirmed expansion, not scalp one-minute noise.

Existing commission and slippage models remain unchanged.

## Catalog generation

A dedicated deterministic generator creates only Edge C experiments. It must not call or alter the original global sampler allocation.

Each catalog row must have:

- family `edge_c_breakout_retest`
- strategy `compression_breakout_retest`
- sampler version `edge_c_v1`
- direction `combined`
- seed lineage from default seed 9,216,200
- one parameter set sampled from the frozen seven-dimensional strategy domain
- one exit specification sampled from the frozen Edge C exit domain
- unchanged repository cost model

The generator must be deterministic for the same seed/budget and must produce unique experiment IDs and fingerprints.

## Campaign execution

Use the existing campaign engine, manifest validation, checkpointing, backtester, risk model, metrics, and result schema.

Default paths:

- catalog: `edge_c/results/EXPERIMENT_CATALOG.csv`
- features: `data/features/XAUUSD_M1_FEATURES.parquet`
- results: `edge_c/results/`

The runner defaults to exactly 2 workers and supports a small `--limit` smoke run followed by resumable full execution without deleting the result directory.

## Viability and robustness contract

Trade count is unrestricted. Frequency is observed and reported but never rewarded or capped.

Edge C first has to prove strategy viability after realistic costs. A candidate cannot advance unless it satisfies all existing hard robustness requirements used by the repository, including adequate sample size/activity, concentration controls, and stability screens, plus these minimum economic requirements:

- `net_profit > 0`
- `profit_factor >= 1.10`
- `expectancy_usd > 0`
- `max_drawdown_pct <= 5.0`

The existing robustness promotion pipeline remains authoritative where its thresholds are stricter.

No threshold may be weakened merely because few or zero candidates survive.

## Coverage and business-objective analysis

Only robust Edge C survivors are evaluated against Edge A for portfolio usefulness.

For each survivor, report:

- average net P/L per active day
- median net P/L per active day
- positive active-day fraction
- completed trades
- trades per active day
- active days and active-day fraction over eligible market days
- worst active day when trade-level data is available
- net profit, PF, expectancy, and max DD
- overlap of active trading days with frozen Edge A candidates
- daily P/L correlation with frozen Edge A candidates when trade-level daily series are available

The $50/day objective is evaluated only after this stage. For a survivor, estimate the simple linear risk multiplier required for `$50 / active day` and the corresponding linearly scaled historical DD as a diagnostic. This scaling is not a backtest result and must be labeled as an approximation.

No strategy is accepted merely because scaling could mathematically reach $50/day. Positive after-cost expectancy and robustness are prerequisites.

## Shortlist and OOS

If robust Edge C survivors exist, shortlist at most 6 candidates. Selection should favor economic robustness and portfolio complementarity to Edge A, not just the highest historical net profit.

A frozen Edge C shortlist gets its own prospective OOS boundary after the shortlist is frozen. Edge C must not inherit Edge A's 2026-09-17 holdout start retroactively.

If no candidate survives, Edge C v1 is rejected and preserved unchanged as a historical research result. A future Edge C v2 would require a new structural hypothesis and sampler version rather than threshold loosening.

## Test contract

Implementation is test-driven. Tests must cover at least:

1. compression uses only prior bars;
2. episode ATR is lagged and fixed after arming;
3. wick-only breakout does not qualify;
4. decisive long and short breakouts enter retest state;
5. close back inside tolerance invalidates the episode;
6. no retest before timeout produces no signal;
7. valid retest plus later confirmation emits exactly one signal;
8. same-bar retest plus confirmation emits exactly one signal when all hold conditions are satisfied;
9. repeated retests from the same episode never emit a second signal;
10. rearm requires a false rolling-compression predicate before a fresh true predicate can arm;
11. NaN/invalid ATR or price data blocks progression safely;
12. strategy registration exposes exactly the frozen seven-parameter domain;
13. Edge C catalog is deterministic and unique;
14. original sampler-v1 never samples the Edge C family;
15. campaign CLI defaults are isolated under `edge_c/results/` and use 2 workers;
16. robustness/economic selector does not impose a trade-frequency cap;
17. zero qualifying candidates remains a valid outcome and does not relax thresholds.

## Success and failure interpretation

A successful Edge C v1 campaign is not defined as reaching $50/day. Success at discovery means at least one candidate demonstrates credible positive after-cost expectancy and survives the frozen robustness contract. Portfolio-level coverage and dollar-target analysis happen afterward.

A campaign with zero robust survivors is a valid negative research result. The repository must preserve that result without retuning the frozen v1 hypothesis to manufacture a winner.
