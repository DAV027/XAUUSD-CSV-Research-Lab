# Edge C v2 Volatility-Scaled Compression Design

## Status

Frozen design for implementation. No production Edge C v2 code may be written until this specification has been reviewed and approved.

## Research context

Edge C v1 tested a stateful XAUUSD M1 compression -> breakout -> retest/hold -> confirmation hypothesis across 10,000 experiments. The campaign completed cleanly, but activation failed structurally: 9,990 experiments produced zero completed trades and only 10 experiments produced any trade at all. Each nonzero experiment produced exactly one short trade with zero-minute holding time and identical after-cost loss, consistent with an end-of-data forced-close artifact rather than a usable sample of economic outcomes.

The v1 compression predicate compared the full high-low range of a 20-120 bar window directly with one lagged ATR:

```text
range_width / ATR <= compression_atr_ratio
```

with `compression_atr_ratio` in 0.35-0.80. That construction asks a multi-bar range to fit inside substantially less than one single-bar ATR and made valid compression episodes almost nonexistent.

Edge C v2 is a new hypothesis. It does not alter, overwrite, reinterpret, or continue Edge C v1. Edge C v1 remains preserved as an activation-failed / economically inconclusive research result.

## Objective

Test whether the same causal compression -> decisive breakout -> retest/hold -> continuation structure becomes scientifically testable when compression is normalized by a volatility scale appropriate for a multi-bar window.

Edge C v2 must answer two questions in sequence:

1. **Activation:** Does the frozen parameter space generate enough independent completed trades to support meaningful economic testing?
2. **Economics:** If activation passes, does any configuration show robust positive after-cost expectancy under the repository's existing cost and risk model?

The `$50/day` objective remains a later portfolio/business diagnostic. It is not a strategy parameter, sampler objective, activation target, or guarantee.

## Research identity

```text
family:        edge_c_breakout_retest_v2
strategy:      compression_breakout_retest_v2
sampler:       edge_c_v2
default seed:  9216300
default budget: 10000
workers:       2
result root:   edge_c_v2/results/
direction:     combined only
```

All Edge C v2 artifacts must be isolated from:

- frozen Edge A configs and OOS artifacts;
- canonical `data/`;
- original `results/`;
- Edge B v1/v2 code and results;
- Edge C v1 code and results;
- diagnostic and prospective-OOS artifacts;
- sampler-v1 allocation and hashes.

## Core strategy state machine

Edge C v2 preserves Edge C v1's state-machine architecture. Only the compression normalization and its threshold domain change in the core hypothesis.

The strategy processes bars causally and emits at most one signal per compression-breakout episode.

### State 1: SEEK

Use only strict-prior bars to evaluate whether a fresh compression structure exists.

For bar index `i` and compression lookback `L`:

```text
range_high = max(high[i-L : i])
range_low  = min(low[i-L : i])
range_width = range_high - range_low
lagged_atr = atr14[i-1]
volatility_scale = lagged_atr * sqrt(L)
compression_score = range_width / volatility_scale
compression = compression_score <= compression_atr_ratio
```

Requirements:

- no current-bar OHLC may be used to form `range_high` or `range_low`;
- `atr14[i-1]` must be finite and positive;
- all high/low values in the prior window must be finite;
- `sqrt(L)` is deterministic and is not sampled;
- a fresh compression may arm only after the previous episode has left compression and rearm is permitted.

### State 2: ARMED

At the first qualifying compression bar, freeze for the episode:

- `range_high`;
- `range_low`;
- `episode_atr = atr14[i-1]`;
- `compression_lookback` used for that experiment.

The episode ATR remains fixed through breakout, retest, invalidation, and confirmation.

A decisive long breakout requires:

```text
close > range_high + breakout_buffer_atr * episode_atr
close > open
abs(close - open) >= breakout_body_atr * episode_atr
```

A decisive short breakout is symmetric below `range_low`.

If the quiet regime ends before a decisive breakout, discard the stale range and require a later fresh compression to arm again.

### State 3: RETEST

The breakout bar may not count as its own retest.

For a long breakout:

- a retest is seen when a later bar's low is `<= broken_level + retest_tolerance_atr * episode_atr`;
- invalidate if a later close is `< broken_level - retest_tolerance_atr * episode_atr`;
- the retest must occur and confirmation must complete within `retest_window` bars after the breakout.

Short logic is symmetric.

### State 4: CONFIRM / SIGNAL

After a valid retest, emit one long signal when:

```text
close >= broken_level + confirmation_atr * episode_atr
close > open
```

Emit one symmetric short signal below the broken level.

A bar may both establish the retest and satisfy confirmation if it is later than the breakout bar and all retest/invalidation/confirmation conditions are satisfied in causal order on that bar.

After a signal, the episode locks. No second signal may be emitted from the same compression-breakout structure.

### Rearm

After signal, invalidation, or retest-window expiry:

- return to SEEK;
- rearm must remain disabled while compression is continuously true;
- rearm becomes eligible only after at least one non-compressed bar;
- a later fresh compression may then arm a new episode.

This prevents repeated signals from one persistent structure.

## Frozen strategy parameter domains

Exactly seven strategy parameters are sampled:

```text
compression_lookback:       integer 20-120
compression_atr_ratio:      float   0.50-1.10
breakout_buffer_atr:        float   0.25-1.50
breakout_body_atr:          float   0.25-1.50
retest_window:              integer 2-20
retest_tolerance_atr:       float   0.10-0.75
confirmation_atr:           float   0.10-1.00
```

Do not expand `retest_window` in v2. Edge C v1 showed boundary concentration at 20, but changing both compression normalization and retest duration at once would confound attribution.

Do not add RSI, MACD, ADX, session filters, news filters, higher-timeframe filters, or other confirmation dimensions in v2.

## Frozen exit search

Use payoff-oriented exits only:

```text
stop_atr:   1.0, 1.5, 2.0, 3.0
exit_type:  target_r or atr_trail
target_r:   1.5, 2.0, 3.0, 4.0
atr_trail:  1.0, 1.5, 2.0
```

No time exits.

Existing commission, slippage, symbol, position-sizing, risk, next-bar-entry, and entry-integrity behavior remain authoritative.

## Sampler

Use an Edge C v2-specific deterministic sampler.

Requirements:

- default budget: exactly 10,000 experiments;
- default seed: exactly 9,216,300;
- sampler version: exactly `edge_c_v2`;
- family: exactly `edge_c_breakout_retest_v2`;
- strategy: exactly `compression_breakout_retest_v2`;
- direction mode: exactly `combined`;
- seven-dimensional Latin Hypercube sampling for strategy parameters;
- deterministic independent exit assignment using the repository's existing experiment model;
- unique experiment IDs and fingerprints;
- no modification of the original sampler-v1, Edge B samplers, or Edge C v1 sampler.

## Activation diagnostic gate

### Purpose

The activation diagnostic prevents another full 10,000-experiment campaign from running when the frozen strategy space is structurally inactive.

The activation gate is **not** a profitability screen and must not be used to choose or favor parameter values based on P/L, PF, expectancy, drawdown, win rate, or returns.

### Deterministic diagnostic sample

Generate the full frozen 10,000-row Edge C v2 catalog first.

Select exactly 256 experiments deterministically across catalog order so the sample spans the full catalog rather than taking the first 256 rows. The selection algorithm must be fixed in code and tested. Recommended definition:

```text
indices = round(linspace(0, budget - 1, 256))
```

with duplicate indices prohibited; for the default 10,000-row budget this must yield exactly 256 unique experiments.

The activation diagnostic runs those 256 experiments on the same canonical discovery feature set and with the same execution/risk/cost model as the eventual campaign.

### Activation metrics

For each sampled experiment record only structural activation quantities needed for the gate:

- experiment ID;
- completed trades;
- risk-skip count;
- zero-trade status;
- end-of-data completed-trade count;
- total completed-trade count.

Aggregate and report:

- sample size;
- number and fraction with zero completed trades;
- number and fraction with at least 10 completed trades;
- number and fraction with at least 300 completed trades;
- completed-trade distribution: minimum, median, p90, maximum;
- total risk skips;
- total completed trades;
- total end-of-data exits;
- end-of-data exit fraction among all completed diagnostic trades.

The diagnostic report must not include economic ranking fields and must not select "best" experiments.

### Gate criteria

Proceed to the full 10,000-experiment Edge C v2 campaign only if all conditions pass:

```text
1. at least 80% of the 256 sampled experiments have >= 10 completed trades
2. at least 10% of the 256 sampled experiments have >= 300 completed trades
3. end-of-data exits <= 5% of all completed diagnostic trades
```

Integer interpretation for the default sample:

```text
>= 205 experiments with >= 10 completed trades
>= 26 experiments with >= 300 completed trades
```

The end-of-data fraction is defined as:

```text
end_of_data_exit_count / total_completed_trades
```

If `total_completed_trades == 0`, the gate fails.

### Failure behavior

If any activation criterion fails:

- mark Edge C v2 activation as failed;
- do not launch the 10,000-experiment campaign;
- preserve the activation report as the negative research result;
- do not weaken thresholds, change domains, or rerun different diagnostic subsets under the same v2 identity.

Any subsequent structural change becomes Edge C v3 or another explicitly new hypothesis.

## Full campaign

Only after the activation gate passes may the normal Edge C v2 campaign run.

Execution requirements:

- output root: `edge_c_v2/results/`;
- workers: 2 by default;
- campaign seed: 9,216,300;
- canonical feature file unchanged;
- resumable/checkpoint behavior must reuse the repository's existing campaign engine;
- smoke/full execution must use the same catalog and result root;
- do not delete the result directory between smoke and resume.

## Economic and robustness evaluation

Activation success means only that the hypothesis is testable. It does not imply profitability.

If and only if the 10,000 campaign completes, apply the repository's unchanged robustness/promotion pipeline.

The defensive Edge C v2 post-robustness selector requires:

```text
net_profit > 0
profit_factor >= 1.10
expectancy_usd > 0 after costs
max_drawdown_pct <= 5.0
finite robustness/final score
```

Trade frequency is neither capped nor rewarded.

Zero robust/viable candidates is a valid negative result and must not cause threshold loosening.

Only after robust after-cost candidates exist may research evaluate:

- average net P/L per active trading day;
- median active-day P/L;
- positive-day fraction;
- worst active day;
- trading-day coverage;
- overlap/correlation with frozen Edge A;
- risk multiplier required to approach approximately $50/day at portfolio level;
- projected portfolio drawdown under that scaling.

Do not use the `$50/day` target to choose strategy parameters.

## Required implementation boundaries

Prefer new versioned files rather than modifying Edge C v1 production code.

Expected units:

```text
xau_lab/strategies/edge_c_v2.py
xau_lab/experiments/edge_c_v2_sampler.py
xau_lab/validation/edge_c_v2_activation.py
xau_lab/validation/edge_c_v2.py
scripts/create_edge_c_v2_catalog.py
scripts/run_edge_c_v2_activation.py
scripts/run_edge_c_v2_campaign.py
scripts/select_edge_c_v2_candidates.py
```

Tests should mirror those units.

The only existing source file expected to change is strategy registration if necessary. Edge C v1 files remain byte-for-byte unchanged unless a shared bug unrelated to hypothesis behavior is independently demonstrated and handled in a separate change.

## Testing requirements

Use test-driven development.

Strategy tests must cover at minimum:

- strict-prior compression window;
- exact `ATR * sqrt(lookback)` normalization;
- inclusive compression threshold boundary;
- no lookahead through current-bar range construction;
- fixed episode ATR;
- long breakout -> later retest -> confirmation;
- symmetric short behavior;
- invalidation;
- retest expiry;
- breakout bar cannot self-retest;
- same later bar may retest and confirm;
- one signal per episode;
- non-compressed leave-before-rearm behavior;
- NaN / invalid ATR handling.

Sampler tests must cover:

- default seed/budget/version/family/strategy;
- deterministic reproducibility;
- 10,000 unique IDs/fingerprints;
- seven frozen strategy domains;
- only frozen exit types/values;
- v1 sampler isolation.

Activation tests must cover:

- deterministic 256-row spread selection;
- exactly 256 unique indices at budget 10,000;
- threshold boundaries at exactly 205 configs >=10 trades and 26 configs >=300 trades;
- pass/fail end-of-data boundary at exactly 5%;
- failure when total completed trades is zero;
- no use of profitability fields in activation decision;
- malformed/partial diagnostic rows fail closed;
- report is deterministic.

Integration tests must cover CLI defaults and isolated output paths.

## Research integrity rules

- Edge C v1 remains immutable historical evidence.
- Edge A remains frozen.
- Future prospective OOS data must not be used to develop Edge C v2.
- Canonical discovery data may be used because Edge C v2 is explicitly a new discovery hypothesis.
- Activation diagnostics may determine whether the parameter space is testable, but may not be used to tune toward profit.
- Do not inspect economic results from the 256 diagnostic subset to alter domains.
- Parameter/domain changes after the v2 freeze create a new hypothesis.
- Do not widen acceptance thresholds to force survivors.
- Do not promote a candidate to live trading from discovery results alone.

## Definition of done

Edge C v2 implementation is complete when:

1. the frozen v2 strategy and sampler are implemented under separate versioned identities;
2. Edge C v1 remains unchanged;
3. the deterministic 256-experiment activation diagnostic exists and is tested;
4. activation cannot inspect or rank economic performance;
5. the full campaign command is operational but documented as gated on activation PASS;
6. promotion/viability paths remain after-cost and frequency-neutral;
7. all repository tests pass;
8. branch diff confirms no frozen Edge A, Edge B, canonical data/results, Edge C v1 results, OOS, or diagnostic artifacts are modified;
9. operator documentation states that activation failure ends v2 without a 10,000-run.