# Signal Kernel Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Python-per-bar strategy hot paths with exact-semantics Numba kernels so the canonical 50,000-experiment campaign projects to 24 hours or less without changing any experiment result semantics.

**Architecture:** Preserve every public strategy function and registry contract, but move repeated rolling/window math into a focused private `xau_lab.strategies._kernels` module and move bar loops into private `@njit(cache=True)` functions. A test-only reference module freezes the pre-optimization Python behavior and every converted strategy must produce an identical `np.int8` signal vector before the implementation is accepted.

**Tech Stack:** Python 3.12, NumPy, Numba, pytest, Polars/Parquet for canonical runtime verification.

**Spec:** `docs/superpowers/specs/2026-09-13-signal-kernel-acceleration-design.md`

## Global Constraints

- Preserve `results/EXPERIMENT_CATALOG.csv` exactly, including all 50,000 IDs/fingerprints and SHA256 `AB2AD9935FB4E9A229CE1D84E2E9B80C4D670A8CEB7E577BF3CAE0B70F0706A9`.
- Preserve canonical feature semantics and the current 72-column feature artifact SHA256 `5D3680D043E1F8D4C759124AD177BC73E47C055101DD3BE7558564187B0C10EF`.
- Preserve warmups, window endpoints, equality/inequality operators, NaN/Inf handling, zero-denominator behavior, and fail-closed behavior exactly.
- Preserve direction filtering, `entry_allowed`, costs, risk sizing, exit semantics, backtest ordering, and summary metrics.
- No approximate quantile, percentile, autocorrelation, or rolling-statistic implementation.
- Stage 1 must not add worker-global lookback caches or `(lookback_count, bars)` matrices.
- Every family conversion follows RED → GREEN and receives an exact signal-oracle comparison.
- PR #16 remains draft and unmerged until the complete verification gate passes.
- Full 50,000 campaign remains blocked until a fresh 100-smoke reports `projected_50000_hours <= 24` with zero errors and stable resources.
- If Stage 1 remains above 24 hours, stop and design Stage 2 caching separately rather than weakening any gate.

## File Structure

- Create `xau_lab/strategies/_kernels.py`: reusable Numba-safe scalar/window primitives only; no strategy registry or trading-policy logic.
- Create `tests/strategies/reference_signals.py`: literal slow pre-optimization reference implementations used only by tests.
- Create `tests/strategies/test_signal_kernel_equivalence.py`: randomized and adversarial reference-vs-optimized signal equality tests across all converted families.
- Modify `xau_lab/strategies/volatility.py`: compiled volatility kernels and thin public wrappers.
- Modify `xau_lab/strategies/mean_reversion.py`: compiled mean-reversion kernels and thin public wrappers.
- Modify `xau_lab/strategies/statistical.py`: compiled statistical kernels and thin public wrappers.
- Modify `xau_lab/strategies/trend.py`: compiled trend/momentum kernels and thin public wrappers.
- Modify `xau_lab/strategies/breakout.py`: compile remaining Python bar loops while retaining the proven exact expanding-quantile implementation.
- Modify `xau_lab/strategies/price_action.py`: compile candle/level pattern loops.
- Modify `xau_lab/strategies/session.py`: compile session-state loops while retaining explicit session-feature lookup in the Python wrapper.
- Modify existing family tests only when a current golden test needs an explicit regression case; do not duplicate broad equivalence coverage there.

---

### Task 1: Freeze semantic oracles and add reusable compiled window primitives

**Files:**
- Create: `xau_lab/strategies/_kernels.py`
- Create: `tests/strategies/reference_signals.py`
- Create: `tests/strategies/test_signal_kernel_equivalence.py`

**Interfaces:**
- Consumes: one-dimensional contiguous NumPy market arrays plus integer window bounds.
- Produces: private Numba helpers `_window_all_finite`, `_window_mean_std`, `_window_min_max`, `_linear_quantile_sorted_window`, `_body_direction`, all callable from strategy kernels without Python object allocation.
- Test oracle functions use names `reference_<strategy_name>(ctx, params)` and must remain pure Python/NumPy copies of the pre-optimization behavior.

- [ ] **Step 1: Add the test-only reference module**

Create `tests/strategies/reference_signals.py`. Copy the current implementation bodies at approved baseline commit `d6dbc5bc8a261c70373ba59c05de0414f143bb0a` into functions prefixed with `reference_`. Keep helper behavior literal, including slice bounds and NumPy defaults. Start with the eight profiler representatives:

```python
reference_regression_slope
reference_compression_breakout
reference_bollinger_reversion
reference_rejection_candle
reference_opening_range_breakout
reference_volatility_contraction_reversion
reference_standardized_return_signal
reference_zscore_reversion
```

For `reference_compression_breakout`, use the current optimized expanding-quantile helper only for the already-proven strict-prefix threshold calculation, but preserve its current Python post-threshold loop literally. Do not reintroduce the obsolete quadratic expanding `np.quantile(ranges[:i])` implementation.

- [ ] **Step 2: Add a deterministic randomized-market fixture and equality helper**

In `tests/strategies/test_signal_kernel_equivalence.py`, add:

```python
import numpy as np

from xau_lab.strategies.base import StrategyContext


def random_context(seed: int = 9215000, n: int = 4096) -> StrategyContext:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.8, n))
    open_ = close + rng.normal(0.0, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0.01, 1.2, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 1.2, n)
    atr = rng.uniform(0.1, 5.0, n)
    flags = np.zeros(n, dtype=bool)
    for start in range(100, n, 500):
        flags[start:min(start + 180, n)] = True
    return StrategyContext(
        open=np.ascontiguousarray(open_),
        high=np.ascontiguousarray(high),
        low=np.ascontiguousarray(low),
        close=np.ascontiguousarray(close),
        spread=np.zeros(n, dtype=float),
        atr14=np.ascontiguousarray(atr),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={"session_london": flags, "session_asia": flags, "session_new_york": flags},
    )


def assert_signal_equal(reference, optimized, ctx, params):
    expected = reference(ctx, params)
    actual = optimized(ctx, params)
    assert expected.dtype == np.int8
    assert actual.dtype == np.int8
    assert np.array_equal(actual, expected)
```

- [ ] **Step 3: Add adversarial mutation cases to the fixture**

Add a helper that copies a context and injects NaN/Inf/flat regions without changing array lengths:

```python
def adversarial_context() -> StrategyContext:
    ctx = random_context(n=1024)
    open_ = ctx.open.copy(); high = ctx.high.copy(); low = ctx.low.copy()
    close = ctx.close.copy(); atr = ctx.atr14.copy()
    close[220] = np.nan
    high[350] = np.inf
    low[351] = -np.inf
    atr[500] = np.nan
    close[700:710] = 2100.0
    open_[700:710] = 2100.0
    high[700:710] = 2100.0
    low[700:710] = 2100.0
    return StrategyContext(
        open=open_, high=high, low=low, close=close,
        spread=np.zeros(len(close)), atr14=atr,
        time_epoch=ctx.time_epoch.copy(), features=dict(ctx.features),
    )
```

- [ ] **Step 4: Write RED tests for the future kernel helpers**

Add imports that intentionally fail until `_kernels.py` is created:

```python
from xau_lab.strategies._kernels import (
    _body_direction,
    _linear_quantile_sorted_window,
    _window_all_finite,
    _window_mean_std,
    _window_min_max,
)
```

Add exact helper assertions:

```python
def test_linear_quantile_window_matches_numpy_default_linear():
    values = np.array([4.0, 1.0, 9.0, 2.0, 2.0, 8.0], dtype=np.float64)
    scratch = np.empty(5, dtype=np.float64)
    for q in (0.05, 0.236588625, 0.4, 0.5, 0.95):
        got = _linear_quantile_sorted_window(values, 1, 6, q, scratch)
        expected = float(np.quantile(values[1:6], q))
        assert got == expected


def test_window_helpers_fail_closed_on_nonfinite_values():
    values = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
    assert not _window_all_finite(values, 0, 4)
    valid, mean, std = _window_mean_std(values, 0, 4)
    assert not valid
    assert np.isnan(mean)
    assert np.isnan(std)
```

- [ ] **Step 5: Run RED verification**

Run:

```powershell
python -m pytest tests/strategies/test_signal_kernel_equivalence.py -v
```

Expected: collection/import failure because `xau_lab.strategies._kernels` does not exist.

- [ ] **Step 6: Implement the private kernel module**

Create `xau_lab/strategies/_kernels.py` with Numba-safe helpers. Use explicit loops to keep finite checking and arithmetic ordering inspectable:

```python
from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def _window_all_finite(values: np.ndarray, start: int, end: int) -> bool:
    if start < 0 or end > len(values) or start >= end:
        return False
    for j in range(start, end):
        if not np.isfinite(values[j]):
            return False
    return True


@njit(cache=True)
def _window_mean_std(values: np.ndarray, start: int, end: int):
    if not _window_all_finite(values, start, end):
        return False, np.nan, np.nan
    n = end - start
    total = 0.0
    for j in range(start, end):
        total += values[j]
    mean = total / n
    sq = 0.0
    for j in range(start, end):
        d = values[j] - mean
        sq += d * d
    return True, mean, math.sqrt(sq / n)


@njit(cache=True)
def _window_min_max(values: np.ndarray, start: int, end: int):
    if not _window_all_finite(values, start, end):
        return False, np.nan, np.nan
    low = values[start]
    high = values[start]
    for j in range(start + 1, end):
        v = values[j]
        if v < low:
            low = v
        if v > high:
            high = v
    return True, low, high


@njit(cache=True)
def _linear_quantile_sorted_window(
    values: np.ndarray,
    start: int,
    end: int,
    q: float,
    scratch: np.ndarray,
) -> float:
    n = end - start
    if n <= 0 or n > len(scratch) or q < 0.0 or q > 1.0:
        return np.nan
    for j in range(n):
        v = values[start + j]
        if not np.isfinite(v):
            return np.nan
        scratch[j] = v
    ordered = np.sort(scratch[:n])
    h = (n - 1) * q
    lo = int(math.floor(h))
    frac = h - lo
    if frac == 0.0:
        return ordered[lo]
    hi = lo + 1
    diff = ordered[hi] - ordered[lo]
    if frac >= 0.5:
        return ordered[hi] - diff * (1.0 - frac)
    return ordered[lo] + diff * frac


@njit(cache=True)
def _body_direction(open_: float, close: float) -> np.int8:
    if not np.isfinite(open_) or not np.isfinite(close):
        return np.int8(0)
    if close > open_:
        return np.int8(1)
    if close < open_:
        return np.int8(-1)
    return np.int8(0)
```

If exact helper tests expose a floating-order difference versus NumPy, adjust only the helper implementation until exact test expectations pass; do not loosen tests to `allclose` for signal-equivalence work.

- [ ] **Step 7: Run GREEN helper tests**

Run:

```powershell
python -m pytest tests/strategies/test_signal_kernel_equivalence.py -v
```

Expected: helper tests pass; representative strategy oracle tests may still be pending until later tasks.

- [ ] **Step 8: Commit**

```powershell
git add xau_lab/strategies/_kernels.py tests/strategies/reference_signals.py tests/strategies/test_signal_kernel_equivalence.py
git commit -m "test: freeze signal semantics and add kernel primitives"
```

---

### Task 2: Compile the volatility family first

**Files:**
- Modify: `xau_lab/strategies/volatility.py`
- Modify: `tests/strategies/reference_signals.py`
- Modify: `tests/strategies/test_signal_kernel_equivalence.py`
- Test: `tests/strategies/test_misc_families.py`

**Interfaces:**
- Public functions remain `atr_percentile_regime(ctx, params) -> np.ndarray`, `volatility_expansion_direction(ctx, params) -> np.ndarray`, and `volatility_contraction_reversion(ctx, params) -> np.ndarray`.
- Private kernels consume raw contiguous arrays and scalar parameters; no `StrategyContext` or `dict` crosses the Numba boundary.

- [ ] **Step 1: Add literal reference functions for all three volatility strategies**

In `tests/strategies/reference_signals.py`, copy the current bodies from baseline `d6dbc5...` as:

```python
reference_atr_percentile_regime
reference_volatility_expansion_direction
reference_volatility_contraction_reversion
```

- [ ] **Step 2: Add RED equivalence tests across finite and adversarial contexts**

Add:

```python
@pytest.mark.parametrize(
    "optimized,reference,params",
    [
        (atr_percentile_regime, reference_atr_percentile_regime,
         {"lookback": 37, "percentile": 0.731}),
        (volatility_expansion_direction, reference_volatility_expansion_direction,
         {"lookback": 41, "range_multiple": 1.83}),
        (volatility_contraction_reversion, reference_volatility_contraction_reversion,
         {"lookback": 53, "percentile": 0.236588625}),
    ],
)
def test_volatility_kernel_signal_equivalence(optimized, reference, params):
    for market in (random_context(), adversarial_context()):
        assert_signal_equal(reference, optimized, market, params)
```

Add boundary parameter cases using lookbacks `5`, `10`, `100`, `200` only where valid for that strategy, and percentile boundaries `0.05`, `0.40`, `0.5`, `0.95`.

- [ ] **Step 3: Run the volatility equivalence tests before implementation**

Run:

```powershell
python -m pytest tests/strategies/test_signal_kernel_equivalence.py -k volatility -v
```

Expected: PASS because optimized names still refer to the baseline implementations. Record this as the semantic baseline; the RED condition for this refactor is supplied by the next structural test.

- [ ] **Step 4: Add a structural RED test requiring a compiled private kernel**

Import the future dispatcher:

```python
from xau_lab.strategies.volatility import _volatility_contraction_reversion_kernel


def test_volatility_contraction_uses_numba_dispatcher():
    market = random_context(n=512)
    volatility_contraction_reversion(market, {"lookback": 25, "percentile": 0.25})
    assert _volatility_contraction_reversion_kernel.signatures
```

Run the single test and require import/attribute failure before implementation.

- [ ] **Step 5: Implement compiled volatility kernels**

In `xau_lab/strategies/volatility.py`, import `njit`, `_body_direction`, and `_linear_quantile_sorted_window`. The contraction kernel must allocate one scratch buffer per experiment, not per bar:

```python
@njit(cache=True)
def _volatility_contraction_reversion_kernel(open_, high, low, close, lookback, percentile):
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    ranges = high - low
    scratch = np.empty(lookback, dtype=np.float64)
    for i in range(lookback, n):
        current = ranges[i]
        if not np.isfinite(current):
            continue
        threshold = _linear_quantile_sorted_window(
            ranges, i - lookback, i, percentile, scratch
        )
        if not np.isfinite(threshold) or current > threshold:
            continue
        out[i] = np.int8(-_body_direction(open_[i], close[i]))
    return out
```

Use the same architecture for ATR percentile and expansion median. For the median strategy call `_linear_quantile_sorted_window(..., 0.5, ...)` so the median semantics remain NumPy-compatible.

Keep public wrappers as scalar extraction only:

```python
def volatility_contraction_reversion(ctx, params):
    return _volatility_contraction_reversion_kernel(
        ctx.open, ctx.high, ctx.low, ctx.close,
        int(params["lookback"]), float(params["percentile"]),
    )
```

- [ ] **Step 6: Run volatility GREEN tests**

Run:

```powershell
python -m pytest tests/strategies/test_signal_kernel_equivalence.py -k volatility -v
python -m pytest tests/strategies/test_misc_families.py -k volatility -v
```

Expected: exact signal arrays and existing golden tests all pass.

- [ ] **Step 7: Run a canonical single-strategy timing probe**

Use `EXP37889369D2F1` only for a sanity timing of the already-profiled breakout? No: this task must time the exact volatility representative from `multi_probe/PROBE_CATALOG.csv` (`EXPA6DA73D5B2BA`) using the read-only profiler pattern. Record signal-generation seconds; do not create campaign results yet.

Target: signal time materially below the old `189.002s`. This is an intermediate diagnostic, not a correctness criterion.

- [ ] **Step 8: Commit**

```powershell
git add xau_lab/strategies/volatility.py tests/strategies/reference_signals.py tests/strategies/test_signal_kernel_equivalence.py
git commit -m "perf: compile volatility signal kernels"
```

---

### Task 3: Compile mean-reversion and statistical hot paths

**Files:**
- Modify: `xau_lab/strategies/mean_reversion.py`
- Modify: `xau_lab/strategies/statistical.py`
- Modify: `tests/strategies/reference_signals.py`
- Modify: `tests/strategies/test_signal_kernel_equivalence.py`
- Test: `tests/strategies/test_mean_reversion.py`
- Test: `tests/strategies/test_misc_families.py`

**Interfaces:**
- All existing public strategy signatures and registry names remain unchanged.
- Private kernels receive raw arrays/scalars and return contiguous `np.int8` vectors.

- [ ] **Step 1: Extend reference oracles to all mean-reversion strategies**

Add literal baseline functions:

```python
reference_zscore_reversion
reference_bollinger_reversion
reference_rsi_extreme
reference_stochastic_extreme
reference_cci_extreme
reference_williams_r_extreme
```

- [ ] **Step 2: Extend reference oracles to all statistical strategies**

Add:

```python
reference_return_continuation
reference_return_reversal
reference_rolling_autocorr_direction
reference_range_position_reversal
reference_standardized_return_signal
```

- [ ] **Step 3: Add parameterized exact-equality tests**

Use random and adversarial contexts. Include threshold-boundary fixtures where the current value is exactly at `z_threshold`, `std_multiplier`, RSI lower/upper, stochastic lower/upper, CCI threshold, Williams %R edge band, and autocorrelation `min_abs_autocorr`. Equality operators must remain identical to baseline (`>=`, `<=`, `>`, `<` as currently implemented).

- [ ] **Step 4: Add structural RED imports for representative kernels**

Require at least:

```python
from xau_lab.strategies.mean_reversion import _zscore_reversion_kernel
from xau_lab.strategies.statistical import _standardized_return_signal_kernel
```

Call public wrappers on a 512-row context and assert each dispatcher has non-empty `.signatures`. Expected pre-implementation: import/attribute failure.

- [ ] **Step 5: Implement mean/std window kernels without changing current window endpoints**

For `zscore_reversion` and `bollinger_reversion`, call `_window_mean_std(close, i-lookback+1, i+1)`. For `standardized_return_signal`, create the same returns vector as baseline and call `_window_mean_std(returns, i-lookback, i)`, comparing the current `returns[i]` against that strict prior window.

Representative z-score kernel:

```python
@njit(cache=True)
def _zscore_reversion_kernel(close, lookback, threshold):
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback - 1, len(close)):
        valid, mean, std = _window_mean_std(close, i - lookback + 1, i + 1)
        if not valid or std <= 0.0:
            continue
        z = (close[i] - mean) / std
        if z >= threshold:
            out[i] = -1
        elif z <= -threshold:
            out[i] = 1
    return out
```

Implement RSI, stochastic, CCI, Williams %R and range-position with explicit compiled loops. For `rolling_autocorr_direction`, reproduce the current return vector, population std, and correlation numerator/denominator in the same logical window; verify only final signal equality, not approximate correlation equality.

- [ ] **Step 6: Run GREEN family tests**

Run:

```powershell
python -m pytest tests/strategies/test_signal_kernel_equivalence.py -k "mean_reversion or statistical" -v
python -m pytest tests/strategies/test_mean_reversion.py tests/strategies/test_misc_families.py -v
```

Expected: all exact oracle and existing golden tests pass.

- [ ] **Step 7: Benchmark the three old hot representatives**

Time only signal generation for:

```text
EXPB62F70031929 bollinger_reversion       old 63.789s
EXPB1A10E7A4435 standardized_return_signal old 66.547s
EXP2053D4626356 zscore_reversion           old 64.508s
```

Record new times in the PR conversation after correctness tests pass.

- [ ] **Step 8: Commit**

```powershell
git add xau_lab/strategies/mean_reversion.py xau_lab/strategies/statistical.py tests/strategies/reference_signals.py tests/strategies/test_signal_kernel_equivalence.py
git commit -m "perf: compile mean reversion and statistical signals"
```

---

### Task 4: Compile trend/momentum and remaining breakout loops

**Files:**
- Modify: `xau_lab/strategies/trend.py`
- Modify: `xau_lab/strategies/breakout.py`
- Modify: `tests/strategies/reference_signals.py`
- Modify: `tests/strategies/test_signal_kernel_equivalence.py`
- Test: `tests/strategies/test_trend.py`
- Test: `tests/strategies/test_breakout.py`

**Interfaces:**
- Preserve public names and outputs.
- Keep `_expanding_linear_quantiles()` and its existing exact tests unchanged unless a wrapper-only refactor is necessary; the task is to compile the remaining post-threshold loop and other breakout loops.

- [ ] **Step 1: Add literal trend references**

Add:

```python
reference_sma_slope
reference_ema_slope
reference_regression_slope
reference_roc_momentum
reference_efficiency_trend
```

- [ ] **Step 2: Add remaining breakout references**

Add:

```python
reference_nbar_breakout
reference_nbar_failed_breakout
reference_range_expansion
reference_bollinger_expansion
reference_compression_breakout
```

The compression reference must call the current `_expanding_linear_quantiles` helper and then use the baseline Python loop so the already-approved expanding-quantile semantics stay fixed.

- [ ] **Step 3: Add exact randomized/adversarial equality cases**

Cover lookback and horizon boundaries, zero ATR, NaN/Inf, flat prices, repeated highs/lows, exact breakout equality (`close == prior_high` must not become a breakout), and exact threshold equality for momentum directions.

- [ ] **Step 4: Add structural RED tests**

Require:

```python
from xau_lab.strategies.trend import _regression_slope_kernel
from xau_lab.strategies.breakout import _compression_breakout_signal_kernel
```

Call public functions and assert Numba dispatchers have compiled signatures. Expected before implementation: import/attribute failure.

- [ ] **Step 5: Implement regression and trend kernels**

Regression must preserve the baseline least-squares definition. Compute `x_centered` and denominator once per call in the Python wrapper or inside the kernel before the bar loop, then for each finite window compute its mean and dot product in compiled loops. Do not replace the regression strategy with an existing feature column unless that feature's lookback and exact arithmetic are guaranteed identical for every sampled parameter.

For EMA slope, reproduce `_ema_of_window` exactly inside Numba; do not substitute the canonical EMA feature because the strategy recomputes EMA over a finite window starting from that window's first value.

- [ ] **Step 6: Compile breakout bar loops**

Keep `_expanding_linear_quantiles(ranges, q)` as the threshold producer. Add a private kernel that receives the precomputed thresholds and evaluates compressed-window mean plus breakout high/low using compiled loops:

```python
@njit(cache=True)
def _compression_breakout_signal_kernel(
    high, low, close, ranges, thresholds, compression_lookback, breakout_lookback
):
    out = np.zeros(len(close), dtype=np.int8)
    warmup = max(compression_lookback, breakout_lookback)
    for i in range(warmup, len(close)):
        # validate and sum ranges[i-compression_lookback:i]
        # validate/max high[i-breakout_lookback:i]
        # validate/min low[i-breakout_lookback:i]
        # preserve `mean > threshold` skip and strict close breakout comparisons
        pass
    return out
```

In the actual implementation, replace the comment block and `pass` with explicit compiled loops; no Python bar loop may remain in `compression_breakout`.

- [ ] **Step 7: Run GREEN tests**

Run:

```powershell
python -m pytest tests/strategies/test_signal_kernel_equivalence.py -k "trend or breakout" -v
python -m pytest tests/strategies/test_trend.py tests/strategies/test_breakout.py -v
```

Expected: exact equality and all existing exact expanding-quantile tests pass.

- [ ] **Step 8: Benchmark old representatives**

Time signal generation for:

```text
EXP1E2D424171EC regression_slope    old 36.541s
EXP37889369D2F1 compression_breakout old 35.094s
```

- [ ] **Step 9: Commit**

```powershell
git add xau_lab/strategies/trend.py xau_lab/strategies/breakout.py tests/strategies/reference_signals.py tests/strategies/test_signal_kernel_equivalence.py
git commit -m "perf: compile trend and breakout signal loops"
```

---

### Task 5: Compile price-action and session state machines

**Files:**
- Modify: `xau_lab/strategies/price_action.py`
- Modify: `xau_lab/strategies/session.py`
- Modify: `tests/strategies/reference_signals.py`
- Modify: `tests/strategies/test_signal_kernel_equivalence.py`
- Test: `tests/strategies/test_misc_families.py`

**Interfaces:**
- Public session wrappers continue to call `_session_flag(ctx, session)` before entering Numba so missing session features still raise the same `KeyError` at the Python boundary.
- Numba session kernels consume boolean flags plus market arrays/scalars.

- [ ] **Step 1: Add literal references for price-action strategies**

Add:

```python
reference_engulfing
reference_rejection_candle
reference_inside_bar_break
reference_outside_bar
reference_level_sweep_reclaim
```

- [ ] **Step 2: Add literal references for session strategies**

Add:

```python
reference_session_open_momentum
reference_prior_session_high_low_break
reference_opening_range_breakout
```

- [ ] **Step 3: Add exact equality tests**

Include finite random contexts, NaN/Inf candle fields, zero-body candles, equal wick lengths, exact mother-bar boundaries, session false→true transitions, gaps between session blocks, invalid opening-range rows, and `opening_range_bars` boundaries.

- [ ] **Step 4: Add structural RED tests**

Require:

```python
from xau_lab.strategies.price_action import _rejection_candle_kernel
from xau_lab.strategies.session import _opening_range_breakout_kernel
```

After a public call, assert both dispatchers have `.signatures`. Expected before implementation: import/attribute failure.

- [ ] **Step 5: Implement price-action kernels**

Do not allocate `np.array([...])` inside each bar. Read scalar values directly and test `np.isfinite` on each scalar. Preserve all strict/weak comparisons exactly.

- [ ] **Step 6: Implement session kernels**

Compile the existing state machines directly. For `opening_range_breakout`, preserve the rule that a non-finite session bar increments `session_bar`; do not silently omit it from the opening-bar count. For `prior_session_high_low_break`, only promote the active high/low to previous-session levels when the session transitions to false and both active values exist.

- [ ] **Step 7: Run GREEN tests**

Run:

```powershell
python -m pytest tests/strategies/test_signal_kernel_equivalence.py -k "price_action or session" -v
python -m pytest tests/strategies/test_misc_families.py -v
```

- [ ] **Step 8: Benchmark the final representatives**

Time:

```text
EXPB8942B9FF195 rejection_candle       old 14.466s
EXPB8F12709D5FE opening_range_breakout old 3.936s
```

- [ ] **Step 9: Commit**

```powershell
git add xau_lab/strategies/price_action.py xau_lab/strategies/session.py tests/strategies/reference_signals.py tests/strategies/test_signal_kernel_equivalence.py
git commit -m "perf: compile price action and session signals"
```

---

### Task 6: Full semantic regression and canonical 8-experiment profile

**Files:**
- No production changes unless verification reveals a specific bug.
- Use: `multi_probe/PROBE_CATALOG.csv` locally.

**Interfaces:**
- Produces: proof that Stage 1 preserves tests/results and a measured runtime profile under the new commit.

- [ ] **Step 1: Run complete local test suite**

Run:

```powershell
python -m pytest -v
```

Expected: zero failures. Record exact test count and duration.

- [ ] **Step 2: Reconfirm immutable artifact hashes**

Run:

```powershell
Get-FileHash .\results\EXPERIMENT_CATALOG.csv -Algorithm SHA256
Get-FileHash .\data\features\XAUUSD_M1_FEATURES.parquet -Algorithm SHA256
```

Require exactly the hashes in Global Constraints.

- [ ] **Step 3: Run the same eight-experiment phase profiler**

Use the previously established profiler script against `multi_probe/PROBE_CATALOG.csv`. Record market load, per-strategy signal/backtest/metrics/total times, and aggregate shares.

Compare directly against baseline signal times:

```text
regression_slope                    36.541s
compression_breakout               35.094s
bollinger_reversion                63.789s
rejection_candle                   14.466s
opening_range_breakout              3.936s
volatility_contraction_reversion  189.002s
standardized_return_signal         66.547s
zscore_reversion                   64.508s
```

- [ ] **Step 4: Run exact canonical result parity for all eight representatives**

Create a fresh result namespace, run `scripts.run_campaign` with the same 8-row probe catalog and `--workers 2`, then compare each new `MASTER_RESULTS.csv` row against the last known-good 8-row run for all common stored fields. Require zero differences for every experiment and zero `ERRORS.csv` rows.

If the old `multi_probe/results` is no longer available locally, compare the seven non-session experiments against the pre-session 8-probe output only if that artifact was retained and compare the session experiment against the approved post-session output; otherwise rerun the approved baseline commit in an isolated worktree before accepting parity. Do not guess historical values.

- [ ] **Step 5: Evaluate the intermediate gate**

Compute aggregate profile seconds. If signal generation has not dropped enough to make the 24-hour target plausible, stop before 100-smoke and profile the remaining hot strategy implementations. Do not add shared caching without a new design review.

- [ ] **Step 6: Commit verification notes only if repository documentation is intentionally updated**

No code commit is required solely for local benchmark output. Record canonical evidence in the PR conversation rather than committing machine-specific result files.

---

### Task 7: Fresh 100-smoke and final Stage 1 runtime gate

**Files:**
- No production changes unless a verification failure is diagnosed and fixed through a new RED/GREEN cycle.

**Interfaces:**
- Produces: authoritative Stage 1 throughput decision for the full campaign.

- [ ] **Step 1: Confirm clean tracked worktree and exact branch head**

Run:

```powershell
git status --short --untracked-files=no
git rev-parse HEAD
```

Require no tracked modifications. Record the exact commit SHA in the smoke manifest evidence.

- [ ] **Step 2: Run authoritative PR CI and require GREEN**

Require GitHub Actions full pytest to pass at the exact smoke commit. Record run ID, exact test count, and duration before launching the canonical smoke.

- [ ] **Step 3: Run a fresh 100-smoke namespace**

Use a new path such as:

```powershell
$smoke = ".\results\SMOKE_100_SIGNAL_KERNEL_V1"
Remove-Item -Recurse -Force $smoke -ErrorAction SilentlyContinue
python -m scripts.run_smoke `
  --catalog .\results\EXPERIMENT_CATALOG.csv `
  --features .\data\features\XAUUSD_M1_FEATURES.parquet `
  --result-root $smoke `
  --count 100 `
  --workers 2 `
  --seed 9215000
```

- [ ] **Step 4: Verify smoke completeness and quotas**

Require:

```text
COMPLETED=100
UNIQUE_IDS=100
breakout=16
exit_execution=8
mean_reversion=16
price_action=14
session=10
statistical=10
trend_momentum=16
volatility=10
NO ERRORS
```

- [ ] **Step 5: Verify provenance**

`RUN_MANIFEST.json` must contain:

```text
catalog_sha256 = ab2ad9935fb4e9a229ce1d84e2e9b80c4d670a8ceb7e577bf3cae0b70f0706a9
source_data_sha256 = 5d3680d043e1f8d4c759124ad177bc73e47c055101dd3be7558564187b0c10ef
software_git_commit = exact current smoke commit
workers = 2
campaign_seed = 9215000
```

- [ ] **Step 6: Verify resource stability**

Record C:/E: free space and `Win32_PageFileUsage`. Reject the gate on any `MemoryError`, uncontrolled disk growth, or a material return of the previous memory-exhaustion behavior.

- [ ] **Step 7: Apply the non-negotiable throughput decision**

Read `THROUGHPUT_BENCHMARK.json`.

If:

```text
projected_50000_hours <= 24
```

then Stage 1 passes the runtime gate and the full campaign may be planned as the next separate action.

If:

```text
projected_50000_hours > 24
```

then the full campaign remains blocked. Stop and create a new Stage 2 shared-cache design; do not merge PR #16 or launch 50k merely because Stage 1 is faster.

- [ ] **Step 8: Update PR #16 evidence, but do not merge**

Update the draft PR body/comment with exact CI, parity, smoke, hash, memory, and throughput evidence. Keep the PR draft and unmerged unless the user explicitly says `merge #16`.
