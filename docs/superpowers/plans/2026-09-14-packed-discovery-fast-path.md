# Stage 2 Packed Discovery Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Python `Trade` object materialization in normal discovery runs with an exact packed-results path so the canonical 50,000-experiment campaign can project to 24 hours or less without changing any stored experiment result.

**Architecture:** Preserve `run_fast_backtest()` and the detailed `Trade` API as the semantic oracle. Add a packed backtest result carrying the existing Numba kernel arrays, summarize those arrays through a focused exact packed-metrics module, and route `run_experiment(..., include_trades=False)` through that path. Keep `include_trades=True` on the current detailed path. Precompute compact broker-calendar group IDs once per loaded worker market so packed metrics never perform per-trade timezone/date parsing.

**Tech Stack:** Python 3.12, NumPy 2.x, Numba 0.61+, Polars 1.x, PyArrow, pytest 8.x.

**Spec:** `docs/superpowers/specs/2026-09-14-packed-discovery-fast-path-design.md`

## Global Constraints

- Preserve canonical feature SHA256 `5D3680D043E1F8D4C759124AD177BC73E47C055101DD3BE7558564187B0C10EF` and 3,403,685 rows.
- Preserve canonical catalog SHA256 `AB2AD9935FB4E9A229CE1D84E2E9B80C4D670A8CEB7E577BF3CAE0B70F0706A9`, 50,000 IDs, and seed `9215000`.
- Preserve signal arrays, direction filtering, `entry_allowed`, costs, risk sizing, entry/exit ordering, same-bar stop/target precedence, time exits, trailing stops, and end-of-data handling exactly.
- Preserve `run_fast_backtest(...) -> BacktestResult` and detailed `Trade` semantics for all existing callers.
- Packed discovery is used only when `include_trades=False`.
- Complete packed-vs-detailed master-result equality is required; do not replace exact checks with `allclose`, tolerances, or ignored fields.
- Do not use `fastmath`, parallel floating reductions, approximate medians, approximate quantiles, or unordered accumulation.
- Do not introduce giant worker-global matrices, object-dtype per-trade arrays, or retained per-experiment packed state.
- Keep parent-only result writing and existing per-experiment exception isolation.
- The authoritative runtime gate remains a fresh 100-experiment smoke at exactly 2 workers.
- Do not launch 50,000 experiments and do not use `--allow-slow` unless a separately approved research-protocol change explicitly authorizes it.
- If the new 100-smoke still projects above 24 hours, stop and re-profile instead of broadening this implementation.

## File Structure

- Create `xau_lab/backtest/packed.py`: immutable packed-result container only; no trading policy.
- Modify `xau_lab/backtest/fast.py`: expose `run_packed_backtest()` around the existing `_kernel`; make `run_fast_backtest()` materialize detailed trades from the packed result without changing behavior.
- Create `xau_lab/metrics/packed.py`: exact compiled trade-economics finalization and exact packed summary metrics.
- Modify `xau_lab/runner/single.py`: add compact broker calendar group IDs to `MarketBundle`; route discovery to packed backtest + packed summary.
- Modify `xau_lab/runner/campaign.py`: derive broker day/month/year group IDs once when loading a worker market, using canonical broker-local calendar columns.
- Add `tests/backtest/test_packed_backtest.py`: packed-kernel result shape, validation, and detailed-path compatibility.
- Add `tests/metrics/test_packed_performance.py`: exact packed-vs-`summarize_trades()` metric parity.
- Modify `tests/runner/test_single.py`: discovery structural test and complete master-result parity.
- Modify `tests/runner/test_campaign.py`: loaded market calendar-group integrity and worker determinism remain intact.
- Create `scripts/verify_packed_parity.py`: reproducible frozen 8/100 complete-result parity harness; verification only, not campaign logic.

---

### Task 1: Expose the Existing Numba Kernel as a Packed Backtest Result

**Files:**
- Create: `xau_lab/backtest/packed.py`
- Modify: `xau_lab/backtest/fast.py`
- Create: `tests/backtest/test_packed_backtest.py`
- Test: `tests/backtest/test_fast_parity.py`

**Interfaces:**
- Produces `PackedBacktestResult` with exact kernel output arrays and `trade_count` / `risk_skip_count`.
- Produces `run_packed_backtest(bars, signals, symbol, cost, risk, exit_spec) -> PackedBacktestResult`.
- `run_fast_backtest(...) -> BacktestResult` remains unchanged and becomes a detailed materializer over `run_packed_backtest()`.

- [ ] **Step 1: Add a RED test requiring the packed API**

Create `tests/backtest/test_packed_backtest.py` with the same deterministic random-market shape used by `test_fast_parity.py` and add:

```python
import numpy as np

from xau_lab.backtest.fast import run_fast_backtest, run_packed_backtest
from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.packed import PackedBacktestResult

SPEC = SymbolSpec(point=0.01, digits=2, contract_size=100.0, volume_min=0.01, volume_step=0.01)
COST = CostModel()
RISK = RiskModel()


def test_packed_backtest_exposes_kernel_rows_without_trade_objects():
    n = 256
    bars = MarketBars(
        time_epoch=np.arange(n, dtype=np.int64) * 60 + 1_700_000_000,
        open=np.full(n, 2000.0),
        high=np.full(n, 2000.7),
        low=np.full(n, 1999.3),
        close=np.full(n, 2000.0),
        spread=np.full(n, 20, dtype=np.int64),
        atr=np.full(n, 1.0),
    )
    signals = np.zeros(n, dtype=np.int8)
    signals[::5] = 1

    packed = run_packed_backtest(
        bars, signals, SPEC, COST, RISK, ExitSpec(stop_atr=1.0, target_r=1.0)
    )

    assert isinstance(packed, PackedBacktestResult)
    assert 0 <= packed.trade_count <= np.count_nonzero(signals)
    assert packed.risk_skip_count >= 0
    assert len(packed.direction) >= packed.trade_count
    assert len(packed.entry_index) >= packed.trade_count
    assert len(packed.exit_index) >= packed.trade_count
```

- [ ] **Step 2: Run RED verification**

Run:

```powershell
python -m pytest tests/backtest/test_packed_backtest.py -v
```

Expected: collection/import failure because `xau_lab.backtest.packed` and `run_packed_backtest` do not exist.

- [ ] **Step 3: Add the packed result container**

Create `xau_lab/backtest/packed.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedBacktestResult:
    trade_count: int
    risk_skip_count: int
    direction: np.ndarray
    signal_index: np.ndarray
    entry_index: np.ndarray
    exit_index: np.ndarray
    raw_entry: np.ndarray
    entry_price: np.ndarray
    initial_stop: np.ndarray
    target: np.ndarray
    lot: np.ndarray
    planned_risk: np.ndarray
    raw_exit: np.ndarray
    reason: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.trade_count)
        if count < 0 or int(self.risk_skip_count) < 0:
            raise ValueError("packed counts must be nonnegative")
        arrays = (
            self.direction,
            self.signal_index,
            self.entry_index,
            self.exit_index,
            self.raw_entry,
            self.entry_price,
            self.initial_stop,
            self.target,
            self.lot,
            self.planned_risk,
            self.raw_exit,
            self.reason,
        )
        if any(np.asarray(values).ndim != 1 for values in arrays):
            raise ValueError("packed trade arrays must be one-dimensional")
        if any(len(values) < count for values in arrays):
            raise ValueError("packed trade array shorter than trade_count")
```

Do not trim/copy arrays here; the kernel already owns them and the packed result must remain allocation-light.

- [ ] **Step 4: Refactor the kernel invocation into `run_packed_backtest()`**

In `xau_lab/backtest/fast.py`, keep `_kernel` byte-for-byte semantically unchanged. Move the existing signal validation, `target_r` / time / trail sentinel conversion, output-capacity calculation, and `_kernel(...)` call into:

```python
def run_packed_backtest(
    bars: MarketBars,
    signals: Sequence[int],
    symbol: SymbolSpec,
    cost: CostModel,
    risk: RiskModel,
    exit_spec: ExitSpec,
) -> PackedBacktestResult:
    raw_signals = np.asarray(signals)
    if raw_signals.ndim != 1 or len(raw_signals) != len(bars):
        raise ValueError("signals length must match MarketBars")
    signal_array = normalize_signal_array(raw_signals)

    target_r = -1.0 if exit_spec.target_r is None else float(exit_spec.target_r)
    time_exit = -1.0 if exit_spec.time_exit_minutes is None else float(exit_spec.time_exit_minutes)
    trail = -1.0 if exit_spec.atr_trail is None else float(exit_spec.atr_trail)
    output_capacity = int(np.count_nonzero(signal_array))

    values = _kernel(
        bars.time_epoch, bars.open, bars.high, bars.low, bars.close,
        bars.spread, bars.atr, signal_array,
        float(symbol.point), int(symbol.digits), float(symbol.contract_size),
        float(symbol.volume_min), float(symbol.volume_step), float(symbol.volume_max),
        float(cost.commission_round_trip_per_lot), float(cost.slippage_points_per_fill),
        float(risk.preferred_risk_usd), float(risk.hard_risk_usd), float(risk.max_lot),
        float(exit_spec.stop_atr), target_r, time_exit, trail, output_capacity,
    )
    return PackedBacktestResult(*values)
```

Import `PackedBacktestResult` from `xau_lab.backtest.packed`.

- [ ] **Step 5: Make the detailed path consume the packed result**

Extract the current Python object loop into `_materialize_packed_trades(...)`. Preserve the exact `_Position` construction and `_finalize_trade()` call order:

```python
def _materialize_packed_trades(packed, bars, symbol, cost):
    trades = []
    for i in range(packed.trade_count):
        target = None if np.isnan(packed.target[i]) else float(packed.target[i])
        position = _Position(
            direction=int(packed.direction[i]),
            signal_index=int(packed.signal_index[i]),
            entry_index=int(packed.entry_index[i]),
            raw_entry_bid=float(packed.raw_entry[i]),
            entry_price=float(packed.entry_price[i]),
            initial_stop=float(packed.initial_stop[i]),
            active_stop=float(packed.initial_stop[i]),
            target_price=target,
            lot=float(packed.lot[i]),
            planned_risk_usd=float(packed.planned_risk[i]),
        )
        trades.append(
            _finalize_trade(
                position,
                bars,
                int(packed.exit_index[i]),
                float(packed.raw_exit[i]),
                _REASON[int(packed.reason[i])],
                symbol,
                cost,
            )
        )
    return tuple(trades)
```

Then implement `run_fast_backtest()` as:

```python
packed = run_packed_backtest(bars, signals, symbol, cost, risk, exit_spec)
trades = _materialize_packed_trades(packed, bars, symbol, cost)
return BacktestResult(trades, risk_skip_count=packed.risk_skip_count)
```

- [ ] **Step 6: Run packed and existing detailed parity tests**

Run:

```powershell
python -m pytest tests/backtest/test_packed_backtest.py tests/backtest/test_fast_parity.py tests/backtest/test_reference_engine.py -v
```

Expected: all PASS. Existing fast/reference trade parity is the compatibility guard.

- [ ] **Step 7: Commit Task 1**

```powershell
git add xau_lab/backtest/packed.py xau_lab/backtest/fast.py tests/backtest/test_packed_backtest.py tests/backtest/test_fast_parity.py
git commit -m "refactor: expose packed fast-backtest results"
```

---

### Task 2: Precompute Compact Broker Calendar Group IDs Once Per Worker Market

**Files:**
- Modify: `xau_lab/runner/single.py`
- Modify: `xau_lab/runner/campaign.py`
- Modify: `tests/runner/test_single.py`
- Modify: `tests/runner/test_campaign.py`

**Interfaces:**
- `MarketBundle` gains read-only `broker_day_id`, `broker_month_id`, and `broker_year_id` contiguous `np.int32` arrays aligned 1:1 with bars.
- `load_market_bundle()` computes IDs once from canonical broker-local calendar columns.
- Tests/manual `MarketBundle` construction may omit IDs; `MarketBundle.__post_init__()` derives them from `broker_date` as a fallback.

- [ ] **Step 1: Add RED calendar-group tests**

In `tests/runner/test_single.py`, add a market crossing day/month/year boundaries:

```python
def test_market_bundle_derives_contiguous_broker_calendar_group_ids():
    market = _market()
    dates = np.array(
        [
            "2025-12-31", "2025-12-31",
            "2026-01-01", "2026-01-01",
            "2026-02-01", "2026-02-01",
        ],
        dtype=object,
    )
    bars = MarketBars(
        time_epoch=np.arange(6, dtype=np.int64) * 60 + 1_767_312_000,
        open=np.ones(6), high=np.ones(6) + 1, low=np.ones(6) - 1,
        close=np.ones(6), spread=np.zeros(6, dtype=np.int64), atr=np.ones(6),
    )
    bundle = MarketBundle(bars=bars, symbol=market.symbol, broker_date=dates, features={})

    assert bundle.broker_day_id.tolist() == [0, 0, 1, 1, 2, 2]
    assert bundle.broker_month_id.tolist() == [0, 0, 1, 1, 2, 2]
    assert bundle.broker_year_id.tolist() == [0, 0, 1, 1, 1, 1]
    assert bundle.broker_day_id.dtype == np.int32
```

- [ ] **Step 2: Run RED verification**

```powershell
python -m pytest tests/runner/test_single.py::test_market_bundle_derives_contiguous_broker_calendar_group_ids -v
```

Expected: FAIL because the fields do not exist.

- [ ] **Step 3: Add `MarketBundle` calendar fields and fallback derivation**

In `xau_lab/runner/single.py`, extend the dataclass after `features`:

```python
broker_day_id: np.ndarray | None = None
broker_month_id: np.ndarray | None = None
broker_year_id: np.ndarray | None = None
```

Add a private fallback helper that walks only date changes and returns `np.int32` arrays. It must treat `YYYY-MM-DD` strings as canonical and increment IDs when day, month, or year changes. In `__post_init__`, if any calendar ID array is missing, derive all three; otherwise normalize all to contiguous `np.int32`, validate length, set read-only, and store them.

- [ ] **Step 4: Compute IDs vectorially in `load_market_bundle()`**

In `xau_lab/runner/campaign.py`, add:

```python
def _contiguous_group_ids(*columns: np.ndarray) -> np.ndarray:
    if not columns:
        raise ValueError("at least one calendar column is required")
    n = len(columns[0])
    if any(len(values) != n for values in columns):
        raise ValueError("calendar columns must have equal length")
    if n == 0:
        return np.empty(0, dtype=np.int32)
    changed = np.ones(n, dtype=np.bool_)
    changed[1:] = False
    for values in columns:
        changed[1:] |= values[1:] != values[:-1]
    return (np.cumsum(changed, dtype=np.int64) - 1).astype(np.int32)
```

Use canonical `broker_date`, `year`, and `month` columns when available:

```python
year = np.asarray(frame["year"].to_numpy(), dtype=np.int32)
month = np.asarray(frame["month"].to_numpy(), dtype=np.int8)
day_id = _contiguous_group_ids(broker_date)
month_id = _contiguous_group_ids(year, month)
year_id = _contiguous_group_ids(year)
```

If legacy/test feature files lack `year`/`month`, parse them once from `broker_date` before calling `_contiguous_group_ids`. Pass all IDs into `MarketBundle`.

- [ ] **Step 5: Verify loader behavior and campaign determinism**

Add an assertion in `tests/runner/test_campaign.py` that `_write_market()`-loaded bundles receive nondecreasing calendar IDs with correct length, then run:

```powershell
python -m pytest tests/runner/test_single.py tests/runner/test_campaign.py -v
```

Expected: PASS, including deterministic 1-worker vs 2-worker master results.

- [ ] **Step 6: Commit Task 2**

```powershell
git add xau_lab/runner/single.py xau_lab/runner/campaign.py tests/runner/test_single.py tests/runner/test_campaign.py
git commit -m "perf: precompute broker calendar group ids"
```

---

### Task 3: Implement Exact Packed Trade Economics and Summary Metrics

**Files:**
- Create: `xau_lab/metrics/packed.py`
- Create: `tests/metrics/test_packed_performance.py`
- Test: `tests/metrics/test_performance.py`
- Test: `tests/backtest/test_packed_backtest.py`

**Interfaces:**
- Consumes `PackedBacktestResult`, `MarketBars`, `SymbolSpec`, `CostModel`, `RiskModel`, and three aligned calendar-ID arrays.
- Produces `summarize_packed_backtest(...) -> dict[str, float | int | None]` with the same keys and values as `summarize_trades()`.
- No Python `Trade` or `_Position` objects are created.

- [ ] **Step 1: Add RED exact-equivalence tests**

Create `tests/metrics/test_packed_performance.py`. Build small deterministic bars/signals, call both paths, and compare the complete dict:

```python
from xau_lab.backtest.fast import run_fast_backtest, run_packed_backtest
from xau_lab.metrics.packed import summarize_packed_backtest
from xau_lab.metrics.performance import summarize_trades


def assert_packed_summary_exact(bars, signals, symbol, cost, risk, exit_spec, day_id, month_id, year_id):
    detailed = run_fast_backtest(bars, signals, symbol, cost, risk, exit_spec)
    packed = run_packed_backtest(bars, signals, symbol, cost, risk, exit_spec)

    expected = summarize_trades(detailed.trades, starting_equity=risk.account_equity)
    actual = summarize_packed_backtest(
        packed,
        bars,
        symbol,
        cost,
        risk,
        day_id,
        month_id,
        year_id,
    )
    assert actual == expected
```

Add separate tests covering: no trades; only wins; only losses; mixed long/short; commission/spread/slippage; same-bar ambiguity; target/time/trail/end-of-data exits; multiple trades per day; day/month/year changes; odd/even medians; equal P/L; and risk skips.

- [ ] **Step 2: Run RED verification**

```powershell
python -m pytest tests/metrics/test_packed_performance.py -v
```

Expected: import failure because `xau_lab.metrics.packed` does not exist.

- [ ] **Step 3: Implement exact compiled trade economics**

Create `xau_lab/metrics/packed.py` with `@njit(cache=True)` helpers and no `fastmath=True`.

The economics loop must preserve `_finalize_trade()` arithmetic order exactly:

```python
@njit(cache=True)
def _finalize_economics(
    trade_count,
    direction,
    entry_index,
    exit_index,
    raw_entry,
    lot,
    planned_risk,
    raw_exit,
    spread,
    time_epoch,
    point,
    contract_size,
    commission_per_lot,
    slippage_points,
):
    net = np.empty(trade_count, dtype=np.float64)
    pnl_r = np.empty(trade_count, dtype=np.float64)
    hold = np.empty(trade_count, dtype=np.float64)
    commission = np.empty(trade_count, dtype=np.float64)
    spread_cost = np.empty(trade_count, dtype=np.float64)
    slippage_cost = np.empty(trade_count, dtype=np.float64)

    for i in range(trade_count):
        d = int(direction[i])
        entry = int(entry_index[i])
        exit_ = int(exit_index[i])
        size = float(lot[i])
        gross = d * (float(raw_exit[i]) - float(raw_entry[i])) * contract_size * size
        spread_points = int(spread[entry]) if d == 1 else int(spread[exit_])
        sp_cost = float(spread_points) * point * contract_size * size
        slip_cost = 2.0 * slippage_points * point * contract_size * size
        comm = commission_per_lot * size
        value = gross - sp_cost - slip_cost - comm

        net[i] = value
        commission[i] = comm
        spread_cost[i] = sp_cost
        slippage_cost[i] = slip_cost
        pnl_r[i] = value / float(planned_risk[i]) if float(planned_risk[i]) > 0.0 else np.nan
        hold[i] = (int(time_epoch[exit_]) - int(time_epoch[entry])) / 60.0

    return net, pnl_r, hold, commission, spread_cost, slippage_cost
```

Do not compute exit-price fields; discovery metrics do not consume them.

- [ ] **Step 4: Implement exact sequential summary helpers**

Implement compiled helpers for:

- sequential gross profit/loss, wins/losses, repeated `sum(net)` semantics,
- equity/drawdown in trade order,
- max loss streak,
- long/short profit factors in trade order,
- sequential day/month/year accumulation using `entry_index -> group_id`,
- exact median via sorted numeric copy and `(a+b)/2.0` for even counts,
- exact top-five positive trade values kept in descending order and summed descending,
- exact best/worst month values.

Use `np.nan` only as an internal undefined sentinel; the Python wrapper must convert undefined fields to `None` so the returned dict matches `summarize_trades()` exactly.

The wrapper signature must be:

```python
def summarize_packed_backtest(
    packed: PackedBacktestResult,
    bars: MarketBars,
    symbol: SymbolSpec,
    cost: CostModel,
    risk: RiskModel,
    broker_day_id: np.ndarray,
    broker_month_id: np.ndarray,
    broker_year_id: np.ndarray,
) -> dict[str, float | int | None]:
    ...
```

Validate all calendar arrays are one-dimensional and aligned with `bars`; validate packed entry/exit indices are in range before compiled summarization.

- [ ] **Step 5: Run metric exactness tests**

```powershell
python -m pytest tests/metrics/test_packed_performance.py tests/metrics/test_performance.py tests/backtest/test_fast_parity.py -v
```

Expected: exact dict equality PASS for every packed fixture; existing summary tests remain unchanged.

- [ ] **Step 6: Add a deterministic high-trade regression test**

Add one 20,000-bar fixture with alternating/recurring signals that produces thousands of trades and assert complete packed-vs-detailed summary equality. This is a correctness regression test, not a wall-clock assertion.

Run:

```powershell
python -m pytest tests/metrics/test_packed_performance.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```powershell
git add xau_lab/metrics/packed.py tests/metrics/test_packed_performance.py
git commit -m "perf: summarize packed backtests without Trade objects"
```

---

### Task 4: Route Discovery Through Packed Backtest + Packed Metrics

**Files:**
- Modify: `xau_lab/runner/single.py`
- Modify: `tests/runner/test_single.py`
- Test: `tests/runner/test_campaign.py`

**Interfaces:**
- `run_experiment(..., include_trades=False)` uses `run_packed_backtest()` + `summarize_packed_backtest()` and returns `trades=()`.
- `run_experiment(..., include_trades=True)` remains on `run_fast_backtest()` + annotated detailed trades.
- Both modes produce exactly equal `master_result` dictionaries.

- [ ] **Step 1: Strengthen the existing discovery structural RED test**

In `tests/runner/test_single.py`, replace/extend the current annotation-only guard with a detailed-path guard:

```python
def test_discovery_path_never_calls_detailed_trade_materialization(monkeypatch):
    def forbidden_detailed(*args, **kwargs):
        raise AssertionError("discovery path must not materialize Trade objects")

    monkeypatch.setattr(single_module, "run_fast_backtest", forbidden_detailed)

    outcome = run_experiment(_experiment(), _market(), include_trades=False)

    assert outcome.ok
    assert outcome.trades == ()
    assert outcome.master_result is not None
```

Before integration this must FAIL because discovery still calls `run_fast_backtest()`.

- [ ] **Step 2: Run RED verification**

```powershell
python -m pytest tests/runner/test_single.py::test_discovery_path_never_calls_detailed_trade_materialization -v
```

Expected: FAIL with the guard assertion.

- [ ] **Step 3: Split detailed and packed branches in `run_experiment()`**

Keep shared signal generation, direction filtering, cost/risk construction, and `ExitSpec` construction before the branch. Then implement:

```python
if include_trades:
    result = run_fast_backtest(...)
    trades = tuple(_annotate_trade(trade, experiment, market_bundle) for trade in result.trades)
    metrics = summarize_trades(trades, starting_equity=risk.account_equity)
    risk_skip_count = result.risk_skip_count
else:
    packed = run_packed_backtest(...)
    trades = ()
    metrics = summarize_packed_backtest(
        packed,
        market_bundle.bars,
        market_bundle.symbol,
        cost,
        risk,
        market_bundle.broker_day_id,
        market_bundle.broker_month_id,
        market_bundle.broker_year_id,
    )
    risk_skip_count = packed.risk_skip_count
```

Build `master` exactly as before, substituting `risk_skip_count` for `result.risk_skip_count`.

- [ ] **Step 4: Verify complete discovery-vs-full master parity**

Retain the existing test:

```python
assert discovery.master_result == full.master_result
```

Add a second experiment fixture with a time exit and mixed broker dates so calendar metrics are exercised.

Run:

```powershell
python -m pytest tests/runner/test_single.py -v
```

Expected: all PASS, including session annotations on the detailed path.

- [ ] **Step 5: Verify campaign behavior remains deterministic**

```powershell
python -m pytest tests/runner/test_campaign.py tests/runner/test_manifest.py tests/runner/test_smoke_provenance.py tests/runner/test_smoke_selection.py -v
```

Expected: PASS. No manifest/provenance behavior changes are permitted.

- [ ] **Step 6: Commit Task 4**

```powershell
git add xau_lab/runner/single.py tests/runner/test_single.py
git commit -m "perf: use packed path for discovery experiments"
```

---

### Task 5: Full Automated Regression Gate

**Files:**
- No new production files unless a failing test identifies a specific defect.

**Interfaces:**
- All existing APIs and all new packed APIs must be GREEN together before canonical verification.

- [ ] **Step 1: Run focused packed/backtest/runner tests**

```powershell
python -m pytest `
  tests/backtest/test_packed_backtest.py `
  tests/backtest/test_fast_parity.py `
  tests/backtest/test_reference_engine.py `
  tests/metrics/test_packed_performance.py `
  tests/metrics/test_performance.py `
  tests/runner/test_single.py `
  tests/runner/test_campaign.py `
  -v
```

Expected: all PASS.

- [ ] **Step 2: Run the complete suite**

```powershell
python -m pytest
```

Expected: zero failures and zero errors.

- [ ] **Step 3: If anything fails, stop and use systematic debugging**

Do not stack fixes. Reproduce the single failure, identify whether it is packed economics, calendar grouping, aggregation order, or runner routing, then add the smallest failing regression before changing production code.

- [ ] **Step 4: Commit only regression fixes, if any**

Use a focused message naming the root cause. If the suite is already GREEN, make no empty commit.

---

### Task 6: Add a Reproducible Frozen Canonical Packed-vs-Detailed Parity Harness

**Files:**
- Create: `scripts/verify_packed_parity.py`
- Test: `tests/runner/test_single.py`

**Interfaces:**
- CLI consumes canonical catalog/features plus either `--representatives` or `--count 100`.
- For every selected experiment it runs `run_experiment(..., include_trades=False)` and `run_experiment(..., include_trades=True)` on one loaded `MarketBundle` and requires exact `master_result` equality.
- It exits nonzero on the first mismatch and prints field-level differences.

- [ ] **Step 1: Implement the parity script**

The script must:

1. load the market once with `load_market_bundle`,
2. load the deterministic catalog with `_read_catalog`,
3. use the frozen representative IDs when `--representatives` is selected,
4. use `select_stratified_smoke_ids(catalog, 100)` when `--count 100` is selected,
5. run packed and detailed modes for each experiment,
6. compare `master_result` dictionaries by exact Python equality,
7. on mismatch print `experiment_id`, field, packed value, detailed value, then exit 1,
8. print `PARITY_PASS=<count>` on success.

Frozen representative IDs:

```python
REPRESENTATIVE_IDS = (
    "EXP1E2D424171EC",
    "EXP37889369D2F1",
    "EXPB62F70031929",
    "EXPB8942B9FF195",
    "EXPB8F12709D5FE",
    "EXPA6DA73D5B2BA",
    "EXPB1A10E7A4435",
    "EXP2053D4626356",
)
```

- [ ] **Step 2: Verify the eight canonical representatives**

Run:

```powershell
python -m scripts.verify_packed_parity `
  --catalog results/EXPERIMENT_CATALOG.csv `
  --features data/features/XAUUSD_M1_FEATURES.parquet `
  --representatives
```

Expected:

```text
PARITY_PASS=8
PARITY_MISMATCHES=0
```

- [ ] **Step 3: Verify all frozen smoke experiments**

Run:

```powershell
python -m scripts.verify_packed_parity `
  --catalog results/EXPERIMENT_CATALOG.csv `
  --features data/features/XAUUSD_M1_FEATURES.parquet `
  --count 100
```

Expected:

```text
PARITY_PASS=100
PARITY_MISMATCHES=0
```

This is the canonical semantic gate. Any mismatch blocks performance testing.

- [ ] **Step 4: Commit the verification harness**

```powershell
git add scripts/verify_packed_parity.py
git commit -m "test: add canonical packed discovery parity harness"
```

---

### Task 7: Measure Stage-2 Performance and Re-run the Authoritative 100-Smoke Gate

**Files:**
- No production changes unless profiling identifies a new root cause and a separately reviewed change is approved.

**Interfaces:**
- Uses canonical hashes, seed `9215000`, exact frozen 100 selection, and exactly 2 workers.
- Produces fresh `THROUGHPUT_BENCHMARK.json` bound to the implementation commit.

- [ ] **Step 1: Record implementation commit and verify canonical hashes**

Run:

```powershell
$implSha = (git rev-parse HEAD).Trim()
Write-Host "IMPLEMENTATION_SHA=$implSha"
(Get-FileHash .\data\features\XAUUSD_M1_FEATURES.parquet -Algorithm SHA256).Hash
(Get-FileHash .\results\EXPERIMENT_CATALOG.csv -Algorithm SHA256).Hash
```

Expected hashes:

```text
5D3680D043E1F8D4C759124AD177BC73E47C055101DD3BE7558564187B0C10EF
AB2AD9935FB4E9A229CE1D84E2E9B80C4D670A8CEB7E577BF3CAE0B70F0706A9
```

- [ ] **Step 2: Run the complete suite one final time**

```powershell
python -m pytest
```

Expected: zero failures/errors.

- [ ] **Step 3: Run a fresh isolated authoritative smoke**

Use a result namespace derived from the actual implementation SHA:

```powershell
$shortSha = (git rev-parse --short HEAD).Trim()
$smokeRoot = "stage2_smoke_$shortSha"
if (Test-Path $smokeRoot) { throw "$smokeRoot already exists; use a fresh implementation commit or remove only after preserving evidence" }

python -m scripts.run_smoke `
  --catalog results\EXPERIMENT_CATALOG.csv `
  --features data\features\XAUUSD_M1_FEATURES.parquet `
  --result-root "$smokeRoot\results" `
  --count 100 `
  --workers 2 `
  --seed 9215000
```

Required correctness output:

```text
Smoke target complete: 100 selected experiments; executed this invocation: 100
```

`ERRORS.csv` must be absent or contain zero rows.

- [ ] **Step 4: Read and enforce the benchmark gate**

```powershell
Get-Content "$smokeRoot\results\THROUGHPUT_BENCHMARK.json"
Get-Content "$smokeRoot\results\RUN_MANIFEST.json"
```

Required conditions:

```text
completed_experiments == 100
smoke_target_experiments == 100
preexisting_selected_experiments == 0
projected_50000_seconds <= 86400
projected_50000_hours <= 24
requires_profiling_before_full_run == false
workers == 2
```

The manifest must contain the canonical feature/catalog hashes and the current implementation commit.

- [ ] **Step 5: If the gate passes, stop before launching 50k and preserve evidence**

Do **not** start the full campaign in this task. Post the exact smoke timing, projection, hashes, commit, 100/100 completion count, and zero-error result to the implementation PR. The user separately decides whether to launch the 50,000 campaign.

- [ ] **Step 6: If the gate fails, stop and re-profile**

Do not use `--allow-slow`, do not increase workers as a substitute, and do not broaden the packed change. Re-run the same stage decomposition used before Stage 2 and identify the new dominant stage before designing any further optimization.

---

## Completion Checklist

Stage 2 is complete only if all are true:

- [ ] `run_fast_backtest()` detailed compatibility tests still pass.
- [ ] `include_trades=True` detailed trade annotations are unchanged.
- [ ] Discovery does not call detailed trade materialization.
- [ ] Packed metric dict equals `summarize_trades()` exactly across deterministic, adversarial, and high-trade fixtures.
- [ ] Calendar group IDs are broker-local, aligned, compact, and computed once per worker market.
- [ ] Complete pytest suite passes.
- [ ] Frozen eight canonical packed-vs-detailed master results pass exactly.
- [ ] Frozen 100 canonical packed-vs-detailed master results pass exactly.
- [ ] Fresh 100-smoke completes 100 unique experiments with zero errors and correct provenance.
- [ ] Fresh two-worker projection is `<= 24` hours.
- [ ] Full 50,000 campaign has not been launched automatically.
