# Stage 2 Packed Discovery Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Python `Trade` object materialization in normal discovery runs with an exact packed-results path so the canonical 50,000-experiment campaign can project to 24 hours or less without changing any stored experiment result.

**Architecture:** Preserve `run_fast_backtest()` and the detailed `Trade` API as the semantic oracle. Add a packed backtest result that exposes the existing Numba kernel arrays, summarize those arrays with exact sequential compiled arithmetic, and route `run_experiment(..., include_trades=False)` through that path. Keep `include_trades=True` on the existing detailed path. Precompute compact broker-calendar group IDs once per loaded worker market so packed metrics never perform per-trade timezone/date parsing.

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
- Create `xau_lab/metrics/packed.py`: exact scalar trade economics plus exact compiled packed summary metrics.
- Modify `xau_lab/runner/single.py`: add compact broker calendar group IDs to `MarketBundle`; route discovery to packed backtest + packed summary.
- Modify `xau_lab/runner/campaign.py`: derive broker day/month/year group IDs once when loading a worker market, using canonical broker-local calendar columns.
- Add `tests/backtest/test_packed_backtest.py`: packed result shape, packed trade-economics parity, and detailed-path compatibility.
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

- [ ] **Step 1: Write the RED packed API test**

Create `tests/backtest/test_packed_backtest.py`:

```python
import numpy as np

from xau_lab.backtest.fast import run_packed_backtest
from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.packed import PackedBacktestResult

SPEC = SymbolSpec(point=0.01, digits=2, contract_size=100.0, volume_min=0.01, volume_step=0.01)
COST = CostModel()
RISK = RiskModel()


def _market(n: int = 256):
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
    return bars, signals


def test_packed_backtest_exposes_kernel_rows_without_trade_objects():
    bars, signals = _market()
    packed = run_packed_backtest(
        bars, signals, SPEC, COST, RISK, ExitSpec(stop_atr=1.0, target_r=1.0)
    )
    assert isinstance(packed, PackedBacktestResult)
    assert 0 <= packed.trade_count <= np.count_nonzero(signals)
    assert packed.risk_skip_count >= 0
    for values in (
        packed.direction,
        packed.signal_index,
        packed.entry_index,
        packed.exit_index,
        packed.raw_entry,
        packed.entry_price,
        packed.initial_stop,
        packed.target,
        packed.lot,
        packed.planned_risk,
        packed.raw_exit,
        packed.reason,
    ):
        assert values.ndim == 1
        assert len(values) >= packed.trade_count
```

- [ ] **Step 2: Run RED verification**

```powershell
python -m pytest tests/backtest/test_packed_backtest.py -v
```

Expected: import/collection failure because the packed API does not exist.

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
            self.direction, self.signal_index, self.entry_index, self.exit_index,
            self.raw_entry, self.entry_price, self.initial_stop, self.target,
            self.lot, self.planned_risk, self.raw_exit, self.reason,
        )
        if any(np.asarray(values).ndim != 1 for values in arrays):
            raise ValueError("packed trade arrays must be one-dimensional")
        if any(len(values) < count for values in arrays):
            raise ValueError("packed trade array shorter than trade_count")
```

Do not trim or copy arrays in the dataclass.

- [ ] **Step 4: Add `run_packed_backtest()` without changing `_kernel`**

In `xau_lab/backtest/fast.py`, import `PackedBacktestResult` and move the existing validation/sentinel/kernel-call section into exactly this wrapper shape:

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

Do not edit `_kernel` in this task.

- [ ] **Step 5: Refactor detailed materialization into one helper**

Add to `fast.py`:

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

Replace the body of `run_fast_backtest()` after validation with:

```python
packed = run_packed_backtest(bars, signals, symbol, cost, risk, exit_spec)
trades = _materialize_packed_trades(packed, bars, symbol, cost)
return BacktestResult(trades, risk_skip_count=packed.risk_skip_count)
```

- [ ] **Step 6: Run GREEN compatibility tests**

```powershell
python -m pytest tests/backtest/test_packed_backtest.py tests/backtest/test_fast_parity.py tests/backtest/test_reference_engine.py -v
```

Expected: all PASS, including existing fast/reference trade parity and output-capacity tests.

- [ ] **Step 7: Commit Task 1**

```powershell
git add xau_lab/backtest/packed.py xau_lab/backtest/fast.py tests/backtest/test_packed_backtest.py
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
- `MarketBundle` gains read-only `broker_day_id`, `broker_month_id`, `broker_year_id` contiguous `np.int32` arrays aligned 1:1 with bars.
- `load_market_bundle()` computes IDs once from canonical broker-local calendar columns.
- Manual/test `MarketBundle` construction may omit IDs; `__post_init__()` derives them from canonical `YYYY-MM-DD` strings.

- [ ] **Step 1: Write the RED group-ID test**

Add to `tests/runner/test_single.py`:

```python
def test_market_bundle_derives_contiguous_broker_calendar_group_ids():
    base = _market()
    dates = np.array(
        ["2025-12-31", "2025-12-31", "2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"],
        dtype=object,
    )
    bars = MarketBars(
        time_epoch=np.arange(6, dtype=np.int64) * 60 + 1_767_312_000,
        open=np.ones(6), high=np.ones(6) + 1, low=np.ones(6) - 1,
        close=np.ones(6), spread=np.zeros(6, dtype=np.int64), atr=np.ones(6),
    )
    bundle = MarketBundle(bars=bars, symbol=base.symbol, broker_date=dates, features={})
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

- [ ] **Step 3: Implement the exact fallback derivation in `single.py`**

Add:

```python
def _calendar_group_ids_from_dates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = np.asarray(values, dtype=object)
    n = len(dates)
    day = np.empty(n, dtype=np.int32)
    month = np.empty(n, dtype=np.int32)
    year = np.empty(n, dtype=np.int32)
    if n == 0:
        return day, month, year

    day_id = month_id = year_id = 0
    previous = str(dates[0])
    if len(previous) != 10 or previous[4] != "-" or previous[7] != "-":
        raise ValueError("broker_date must use YYYY-MM-DD")
    day[0] = month[0] = year[0] = 0

    for index in range(1, n):
        current = str(dates[index])
        if len(current) != 10 or current[4] != "-" or current[7] != "-":
            raise ValueError("broker_date must use YYYY-MM-DD")
        if current != previous:
            day_id += 1
        if current[:7] != previous[:7]:
            month_id += 1
        if current[:4] != previous[:4]:
            year_id += 1
        day[index] = day_id
        month[index] = month_id
        year[index] = year_id
        previous = current
    return day, month, year
```

Extend `MarketBundle`:

```python
broker_day_id: np.ndarray | None = None
broker_month_id: np.ndarray | None = None
broker_year_id: np.ndarray | None = None
```

Inside `__post_init__()`:

```python
if self.broker_day_id is None or self.broker_month_id is None or self.broker_year_id is None:
    day, month, year = _calendar_group_ids_from_dates(broker_date)
else:
    day = np.ascontiguousarray(self.broker_day_id, dtype=np.int32)
    month = np.ascontiguousarray(self.broker_month_id, dtype=np.int32)
    year = np.ascontiguousarray(self.broker_year_id, dtype=np.int32)
for name, values in (("broker_day_id", day), ("broker_month_id", month), ("broker_year_id", year)):
    if len(values) != len(self.bars):
        raise ValueError(f"{name} length must match market bars")
    values.setflags(write=False)
    object.__setattr__(self, name, values)
```

- [ ] **Step 4: Implement vectorized IDs in `campaign.py`**

Add:

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

After obtaining `broker_date` in `load_market_bundle()`:

```python
if "year" in frame.columns and "month" in frame.columns:
    broker_year = np.asarray(frame["year"].to_numpy(), dtype=np.int32)
    broker_month = np.asarray(frame["month"].to_numpy(), dtype=np.int16)
else:
    broker_year = np.fromiter((int(str(value)[:4]) for value in broker_date), dtype=np.int32, count=len(broker_date))
    broker_month = np.fromiter((int(str(value)[5:7]) for value in broker_date), dtype=np.int16, count=len(broker_date))

broker_day_id = _contiguous_group_ids(broker_date)
broker_month_id = _contiguous_group_ids(broker_year, broker_month)
broker_year_id = _contiguous_group_ids(broker_year)
```

Pass all three IDs to `MarketBundle(...)`.

- [ ] **Step 5: Add loader integrity assertions and run GREEN tests**

In `tests/runner/test_campaign.py`, import `load_market_bundle`, load `_write_market(...)`, then assert:

```python
bundle = load_market_bundle(feature_path)
assert len(bundle.broker_day_id) == len(bundle.bars)
assert np.all(bundle.broker_day_id[1:] >= bundle.broker_day_id[:-1])
assert np.all(bundle.broker_month_id[1:] >= bundle.broker_month_id[:-1])
assert np.all(bundle.broker_year_id[1:] >= bundle.broker_year_id[:-1])
```

Run:

```powershell
python -m pytest tests/runner/test_single.py tests/runner/test_campaign.py -v
```

Expected: PASS, including 1-worker vs 2-worker campaign determinism.

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
- Modify: `tests/backtest/test_packed_backtest.py`
- Test: `tests/metrics/test_performance.py`

**Interfaces:**
- Private compiled `_trade_economics(...)` returns the exact metric-relevant economics for one packed trade.
- Public `summarize_packed_backtest(...) -> dict[str, float | int | None]` returns exactly the same metric dictionary as `summarize_trades()`.
- Discovery summary allocates only one `hold_minutes` array of length `trade_count` plus day/month/year aggregate arrays; it does not allocate Python `Trade` objects or six per-trade economics arrays.

- [ ] **Step 1: Write RED scalar economics parity tests**

In `tests/backtest/test_packed_backtest.py`, import the future `_trade_economics`. For a packed run and its detailed `run_fast_backtest()` oracle, compare each metric-relevant field exactly:

```python
for i, trade in enumerate(detailed.trades):
    net, pnl_r, hold, commission, spread_cost, slippage_cost = _trade_economics(
        i,
        packed.direction,
        packed.entry_index,
        packed.exit_index,
        packed.raw_entry,
        packed.lot,
        packed.planned_risk,
        packed.raw_exit,
        bars.spread,
        bars.time_epoch,
        SPEC.point,
        SPEC.contract_size,
        COST.commission_round_trip_per_lot,
        COST.slippage_points_per_fill,
    )
    assert net == trade.net_pnl
    assert pnl_r == trade.pnl_R
    assert hold == trade.hold_minutes
    assert commission == trade.commission
    assert spread_cost == trade.spread_cost
    assert slippage_cost == trade.slippage_cost
```

Run and confirm import failure before implementation.

- [ ] **Step 2: Implement `_trade_economics` exactly**

Create `xau_lab/metrics/packed.py` with:

```python
from __future__ import annotations

import math
import numpy as np
from numba import njit

from xau_lab.backtest.models import CostModel, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.packed import PackedBacktestResult


@njit(cache=True)
def _trade_economics(
    i,
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
    d = int(direction[i])
    entry = int(entry_index[i])
    exit_ = int(exit_index[i])
    size = float(lot[i])
    gross = d * (float(raw_exit[i]) - float(raw_entry[i])) * contract_size * size
    spread_points = int(spread[entry]) if d == 1 else int(spread[exit_])
    spread_cost = float(spread_points) * point * contract_size * size
    slippage_cost = 2.0 * slippage_points * point * contract_size * size
    commission = commission_per_lot * size
    net = gross - spread_cost - slippage_cost - commission
    pnl_r = net / float(planned_risk[i]) if float(planned_risk[i]) > 0.0 else np.nan
    hold = (int(time_epoch[exit_]) - int(time_epoch[entry])) / 60.0
    return net, pnl_r, hold, commission, spread_cost, slippage_cost
```

Do not use `fastmath`.

- [ ] **Step 3: Add exact median and optional-value helpers**

Add:

```python
@njit(cache=True)
def _median_exact(values, count):
    if count <= 0:
        return np.nan
    ordered = np.sort(values[:count].copy())
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0


@njit(cache=True)
def _pf(gross_profit, gross_loss, count):
    if count == 0 or gross_loss == 0.0:
        return np.nan
    return gross_profit / abs(gross_loss)
```

- [ ] **Step 4: Implement one sequential packed summary kernel**

Implement `_summary_kernel(...)` with these exact rules:

```python
# Allocate only these variable-size helpers:
hold_values = np.empty(trade_count, dtype=np.float64)
daily = np.zeros(day_group_count, dtype=np.float64)
monthly = np.zeros(month_group_count, dtype=np.float64)
yearly = np.zeros(year_group_count, dtype=np.float64)
daily_active = np.zeros(day_group_count, dtype=np.bool_)
monthly_active = np.zeros(month_group_count, dtype=np.bool_)
yearly_active = np.zeros(year_group_count, dtype=np.bool_)
top5 = np.full(5, -np.inf, dtype=np.float64)
```

Then loop `for i in range(trade_count)` in trade order. For every trade:

1. call `_trade_economics(...)`,
2. add positive/negative values to `gross_profit` / `gross_loss`,
3. update wins/losses and max-loss streak,
4. update equity, peak, max drawdown USD and percent,
5. accumulate `sum_net`, `sum_pnl_r`, commission/spread/slippage totals in trade order,
6. accumulate long/short profit/loss and trade counts in trade order,
7. save `hold_values[i]`,
8. get `entry = int(entry_index[i])`, then `day_id = broker_day_id[entry]`, `month_id = broker_month_id[entry]`, `year_id = broker_year_id[entry]`,
9. add net P/L to those group arrays in trade order and mark them active,
10. if `net > 0`, insert into `top5` in descending order by shifting lower entries.

Use this exact top-five insertion:

```python
if net > 0.0:
    for slot in range(5):
        if net > top5[slot]:
            for shift in range(4, slot, -1):
                top5[shift] = top5[shift - 1]
            top5[slot] = net
            break
```

After the trade loop, copy active daily totals into a compact `daily_values` array in ascending group-ID order and compute `_median_exact(daily_values, active_days)`. Compute monthly/yearly positive fractions by scanning active groups. Compute `worst_month` and `best_month` by scanning active monthly groups. Compute median hold with `_median_exact(hold_values, trade_count)`. Sum finite `top5` values in slot order.

Return a fixed tuple in this exact order:

```text
completed_trades, wins, losses, win_rate,
gross_profit, gross_loss, profit_factor, after_cost_profit, expectancy_usd,
max_drawdown_usd, max_drawdown_pct, max_loss_streak, expectancy_R,
profit_per_active_day, median_profit_per_active_day,
positive_year_fraction, positive_month_fraction, active_months, worst_month,
long_PF, short_PF, long_trades, short_trades, trades_per_active_day,
median_hold_minutes, top_5_trade_profit_fraction, best_month_profit_fraction,
net_profit, commission_cost, spread_cost, slippage_cost
```

Use `np.nan` only for values that are `None` in `summarize_trades()`.

- [ ] **Step 5: Implement the Python wrapper and exact dict mapping**

Add:

```python
def _optional(value: float):
    return None if math.isnan(float(value)) else float(value)


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
    day = np.asarray(broker_day_id, dtype=np.int32)
    month = np.asarray(broker_month_id, dtype=np.int32)
    year = np.asarray(broker_year_id, dtype=np.int32)
    if any(values.ndim != 1 or len(values) != len(bars) for values in (day, month, year)):
        raise ValueError("broker calendar group arrays must be one-dimensional and match market bars")
    count = packed.trade_count
    if count:
        entries = packed.entry_index[:count]
        exits = packed.exit_index[:count]
        if np.any(entries < 0) or np.any(entries >= len(bars)) or np.any(exits < 0) or np.any(exits >= len(bars)):
            raise ValueError("packed trade index exceeds market bars")
    values = _summary_kernel(
        count, packed.direction, packed.entry_index, packed.exit_index,
        packed.raw_entry, packed.lot, packed.planned_risk, packed.raw_exit,
        bars.spread, bars.time_epoch, float(symbol.point), float(symbol.contract_size),
        float(cost.commission_round_trip_per_lot), float(cost.slippage_points_per_fill),
        float(risk.account_equity), day, month, year,
    )
    (
        completed, wins, losses, win_rate,
        gross_profit, gross_loss, profit_factor, after_cost_profit, expectancy_usd,
        max_dd_usd, max_dd_pct, max_loss_streak, expectancy_r,
        profit_per_day, median_day, positive_year, positive_month, active_months, worst_month,
        long_pf, short_pf, long_trades, short_trades, trades_per_day,
        median_hold, top5_fraction, best_month_fraction,
        net_profit, commission_cost, spread_cost, slippage_cost,
    ) = values
    return {
        "completed_trades": int(completed),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": _optional(win_rate),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": _optional(profit_factor),
        "after_cost_profit": float(after_cost_profit),
        "expectancy_usd": _optional(expectancy_usd),
        "max_drawdown_usd": float(max_dd_usd),
        "max_drawdown_pct": float(max_dd_pct),
        "max_loss_streak": int(max_loss_streak),
        "expectancy_R": _optional(expectancy_r),
        "profit_per_active_day": _optional(profit_per_day),
        "median_profit_per_active_day": _optional(median_day),
        "positive_year_fraction": _optional(positive_year),
        "positive_month_fraction": _optional(positive_month),
        "active_months": int(active_months),
        "worst_month": _optional(worst_month),
        "long_PF": _optional(long_pf),
        "short_PF": _optional(short_pf),
        "long_trades": int(long_trades),
        "short_trades": int(short_trades),
        "trades_per_active_day": _optional(trades_per_day),
        "median_hold_minutes": _optional(median_hold),
        "top_5_trade_profit_fraction": _optional(top5_fraction),
        "best_month_profit_fraction": _optional(best_month_fraction),
        "net_profit": float(net_profit),
        "commission_cost": float(commission_cost),
        "spread_cost": float(spread_cost),
        "slippage_cost": float(slippage_cost),
    }
```

- [ ] **Step 6: Write complete packed summary equivalence tests**

Create `tests/metrics/test_packed_performance.py` with a helper that receives `broker_dates`, derives group IDs through `MarketBundle`, and uses the same canonical dates for the detailed oracle:

```python
def assert_exact_summary(bars, signals, symbol, cost, risk, exit_spec, broker_dates):
    bundle = MarketBundle(bars=bars, symbol=symbol, broker_date=np.asarray(broker_dates, dtype=object), features={})
    detailed = run_fast_backtest(bars, signals, symbol, cost, risk, exit_spec)
    packed = run_packed_backtest(bars, signals, symbol, cost, risk, exit_spec)
    expected = summarize_trades(
        detailed.trades,
        starting_equity=risk.account_equity,
        broker_dates=bundle.broker_date,
    )
    actual = summarize_packed_backtest(
        packed, bars, symbol, cost, risk,
        bundle.broker_day_id, bundle.broker_month_id, bundle.broker_year_id,
    )
    assert actual == expected
```

Add concrete fixtures for: no trades; only wins; only losses; mixed long/short; spread/slippage/commission; same-bar ambiguity; target/time/trail/end-of-data; multiple trades per day; day/month/year boundaries; odd/even median counts; equal P/L; risk skips; and a deterministic 20,000-bar high-trade fixture.

- [ ] **Step 7: Run GREEN metric tests**

```powershell
python -m pytest tests/backtest/test_packed_backtest.py tests/metrics/test_packed_performance.py tests/metrics/test_performance.py tests/backtest/test_fast_parity.py -v
```

Expected: exact equality PASS everywhere.

- [ ] **Step 8: Commit Task 3**

```powershell
git add xau_lab/metrics/packed.py tests/metrics/test_packed_performance.py tests/backtest/test_packed_backtest.py
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

- [ ] **Step 1: Write the structural RED test**

Add to `tests/runner/test_single.py`:

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

Run it before implementation and require the guard failure.

- [ ] **Step 2: Implement the packed/detailed branch in `run_experiment()`**

Keep signal generation, direction filtering, `CostModel`, `RiskModel`, and `ExitSpec` construction shared. Replace the unconditional detailed backtest with:

```python
exit_spec = _exit_spec(experiment)
if include_trades:
    result = run_fast_backtest(
        market_bundle.bars, signals, market_bundle.symbol, cost, risk, exit_spec
    )
    trades = tuple(_annotate_trade(trade, experiment, market_bundle) for trade in result.trades)
    metrics = summarize_trades(trades, starting_equity=risk.account_equity)
    risk_skip_count = result.risk_skip_count
else:
    packed = run_packed_backtest(
        market_bundle.bars, signals, market_bundle.symbol, cost, risk, exit_spec
    )
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

Build `master` exactly as today except use `int(risk_skip_count)`.

- [ ] **Step 3: Add a calendar-sensitive complete master parity test**

Retain the existing simple equality test and add a second market whose `broker_date` spans two months. Run both modes and require:

```python
assert discovery.master_result == full.master_result
assert discovery.trades == ()
assert len(full.trades) > 0
```

- [ ] **Step 4: Run GREEN runner and campaign tests**

```powershell
python -m pytest tests/runner/test_single.py tests/runner/test_campaign.py tests/runner/test_manifest.py tests/runner/test_smoke_provenance.py tests/runner/test_smoke_selection.py -v
```

Expected: all PASS; no provenance behavior changes.

- [ ] **Step 5: Commit Task 4**

```powershell
git add xau_lab/runner/single.py tests/runner/test_single.py
git commit -m "perf: use packed path for discovery experiments"
```

---

### Task 5: Run the Complete Automated Regression Gate

**Files:**
- No production changes unless a specific failing regression is diagnosed first.

- [ ] **Step 1: Run focused tests**

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

- [ ] **Step 3: Handle failures through root-cause-first TDD only**

For any failure: reproduce just that test; determine whether the defect is kernel packaging, trade economics, calendar grouping, aggregate order, or runner routing; add/retain the smallest regression; make one production fix; rerun focused + full suite. Do not stack speculative fixes.

---

### Task 6: Add a Reproducible Frozen Canonical Packed-vs-Detailed Parity Harness

**Files:**
- Create: `scripts/verify_packed_parity.py`

**Interfaces:**
- CLI accepts `--catalog`, `--features`, and exactly one of `--representatives` or `--count`.
- It loads one market bundle, selects deterministic experiments, runs packed and detailed modes, compares complete master dictionaries exactly, prints field-level differences, and exits nonzero on mismatch.

- [ ] **Step 1: Implement the parity CLI**

Create `scripts/verify_packed_parity.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_smoke import select_stratified_smoke_ids
from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.runner.single import run_experiment

REPRESENTATIVE_IDS = (
    "EXP1E2D424171EC", "EXP37889369D2F1", "EXPB62F70031929", "EXPB8942B9FF195",
    "EXPB8F12709D5FE", "EXPA6DA73D5B2BA", "EXPB1A10E7A4435", "EXP2053D4626356",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact packed-vs-detailed canonical result parity")
    parser.add_argument("--catalog", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/XAUUSD_M1_FEATURES.parquet"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--representatives", action="store_true")
    group.add_argument("--count", type=int)
    args = parser.parse_args()

    catalog = _read_catalog(args.catalog)
    by_id = {experiment.experiment_id: experiment for experiment in catalog}
    if args.representatives:
        selected_ids = list(REPRESENTATIVE_IDS)
    else:
        if args.count is None or args.count <= 0:
            parser.error("--count must be positive")
        selected_ids = select_stratified_smoke_ids(args.catalog, args.count)

    market = load_market_bundle(args.features)
    mismatches = 0
    for index, experiment_id in enumerate(selected_ids, 1):
        experiment = by_id[experiment_id]
        packed = run_experiment(experiment, market, include_trades=False)
        detailed = run_experiment(experiment, market, include_trades=True)
        if packed.master_result != detailed.master_result:
            mismatches += 1
            left = packed.master_result or {}
            right = detailed.master_result or {}
            for field in sorted(set(left) | set(right)):
                if left.get(field) != right.get(field):
                    print("MISMATCH", experiment_id, field, repr(left.get(field)), repr(right.get(field)))
            break
        print(f"PARITY_PROGRESS={index}/{len(selected_ids)} {experiment_id}")

    print(f"PARITY_PASS={len(selected_ids) - mismatches}")
    print(f"PARITY_MISMATCHES={mismatches}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run eight canonical representatives**

```powershell
python -m scripts.verify_packed_parity `
  --catalog results/EXPERIMENT_CATALOG.csv `
  --features data/features/XAUUSD_M1_FEATURES.parquet `
  --representatives
```

Expected final output:

```text
PARITY_PASS=8
PARITY_MISMATCHES=0
```

- [ ] **Step 3: Run frozen canonical 100**

```powershell
python -m scripts.verify_packed_parity `
  --catalog results/EXPERIMENT_CATALOG.csv `
  --features data/features/XAUUSD_M1_FEATURES.parquet `
  --count 100
```

Expected final output:

```text
PARITY_PASS=100
PARITY_MISMATCHES=0
```

Any mismatch blocks performance testing.

- [ ] **Step 4: Commit Task 6**

```powershell
git add scripts/verify_packed_parity.py
git commit -m "test: add canonical packed discovery parity harness"
```

---

### Task 7: Re-run the Authoritative Stage-2 Runtime Gate

**Files:**
- No production changes unless a new measured bottleneck is separately diagnosed and approved.

**Interfaces:**
- Canonical feature/catalog hashes, seed `9215000`, frozen 100 selection, exactly 2 workers.
- Produces a fresh `THROUGHPUT_BENCHMARK.json` bound to the implementation commit.

- [ ] **Step 1: Record implementation commit and hashes**

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

- [ ] **Step 2: Run full suite one final time**

```powershell
python -m pytest
```

Expected: zero failures/errors.

- [ ] **Step 3: Run a fresh isolated 100-smoke**

```powershell
$shortSha = (git rev-parse --short HEAD).Trim()
$smokeRoot = "stage2_smoke_$shortSha"
if (Test-Path $smokeRoot) { throw "$smokeRoot already exists; preserve old evidence and use a fresh commit/namespace" }

python -m scripts.run_smoke `
  --catalog results\EXPERIMENT_CATALOG.csv `
  --features data\features\XAUUSD_M1_FEATURES.parquet `
  --result-root "$smokeRoot\results" `
  --count 100 `
  --workers 2 `
  --seed 9215000
```

Required: `100` selected completed, `100` unique IDs, zero errors.

- [ ] **Step 4: Enforce the benchmark and provenance gate**

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

Manifest feature/catalog hashes must be canonical and `software_git_commit` must equal `$implSha`.

- [ ] **Step 5: Stop at the gate result**

If PASS: post exact timing, projection, hashes, commit, 100/100 completion, and zero errors to the implementation PR. Do not automatically launch 50,000 experiments.

If FAIL: do not use `--allow-slow`, do not substitute a worker-count increase, and do not add more optimizations. Re-profile the same frozen 100 and design the next change from the newly measured dominant stage.

---

## Completion Checklist

- [ ] Detailed `run_fast_backtest()` compatibility remains intact.
- [ ] `include_trades=True` detailed trade annotations remain intact.
- [ ] Discovery never materializes Python `Trade` objects.
- [ ] Scalar packed trade economics equal detailed `Trade` economics exactly.
- [ ] Packed metric dict equals `summarize_trades()` exactly across deterministic, calendar-boundary, exit-mode, and high-trade fixtures.
- [ ] Broker calendar group IDs are aligned, compact, read-only, and computed once per loaded worker market in production.
- [ ] Full pytest suite passes.
- [ ] Frozen eight canonical complete master-result parity passes exactly.
- [ ] Frozen 100 canonical complete master-result parity passes exactly.
- [ ] Fresh authoritative 100-smoke completes 100 unique experiments with zero errors and correct provenance.
- [ ] Fresh two-worker projection is `<= 24` hours.
- [ ] Full 50,000 campaign has not been launched automatically.
