# H16 MQL5 calendar consensus-source assessment

## Decision

Use the MetaTrader 5 / MQL5 Economic Calendar as a **candidate consensus source only** for H16 development data.

Do not replace H16 official BLS/BEA actuals with MQL5 actual values. The existing frozen H16 source policy remains unchanged.

## Why this source is useful

The MQL5 calendar API exposes, per release:

- stable value ID and event ID,
- event timestamp and reporting period,
- actual value,
- previous value,
- revised previous value,
- forecast value,
- event metadata including event code, name, unit, importance and source URL.

The calendar is queried from a connected terminal with `CalendarValueHistory()`. Calendar values are not natively available inside Strategy Tester, so research data must be exported first and then frozen for replay/analysis.

Primary references:

- MQL5 CodeBase 52977 — Economic Calendar CSV:
  https://www.mql5.com/en/code/52977
- MQL5 CodeBase 53393 — Economic Calendar Monitor and Cache for Backtesting on History:
  https://www.mql5.com/en/code/53393
- MQL5 CodeBase 76951 — CalendarExport:
  https://www.mql5.com/en/code/76951
- MQL5 calendar API:
  https://www.mql5.com/en/docs/calendar/calendarvaluehistory
- MQL5 calendar structures:
  https://www.mql5.com/en/docs/constants/structures/mqlcalendar

## Frozen H16 extraction scope

Export **development data only**:

- country: `US`
- currency: `USD`
- from: `2023-01-01`
- to exclusive: `2025-09-01`

Do not export or inspect H16 return relationships in the reserved OOS period beginning 2025-09-01.

Target event families and exact metric forms:

| H16 family | MQL5 event |
| --- | --- |
| CPI | Consumer Price Index (CPI) m/m |
| CORE_CPI | Core Consumer Price Index (CPI) m/m |
| NFP | Nonfarm Payrolls |
| UNEMPLOYMENT | Unemployment Rate |
| PCE | Personal Consumption Expenditures (PCE) Price Index m/m |
| CORE_PCE | Core Personal Consumption Expenditures (PCE) Price Index m/m |

Exclude YoY, q/q, NSA and similarly named variants.

## Required raw fields

The export used for H16 must preserve at least:

- MQL5 value ID
- MQL5 event ID
- event code
- event name
- release timestamp in broker/server time
- historical server UTC offset, or enough information to reconstruct it
- reporting period
- actual
- forecast
- previous
- revised previous
- unit
- multiplier
- importance
- source URL

The H16 merge should consume only the **forecast** as `consensus`, plus timestamp/identity metadata. Official actual values remain sourced from BLS/BEA release archives.

## Time handling

MQL5 calendar functions use trade-server time, not UTC. Historical broker daylight-saving changes can shift old events by one hour if a current offset is naively applied.

For that reason, use the timezone-correction path demonstrated by CodeBase 52977 / 53393. Preserve the historical server offset and convert each accepted release to canonical UTC before joining with H16 official actuals.

No XAUUSD-return join should occur until timestamp conversion passes structural checks.

## Forecast acceptance checks

MQL5 `forecast_value` is treated as a candidate pre-release consensus field, not automatically trusted.

Before canonical H16 generation:

1. Require a non-missing forecast for the chosen release.
2. Confirm event metric/unit exactly matches the frozen H16 metric.
3. Reject duplicate release/value IDs.
4. Reject rows on or after the OOS boundary unless explicitly unlocked for the preregistered final evaluation.
5. Compare a sample of development-period MQL5 forecasts against independently archived pre-release consensus reports where feasible.
6. Flag any release whose forecast appears inconsistent with the release-period market consensus instead of silently correcting it.
7. Keep the original exported MQL5 row in the raw-data audit trail.

## Next concrete step

Run CodeBase 52977 (`Economic Calendar CSV`) on a connected MT5 terminal, preferably from an XAUUSD H1 chart so its historical server/DST correction can operate on the same symbol family used by this research.

Export only the H16 development window and USD/US calendar records. Then inspect the produced CSV header and a small sample of the six target event families before writing the repository importer.

At this stage, do **not** inspect H16 XAUUSD returns and do not touch the reserved OOS period.
