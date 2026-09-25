# H16 official actuals collector

This stage collects only **as-released official actual values** for the H16 development period.

Sources:

- BLS archived Consumer Price Index releases
- BLS archived Employment Situation releases
- BEA archived Personal Income and Outlays releases

Frozen metrics:

- CPI: seasonally adjusted month-over-month percent change
- Core CPI: seasonally adjusted month-over-month percent change
- NFP: monthly total nonfarm payroll change, in thousands of jobs
- Unemployment: U-3 unemployment rate, percent
- PCE: month-over-month PCE price-index change
- Core PCE: month-over-month PCE price-index change excluding food and energy

The collector writes:

`data/macro/h16_official_actuals.csv`

It intentionally leaves `consensus` and `previous` blank. H16 must not substitute a current revised time series for the number that market participants saw at the release timestamp.

Run:

```bash
python scripts/collect_h16_official_actuals.py
```

Then inspect only structural/data-quality properties: row counts, missing fields, duplicate IDs, timestamps, source URLs, and parser failures.

Do **not** join XAUUSD returns or inspect H16 OOS performance at this stage.

The frozen OOS starts at `2025-09-01T00:00:00Z`. The official collector derives its end boundary from the H16 frozen config and does not fetch an H16 development event on or after that boundary.

## Consensus dependency

The official agencies publish outcomes, not our required market-consensus snapshot. Consensus must therefore be ingested separately from a source that records the estimate **before the release time** and uses the exact same metric/unit.

Never:

- use `previous` as a replacement for consensus,
- mix YoY consensus with MoM actuals,
- use a forecast captured after the official release,
- overwrite an as-released actual with a later revision.

Once consensus is merged, pass the resulting raw CSV through:

```bash
python scripts/validate_h16_macro_events.py --input <merged.csv>
```

Only that validated canonical file can feed H16 strategy research.
