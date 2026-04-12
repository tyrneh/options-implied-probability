---
title: Examples
nav_order: 4
---

# Examples

The notebooks below mirror the workflows in the User Guide.

- [`examples/quickstart_yfinance.ipynb`](https://github.com/Open-Lemma/options-implied-probability/blob/main/examples/quickstart_yfinance.ipynb): live-data probability workflow (vendor fetch).
- [`examples/quickstart_VolCurve.ipynb`](https://github.com/Open-Lemma/options-implied-probability/blob/main/examples/quickstart_VolCurve.ipynb): single-expiry volatility workflow.
- [`examples/quickstart_VolSurface.ipynb`](https://github.com/Open-Lemma/options-implied-probability/blob/main/examples/quickstart_VolSurface.ipynb): multi-expiry surface workflow.

`MarketInputs.valuation_date` accepts both dates and datetimes. For intraday
precision when using vendor data, prefer `snapshot.asof`.

Object-model contract used throughout the examples:

- `valuation_date` remains the public field name and stores the canonical
  normalized timestamp.
- `valuation_calendar_date` is the explicit date-only convenience view.
- `valuation_timestamp` remains as a temporary compatibility alias to the same
  canonical timestamp value.

Across the examples, the canonical maturity field is `expiry`. Exact
`time_to_expiry_years` is used internally for pricing and probability
calculations. Reporting in day units uses explicit `time_to_expiry_days`,
while `calendar_days_to_expiry` is the explicit integer calendar bucket. If
you pass intraday timestamps, plots will surface that time-of-day where it
matters.

Developer note:

- `oipd.core.maturity` is the canonical home of maturity logic.

Migration note:

- old `days_to_expiry` inputs should move to `time_to_expiry_years` for
  pricing or `time_to_expiry_days` for reporting
- old `days_to_expiry` metadata reads should move to
  `calendar_days_to_expiry` if integer calendar-bucket semantics were intended

If you provide a Black-Scholes `dividend_schedule`, its `ex_date` column can
also be date-only or full timestamp values. Same-day timestamp ordering matters
for pricing. Timezone display redesign remains out of scope for this cycle.
