---
title: Reference: MarketInputs and Sources
parent: User Guide
nav_order: 10
---

# Reference: `MarketInputs`, `VendorSnapshot`, and `sources`

This page covers the non-estimator interfaces used to feed data into OIPD.

## `MarketInputs`

`MarketInputs` is the explicit input contract for fitting.

```python
MarketInputs(
    risk_free_rate: float,
    valuation_date: date,
    risk_free_rate_mode: Literal["annualized", "continuous"] = "annualized",
    underlying_price: float | None = None,
    dividend_yield: float | None = None,
    dividend_schedule: pd.DataFrame | None = None,
)
```

Key points:

- `underlying_price` must be set before fitting.
- `valuation_date` is normalized from `str`/`datetime` if needed.
- Dividends can be specified as yield or schedule.

## `VendorSnapshot`

Returned by `sources.fetch_chain(...)`:

- `asof`
- `vendor`
- `underlying_price`
- optional dividends metadata
- convenience property: `snapshot.date`

## `sources` module

Implemented functions:

- `sources.from_csv(path, column_mapping=None)`
- `sources.from_dataframe(df, column_mapping=None)`
- `sources.list_expiry_dates(ticker, vendor="yfinance", **vendor_kwargs)`
- `sources.fetch_chain(ticker, expiries=None, horizon=None, vendor="yfinance", ...)`

## `resolve_market(inputs)`

`resolve_market` produces a `ResolvedMarket` with provenance metadata used by pricing and probability pipelines.

## Validated Commands (Offline)

- [x] `MarketInputs(...)` construction executed in `.venv`.
- [x] `resolve_market(...)` path exercised indirectly by `VolCurve.fit(...)` and `VolSurface.fit(...)` snippets.
