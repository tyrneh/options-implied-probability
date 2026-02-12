---
title: Troubleshooting and FAQ
parent: User Guide
nav_order: 11
---

# Troubleshooting and FAQ

## "Input DataFrame must contain an 'expiry' column"

Cause: your chain column names do not match OIPD expectations.

Fix:

- Provide `column_mapping` in `fit(...)`, or
- Rename your DataFrame columns before fitting.

## "VolCurve.fit requires a single expiry"

Cause: multi-expiry chain passed to `VolCurve`.

Fix: filter to one expiry, or use `VolSurface`.

## "VolSurface.fit requires at least two unique expiries"

Cause: single-expiry chain passed to `VolSurface`.

Fix: append at least one additional expiry.

## "Expiry must be strictly after valuation_date"

Cause: expiry is same day or in the past.

Fix: update chain expiry or valuation date.

## Vendor fetch errors (network)

Typical issue in restricted environments:

- DNS/network failures during `sources.list_expiry_dates` or `sources.fetch_chain`.

Fix:

1. Validate connectivity to vendor endpoints.
2. Retry with `cache_enabled=False` for debugging.
3. Use local CSV/DataFrame ingestion until connectivity is restored.

## Numerical sanity checks before interpretation

- Compare `prob_below(spot)` against your intuition.
- Check moment scale (`mean`, `variance`) for outliers.
- Inspect volatility diagnostics (`iv_results`, `diagnostics`) if probabilities look unstable.

## Validated Commands (Offline)

- [x] Error scenarios referenced here were reproduced in interface tests and smoke scripts.
