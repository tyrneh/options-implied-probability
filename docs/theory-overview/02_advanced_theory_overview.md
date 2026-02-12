---
title: Advanced Theory Overview
parent: Theory Overview
nav_order: 2
---

# Advanced Theory Overview

This page explains how OIPD translates financial theory into a robust software architecture.

## 1. Conceptual decomposition

OIPD separates the workflow into two layers:

1. **Volatility fitting** (`VolCurve`, `VolSurface`)
2. **Probability derivation** (`ProbCurve`, `ProbSurface`)

This separation is deliberate. Many users need only the volatility layer for pricing and risk, while others continue to probability analysis.


## 3. Single-expiry theory in implementation (`VolCurve`)

At fit time, `VolCurve`:

1. Validates a single-expiry chain.
2. Resolves market inputs (`MarketInputs` -> `ResolvedMarket`).
3. Calibrates the volatility model.
4. Stores diagnostics and metadata (forward, ATM vol, fit details).

Probability calls (`implied_distribution()`) are then derived from the fitted curve, not from raw quotes.

## 4. Multi-expiry theory in implementation (`VolSurface`)

`VolSurface` fits independent expiry slices and builds an interpolator in total-variance space.

### Why total variance interpolation?

Interpolating total variance, instead of raw volatility, is a common practical approach for improved stability across maturities.

### Surface slicing behavior

- Exact fitted expiry -> returns original fitted `VolCurve` slice.
- In-between expiry -> returns interpolated synthetic `VolCurve`.
- Beyond last fitted pillar -> rejected for `slice(...)` long-end extrapolation.

## 5. Probability object behavior (`ProbCurve`, `ProbSurface`)

`ProbCurve` supports two usage patterns:

- direct construction from chain (`ProbCurve.from_chain`), or
- conversion from fitted volatility (`vol_curve.implied_distribution()`).

It exposes:

- local density queries (`pdf`),
- tail/cdf queries (`prob_below`, `prob_above`, `prob_between`),
- distribution moments and quantiles.

`ProbSurface` is a collection of expiry-specific `ProbCurve` slices with fan-chart visualization.

## 6. Interpretation and risk caveats

- OIPD outputs are **risk-neutral** distributions, not direct forecasts of real-world frequencies.
- Numerical outputs depend on input quality (spread noise, stale quotes, sparse strikes).
- For production use, inspect diagnostics and compare results across data windows and model settings.

## 7. What to review for rigor

If you are auditing implementation quality, focus on:

1. Input validation and expiry handling.
2. Fit diagnostics and calibration stability.
3. Interpolated-slice behavior and edge-case guards.
4. Regression tests for numerical drift.
