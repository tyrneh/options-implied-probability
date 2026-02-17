---
title: Algorithm Details
nav_order: 5
---

# Algorithm Details

This page summarizes the target architecture for a conceptually correct and performant probability surface API.

## 1. High-Level Overview

`VolSurface` is the canonical continuous model:

- `VolSurface`: `(K, t) -> sigma(K, t)`.

`ProbSurface` should be defined as a transformation of that same model:

- `ProbSurface`: `(K, t) -> q(K, t)`.
- `q(K, t) = exp(r(t) * t) * d2C(K, t) / dK2`.
- `C(K, t)` is obtained from the pricing model (Black-76 or Black-Scholes) using `sigma(K, t)`.

Design rule: interpolate in volatility/price space first, then derive probability.

## 2. Rationale

- **Financial consistency**: All probability outputs come from one calibrated volatility surface.
- **Stability**: Total-variance interpolation in vol space is more robust than direct interpolation of PDF/CDF/quantiles across expiries.
- **Interpretability**: The probability layer remains an explicit transformation of option prices (Breeden-Litzenberger), which is auditable.
- **Performance**: Core work can be vectorized over maturity and strike grids.

## 3. Implementation Summary

### 3.1 Interface Layer (`oipd/interface`)

Implement `ProbSurface` as a function-like object that stores:

- Reference to fitted `VolSurface`.
- Numerical settings (grid resolution, derivative controls).
- Lazy cache keyed by maturity.

Public methods:

- `pdf(K, t)` and `__call__(K, t)`.
- `cdf(K, t)`.
- `quantile(p, t)`.
- `slice(expiry)` returning a `ProbCurve`.
- `plot_fan(num_points=...)` using dense maturity sampling.

### 3.2 Pipeline Layer (`oipd/pipelines`)

Add batched probability builders that:

1. Sample maturity and strike grids.
2. Compute implied vols and call prices in matrix form.
3. Compute PDF via second derivatives in strike.
4. Integrate to CDF and extract quantiles.

### 3.3 Core Layer (`oipd/core`)

Keep kernels pure and vectorized:

- Axis-wise finite-difference second derivative.
- Axis-wise cumulative integration.

### 3.4 Performance Policy

- Avoid constructing interface wrapper objects in internal hot paths.
- Keep wrapper creation for user-facing ergonomics only.
- Use lazy cache for interactive queries and optional eager precompute for plotting/reporting.

## 4. Practical Outcome

Users get a probability surface that is:

- Conceptually correct: a direct transformation of `VolSurface`.
- Fast: vectorized where it matters.
- Backward-compatible: existing entry points can remain while internals improve.

