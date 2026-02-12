---
title: Reference: VolSurface
parent: User Guide
nav_order: 7
---

# Reference: `VolSurface`

`VolSurface` is the multi-expiry stateful volatility estimator.

## Constructor

```python
VolSurface(
    *,
    method: str = "svi",
    pricing_engine: str = "black76",
    price_method: str = "mid",
    max_staleness_days: int = 3,
)
```

## Fit Contract

```python
fit(chain, market, *, column_mapping=None, horizon=None) -> VolSurface
```

Requirements:

- `chain` must parse to at least two expiries.
- `expiry` column is required after mapping.
- Same-day or past expiries are filtered/rejected in fit.

## Main Methods and Properties

| Category | API |
|---|---|
| Surface evaluation | `implied_vol(K, t)`, `__call__(K, t)` |
| Total variance | `total_variance(K, t)` |
| Forward interpolation | `forward_price(t)` |
| Pricing | `price(strikes, t, call_or_put="call")` |
| Greeks | `delta`, `gamma`, `vega`, `theta`, `rho`, `greeks` (all with `t`) |
| Slicing | `slice(expiry)` |
| Fit outputs | `expiries`, `iv_results()`, `params` |
| Probability bridge | `implied_distribution()` |
| Plotting | `plot(...)`, `plot_3d(...)`, `plot_term_structure(...)` |

## Slice Behavior

- Exact fitted expiry: returns a fitted `VolCurve` snapshot.
- In-between expiry: returns an interpolated synthetic `VolCurve`.
- Beyond last fitted expiry: raises `ValueError` (long-end extrapolation is not supported in `slice`).

## Minimal Lifecycle Example

Assume `surface_chain` and `market` are prepared.

```python
surface = VolSurface().fit(surface_chain, market)
iv = surface.implied_vol(100.0, 0.25)
slice_curve = surface.slice(surface.expiries[0])
prob_surface = surface.implied_distribution()
```

## Validated Commands (Offline)

- [x] `VolSurface` lifecycle snippet executed in `.venv`.
