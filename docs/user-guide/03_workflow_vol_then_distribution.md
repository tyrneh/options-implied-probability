---
title: Workflow B: Fit Volatility Then Derive Distribution
parent: User Guide
nav_order: 3
---

# Workflow B: Fit Volatility Then Derive Distribution

Use this workflow when you want both:

1. A volatility object for pricing/risk work.
2. A probability object derived from that fitted volatility.

This is the standard quant workflow in OIPD.

## Single Expiry Path

Assume `chain` and `market` are prepared as in the quickstart page.

```python
from oipd import VolCurve

vol_curve = VolCurve(method="svi", pricing_engine="black76")
vol_curve.fit(chain, market)

# Volatility-layer outputs
print("ATM vol:", vol_curve.atm_vol)
print("Forward:", vol_curve.forward_price)
print("Params keys:", vol_curve.params.keys())

# Probability-layer output
prob_curve = vol_curve.implied_distribution()
print("P(S < 100):", prob_curve.prob_below(100.0))
```

## Multi Expiry Path

Assume `surface_chain` and `market` are prepared as in the quickstart page.

```python
from oipd import VolSurface

vol_surface = VolSurface(method="svi")
vol_surface.fit(surface_chain, market)

print("Fitted expiries:", vol_surface.expiries)
print("Interpolated IV at K=100, t=0.2:", vol_surface.implied_vol(100.0, 0.2))

prob_surface = vol_surface.implied_distribution()
print("Distribution expiries:", prob_surface.expiries)

# Unified maturity queries (same t input pattern as VolSurface price/atm_vol)
print("PDF at K=100, t=45/365:", prob_surface.pdf(100.0, t=45 / 365.0))
print("CDF at K=100, date='2025-02-15':", prob_surface.cdf(100.0, t="2025-02-15"))
print("Median at t=45/365:", prob_surface.quantile(0.5, t=45 / 365.0))
```

## Why choose this path

- You can inspect calibration quality (`iv_results`, `diagnostics`).
- You can use pricing and Greeks before distribution conversion.
- You get one consistent probability engine across pillar and interior maturities.
- You keep a clean architecture split: stateful interface objects on top of
  stateless probability pipelines/core numerics.

## Validated Commands (Offline)

- [x] `VolCurve.fit(...)` + `implied_distribution()` snippet executed in `.venv`.
- [x] `VolSurface.fit(...)` + interpolation + `implied_distribution()` snippet executed in `.venv`.
