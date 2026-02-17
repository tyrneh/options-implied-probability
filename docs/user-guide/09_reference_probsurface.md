---
title: Reference: ProbSurface
parent: User Guide
nav_order: 9
---

# Reference: `ProbSurface`

`ProbSurface` is the multi-expiry probability object. It supports:

- a single `VolSurface`-backed probability engine for all maturities,
- slice retrieval at any supported expiry as `ProbCurve`,
- direct callable probability queries at arbitrary maturities.
- a thin interface layer that delegates numerical work to stateless
  probability pipelines and core math helpers.

## Constructor Patterns

1. From fitted volatility surface:

```python
prob_surface = vol_surface.implied_distribution()
```

`ProbSurface` is constructed from a fitted `VolSurface` only:

```python
from oipd.interface.probability import ProbSurface
prob_surface = ProbSurface(vol_surface=vol_surface)
```

2. Convenience constructor:

```python
ProbSurface.from_chain(chain, market, *, column_mapping=None, max_staleness_days=3)
```

## Main Methods and Properties

| Category | API |
|---|---|
| Expiry listing | `expiries` |
| Slice retrieval | `slice(expiry)` -> `ProbCurve` |
| Density queries | `pdf(price, t)`, `__call__(price, t)` |
| CDF queries | `cdf(price, t)` |
| Quantile queries | `quantile(q, t)` |
| Visualization | `plot_fan(lower_percentile=25, upper_percentile=75, ...)` |

## Failure Modes

- `ProbSurface(vol_surface=...)` raises `ValueError` if `vol_surface` is not fitted.
- `slice(expiry)` raises `ValueError` if the expiry is beyond the last fitted pillar.
- `pdf/cdf/quantile(..., t=...)` raise `ValueError` when `t <= 0`.
- `pdf/cdf/quantile(..., t=...)` raise `ValueError` for maturities beyond the last pillar.
- `plot_fan` raises `ValueError` if the underlying volatility surface is unavailable.

## Minimal Query Example

Assume `surface_chain` and `market` are prepared.

```python
ps = ProbSurface.from_chain(surface_chain, market)
curve = ps.slice(ps.expiries[0])
print(curve.prob_below(100.0))

# Direct q(K, t) style queries
print(ps.pdf(100.0, t="2025-02-15"))
print(ps.cdf(100.0, t=45 / 365.0))
print(ps.quantile(0.5, t=45 / 365.0))
```

## Validated Commands (Offline)

- [x] `ProbSurface` slice/query snippet executed in `.venv`.
