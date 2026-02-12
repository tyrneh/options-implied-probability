---
title: Reference: ProbSurface
parent: User Guide
nav_order: 9
---

# Reference: `ProbSurface`

`ProbSurface` is the multi-expiry probability object (a mapping of expiries to `ProbCurve`).

## Constructor Patterns

1. From fitted volatility surface:

```python
prob_surface = vol_surface.implied_distribution()
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
| Visualization | `plot_fan(lower_percentile=25, upper_percentile=75, ...)` |

## Failure Modes

- `slice(expiry)` raises `ValueError` if the expiry is not present.
- `plot_fan` raises `ValueError` if no distributions are available.

## Minimal Query Example

Assume `surface_chain` and `market` are prepared.

```python
ps = ProbSurface.from_chain(surface_chain, market)
curve = ps.slice(ps.expiries[0])
print(curve.prob_below(100.0))
```

## Validated Commands (Offline)

- [x] `ProbSurface` slice/query snippet executed in `.venv`.
