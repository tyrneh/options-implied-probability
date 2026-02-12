---
title: Reference: ProbCurve
parent: User Guide
nav_order: 8
---

# Reference: `ProbCurve`

`ProbCurve` is the single-expiry probability object derived from a fitted `VolCurve`.

## Constructor Patterns

1. From fitted volatility object:

```python
prob = vol_curve.implied_distribution()
```

2. Convenience constructor from chain:

```python
ProbCurve.from_chain(chain, market, *, column_mapping=None, max_staleness_days=3)
```

## Main Methods and Properties

| Category | API |
|---|---|
| Density queries | `pdf(price)`, `__call__(price)` |
| CDF/tails | `prob_below`, `prob_above`, `prob_between` |
| Moments | `mean`, `variance`, `skew`, `kurtosis` |
| Quantile | `quantile(q)` |
| Cached arrays | `prices`, `pdf_values`, `cdf_values` |
| Metadata | `resolved_market`, `metadata` |
| Plot | `plot(kind="pdf"|"cdf"|"both", ...)` |

## Notes on Computation

- `ProbCurve` computes array grids lazily when needed for plotting and moment calculations.
- Tail and quantile results depend on the fitted volatility model and numerical grid.

## Minimal Query Example

Assume `chain` and `market` are prepared.

```python
prob = ProbCurve.from_chain(chain, market)
print(prob.prob_below(100.0))
print(prob.quantile(0.5))
print(prob.mean(), prob.variance())
```

## Validated Commands (Offline)

- [x] `ProbCurve` query snippet executed in `.venv`.
