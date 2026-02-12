---
title: Workflow A: Direct Distribution from Chain
parent: User Guide
nav_order: 2
---

# Workflow A: Direct Distribution from Chain

Use this workflow when your main output is probability metrics, not volatility diagnostics.

- Single expiry: `ProbCurve.from_chain(chain, market)`
- Multi expiry: `ProbSurface.from_chain(chain, market)`

## When this workflow is a good fit

- You want probabilities quickly (`prob_below`, quantiles, moments).
- You do not need per-slice volatility diagnostics.
- You already have a cleaned chain with an `expiry` column.

## Core Query Methods

For `ProbCurve`:

- Density and tails: `pdf`, `prob_below`, `prob_above`, `prob_between`
- Summary statistics: `mean`, `variance`, `skew`, `kurtosis`, `quantile`
- Cached arrays: `prices`, `pdf_values`, `cdf_values`

For `ProbSurface`:

- `expiries`
- `slice(expiry)` -> returns `ProbCurve`
- `plot_fan(...)`

## Minimal Example

Assume `prob` is already created via `ProbCurve.from_chain(chain, market)`.

```python
summary = {
    "p_below_95": prob.prob_below(95.0),
    "p_above_110": prob.prob_above(110.0),
    "median": prob.quantile(0.5),
    "mean": prob.mean(),
    "variance": prob.variance(),
}
print(summary)
```

## Constraints to remember

- `ProbCurve.from_chain` expects exactly one expiry.
- `ProbSurface.from_chain` expects at least two expiries.
- These are risk-neutral probabilities implied by option prices.

## Validated Commands (Offline)

- [x] `ProbCurve` query-method snippet executed in `.venv`.
