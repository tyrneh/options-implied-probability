---
title: API Comparison
parent: User Guide
nav_order: 5
---

# API Comparison

This page is a quick decision aid before you open full reference pages.

## Volatility Objects

| Capability | `VolCurve` (single expiry) | `VolSurface` (multi expiry) |
|---|---|---|
| Fit | `fit(chain, market, column_mapping=...)` | `fit(chain, market, column_mapping=..., horizon=...)` |
| Evaluate IV | `implied_vol(strikes)` | `implied_vol(K, t)` |
| Evaluate total variance | `total_variance(strikes)` | `total_variance(K, t)` |
| Pricing | `price(strikes, call_or_put=...)` | `price(strikes, t, call_or_put=...)` |
| Greeks | `delta/gamma/vega/theta/rho`, `greeks` | same set with extra `t` argument |
| Fitted metadata | `atm_vol`, `params`, `forward_price`, `diagnostics`, `resolved_market` | `params`, `expiries`, `iv_results()` |
| Distribution bridge | `implied_distribution()` -> `ProbCurve` | `implied_distribution()` -> `ProbSurface` |
| Plotting | `plot(...)` | `plot(...)`, `plot_3d(...)`, `plot_term_structure(...)` |
| Slice support | N/A | `slice(expiry)` -> `VolCurve` |

## Probability Objects

| Capability | `ProbCurve` | `ProbSurface` |
|---|---|---|
| Constructor from chain | `ProbCurve.from_chain(...)` | `ProbSurface.from_chain(...)` |
| Primary queries | `pdf`, `prob_below`, `prob_above`, `prob_between` | `slice(expiry)` then same as `ProbCurve` |
| Moments | `mean`, `variance`, `skew`, `kurtosis` | via slice |
| Quantiles | `quantile(q)` | via slice |
| Cached arrays | `prices`, `pdf_values`, `cdf_values` | via slice |
| Plotting | `plot(kind=...)` | `plot_fan(...)` |
| Expiry access | metadata from curve | `expiries` |

## Which path should you use?

- Use `ProbCurve.from_chain` / `ProbSurface.from_chain` for fastest probability workflow.
- Use `VolCurve` / `VolSurface` first when you need volatility diagnostics, pricing, or Greeks before probability conversion.

## Validated Commands (Offline)

- [x] No executable commands on this page.
