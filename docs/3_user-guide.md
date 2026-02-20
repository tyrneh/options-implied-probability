---
title: User Guide
nav_order: 3
has_children: true
---

# User Guide

This page gives a practical map of how to use OIPD.

## 1. High-Level Mental Model

OIPD has four core objects. 

A simple way to understand the package is by use case: fitting at a single future date vs over time, and working in implied volatility vs probabilities.

| Scope | Volatility Layer | Probability Layer |
| --- | --- | --- |
| Single future date | `VolCurve` | `ProbCurve` |
| Future time horizon | `VolSurface` | `ProbSurface` |

You can think about the lifecycle in three steps:

1. Initialize the estimator object with configuration.
2. Call `.fit(chain, market)` to calibrate.
3. Query/plot the fitted object, or convert from vol to probability via `.implied_distribution()`.

If you're familiar with scikit-learn, this is the same mental model: configure an estimator, call `fit`, then inspect outputs.

Conceptual flow:

```text
Step 1: Fit volatility
  Initialize VolCurve / VolSurface object
      + options chain + market inputs
      -> .fit(...)
      -> fitted VolCurve / VolSurface object (inspect IV, prices, greeks, etc.)

Step 2: Convert fitted volatility to probability
  Use fitted VolCurve / VolSurface
      -> .implied_distribution()
      -> ProbCurve / ProbSurface object (inspect PDF, CDF, quantiles, moments, etc.)
```

**Shortcut for computing probabilities:**

If you only care about probability and do not want to manually manage the volatility step, use the convenience constructors shown in `README.md` (`ProbCurve.from_chain(...)` and `ProbSurface.from_chain(...)`): OIPD will run the volatility fitting and probability distribution conversion in the background.

<br>

## 2. How To Call `fit` In Practice

### 2.1 Load option data

Local CSV/DataFrame route:

```python
from oipd import sources

COLUMN_MAPPING = {
    "strike": "strike",
    "last_price": "last_price",
    "bid": "bid",
    "ask": "ask",
    "expiration": "expiry",
    "type": "option_type",
}

chain = sources.from_csv("data/AAPL_data.csv", column_mapping=COLUMN_MAPPING)
```

Vendor route (automated yfinance fetch via `sources`):

```python
from oipd import sources

# choose one: explicit expiries or horizon
chain, snapshot = sources.fetch_chain(
    ticker="AAPL",
    horizon="3m",
    vendor="yfinance",
)
```

### 2.2 Define `MarketInputs` 

`MarketInputs` stores the required parameters for the Black-Scholes formula. It always needs:

1. `risk_free_rate`
2. `valuation_date`
3. `underlying_price`

Example:

```python
from oipd import MarketInputs

market = MarketInputs(
    risk_free_rate=0.04,
    valuation_date=snapshot.date,              # or a fixed date
    underlying_price=snapshot.underlying_price, # set explicitly if snapshot is unavailable
    # optional:
    # risk_free_rate_mode="annualized",  # default
    # dividend_yield=0.01,
)
```

### 2.3 Fit single-expiry (`VolCurve`) or multi-expiry (`VolSurface`)

Single-expiry example:

```python
from oipd import VolCurve

expiry = "2026-01-16"
chain_one_expiry = chain[chain["expiry"] == expiry].copy()

vol_curve = VolCurve()
vol_curve.fit(chain_one_expiry, market)

# optional downstream conversion to probability
prob_curve = vol_curve.implied_distribution()
```

Multi-expiry example:

```python
from oipd import VolSurface

vol_surface = VolSurface()
vol_surface.fit(chain, market)  # chain contains multiple expiries

# optional downstream conversion to probability
prob_surface = vol_surface.implied_distribution()
```

## 3. Surface Objects and `slice(...)`

Both surface objects support *slicing*:

1. `VolSurface.slice(expiry)` returns a `VolCurve`.
2. `ProbSurface.slice(expiry)` returns a `ProbCurve`.

After slicing, you can use the same methods you would use on regular curve objects (`implied_vol`, `price`, `greeks`, `iv_results` for volatility curves; `pdf`, `prob_below`, `quantile`, `plot` for probability curves).

```python
# volatility surface -> volatility curve snapshot
vol_curve_slice = vol_surface.slice("2026-01-16")
iv_at_250 = vol_curve_slice.implied_vol(250)

# probability surface -> probability curve snapshot
prob_curve_slice = prob_surface.slice("2026-01-16")
p_below_240 = prob_curve_slice.prob_below(240)
```

## 4. Overview of all API methods

### Volatility API Methods Comparison

| Feature / Action | `VolCurve` (Single Expiry) | Description (Curve) | `VolSurface` (Multi Expiry) | Description (Surface) |
| :--- | :--- | :--- | :--- | :--- |
| **Calibration** | `fit(chain, market)` | Calibrate model to one expiry. | `fit(chain, market)` | Calibrate multiple expiries. |
| **Implied Volatility** | `implied_vol(K)` | Get $\sigma$ for strike $K$. | `implied_vol(K, t)` | Get $\sigma$ for strike $K$ and time $t$. |
| **Total Variance** | `total_variance(K)` | Get $w = \sigma^2 t$. | `total_variance(K, t)` | Get $w = \sigma^2 t$ (interpolated). |
| **Option Pricing** | `price(K, call_put)` | Theoretical option price. | `price(K, t, call_put)` | Price at arbitrary time $t$. |
| **ATM Volatility** | `atm_vol` (property) | ATM Vol level. | `atm_vol(t)` | ATM Vol term structure at $t$. |
| **Forward Price** | `forward_price` (property) | Parity-implied forward $F$. | `forward_price(t)` | Interpolated forward $F(t)$. |
| **Greeks (Bulk)** | `greeks(K)` | DataFrame of all Greeks. | `greeks(K, t)` | DataFrame of interpolated Greeks. |
| **Individual Greeks** | `delta`, `gamma`, `vega`... | Partials w.r.t $S, \sigma, t, r$. | `delta`, `gamma`, `vega`... | Interpolated Greeks at $(K, t)$. |
| **Calibration Data** | `iv_results()` | DataFrame of fitted vs market. | `iv_results()` | Long-format DataFrame of all fits. |
| **Parameters** | `params` | Fitted coeffs (SVI `a, b...`). | `params` (dict) | Dict of `{expiry: params}`. |
| **Slicing** | | | `slice(expiry)` | Extract a `VolCurve` snapshot. |
| **Expiries** | `expiries` (1-tuple) | Single expiry date. | `expiries` (list) | List of fitted expiry dates. |
| **Distributions** | `implied_distribution()` | Get `ProbCurve` (RND). | `implied_distribution()` | Get `ProbSurface` (RND Surface). |
| **Visualization (2D curve)** | `plot()` | Plot fitted smile vs market. | `plot()` | Overlayed IV smiles. |
| **Visualization (3D surface)** | | | `plot_3d()` | Isometric 3D volatility surface. |
| **Term Structure** | | | `plot_term_structure()` | Interpolated ATM-forward IV term structure vs days to expiry. |

### Probability API Methods Comparison

| Feature / Action | `ProbCurve` (Single Expiry) | Description (Curve) | `ProbSurface` (Multi Expiry) | Description (Surface) |
| :--- | :--- | :--- | :--- | :--- |
| **Construction (Convenience)** | `from_chain(chain, market, ...)` | One-line constructor: fits SVI on a single-expiry chain, then builds `ProbCurve`. | `from_chain(chain, market, ...)` | One-line constructor: fits SVI across expiries, then builds `ProbSurface`. |
| **Construction (From Vol)** | via `VolCurve.implied_distribution()` | Build a `ProbCurve` from a fitted `VolCurve`. | via `VolSurface.implied_distribution()` | Build a `ProbSurface` from a fitted `VolSurface`. |
| **Density (PDF)** | `pdf(S)` | Probability density at price $S$. | `pdf(S, t)` | Probability density at price $S$ for a given maturity `t`. |
| **Callable Alias** | `__call__(S)` | Alias for `pdf(S)`. | `__call__(S, t)` | Alias for `pdf(S, t)`. |
| **CDF** | `prob_below(S)` | Cumulative distribution function at price $S$. | `cdf(S, t)` | CDF at price $S$ for maturity `t`. |
| **Tail Probabilities** | `prob_above(S)`, `prob_between(L, H)` | Upper tail and interval probabilities. | `slice(expiry).prob_above(S)`, `slice(expiry).prob_between(L, H)` | Tail methods are exposed on the sliced `ProbCurve`. |
| **Quantile** | `quantile(q)` | Inverse CDF (price at probability $q$). | `quantile(q, t)` | Direct surface quantile query at maturity `t` (also available via `slice(expiry).quantile(q)`). |
| **Moments** | `mean()`, `variance()`, `skew()`, `kurtosis()` | Distribution moments. | `slice(expiry).mean()` etc. | Moments for a selected expiry. |
| **Grid Access** | `prices`, `pdf_values`, `cdf_values` | Cached evaluation grid for plots and queries. | `slice(expiry).prices` etc. | Grid for a selected expiry. |
| **Visualization (2D)** | `plot(kind=...)` | PDF/CDF plot for one expiry. | `plot_fan()` | Fan chart of quantile bands over expiries. |
| **Metadata / Expiries** | `resolved_market`, `metadata` | Market snapshot + fit metadata. | `expiries` | Available expiry dates. |
| **Slicing** | | | `slice(expiry)` | Extract a `ProbCurve` snapshot. |
