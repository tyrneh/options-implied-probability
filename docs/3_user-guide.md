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
      -> fitted VolCurve / VolSurface object (inspect IV, prices, forward-space greeks, etc.)

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

For local CSV or DataFrame uploads, use one long-form option-chain table: one
row per option contract. OIPD expects these standard input columns:

| Column | Required? | Notes |
| --- | --- | --- |
| `strike` | Yes | Option strike price. |
| `expiry` | Yes | Option expiry date or timestamp. |
| `option_type` | Yes | Call/put marker. Values are normalized to `C` or `P`; `call` and `put` are accepted. |
| `last_price` | Yes | Last traded option price. This is required for fitting in the current input contract. |
| `bid` | Recommended | Best bid when available; useful quote metadata, but not a standalone replacement for `last_price`. |
| `ask` | Recommended | Best ask when available; useful quote metadata, but not a standalone replacement for `last_price`. |
| `volume` | Optional | Lowercase column name. Can be used downstream for weighting. |
| `last_trade_date` | Optional | Last quote/trade timestamp. Can be used downstream to filter stale quotes. |

Example local CSV route:

```python
from oipd import sources

COLUMN_MAPPING = {
    "strike": "strike",
    "last_price": "last_price",
    "bid": "bid",
    "ask": "ask",
    "volume": "volume",
    "last_trade_date": "last_trade_date",
    "expiration": "expiry",
    "type": "option_type",
}

chain = sources.from_csv("data/AAPL_data.csv", column_mapping=COLUMN_MAPPING)
```

The same schema applies to `sources.from_dataframe(...)`. Use
`column_mapping` when your input file uses different column names.

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

`MarketInputs` stores the public market context used by the default fitting
workflow. It always needs:

1. `risk_free_rate`
2. `valuation_date`
3. `underlying_price`

`underlying_price` should be the raw, unadjusted current price of the
underlying in the same contract units as the option strikes. For US equity
and yfinance-style workflows, use the live or snapshot stock price, not a
split- or dividend-adjusted historical close. OIPD uses this value for
diagnostics and forward-implied carry calculations; implied-volatility fitting
uses the forward inferred from option prices.

Example:

```python
from oipd import MarketInputs

market = MarketInputs(
    risk_free_rate=0.04,
    valuation_date=snapshot.asof,              # recommended for intraday precision
    underlying_price=snapshot.underlying_price, # raw current underlying price
    # optional:
    # risk_free_rate_mode="annualized",  # default
)
```

`valuation_date` accepts both plain dates and full datetimes. Date-only values
remain fully supported, and full datetimes (for example `snapshot.asof`) should
be preferred when intraday precision matters.

Market-object contract:

- `valuation_date` remains the public field name and stores the canonical
  normalized timestamp.
- `valuation_calendar_date` is the explicit date-only convenience view.
- `valuation_timestamp` remains as a temporary compatibility alias to the same
  canonical timestamp value.

The option maturity field remains `expiry`. OIPD keeps `valuation_date` as the
public input name for backwards compatibility, but internally it resolves both
`valuation_date` and `expiry` to full timestamps before computing maturity.
That means exact `time_to_expiry_years` is now the source-of-truth input for
pricing, calibration, and probability calculations. Reporting in day units now
uses explicit `time_to_expiry_days`, while `calendar_days_to_expiry` is the
explicit integer calendar bucket.

Maturity contract:

- `oipd.core.maturity` is the canonical home of maturity logic.
- `time_to_expiry_years` is the pricing truth.
- `time_to_expiry_days` is the continuous reporting truth.
- `calendar_days_to_expiry` is the integer calendar bucket.

Migration note:

- replace old `days_to_expiry` inputs with `time_to_expiry_years` for pricing
  or `time_to_expiry_days` for reporting
- replace old `days_to_expiry` metadata reads with
  `calendar_days_to_expiry` if you intended integer calendar-bucket semantics

Timezone display redesign is still out of scope for this cycle. Intraday
arithmetic is supported, but timezone-aware display semantics are unchanged.

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

### 2.4 Forward inference, carry, and Greeks

The public workflow fits in Black-76 forward space. OIPD infers the stored
`forward_price` from same-strike put-call parity across usable call/put pairs.
That means each fitted expiry needs both calls and puts at matching strikes
with usable prices. Call-only or put-only chains are not enough for the main
public workflow.

The gap between `underlying_price` and the parity-implied forward is best read
as `forward-implied carry` or `dividend-equivalent carry`. It is useful model
metadata, but it is not a clean dividend forecast. Treat it with extra caution
for American single-name or ETF options, special dividends, hard-to-borrow
names, sparse quotes, stale quotes, and raw-versus-adjusted spot mismatches.

Internally retained fit diagnostics include a `parity_report` with pair counts,
confidence, outlier counts, liquidity indicators, and the strikes used or
excluded during the forward estimate. The public diagnostics surface for this
report is still stabilizing, so treat it as debugging metadata rather than a
stable user-facing API.

Prices and Greeks reported by `VolCurve` and `VolSurface` are Black-76 outputs.
For Greeks, delta is a forward delta: it is sensitivity to the inferred forward
price, not necessarily the same object as cash-equity spot delta.

### 2.5 Warning diagnostics

OIPD tries to keep notebooks readable. When a public operation sees multiple
related issues, it emits one summarized warning per broad category instead of
printing one long warning per row, expiry, or numerical repair.

The detailed audit trail is stored on every public curve and surface object:

```python
surface = ProbSurface.from_chain(chain, market)
fig = surface.plot_fan()  # may emit one concise ModelRiskWarning or WorkflowWarning

surface.warning_diagnostics.summary
surface.warning_diagnostics.events
```

Each event has stable fields:

```python
event = surface.warning_diagnostics.events[0]
event.category     # e.g. "model_risk"
event.event_type   # e.g. "cdf_repair"
event.severity     # "info", "warning", or "severe"
event.message
event.details      # compact JSON-like audit facts
```

The broad warning classes live in `oipd.warnings` and are standard
`UserWarning` subclasses, so you can filter them with Python's `warnings`
module:

```python
import warnings

from oipd.warnings import DataQualityWarning, ModelRiskWarning, WorkflowWarning

warnings.simplefilter("always", ModelRiskWarning)
warnings.simplefilter("error", WorkflowWarning)
```

The warning taxonomy is:

- `DataQualityWarning`: input quote issues such as stale rows or fallback prices
- `ModelRiskWarning`: fitted-model or probability-shape issues such as CDF repair
- `NumericalWarning`: numerical fragility or instability
- `WorkflowWarning`: best-effort continuation, such as skipped expiries

Probability CDF validation is controlled by `cdf_violation_policy`. The default
is `"warn"`: OIPD repairs material direct-CDF monotonicity violations, emits a
summarized `ModelRiskWarning`, and records a `cdf_repair` event. Use `"raise"`
for stricter workflows:

```python
prob = ProbSurface.from_chain(
    chain,
    market,
    cdf_violation_policy="raise",
)
```

With `"raise"`, strict CDF violations propagate as errors instead of being
converted into repair diagnostics or fan-chart skipped-expiry warnings.

## 3. Surface Objects and `slice(...)`

Both surface objects support *slicing*:

1. `VolSurface.slice(expiry)` returns a `VolCurve`.
2. `ProbSurface.slice(expiry)` returns a `ProbCurve`.

After slicing, you can use the same methods you would use on regular curve objects (`implied_vol`, `price`, `greeks`, `iv_results` for volatility curves; `pdf`, `prob_below`, `quantile`, `density_results`, `plot` for probability curves).

When `expiry` or `valuation_date` includes a non-midnight timestamp, OIPD
preserves that intraday precision through surface queries and will show the
time-of-day in labels/plots where relevant. Midnight timestamps continue to
render as date-only labels, so older date-based workflows remain visually
stable.

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
| **Option Pricing** | `price(K, call_put)` | Black-76 theoretical option price. | `price(K, t, call_put)` | Black-76 price at arbitrary time $t$. |
| **ATM Volatility** | `atm_vol` (property) | ATM Vol level. | `atm_vol(t)` | ATM Vol term structure at $t$. |
| **Forward Price** | `forward_price` (property) | Parity-implied forward $F$. | `forward_price(t)` | Interpolated forward $F(t)$. |
| **Greeks (Bulk)** | `greeks(K)` | DataFrame of forward-space Black-76 Greeks. | `greeks(K, t)` | DataFrame of interpolated forward-space Greeks. |
| **Individual Greeks** | `delta`, `gamma`, `vega`... | Partials w.r.t. forward $F$, $\sigma$, $t$, and $r$. | `delta`, `gamma`, `vega`... | Interpolated Greeks at $(K, t)$. |
| **Calibration Data** | `iv_results(domain=None, points=200, include_observed=True)` | Single-slice DataFrame of fitted IV vs observed quotes on a configurable strike grid. | `iv_results(domain=None, points=200, include_observed=True, start=None, end=None, step_days=1)` | Long-format IV export on a daily grid by default; omitted `start`/`end` use the first/last fitted pillar and fitted pillars are always preserved. |
| **Parameters** | `params` | Fitted coeffs (SVI `a, b...`). | `params` (dict) | Dict of `{expiry: params}`. |
| **Slicing** | | | `slice(expiry)` | Extract a `VolCurve` snapshot. |
| **Expiries** | `expiries` (1-tuple) | Single expiry date. | `expiries` (list) | List of fitted expiry dates. |
| **Distributions** | `implied_distribution(grid_points=None)` | Get `ProbCurve` (RND). `grid_points=None` uses the smart native grid policy. | `implied_distribution(grid_points=None)` | Get `ProbSurface` (RND Surface). `grid_points=None` uses the smart native grid policy for each materialized slice. |
| **Visualization (2D curve)** | `plot()` | Plot fitted smile vs market. | `plot()` | Overlayed IV smiles. |
| **Visualization (3D surface)** | | | `plot_3d()` | Isometric 3D volatility surface. `expiry_range` accepts date-like bounds and is converted to continuous `time_to_expiry_days` internally. |
| **Term Structure** | | | `plot_term_structure()` | Interpolated ATM-forward IV term structure vs days to expiry. |

### Probability API Methods Comparison

| Feature / Action | `ProbCurve` (Single Expiry) | Description (Curve) | `ProbSurface` (Multi Expiry) | Description (Surface) |
| :--- | :--- | :--- | :--- | :--- |
| **Construction (Convenience)** | `from_chain(chain, market, ...)` | One-line constructor: fits SVI on a single-expiry chain, then builds `ProbCurve`. | `from_chain(chain, market, ...)` | One-line constructor: fits SVI across expiries, then builds `ProbSurface`. |
| **Construction (From Vol)** | via `VolCurve.implied_distribution(grid_points=None)` | Build a `ProbCurve` from a fitted `VolCurve`; `grid_points` controls native numerical resolution. | via `VolSurface.implied_distribution(grid_points=None)` | Build a `ProbSurface` from a fitted `VolSurface`; `grid_points` controls native numerical resolution for materialized slices. |
| **Density (PDF)** | `pdf(S)` | Probability density at price $S$. | `pdf(S, t)` | Probability density at price $S$ for a given maturity `t`. |
| **Callable Alias** | `__call__(S)` | Alias for `pdf(S)`. | `__call__(S, t)` | Alias for `pdf(S, t)`. |
| **CDF** | `prob_below(S)` | Cumulative distribution function at price $S$. | `cdf(S, t)` | CDF at price $S$ for maturity `t`. |
| **Tail Probabilities** | `prob_above(S)`, `prob_between(L, H)` | Upper tail and interval probabilities. | `slice(expiry).prob_above(S)`, `slice(expiry).prob_between(L, H)` | Tail methods are exposed on the sliced `ProbCurve`. |
| **Quantile** | `quantile(q)` | Inverse CDF (price at probability $q$). | `quantile(q, t)` | Direct surface quantile query at maturity `t` (also available via `slice(expiry).quantile(q)`). |
| **Moments** | `mean()`, `variance()`, `skew()`, `kurtosis()` | Distribution moments. | `slice(expiry).mean()` etc. | Moments for a selected expiry. |
| **Grid Access** | `prices`, `pdf_values`, `cdf_values` | Cached evaluation grid for plots and queries. | `slice(expiry).prices` etc. | Grid for a selected expiry. |
| **Data Export** | `density_results(domain=None, points=200, full_domain=False)` | DataFrame with `price`, `pdf`, and `cdf`; defaults to a compact `default_view_domain` with 200 rows. Use `full_domain=True` for native full-domain arrays or `domain=(a, b)` for an explicit export range. | `density_results(domain=None, points=200, start=None, end=None, step_days=1, full_domain=False)` | Long-format export on a daily grid by default, with 200 compact rows per expiry. Use `full_domain=True` to export native full-domain arrays for each slice. |
| **Visualization (2D)** | `plot(kind=..., points=800, full_domain=False, xlim=None)` | PDF/CDF plot for one expiry. Defaults to the compact view domain; `full_domain=True` plots the native full domain, and `xlim=(a, b)` explicitly controls the visible plot range. | `plot_fan()` | Fixed multi-band fan chart over expiries with four shaded bands, a dashed median, and dots at fitted expiry pillars. |
| **Metadata / Expiries** | `resolved_market`, `metadata` | Market snapshot + fit metadata. | `expiries` | Available expiry dates. |
| **Slicing** | | | `slice(expiry)` | Extract a `ProbCurve` snapshot. |

Probability views now separate the native numerical grid from display/export
resolution:

- `grid_points` controls native probability materialization: stored `prices`,
  `pdf_values`, and `cdf_values`. The default is `grid_points=None`, which uses
  a smart grid policy based on domain width, spot/forward scale, and observed
  strike spacing.
- `points` controls view/export/plot resolution. It does not change the fitted
  native probability arrays.
- `density_results()` defaults to the compact `default_view_domain` with
  `points=200`, so wide probability tails do not automatically create very
  large or very sparse DataFrames.
- `density_results(full_domain=True)` returns the native full-domain arrays
  exactly when `domain` is omitted.
- `density_results(domain=(a, b), points=N)` uses the explicit export range and
  row count regardless of `full_domain`.
