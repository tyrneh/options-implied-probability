---
title: Reference: VolCurve
parent: User Guide
nav_order: 6
---

# Reference: `VolCurve`

`VolCurve` is the single-expiry stateful volatility estimator.

## Constructor

```python
VolCurve(
    *,
    method: str = "svi",
    pricing_engine: str = "black76",
    price_method: str = "mid",
    max_staleness_days: int = 3,
)
```

## Fit Contract

```python
fit(chain, market, *, column_mapping=None) -> VolCurve
```

Requirements:

- `chain` must have exactly one unique expiry after parsing.
- `expiry` must be strictly after `market.valuation_date`.

## Main Methods and Properties

| Category | API |
|---|---|
| Vol evaluation | `implied_vol(strikes)`, `__call__(strikes)` |
| Total variance | `total_variance(strikes)` |
| Pricing | `price(strikes, call_or_put="call")` |
| Greeks | `delta`, `gamma`, `vega`, `theta`, `rho`, `greeks` |
| Fit outputs | `atm_vol`, `expiries`, `iv_results`, `params`, `forward_price`, `diagnostics`, `resolved_market` |
| Probability bridge | `implied_distribution()` |
| Plotting | `plot(...)` |

## Common Errors

- Access before fit: methods raise `ValueError` with "Call fit before ...".
- Wrong expiry cardinality: `fit` raises `ValueError` if multiple expiries are present.
- Invalid expiry timing: `fit` raises `CalculationError` if expiry is on/before valuation date.

## Minimal Lifecycle Example

Assume `chain` and `market` are prepared.

```python
vol = VolCurve().fit(chain, market)
ivs = vol.implied_vol([95.0, 100.0, 105.0])
prices = vol.price([95.0, 100.0, 105.0], call_or_put="call")
prob = vol.implied_distribution()
```

## Validated Commands (Offline)

- [x] `VolCurve` lifecycle snippet executed in `.venv`.
