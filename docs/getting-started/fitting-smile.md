---
title: Fitting a Volatility Smile
parent: Getting Started
nav_order: 2
---

## 2.2. Fitting a Volatility Smile

The `oipd.VolCurve` class is highly configurable. You can specify the fitting method, pricing engine, and more.

```python
# Example of a more customised VolCurve
vol_curve_bs = oipd.VolCurve(
    method="svi",
    pricing_engine="bs",  # Use Black-Scholes instead of Black-76
    price_method="mid"
).fit(options_df, market)

# Accessing fitted parameters and diagnostics
print("Fitted SVI parameters:", vol_curve_bs.params)
print("ATM Volatility:", vol_curve_bs.at_money_vol)
print("Fit Diagnostics:", vol_curve_bs.diagnostics)
```