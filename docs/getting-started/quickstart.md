---
title: Quickstart Guide
parent: Getting Started
nav_order: 1
---

## 2.1. Quickstart Guide

This example shows how to get a risk-neutral probability distribution for Apple (AAPL) stock.

```python
import oipd
import yfinance as yf
from datetime import date
import pandas as pd

# 1. Fetch Data
# We use yfinance to get the option chain for a specific expiry.
ticker = yf.Ticker("AAPL")
expiry = date(2025, 12, 19)
chain = ticker.option_chain(expiry.strftime("%Y-%m-%d"))
# We'll use the call options for this example.
options_df = chain.calls

# 2. Define Market Conditions
# The MarketInputs object holds all the necessary market data.
market = oipd.MarketInputs(
    underlying_price=ticker.info["currentPrice"],
    valuation_date=pd.to_datetime(date.today()),
    expiry_date=pd.to_datetime(expiry),
    risk_free_rate=0.05,  # Use a realistic rate for your analysis
)

# 3. Fit the Volatility Curve
# The VolCurve class is the main entry point.
# We create an instance and call the fit() method.
vol_curve_estimator = oipd.VolCurve()
vol_curve_estimator.fit(options_df, market)

# 4. Derive the Probability Distribution
# The implied_distribution() method returns a Distribution object.
dist = vol_curve_estimator.implied_distribution()

# 5. Query Probabilities
# Now you can query the distribution for insights.
price_target = 250.0
prob_below = dist.prob_below(price_target)
prob_above = dist.prob_above(price_target)
prob_between = dist.prob_between(240.0, 260.0)

print(f"Probability of AAPL being below ${price_target}: {prob_below:.2%}")
print(f"Probability of AAPL being above ${price_target}: {prob_above:.2%}")
print(f"Probability of AAPL being between $240 and $260: {prob_between:.2%}")

# You can also get the expected value (mean) of the distribution.
ev = dist.expected_value()
print(f"Expected value of AAPL at expiry: ${ev:.2f}")
```