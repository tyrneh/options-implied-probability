---
title: Quickstart: Probability in 5 Minutes
parent: User Guide
nav_order: 1
---

# Quickstart: Probability in 5 Minutes

This quickstart shows the shortest path from an option chain to probability queries.

It uses deterministic synthetic data so you can run it offline and reproduce the same result.

## Single Expiry: `ProbCurve`

```python
from datetime import date
import numpy as np
import pandas as pd
from oipd import MarketInputs, ProbCurve

strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
expiry = pd.Timestamp("2025-03-21")
S, r = 100.0, 0.05
T = (expiry.date() - date(2025, 1, 1)).days / 365.0

df = np.exp(-r * T)
call_last = np.array([12.5, 8.2, 5.1, 3.1, 1.6])
put_last = np.abs(call_last - S + strikes * df)

calls = pd.DataFrame(
    {
        "expiry": [expiry] * len(strikes),
        "strike": strikes,
        "last_price": call_last,
        "bid": call_last - 0.2,
        "ask": call_last + 0.2,
        "option_type": ["call"] * len(strikes),
    }
)
puts = pd.DataFrame(
    {
        "expiry": [expiry] * len(strikes),
        "strike": strikes,
        "last_price": put_last,
        "bid": np.maximum(put_last - 0.2, 0.01),
        "ask": put_last + 0.2,
        "option_type": ["put"] * len(strikes),
    }
)
chain = pd.concat([calls, puts], ignore_index=True)

market = MarketInputs(
    valuation_date=date(2025, 1, 1),
    risk_free_rate=0.05,
    underlying_price=100.0,
    dividend_yield=0.0,
)

prob = ProbCurve.from_chain(chain, market)

print("P(S < 100):", round(prob.prob_below(100.0), 4))
print("P(S >= 120):", round(prob.prob_above(120.0), 4))
print("Median:", round(prob.quantile(0.50), 4))
```

## Multiple Expiries: `ProbSurface`

```python
from oipd import ProbSurface

# Reuse the same chain construction pattern and append a second expiry.
expiry_2 = pd.Timestamp("2025-04-18")
T2 = (expiry_2.date() - date(2025, 1, 1)).days / 365.0
call_last2 = np.array([13.4, 9.3, 6.4, 4.4, 2.8])
put_last2 = np.abs(call_last2 - S + strikes * np.exp(-r * T2))

calls2 = calls.copy()
calls2["expiry"] = expiry_2
calls2["last_price"] = call_last2
calls2["bid"] = call_last2 - 0.2
calls2["ask"] = call_last2 + 0.2

puts2 = puts.copy()
puts2["expiry"] = expiry_2
puts2["last_price"] = put_last2
puts2["bid"] = np.maximum(put_last2 - 0.2, 0.01)
puts2["ask"] = put_last2 + 0.2

surface_chain = pd.concat([chain, calls2, puts2], ignore_index=True)

prob_surface = ProbSurface.from_chain(surface_chain, market)
first_expiry = prob_surface.expiries[0]
first_curve = prob_surface.slice(first_expiry)
print("First expiry:", first_expiry)
print("P(S < 100) at first expiry:", round(first_curve.prob_below(100.0), 4))
```

## Validated Commands (Offline)

- [x] Single-expiry `ProbCurve.from_chain(...)` snippet executed in `.venv`.
- [x] Multi-expiry `ProbSurface.from_chain(...)` snippet executed in `.venv`.
