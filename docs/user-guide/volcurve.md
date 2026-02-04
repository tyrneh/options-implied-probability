---
title: The VolCurve Interface
parent: User Guidege
nav_order: 1
---

## 3.1. The `VolCurve` Interface

The `VolCurve` is the workhorse of the `oipd` library. Its constructor allows you to configure the estimation process.

`VolCurve(method='svi', method_options=None, solver='brent', pricing_engine='black76', price_method='mid', max_staleness_days=3)`

*   **`method`**: The smile fitting algorithm.
    *   `'svi'`: (Default) The SVI (Stochastic Volatility Inspired) model. Recommended for most use cases.
    *   `'bspline'`: Fits a B-spline to the implied volatility data.
*   **`method_options`**: A dictionary of options for the chosen method. For SVI, this can include `random_seed`, `max_iter`, etc.
*   **`solver`**: The numerical solver for backing out implied volatility from option prices.
    *   `'brent'`: (Default) A robust root-finding algorithm.
    *   `'newton'`: Newton's method, which can be faster but less stable.
*   **`pricing_engine`**: The options pricing model to use.
    *   `'black76'`: (Default) The Black-76 model, which prices options on futures. It uses the forward price.
    *   `'bs'`: The Black-Scholes model, which prices options on the underlying stock. It requires dividend information.
*   **`price_method`**: Which price to use from the option chain data. Can be `'mid'`, `'last'`, `'bid'`, or `'ask'`.
*   **`max_staleness_days`**: The maximum age in days for an option quote to be considered valid.