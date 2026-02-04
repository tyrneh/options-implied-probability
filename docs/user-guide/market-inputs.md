---
title: Market Inputs
parent: User Guide
nav_order: 4
---

## 3.4. Data Sources and Market Inputs

OIPD is designed to be data-source agnostic. The `fit` method expects a pandas DataFrame with the following columns (can be remapped using the `column_mapping` parameter):

*   `strike`: The strike price.
*   `expiry`: The expiration date.
*   `option_type`: `'call'` or `'put'`.
*   `last_price`: The last traded price.
*   `bid`: The bid price.
*   `ask`: The ask price.
*   `volume`: The trading volume.
*   `open_interest`: The open interest.
*   `last_trade_date`: The date of the last trade.

The `MarketInputs` class is crucial for providing the context for the valuation:

`MarketInputs(valuation_date, underlying_price, risk_free_rate, expiry_date=None, dividend_yield=0.0, dividend_schedule=None, risk_free_rate_mode='continuous')`