from oipd import (
    VolCurve,
    VolSurface,
    MarketInputs,
    ModelParams,
    sources,
)

import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import time
import numpy as np


# read in csv and convert to dataframe
# Note: You can pass a DataFrame directly to VolCurve/VolSurface.fit.
# Use oipd.sources.from_csv/from_dataframe when you want standardized cleaning/validation.
df_appl = pd.read_csv("data/AAPL_data.csv")

# filter appl for 2026-01-16 expiration date
df_appl_slice = df_appl[df_appl["expiration"] == "2025-10-10"]


# --- Example 1 - AAPL on a single expiry (slice) --- #


# 3 INPUTS:
# 1. market parameters
# 2. model  [optional]
# 3. column mapping

market = MarketInputs(
    valuation_date=date(2025, 10, 6),  # date pulled
    risk_free_rate=0.04,
    underlying_price=256.69,  # closing price on 2025-10-06
)

column_mapping = {
    "strike": "strike",
    "last_price": "last_price",
    "type": "option_type",
    "bid": "bid",
    "ask": "ask",
    "expiration": "expiry",
}
# initialize vol smile and fit
appl_vol_curve = VolCurve()
appl_vol_curve.fit(df_appl_slice, market, column_mapping=column_mapping)

appl_vol_curve._metadata

# plot vol smile
appl_vol_curve.plot(include_observed=True) #, xlim=(200, 300), ylim=(0, 1))
plt.show()

# get fitted vol as dataframe
fitted_vol_df = appl_vol_curve.iv_smile()
fitted_vol_df.head()


# return SVI parameters
svi_params = appl_vol_curve.params
print(svi_params)


# generate ProbCurve from vol smile
prob_appl = appl_vol_curve.implied_distribution()

prob_appl.plot(kind="both")


# --- Example 2 - AAPL surface (multiple expiries) --- #

# 3 INPUTS:
# 1. market parameters
# 2. model  [optional]
# 3. column mapping


market = MarketInputs(
    valuation_date=date(2025, 10, 6),  # date pulled
    risk_free_rate=0.04,
    underlying_price=256.69,  # closing price on 2025-10-06
)

column_mapping = {
    "strike": "strike",
    "last_price": "last_price",
    "type": "option_type",
    "bid": "bid",
    "ask": "ask",
    "expiration": "expiry",
}

appl_surface = VolSurface(method="svi")
appl_surface.fit(df_appl, market, column_mapping=column_mapping, horizon="3m")

# plot IV surface as grid of maturities

# Plot the implied volatility surface (2D overlay)
# Default y_axis="total_variance", x_axis="log_moneyness"
appl_surface.plot(xlim=(-1, 1), ylim=(0, 0.1))



### ----------------------------------------
# Example 3 - fetching from Yfinance 
### ----------------------------------------

# --- using unified sources.fetch_chain --- #

# 1. Fetch Option Chain (Single-Expiry)
# You can pass a string date, a date object, or a list of them.
sources.list_expiry_dates("GME") 
# returns a list of expiry dates:
"""
['2026-01-30',
 '2026-02-06',
 '2026-02-13',
 '2026-02-20',
 '2026-02-27',
 '2026-03-06',
 '2026-03-20',
 '2026-04-17',
 '2026-06-18',
 '2026-07-17',
 '2026-09-18',
 '2026-10-16',
 '2026-12-18',
 '2027-01-15',
 '2027-09-17',
 '2027-12-17',
 '2028-01-21',
 '2028-06-16',
 '2028-12-15']
"""

gme_expiry = sources.list_expiry_dates("GME")[0] # get the first expiry date
gme_chain, gme_snapshot = sources.fetch_chain("GME", expiries=gme_expiry)

# MarketInputs derives "now" and spot price from the snapshot
gme_market = MarketInputs(
    underlying_price=gme_snapshot.underlying_price,
    valuation_date=gme_snapshot.date,
    risk_free_rate=0.04,
)

gme_curve = VolCurve().fit(gme_chain, gme_market)

gme_curve.plot(include_observed=True)
plt.show()


# 2. Fetch Option Surface (Multi-Expry)
# No manual loops needed. Just pass a list of expiries.

# List available dates first if needed
all_dates = sources.list_expiry_dates("GME")
surface_dates = all_dates[:3]

# Fetch everything in one go
gme_surface_chain, gme_surface_snapshot = sources.fetch_chain("GME", expiries=surface_dates)

gme_surface_market = MarketInputs(
    valuation_date=gme_surface_snapshot.date,
    risk_free_rate=0.04,
    underlying_price=gme_surface_snapshot.underlying_price,
)

# Fitting is identical to single-expiry, just use VolSurface
gme_surface = VolSurface().fit(gme_surface_chain, gme_surface_market, )

gme_surface.plot(xlim=(-1, 1), ylim=(0, 0.1))
plt.show()
