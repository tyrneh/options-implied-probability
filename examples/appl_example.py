from oipd import (
    VolCurve,
    VolSurface,
    MarketInputs,
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
fitted_vol_df = appl_vol_curve.iv_results()
fitted_vol_df.head()


# return SVI parameters
svi_params = appl_vol_curve.params
print(svi_params)


# generate ProbCurve from vol smile
prob_appl = appl_vol_curve.implied_distribution()

prob_appl.plot(kind="both")


# 

# --- Pricing Demo (VolCurve) --- #
# Price a theoretical option at Strike=250 using the fitted curve
theoretical_price = appl_vol_curve.price(strikes=[250.0], call_or_put="call")
print(f"Theoretical Call Price (K=250): {theoretical_price[0]:.4f}")

theoretical_put = appl_vol_curve.price(strikes=[250.0], call_or_put="put")
print(f"Theoretical Put Price (K=250): {theoretical_put[0]:.4f}")


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

# Return the IV at an arbitrary strike price and maturity
iv_t = appl_surface.implied_vol(K=100, t=0.1)

appl_surface.plot_term_structure()

appl_surface.plot_3d()


# convert the surface to a probability surface
prob_surface = appl_surface.implied_distribution()
prob_surface.plot()



# --- Pricing Demo (VolSurface) --- #
# Price an option at an arbitrary time (interpolated)
# e.g., K=100, t=0.1 years
surface_price = appl_surface.price(strikes=[100], t=0.1, call_or_put="call")
print(f"Interpolated Surface Price (K=100, t=0.1y): {surface_price[0]:.4f}")

# get a VolCurve slice at an expiry
appl_slice = appl_surface.slice(expiry="2025-12-19")
#plot the slice
appl_slice.plot()
appl_slice.implied_distribution().plot(kind="both")

# get a VolCurve slice at an ARBITRARY expiry
appl_slice = appl_surface.slice(expiry="2025-12-10")
#plot the slice
appl_slice.plot()
appl_slice.implied_distribution().plot(kind="both")





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

# get a slice
gme_slice_2 = gme_surface.slice(expiry="2026-02-06")
gme_slice_2.plot(include_observed=True)
plt.show()

# get a second slice
gme_slice_3 = gme_surface.slice(expiry="2026-02-13")
gme_slice_3.plot(include_observed=True)
plt.show()

