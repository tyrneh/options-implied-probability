from oipd import (
    VolCurve,
    VolSurface,
    MarketInputs,
    ModelParams,
)

import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import time
import numpy as np


# read in csv and convert to dataframe
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
    expiry_date=date(2025, 10, 10),
    risk_free_rate=0.04,
    underlying_price=256.69,  # closing price on 2025-10-06
)

column_mapping = {
    "strike": "strike",
    "last_price": "last_price",
    "type": "option_type",
    "bid": "bid",
    "ask": "ask",
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
# Default y_metric="total_variance", axis_mode="log_strike_over_forward"
appl_surface.plot(xlim=(-1, 1), ylim=(0, 0.1))



# plot IV surface as 3D plot
# fig = appl_surface.plot_iv_3d()
# fig.show()

# plot probability surface as 3D plot
# fig = appl_surface.plot_probability_3d()
# fig.show()

# return probability surface as dataframe
# prob_surface = appl_surface.density_surface()
# prob_surface.head()

# plot probability of prices over time as 2D plot
# appl_surface.plot()
# plt.show()

# return SSVI parameters
# ssvi_params = appl_surface.ssvi_params()
# VolSurface currently fits independent slices, so it might not have global SSVI params
# unless we implemented the SSVI fitting mode.

# rebuild the surface from the SSVI parameters
# Rebuild every stored maturity slice from the SSVI parameter table
# surface_rebuild = rebuild_surface_from_ssvi(
#     ssvi_params,
#     forward_prices=appl_surface.forward_levels(),
#     risk_free_rate=float(market.risk_free_rate),
# )

# Select the first slice maturity
# first_days = int(ssvi_params["days_to_expiry"].iloc[0])
# first_rebuilt_slice = surface_rebuild.slice(days_to_expiry=first_days)
# first_maturity_data = first_rebuilt_slice.data
# first_maturity_data.head()

# Select an arbitrary maturity
# rebuilt_slice_df = surface_rebuild.slice(days_to_expiry=100).data
# rebuilt_slice_df.head()

# Test Plotting
print("\nTesting Plotting...")
try:
    import matplotlib.pyplot as plt

    # 1. VolCurve Plot
    # Pick the first available expiry
    expiry_to_plot = appl_surface.expiries[0]
    vol_curve = appl_surface.slice(expiry=expiry_to_plot)
    fig_iv = vol_curve.plot(title=f"Fitted IV Smile ({expiry_to_plot.date()})")
    # plt.show() # Uncomment to see plot
    print("VolCurve.plot() successful")

    # 2. ProbCurve Plot
    prob = vol_curve.implied_distribution()
    fig_pdf = prob.plot(kind="pdf", title="Implied PDF (100 days)")
    # plt.show() # Uncomment to see plot
    print("ProbCurve.plot() successful")

except ImportError:
    print("Matplotlib not installed, skipping plotting tests")
except Exception as e:
    print(f"Plotting failed: {e}")
    raise




# --- Example 3 - NVDA on a single expiry (slice) --- #


# 3 INPUTS:
# 1. market parameters
# 2. model  [optional]
# 3. column mapping

market = MarketInputs(
    # valuation_date=date(2025, 10, 6),  # date pulled
    expiry_date=date(2025, 12, 16),
    risk_free_rate=0.04,
    # underlying_price=256.69,  # closing price on 2025-10-06
)

column_mapping = {
    "strike": "strike",
    "last_price": "last_price",
    "type": "option_type",
    "bid": "bid",
    "ask": "ask",
}
# initialize vol smile and fit
nvda_vol_curve = VolCurve()
nvda_vol_curve.fit(df_appl_slice, market, column_mapping=column_mapping)

appl_vol_curve._metadata

# plot vol smile
appl_vol_curve.plot(include_observed=True, xlim=(200, 300), ylim=(0, 1))

# get fitted vol as dataframe
fitted_vol_df = appl_vol_curve.iv_smile()
fitted_vol_df.head()


# return SVI parameters
svi_params = appl_vol_curve.params
print(svi_params)


# generate ProbCurve from vol smile
prob_appl = appl_vol_curve.implied_distribution()

prob_appl.plot(kind="both")