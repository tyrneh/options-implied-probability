from oipd import (
    RND,
    MarketInputs,
    ModelParams,
    RNDSurface,
    rebuild_slice_from_svi,
    rebuild_surface_from_ssvi,
)

import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import time
import numpy as np


# read in csv and convert to dataframe
df_appl = pd.read_csv("data/AAPL_data.csv")

# filter appl for 2026-01-16 expiration date
df_appl_slice = df_appl[df_appl["expiration"] == "2026-01-16"]


# --- Example 1 - AAPL on a single expiry (slice) --- #


# 3 INPUTS:
# 1. market parameters
# 2. model  [optional]
# 3. column mapping

market = MarketInputs(
    valuation_date=date(2025, 10, 6),  # date pulled
    expiry_date=date(2026, 1, 16),
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


# estimate RND and time the operation
start_time = time.perf_counter()
est_appl = RND.from_dataframe(df_appl_slice, market, column_mapping=column_mapping)
end_time = time.perf_counter()
print(f"RND.from_dataframe executed in {end_time - start_time:.4f} seconds")

est_appl.plot(kind="pdf")
plt.show()


# plot fitted IV smile as 2D plot
est_appl.plot_iv(observations="range", x_axis="strike")
plt.show()


# return SVI parameters
svi_params = est_appl.svi_params()
print(svi_params)

# reconstruct vol curve and RND from SVI parameters
rebuild = rebuild_slice_from_svi(
    svi_params,
    forward_price=float(est_appl.meta["forward_price"]),
    days_to_expiry=est_appl.market.days_to_expiry,
    risk_free_rate=float(est_appl.market.risk_free_rate),
    strike_grid=est_appl.meta["vol_curve"].grid[0],
)
rebuild_df = rebuild.data  # THIS CONTAINS THE VOL, PDF, AND CDF
rebuild_df.head()

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

appl_surface = RNDSurface.from_dataframe(df_appl, market, column_mapping=column_mapping)

# plot IV surface as grid of maturities
appl_surface.plot_iv(x_axis="strike", layout="grid")
plt.show()

# plot IV surface as 3D plot
fig = appl_surface.plot_iv_3d()
fig.show()

# plot probability surface as 3D plot
fig = appl_surface.plot_probability_3d()
fig.show()

# return probability surface as dataframe
prob_surface = appl_surface.density_surface()
prob_surface.head()

# plot probability of prices over time as 2D plot
appl_surface.plot()
plt.show()

# return SSVI parameters
ssvi_params = appl_surface.ssvi_params()

# rebuild the surface from the SSVI parameters
# Rebuild every stored maturity slice from the SSVI parameter table
surface_rebuild = rebuild_surface_from_ssvi(
    ssvi_params,
    forwards=appl_surface.forward_levels(),
    risk_free_rate=float(market.risk_free_rate),
)

# Select the first calibrated maturity (expressed in year fractions)
first_maturity = float(ssvi_params["maturity"].iloc[0])

# Get the reconstructed data for the first maturity
first_rebuilt_slice = surface_rebuild.slice(first_maturity)
first_maturity_data = first_rebuilt_slice.data
first_maturity_data.head()

# get the reconstructed data for an arbitrary maturity
arbitrary_maturity = 0.3
arbitrary_rebuilt_data = surface_rebuild.slice(arbitrary_maturity).data
arbitrary_rebuilt_data.head()
