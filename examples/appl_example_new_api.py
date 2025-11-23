"""
New API Example - AAPL
======================

This script replicates the logic of `appl_example.py` using the new modular API.
It demonstrates:
1.  Data Loading via `oipd.sources`
2.  Single Expiry Analysis (`VolCurve` -> `Distribution`)
3.  Multi-Expiry Analysis (`VolSurface` -> `DistributionSurface`)
"""

import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import time
import numpy as np

from oipd import sources, MarketInputs
from oipd.interface.volatility import VolCurve, VolSurface
from oipd.interface.probability import Distribution, DistributionSurface


# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
# Map legacy column names to new standard
column_mapping = {
    "strike": "strike",
    "last_price": "last_price",
    "type": "option_type",
    "bid": "bid",
    "ask": "ask",
    "expiration": "expiry"  # New standard name
}

df_appl = sources.from_csv("data/AAPL_data.csv", column_mapping=column_mapping)

# Filter for single expiry slice
target_expiry = "2026-01-16"
df_appl_slice = df_appl[df_appl["expiry"] == target_expiry].copy()

# ---------------------------------------------------------
# Example 1 - AAPL on a single expiry (slice)
# ---------------------------------------------------------
market = MarketInputs(
    valuation_date=date(2025, 10, 6),
    expiry_date=date(2026, 1, 16),
    risk_free_rate=0.04,
    underlying_price=256.69,
)

# Initialize and Fit VolCurve
vc = VolCurve(method="svi")
vc.fit(df_appl_slice, market)

# Derive Distribution
dist = vc.implied_distribution()

# Plotting (Note: New plotting API is not fully implemented yet, 
# so we use standard matplotlib for now, accessing the raw arrays)
plt.figure(figsize=(10, 6))
plt.plot(dist.prices, dist.pdf, label="Risk-Neutral PDF")
plt.title(f"AAPL PDF for {target_expiry}")
plt.xlabel("Price")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plotting IV Smile
strikes = np.linspace(df_appl_slice["strike"].min(), df_appl_slice["strike"].max(), 100)
ivs = vc(strikes)

plt.figure(figsize=(10, 6))
plt.plot(strikes, ivs, label="Fitted SVI")
# We can access observed IVs from metadata if available
# Note: diagnostics might be a DataFrame directly
obs_iv = vc.diagnostics
if obs_iv is not None and isinstance(obs_iv, pd.DataFrame) and "strike" in obs_iv.columns:
    plt.scatter(obs_iv["strike"], obs_iv["iv"], alpha=0.5, label="Market Data", color="red", s=10)

plt.title(f"AAPL Volatility Smile for {target_expiry}")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Return SVI Parameters
print(vc.params)

# ---------------------------------------------------------
# Example 2 - AAPL surface (multiple expiries)
# ---------------------------------------------------------

market_surface = MarketInputs(
    valuation_date=date(2025, 10, 6),
    risk_free_rate=0.04,
    underlying_price=256.69,
)

# Initialize and Fit VolSurface
vs = VolSurface(method="svi")
vs.fit(df_appl, market_surface)

# Derive DistributionSurface
ds = vs.implied_distribution()

# Plotting IV Surface (Grid)
# Ideally we would use a dedicated plot method, but here is a simple loop
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, expiry in enumerate(vs.expiries[:4]):
    ax = axes[i]
    vc_slice = vs.slice(expiry)
    
    # Grid for plotting
    strikes = np.linspace(200, 350, 50)
    ivs = vc_slice(strikes)
    
    ax.plot(strikes, ivs, label="Fitted")
    ax.set_title(f"Expiry: {expiry.date()}")
    ax.grid(True, alpha=0.3)
    
plt.tight_layout()
plt.show()

# Accessing Distribution Slice
dist_slice = ds.slice("2026-01-16")
print(f"Expected Value: {dist_slice.expected_value():.2f}")
print(f"Prob < 250: {dist_slice.prob_below(250):.2%}")
