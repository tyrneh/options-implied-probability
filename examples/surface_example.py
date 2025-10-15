from oipd import RND, MarketInputs, ModelParams, RNDSurface, VolModel
from oipd.core.svi import svi_options

import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import time


# surface example
market = MarketInputs(
    valuation_date=date.today(),  # date pulled
    risk_free_rate=0.04,
)

est_appl = RNDSurface.from_ticker(
    "gme",
    market,
    horizon="1M",
    vol=VolModel(method="raw_svi"),
)

fig = est_appl.plot_iv()
plt.show()


# --- smile example --- #

# --- Example 1 - GME --- #
# --- using yfinance connection --- #

# 1. Get a list of available expiry dates
expiry_dates = RND.list_expiry_dates("GME")
print(expiry_dates[:])  # ['2025-09-05', '2025-09-12', '2025-09-19',...]

# 2. Use ticker data with market parameters (current price fetched automatically)
market = MarketInputs(
    valuation_date=date.today(),
    expiry_date=date(2025, 11, 14),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

model_gme = ModelParams(price_method="mid", max_staleness_days=None)

# 3. Fetch and estimate - auto-fetched data is available in the result
est_gamestop = RND.from_ticker("GME", market, model=model_gme)

# 4. Check the final market parameters used in estimation
est_gamestop.market

# 5. Plot using the result object
est_gamestop.plot_iv()
plt.show()
