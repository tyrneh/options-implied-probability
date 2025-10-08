from oipd import RND, MarketInputs, ModelParams
from oipd.core.svi import svi_options

import matplotlib.pyplot as plt
from datetime import date
import pandas as pd

# --- Example 1 - AAPL --- #

# read in csv and convert to dataframe
df_appl = pd.read_csv("data/AAPL_data.csv")

# filter appl for 2026-01-16 expiration date
df_appl = df_appl[df_appl["expiration"] == "2026-01-16"]


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

# estimate RND
est_appl = RND.from_dataframe(df_appl, market, column_mapping=column_mapping)

est_appl.plot(kind="pdf")
plt.show()


# fetch fitted IV smile from results object
est_appl.plot(
    "iv_smile",
    title="AAPL IV Smile using SVI, expiry 2026-01-16, data from Yahoo Finance",
)
plt.show()

# get the fitted IV smile as a dataframe
# columns: strike, fitted_iv, bid_iv, ask_iv, last_iv
# bid_iv, ask_iv, last_iv are the observed implied volatilities from the raw bids, asks, and last prices
est_appl.iv_smile()


est_appl.to_frame()



