"""Example script showcasing the RND workflow with the SVI surface fit."""

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

from oipd import ModelParams, RND
from oipd.market_inputs import MarketInputs
from oipd.core.svi import svi_options

# --- Example 3 - Bitcoin --- #
# options data from Deribit BTC-USDC
# https://www.deribit.com/options/BTC_USDC

#     strike , last_price , bid , ask
column_mapping_bitcoin = {
    "Strike": "strike",
    "Last_USD": "last_price",
    "Bid_USD": "bid",
    "Ask_USD": "ask",
    "OptionType": "option_type",
}

# 2️⃣  market parameters
market_bitcoin = MarketInputs(
    underlying_price=108864,  # current price of the underlying instrument
    valuation_date=date(2025, 8, 30),
    expiry_date=date(2025, 12, 26),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

custom_svi = svi_options(max_iter=800, tol=1e-9, regularisation=1e-6)

model_bitcoin = ModelParams(
    price_method="mid",
    max_staleness_days=None,
    surface_method="svi",
    surface_options=custom_svi,
)

# 4️⃣  run using S&P500 e-mini futures options chain
est_bitcoin = RND.from_csv(
    "data/bitcoin_date20250830_strike20251226_price108864.csv",
    market_bitcoin,
    model=model_bitcoin,
    column_mapping=column_mapping_bitcoin,
)

est_bitcoin.prob_at_or_above(100000)


# PDF only
fig = est_bitcoin.plot(
    kind="pdf",
    figsize=(8, 6),
    title="Implied Bitcoin price on Dec 26, 2025",
)
plt.show()

est_bitcoin.prob_at_or_above(300000)
