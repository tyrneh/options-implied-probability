from oipd import RND, MarketInputs, ModelParams
import matplotlib.pyplot as plt
from datetime import date

# 1️⃣  what the library expects internally
#     strike , last_price , bid , ask
column_mapping_sp500 = {
    "Strike": "strike",
    "Price": "last_price",
    "Bid": "bid",
    "Ask": "ask",
}

# 2️⃣  market parameters
market_sp500 = MarketInputs(
    spot_price=6460.26,  # current price of the underlying asset
    valuation_date=date(2025, 8, 30),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_sp500 = ModelParams(fit_kde=False, price_method="last")

# 4️⃣  run using S&P500 e-mini futures options chain
est_sp500 = RND.from_csv(
    "data/s-p-futures_date20250830_strike20251219_price646026.csv",
    market_sp500,
    model=model_sp500,
    column_mapping=column_mapping_sp500,
)

# PDF only
fig = est_sp500.plot(
    kind="pdf", figsize=(6, 4), title="S&P500", source="S&P500 e-mini futures options"
)
plt.show()

# ---- test prob at or above a price X ---- #
prob = est_sp500.prob_at_or_above(6500)
print(prob)

prob = est_sp500.prob_below(6500)
print(prob)

# ============================================
# Crude Oil
# ============================================

#     strike , last_price , bid , ask
column_mapping_wti = {
    "Strike": "strike",
    "Last": "last_price",
    "Bid": "bid",
    "Ask": "ask",
}

# 2️⃣  market parameters
market_wti = MarketInputs(
    spot_price=64.01,  # current price of the underlying asset
    valuation_date=date(2025, 8, 30),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_wti = ModelParams(fit_kde=False, price_method="last", max_staleness_days=None)

# 4️⃣  run using S&P500 e-mini futures options chain
est_wti = RND.from_csv(
    "data/CrudeWTICalls_date20250830_strike20251216_price6401.csv",
    market_wti,
    model=model_wti,
    column_mapping=column_mapping_wti,
)

# PDF only
fig = est_wti.plot(
    kind="pdf",
    figsize=(8, 6),
    xlim=(50, 110),
)
plt.show()


# ============================================
# Bitcoin
# ============================================

#     strike , last_price , bid , ask
column_mapping_bitcoin = {
    "Strike": "strike",
    "Last": "last_price",
    "Bid": "bid",
    "Ask": "ask",
}

# 2️⃣  market parameters
market_bitcoin = MarketInputs(
    spot_price=108864,  # current price of the underlying asset
    valuation_date=date(2025, 8, 30),
    expiry_date=date(2025, 12, 26),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_bitcoin = ModelParams(fit_kde=False, price_method="last", max_staleness_days=None)

# 4️⃣  run using S&P500 e-mini futures options chain
est_bitcoin = RND.from_csv(
    "data/BitcoinCalls_date20250830_strike20251226_price108864.csv",
    market_bitcoin,
    model=model_bitcoin,
    column_mapping=column_mapping_bitcoin,
)

# PDF only
fig = est_bitcoin.plot(
    kind="pdf",
    figsize=(8, 6),
    xlim=(0, 200000),
)
plt.show()

# ============================================
# YFINANCE
# ============================================

# --- testing yfinance --- #
expiry_dates = RND.list_expiry_dates("NVDA")
print(expiry_dates[:])  # '2025-09-05', '2025-09-12', '2025-09-19',...]

# 2. Use ticker data with market parameters (current price fetched automatically)
market = MarketInputs(
    valuation_date=date(2025, 9, 1),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04199,
)

model = ModelParams(fit_kde=True, solver="brent")

# 3. Fetch and estimate - auto-fetched data is available in the result
result = RND.from_ticker("NVDA", market, model=model)

# 4. Access auto-fetched data through the result (NOT through market!)
print(f"Fetched spot price: ${result.market.spot_price:.2f}")
print(f"Fetched dividend yield: {result.market.dividend_yield}")
print(f"Data sources: {result.summary()}")

# 5. Plot using the result object
result.plot()
plt.show()

result.plot(kind="pdf")
plt.show()

result.plot()
plt.show()

# Note: result.plot() uses the resolved market parameters from the calculation.
# The original MarketInputs object remains immutable and unchanged.
# Auto-fetched values are always accessed through result.market, not the original market object.

result.to_frame()
