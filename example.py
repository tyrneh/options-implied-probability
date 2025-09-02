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

# Default plot - overlays PDF and CDF with dual y-axes
fig = est_sp500.plot(
    figsize=(10, 6),
)
plt.show()

# PDF only
fig = est_sp500.plot(
    kind="pdf",
    figsize=(8, 6),
)
plt.show()

# CDF only
fig = est_sp500.plot(
    kind="cdf",
    figsize=(8, 6),
)
plt.show()

# ---- test prob at or above a price X ---- #
prob = est_sp500.prob_at_or_above(6500)
print(prob)

prob = est_sp500.prob_below(6500)
print(prob)


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
