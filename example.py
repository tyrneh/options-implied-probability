from oipd import RND, MarketInputs, ModelParams
import matplotlib.pyplot as plt
from datetime import date

# 1️⃣  what the library expects internally
#     strike , last_price , bid , ask
column_mapping = {
    "Strike": "strike",
    "Last": "last_price",
    "Bid": "bid",
    "Ask": "ask",
}

# 2️⃣  market parameters
market = MarketInputs(
    spot_price=593.83,  # current price of the underlying asset
    valuation_date=date(2025, 3, 3),
    expiry_date=date(2025, 5, 16),
    risk_free_rate=0.04,
)

# 3️⃣  optional model knobs (could omit)
model = ModelParams(fit_kde=True)

# 4️⃣  run
est = RND.from_csv(
    "data/spy_date20250303_strike20250516_price59383.csv",
    market,
    model=model,
    column_mapping=column_mapping,  # ← here
)

# ============================================
# NEW: Publication-ready plots with overlayed PDF/CDF
# ============================================

print("🎨 Creating publication-ready plots...")

# Default plot - overlays PDF and CDF with dual y-axes
fig = est.plot(
    figsize=(10, 6),
)
plt.show()

# PDF only
fig = est.plot(
    kind="pdf",
    figsize=(8, 6),
)
plt.show()

# CDF only
fig = est.plot(
    kind="cdf",
    figsize=(8, 6),
)
plt.show()

# ---- test prob at or above a price X ---- #
prob = est.prob_at_or_above(28)
print(prob)


# --- testing yfinance --- #
expiry_dates = RND.list_expiry_dates("SPY")
print(expiry_dates[:])  # ['2025-07-11', '2025-07-18', '2025-07-25']

# 2. Use ticker data with market parameters (current price fetched automatically)
market = MarketInputs(
    spot_price=None,  # Will be auto-fetched (MarketInputs stays immutable)
    dividend_yield=None,  # Will be auto-fetched (MarketInputs stays immutable)
    valuation_date=date(2025, 7, 10),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04,
)

model = ModelParams(fit_kde=True, solver="brent")

# 3. Fetch and estimate - auto-fetched data is available in the result
result = RND.from_ticker("SPY", market, model=model)

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
