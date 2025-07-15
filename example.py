from oipd import RND, MarketParams, ModelParams
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
market = MarketParams(
    current_price=64.32,
    current_date=date(2025, 3, 3),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04,
)

# 3️⃣  optional model knobs (could omit)
model = ModelParams(fit_kde=True)

# 4️⃣  run
est = RND.from_csv(
    "data/Calls CrudeOilWTI Strike11-17-25 CurrentPrice6432 .csv",
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
    market_params=market,
    figsize=(10, 6),
)
plt.show()

# PDF only
fig = est.plot(
    kind="pdf",
    market_params=market,
    figsize=(8, 6),
)
plt.show()

# CDF only
fig = est.plot(
    kind="cdf",
    market_params=market,
    figsize=(8, 6),
)
plt.show()

# ---- test prob at or above a price X ---- #
prob = est.prob_at_or_above(28)
print(prob)


# --- testing yfinance --- #
expiry_dates = RND.list_expiry_dates("PLTR")
print(expiry_dates[:])  # ['2025-07-11', '2025-07-18', '2025-07-25']

# 2. Use ticker data with market parameters (current price fetched automatically)
market = MarketParams(
    current_price=None,  # Will be auto-fetched and this object will be updated
    dividend_yield=None,  # Will be auto-fetched and this object will be updated
    current_date=date(2025, 7, 10),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04,
)

model = ModelParams(fit_kde=True, solver="brent")

# 3. Fetch and estimate - this will update market.current_price automatically
est = RND.from_ticker("PLTR", market, model=model)

# 4. Now market.current_price has been set, so you can use it anywhere
print(f"Fetched current price: ${market.current_price:.2f}")
print(market.dividend_yield)


# 5. Plot works perfectly with the same market object
est.plot(market_params=market)
plt.show()

est.plot(kind="pdf", market_params=market)
plt.show()

est.plot()
plt.show()

# Both approaches work now:
# - est.plot() uses stored market parameters automatically
# - est.plot(market_params=market) uses the original (now updated) market object
