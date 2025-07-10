from oipd import RND, MarketParams, ModelParams
import matplotlib.pyplot as plt
from datetime import date

# 1️⃣  what the library expects internally
#     strike , last_price , bid , ask
column_mapping = {
    "Strike": "strike",
    "Last Price": "last_price",
    "Bid": "bid",
    "Ask": "ask",
}

# 2️⃣  market parameters
market = MarketParams(
    current_price=39.39,
    current_date=date(2025, 3, 3),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04,
)

# 3️⃣  optional model knobs (could omit)
model = ModelParams(fit_kde=True)

# 4️⃣  run
est = RND.from_csv(
    "data/ussteel_date20250303_strike20251219_price3939.csv",
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
expiry_dates = RND.list_expiry_dates("AAPL")
print(expiry_dates[:])  # ['2025-07-11', '2025-07-18', '2025-07-25']

# 2. Use ticker data with market parameters (current price fetched automatically)
market = MarketParams(
    current_price=None,  # Will be fetched automatically from yfinance
    current_date=date(2025, 7, 10),
    expiry_date=date(2026, 5, 15),
    risk_free_rate=0.04,
)

model = ModelParams(fit_kde=True)

# 3. Fetch and estimate directly from ticker (expiry comes from market params)
est = RND.from_ticker("AAPL", market, model=model)

# Plot with automatically fetched current price - no need to pass market_params!
fig = est.plot()
plt.show()

# You can also access the market parameters that were used (including fetched current price)
if est.market_params and est.market_params.current_price:
    print(f"Current price used: ${est.market_params.current_price:.2f}")
else:
    print("Current price not available")
