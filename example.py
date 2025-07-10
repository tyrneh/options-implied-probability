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
    current_price=62.76,
    current_date=date(2025, 7, 8),
    expiry_date=date(2025, 11, 17),
    risk_free_rate=0.04,
)

# 3️⃣  optional model knobs (could omit)
model = ModelParams(fit_kde=True)

# 4️⃣  run
est = RND.from_csv(
    "data/Calls Crude Oil WTI Dec 2026 Options Prices - 11-17-2026 CurrentPrice 6276.csv",
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
    source="Source: Options market data",
)
plt.show()

# CDF only
fig = est.plot(
    kind="cdf",
    market_params=market,
    figsize=(8, 6),
    source="Source: Options market data",
)
plt.show()
