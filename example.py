from oipd import RND, MarketInputs, ModelParams
import matplotlib.pyplot as plt
from datetime import date

# 1️⃣  what the library expects internally
#     strike , last_price , bid , ask
column_mapping_gamestop = {
    "Strike": "strike",
    "Last": "last_price",
    "Bid": "bid",
    "Ask": "ask",
    "OptionType": "option_type",
}

# 2️⃣  market parameters
market_gamestop = MarketInputs(
    spot_price=23.235,  # current price of the underlying asset
    valuation_date=date(2025, 9, 8),
    expiry_date=date(2026, 1, 16),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_gamestop = ModelParams(fit_kde=False, price_method="mid")

# 4️⃣  run using Gamestop options chain
est_gamestop = RND.from_csv(
    "data/GME_date250908exp260116_current23235.csv",
    market_gamestop,
    model=model_gamestop,
    column_mapping=column_mapping_gamestop,
)

# PDF only
fig = est_gamestop.plot(kind="pdf", figsize=(6, 4), source="Gamestop options")
plt.show()

# ---- test prob at or above a price X ---- #
prob = est_gamestop.prob_at_or_above(120)
print(prob)

prob = est_gamestop.prob_below(150)
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
    "Type": "option_type",
}

# 2️⃣  market parameters
market_wti = MarketInputs(
    spot_price=61.69,  # current price of the underlying asset
    valuation_date=date(2025, 9, 8),
    expiry_date=date(2025, 12, 16),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_wti = ModelParams(
    solver="brent", fit_kde=False, price_method="mid", max_staleness_days=None
)

# 4️⃣  run using S&P500 e-mini futures options chain
est_wti = RND.from_csv(
    "data/WTIfutures_date250908exp251216_spot6169.csv",
    market_wti,
    model=model_wti,
    column_mapping=column_mapping_wti,
)

# PDF only
fig = est_wti.plot(
    kind="pdf",
    figsize=(8, 6),
    xlim=(40, 140),
)
plt.show()

prob = est_wti.prob_below(200)
print(prob)


# ============================================
# Bitcoin
# ============================================

# #     strike , last_price , bid , ask
# column_mapping_bitcoin = {
#     "Strike": "strike",
#     "Last": "last_price",
#     "Bid": "bid",
#     "Ask": "ask",
#     "OptionType": "option_type",
# }

# # 2️⃣  market parameters
# market_bitcoin = MarketInputs(
#     spot_price=108864,  # current price of the underlying asset
#     valuation_date=date(2025, 8, 30),
#     expiry_date=date(2025, 12, 26),
#     risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
# )

# # 3️⃣  optional model knobs (could omit)
# model_bitcoin = ModelParams(fit_kde=False, price_method="mid", max_staleness_days=None)

# # 4️⃣  run using S&P500 e-mini futures options chain
# est_bitcoin = RND.from_csv(
#     "data/bitcoin_date20250830_strike20251226_price108864.csv",
#     market_bitcoin,
#     model=model_bitcoin,
#     column_mapping=column_mapping_bitcoin,
# )

# # PDF only
# fig = est_bitcoin.plot(
#     kind="pdf",
#     figsize=(8, 6),
#     xlim=(0, 200000),
# )
# plt.show()

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
