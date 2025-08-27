from oipd import RND, MarketInputs, ModelParams
import matplotlib.pyplot as plt
from datetime import date

# 1️⃣  Column mapping for CSV files
column_mapping = {
    "Strike": "strike",
    "Last": "last_price",
    "Bid": "bid",
    "Ask": "ask",
}

# ============================================
# CSV-based estimation (strict mode)
# ============================================

# For CSV files, you must provide spot_price and dividend info
market_csv = MarketInputs(
    spot_price=593.83,  # Required for CSV
    valuation_date=date(2025, 3, 3),
    expiry_date=date(2025, 5, 16),
    risk_free_rate=0.04,
    dividend_yield=0.01,  # Or dividend_schedule
)

model = ModelParams(fit_kde=True)

# Run estimation from CSV
result = RND.from_csv(
    "data/spy_date20250303_strike20250516_price59383.csv",
    market_csv,
    model=model,
    column_mapping=column_mapping,
)

print("CSV Result Summary:")
print(result.summary())
print()

result.plot()
plt.show()


# ============================================
# Ticker-based estimation with auto-fetching
# ============================================

# List available expiry dates
expiry_dates = RND.list_expiry_dates("SPY")
print(f"Available expiry dates: {expiry_dates[:5]}")

# For tickers, spot_price and dividends are optional (auto-fetched)
market_ticker = MarketInputs(
    valuation_date=date.today(),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04,
    # spot_price=None,  # Will be auto-fetched
    # dividend_yield=None,  # Will be auto-fetched
)

# Fetch and estimate - will print summary showing what was auto-fetched
result = RND.from_ticker("SPY", market_ticker, model=model)

# Access the resolved market parameters
print(f"\nResolved market parameters:")
print(
    f"  Spot price: ${result.market.spot_price:.2f} (source: {result.market.provenance.spot_price})"
)
print(f"  Dividends: {result.market.provenance.dividends}")
print(f"  Days to expiry: {result.market.days_to_expiry}")
print()

result.plot()
plt.show()
# ============================================
# Override vendor data with your own values
# ============================================

market_override = MarketInputs(
    valuation_date=date.today(),  # Required: analysis date
    spot_price=450.0,  # Your own price (overrides vendor)
    dividend_yield=0.02,  # Your own yield (overrides vendor)
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04,
)

result_override = RND.from_ticker(
    "SPY", market_override, model=model, echo=True  # Print summary of what was used
)

# ============================================
# Different fill modes
# ============================================

# Use only vendor data, ignore user inputs
result_vendor_only = RND.from_ticker(
    "SPY",
    market_ticker,
    model=model,
    fill="vendor_only",  # Ignore user inputs, use only vendor
    echo=True,
)

# ============================================
# Plotting with resolved parameters
# ============================================

# Access PDF/CDF data
print(f"PDF shape: {result.pdf.shape}")
print(f"CDF shape: {result.cdf.shape}")
print(f"Price range: ${result.prices.min():.2f} to ${result.prices.max():.2f}")

# Get probability at or above a price
prob = result.prob_at_or_above(450)
print(f"Probability SPY >= $450: {prob:.2%}")
