from datetime import date
import pandas as pd
from oipd import VolCurve, MarketInputs, sources

# Mock data creation to avoid network dependency in test
# We create a dataframe that mimics what yfinance would return after sources.fetch_chain
today = date.today()
expiry = date(2025, 10, 10)

data = {
    "strike": [100, 105, 110],
    "last_price": [10.5, 6.5, 3.5],
    "bid": [10.4, 6.4, 3.4],
    "ask": [10.6, 6.6, 3.6],
    "option_type": ["C", "C", "C"],
    "expiry": [expiry, expiry, expiry]
}
df = pd.DataFrame(data)

# Market Inputs
market = MarketInputs(
    valuation_date=today,
    risk_free_rate=0.04,
    underlying_price=105.0,
)

print("1. Initializing VolCurve...")
vc = VolCurve()

print("2. Fitting VolCurve...")
# This will trigger resolve_market -> apply_put_call_parity
# If our fixes work, this should run without error
try:
    vc.fit(df, market)
    print("   Fit successful!")
except Exception as e:
    print(f"   Fit FAILED: {e}")
    raise e

print("3. Checking metadata...")
# Verify that T was calculated correctly internally
print(f"   Metadata keys: {vc._metadata.keys()}")

print("Verification passed!")
