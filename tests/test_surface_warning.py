
import warnings
import pandas as pd
import numpy as np
from datetime import date
from oipd import VolSurface, VolCurve, MarketInputs
def test_curve_warning():
    print("\n--- Testing VolCurve Warning ---")
    # Setup single slice with missing mid
    df = pd.DataFrame({
        "strike": [100.0, 110.0, 120.0, 130.0, 140.0],
        "last_price": [10.0, 5.0, 2.0, 1.0, 0.5],
        "bid": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "ask": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "expiry": ["2025-01-01"] * 5,
        "option_type": ["call"] * 5
    })
    
    market = MarketInputs(
        valuation_date=date(2024, 12, 1),
        expiry_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0
    )
    
    column_mapping = {"strike": "strike", "last_price": "last_price", "bid": "bid", "ask": "ask", "type": "option_type"}

    print("Running VolCurve.fit()...")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        
        curve = VolCurve(price_method="mid")
        curve.fit(df, market, column_mapping=column_mapping)
        
        # Analyze
        mid_fill_warnings = [
            str(w.message) for w in captured 
            if "Filled missing mid prices with last_price due to unavailable bid/ask" in str(w.message)
        ]
        
    print(f"Caught {len(mid_fill_warnings)} 'Filled missing...' warnings.")
    if len(mid_fill_warnings) >= 1:
        print(f"Warning message: {mid_fill_warnings[0]}")
        print("SUCCESS: VolCurve emitted the expected warning.")
    else:
        print("FAILURE: VolCurve did NOT emit the warning.")



def test_surface_warning_aggregation():
    # 1. Setup Data
    # Expiry 1 (2025-01-01): Missing bid/ask, will trigger fill
    df1 = pd.DataFrame({
        "strike": [100.0, 110.0, 120.0, 130.0, 140.0],
        "last_price": [10.0, 5.0, 2.0, 1.0, 0.5],
        "bid": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "ask": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "expiry": ["2025-01-01"] * 5,
        "option_type": ["call"] * 5
    })

    # Expiry 2 (2025-02-01): Good data
    df2 = pd.DataFrame({
        "strike": [100.0, 110.0, 120.0, 130.0, 140.0],
        "last_price": [12.0, 6.0, 3.0, 1.5, 0.8],
        "bid": [11.0, 5.5, 2.5, 1.0, 0.5],
        "ask": [13.0, 6.5, 3.5, 2.0, 1.0],
        "expiry": ["2025-02-01"] * 5,
        "option_type": ["call"] * 5
    })
    
    # Expiry 3 (2025-03-01): Also missing bid/ask, to ensure aggregation counts correct
    df3 = pd.DataFrame({
        "strike": [100.0, 110.0, 120.0, 130.0, 140.0],
        "last_price": [15.0, 8.0, 4.0, 2.0, 1.0],
        "bid": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "ask": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "expiry": ["2025-03-01"] * 5,
        "option_type": ["call"] * 5
    })

    chain = pd.concat([df1, df2, df3], ignore_index=True)
    
    market = MarketInputs(
        valuation_date=date(2024, 12, 1),
        risk_free_rate=0.05,
        underlying_price=100.0
    )
    
    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "bid": "bid",
        "ask": "ask",
        "expiry": "expiry",
        "type": "option_type"
    }

    # 2. Run Fit and Capture Warnings
    # We expect warnings about 'price_method=mid' falling back to last, 
    # BUT we want to check specifically for the "Filled missing mid prices..." aggregation.
    
    print("Running VolSurface.fit()...")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always") # Cause all warnings to always be triggered.
        
        surface = VolSurface(price_method="mid") # Request mid to trigger fallback
        surface.fit(chain, market, column_mapping=column_mapping)
        
        # 3. Analyze Warnings
        mid_fill_warnings = [
            str(w.message) for w in captured 
            if "Filled missing mid prices with last_price" in str(w.message)
        ]
        
    print(f"\nCaught {len(captured)} total warnings.")
    print(f"Caught {len(mid_fill_warnings)} 'Filled missing...' warnings.")
    
    if len(mid_fill_warnings) > 0:
        print(f"Warning message: {mid_fill_warnings[0]}")

    # 4. Assertions
    # We expect exactly ONE warning because we aggregated them.
    # If we failed, we would see 2 (one for Jan, one for Mar).
    if len(mid_fill_warnings) == 1:
        print("\nSUCCESS: Warnings were aggregated into a single message.")
        # Check content
        if "2 expiries" in mid_fill_warnings[0]:
            print("SUCCESS: Message count is correct (2 expiries).")
        else:
            print(f"FAILURE: Message content incorrect. Expected '2 expiries', got: {mid_fill_warnings[0]}")
    elif len(mid_fill_warnings) == 0:
        print("\nFAILURE: No warnings caught. Logic might be over-suppressed.")
    else:
        print(f"\nFAILURE: Warnings were NOT aggregated. Count: {len(mid_fill_warnings)}")

if __name__ == "__main__":
    print("Starting tests...")
    try:
        test_surface_warning_aggregation()
        test_curve_warning()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
