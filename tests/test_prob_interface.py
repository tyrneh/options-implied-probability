
import pytest
import pandas as pd
import numpy as np
from datetime import date

from oipd import MarketInputs
from oipd.interface.volatility import VolCurve
from oipd.interface.probability import Distribution, DistributionSurface

def test_distribution_interface():
    # 1. Load Data
    df_appl = pd.read_csv("tests/../data/AAPL_data.csv")
    df_slice = df_appl[df_appl["expiration"] == "2026-01-16"].copy()
    
    # 2. Setup Market
    market = MarketInputs(
        valuation_date=date(2025, 10, 6),
        expiry_date=date(2026, 1, 16),
        risk_free_rate=0.04,
        underlying_price=256.69,
    )
    
    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
    }
    
    # 3. Initialize and Fit Distribution directly
    dist = Distribution(method="svi", method_options={"random_seed": 42})
    
    # Verify state before fit
    with pytest.raises(ValueError, match="Call fit before"):
        _ = dist.pdf
        
    dist.fit(df_slice, market, column_mapping=column_mapping)
    
    # 4. Verify State After Fit
    assert dist.pdf is not None
    assert dist.cdf is not None
    assert dist.prices is not None
    
    # 5. Verify Probability Queries
    # Price is 256.69. Prob below 200 should be low, prob below 300 should be high.
    p_low = dist.prob_below(200.0)
    p_high = dist.prob_below(300.0)
    
    assert 0.0 <= p_low < p_high <= 1.0
    assert p_low < 0.4  # Rough check
    assert p_high > 0.6 # Rough check
    
    # 6. Verify Moments
    ev = dist.expected_value()
    # EV should be close to Forward price (approx Spot * e^(r*T))
    # T approx 0.28 years. F approx 256.69 * e^(0.04*0.28) ~= 259.5
    assert np.isclose(ev, 259.0, atol=5.0)

def test_vol_curve_to_distribution():
    # Verify the wiring from VolCurve -> Distribution
    
    # 1. Load Data
    df_appl = pd.read_csv("tests/../data/AAPL_data.csv")
    df_slice = df_appl[df_appl["expiration"] == "2026-01-16"].copy()
    
    market = MarketInputs(
        valuation_date=date(2025, 10, 6),
        expiry_date=date(2026, 1, 16),
        risk_free_rate=0.04,
        underlying_price=256.69,
    )
    
    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
    }
    
    # 2. Fit VolCurve
    vc = VolCurve(method="svi", method_options={"random_seed": 42})
    vc.fit(df_slice, market, column_mapping=column_mapping)
    
    # 3. Derive Distribution
    dist = vc.implied_distribution()
    
    assert isinstance(dist, Distribution)
    assert dist.pdf is not None
    
    # Check consistency
    ev = dist.expected_value()
    assert np.isclose(ev, 259.0, atol=5.0)

if __name__ == "__main__":
    test_distribution_interface()
    test_vol_curve_to_distribution()
    print("\nTest Passed!")
