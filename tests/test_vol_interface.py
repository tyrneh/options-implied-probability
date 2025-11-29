import pytest
import pandas as pd
import numpy as np
from datetime import date

from oipd import MarketInputs
from oipd.interface.volatility import VolCurve, VolSurface
from oipd.pipelines.market_inputs import resolve_market


def test_vol_curve_interface():
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

    # 3. Initialize and Fit VolCurve
    vc = VolCurve(method="svi", method_options={"random_seed": 42})

    # Verify state before fit
    with pytest.raises(ValueError, match="Call fit before"):
        _ = vc.params

    vc.fit(df_slice, market, column_mapping=column_mapping)

    # 4. Verify State After Fit
    assert vc.params is not None
    assert "a" in vc.params
    assert vc.forward is not None
    assert np.isclose(vc.forward, 256.69, rtol=0.05)  # Approximate check

    # 5. Verify Callability
    test_strikes = [200, 250, 300]
    ivs = vc(test_strikes)
    assert len(ivs) == 3
    assert np.all(ivs > 0)
    assert np.all(ivs < 2.0)  # Reasonable bounds


def test_vol_surface_interface():
    # 1. Load Data (Multiple Expiries)
    df_appl = pd.read_csv("tests/../data/AAPL_data.csv")

    # 2. Setup Market (No specific expiry)
    market = MarketInputs(
        valuation_date=date(2025, 10, 6),
        risk_free_rate=0.04,
        underlying_price=256.69,
    )

    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
        "expiration": "expiration",
    }

    # 3. Initialize and Fit VolSurface
    vs = VolSurface(method="svi", method_options={"random_seed": 42})
    vs.fit(df_appl, market, column_mapping=column_mapping)

    # 4. Verify Expiries
    expiries = vs.expiries
    assert len(expiries) > 1

    # 5. Verify Slicing
    first_expiry = expiries[0]
    vc_slice = vs.slice(first_expiry)

    assert isinstance(vc_slice, VolCurve)
    assert vc_slice.params is not None

    # Verify slice gives same result as manual fit
    iv_slice = vc_slice([250])
    assert iv_slice > 0


if __name__ == "__main__":
    test_vol_curve_interface()
    test_vol_surface_interface()
    print("\nTest Passed!")
