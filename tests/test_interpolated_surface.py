"""Integration test for interpolated VolSurface."""

import warnings
import pandas as pd
from datetime import date
from oipd import VolSurface, MarketInputs


def test_interpolated_surface():
    print("\n--- Testing Interpolated VolSurface ---")

    # Load AAPL data
    try:
        df = pd.read_csv("data/AAPL_data.csv")
    except FileNotFoundError:
        print("Data file not found at data/AAPL_data.csv")
        return

    market = MarketInputs(
        valuation_date=date(2025, 10, 6),
        risk_free_rate=0.04,
        underlying_price=256.69,
    )

    mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
        "expiration": "expiry",  # CSV uses 'expiration', not 'expiry'
    }

    print("Fitting surface with interpolation='linear'...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surface = VolSurface()
        surface.fit(
            df, market, column_mapping=mapping, horizon="3m", interpolation="linear"
        )

    print(f"Fitted expiries: {len(surface.expiries)}")

    # Test 1: Interpolator exists
    print("Testing interpolator exists...")
    assert surface.interpolator is not None, "Interpolator should be available"
    print("  Interpolator: OK")

    # Test 2: Query ATM vol at arbitrary time
    print("Testing implied_vol at arbitrary time...")
    K = 256.69  # ATM
    t = 45 / 365.0  # 45 days
    iv = surface.implied_vol(K, t)
    assert 0.0 < iv < 2.0, f"IV out of range: {iv}"
    print(f"  sigma({K}, t={t:.3f}y) = {iv:.2%} OK")

    # Test 3: Total variance at arbitrary time
    print("Testing total_variance at arbitrary time...")
    w = surface.total_variance(K, t)
    assert w > 0, f"Total variance should be positive: {w}"
    print(f"  w({K}, t={t:.3f}y) = {w:.6f} OK")

    # Test 4: __call__ alias
    print("Testing __call__ alias...")
    iv_call = surface(K, t)
    assert iv_call == iv, f"__call__ should match implied_vol"
    print("  __call__: OK")

    # Test 5: Short-end extrapolation
    print("Testing short-end extrapolation...")
    iv_short = surface.implied_vol(K, 1 / 365.0)  # 1 day
    assert 0.0 < iv_short < 5.0, f"Short-end IV out of range: {iv_short}"
    print(f"  sigma({K}, t=1d) = {iv_short:.2%} OK")

    # Test 6: Long-end extrapolation
    print("Testing long-end extrapolation...")
    iv_long = surface.implied_vol(K, 2.0)  # 2 years
    assert 0.0 < iv_long < 5.0, f"Long-end IV out of range: {iv_long}"
    print(f"  sigma({K}, t=2y) = {iv_long:.2%} OK")

    print("\nInterpolated VolSurface: ALL TESTS PASSED")


if __name__ == "__main__":
    test_interpolated_surface()
