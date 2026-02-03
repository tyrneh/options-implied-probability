"""Tests for Greeks calculation on VolSurface."""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from oipd import VolSurface, MarketInputs


@pytest.fixture
def market_inputs():
    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_yield=0.02,
    )


@pytest.fixture
def multi_expiry_chain():
    """Synthetic chain with 2 expiries for Surface fitting."""
    # Expiry 1: 30 days
    expiry1 = pd.Timestamp("2025-01-31")
    strikes = [80, 90, 100, 110, 120]
    
    # Simple smile data for 30d
    calls1 = pd.DataFrame({
        "expiry": [expiry1] * len(strikes),
        "strike": strikes,
        "bid": [20.5, 11.0, 3.5, 0.8, 0.2],
        "ask": [21.0, 11.5, 4.0, 1.2, 0.4],
        "last_price": [20.75, 11.25, 3.75, 1.0, 0.3],
        "option_type": ["call"] * len(strikes),
    })

    # Expiry 2: 90 days
    expiry2 = pd.Timestamp("2025-04-01")
    # Higher prices due to more time value
    calls2 = pd.DataFrame({
        "expiry": [expiry2] * len(strikes),
        "strike": strikes,
        "bid": [22.5, 13.0, 5.5, 1.8, 0.6],
        "ask": [23.0, 13.5, 6.0, 2.2, 0.8],
        "last_price": [22.75, 13.25, 5.75, 2.0, 0.7],
        "option_type": ["call"] * len(strikes),
    })

    calls = pd.concat([calls1, calls2], ignore_index=True)

    # Generate puts via parity for ALL calls
    # Note: Using simple global r, q approximation for generation
    S = 100.0
    r = 0.05
    # calculate t for each row
    t_array = (calls["expiry"] - pd.Timestamp("2025-01-01")).dt.days / 365.0
    df_array = np.exp(-r * t_array)
    
    puts = calls.copy()
    puts["option_type"] = "put"
    # P = C - S + K * df
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df_array).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df_array).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df_array).abs()

    return pd.concat([calls, puts], ignore_index=True)


class TestVolSurfaceGreeks:
    """Test Greeks methods on VolSurface."""

    def test_surface_delta_interpolated(self, multi_expiry_chain, market_inputs):
        """Surface should calculate Delta at arbitrary time t."""
        vs = VolSurface().fit(multi_expiry_chain, market_inputs)
        
        # Test at t=60 days (interpolated between 30 and 90)
        t_target = 60 / 365.0
        strikes = [90, 100, 110]
        
        delta = vs.delta(strikes, t=t_target)
        
        assert isinstance(delta, np.ndarray)
        assert len(delta) == 3
        # Basic check: Call delta is positive
        assert np.all(delta > 0)
        # Deep ITM (90) > ATM (100) > OTM (110)
        assert delta[0] > delta[1] > delta[2]

    def test_surface_gamma(self, multi_expiry_chain, market_inputs):
        """Gamma should be positive."""
        vs = VolSurface().fit(multi_expiry_chain, market_inputs)
        gamma = vs.gamma([100], t=0.2)
        assert gamma[0] > 0

    def test_surface_theta_decay(self, multi_expiry_chain, market_inputs):
        """Theta should be negative for calls."""
        vs = VolSurface().fit(multi_expiry_chain, market_inputs)
        theta = vs.theta([100], t=0.2, call_or_put="call")
        assert theta[0] < 0

    def test_surface_greeks_dataframe(self, multi_expiry_chain, market_inputs):
        """greeks() should return DataFrame with correct columns."""
        vs = VolSurface().fit(multi_expiry_chain, market_inputs)
        df = vs.greeks([100], t=0.2)
        
        assert isinstance(df, pd.DataFrame)
        expected_cols = ["strike", "delta", "gamma", "vega", "theta", "rho"]
        assert list(df.columns) == expected_cols
        assert len(df) == 1

    def test_black_scholes_engine(self, multi_expiry_chain, market_inputs):
        """Test BS engine execution path."""
        # Use BS engine
        vs = VolSurface(pricing_engine="bs").fit(multi_expiry_chain, market_inputs)
        
        # Should work without error
        delta = vs.delta([100], t=0.2)
        assert np.isfinite(delta[0])
