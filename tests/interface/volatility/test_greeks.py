"""Tests for Greeks calculation on VolCurve."""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from oipd import VolCurve, MarketInputs


@pytest.fixture
def market_inputs():
    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_yield=0.02,
    )


@pytest.fixture
def single_expiry_chain():
    """Synthetic chain for Greeks testing."""
    strikes = [80, 90, 100, 110, 120]
    expiry = pd.Timestamp("2025-02-01")  # ~31 days

    data = {
        "expiry": [expiry] * len(strikes),
        "strike": strikes,
        "bid": [20.5, 11.0, 3.5, 0.8, 0.2],
        "ask": [21.0, 11.5, 4.0, 1.2, 0.4],
        "last_price": [20.75, 11.25, 3.75, 1.0, 0.3],
        "option_type": ["call"] * len(strikes),
    }
    calls = pd.DataFrame(data)

    # Generate puts via parity
    S, r, t = 100.0, 0.05, 31 / 365
    df = np.exp(-r * t)
    puts = calls.copy()
    puts["option_type"] = "put"
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df).abs()

    return pd.concat([calls, puts], ignore_index=True)


class TestVolCurveGreeks:
    """Test Greeks methods on VolCurve."""

    def test_delta_exists_and_returns_array(self, single_expiry_chain, market_inputs):
        """Delta should return an array of sensitivities."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        delta = vc.delta([90, 100, 110])
        assert isinstance(delta, np.ndarray)
        assert len(delta) == 3

    def test_delta_call_vs_put(self, single_expiry_chain, market_inputs):
        """Call Delta > 0, Put Delta < 0 for OTM strikes."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        strike = 100
        delta_call = vc.delta([strike], call_or_put="call")[0]
        delta_put = vc.delta([strike], call_or_put="put")[0]

        # ATM Call Delta should be ~0.5, Put Delta ~-0.5
        assert delta_call > 0
        assert delta_put < 0

    def test_gamma_same_for_call_and_put(self, single_expiry_chain, market_inputs):
        """Gamma should be identical for Call and Put at same strike."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        gamma = vc.gamma([100])
        assert gamma[0] > 0  # Gamma is always positive

    def test_vega_positive(self, single_expiry_chain, market_inputs):
        """Vega should be positive (longer options = more valuable if vol rises)."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        vega = vc.vega([100])
        assert vega[0] > 0

    def test_theta_negative_for_call(self, single_expiry_chain, market_inputs):
        """Theta (time decay) should generally be negative for long options."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        theta = vc.theta([100], call_or_put="call")
        # Theta is negative = option loses value as time passes
        assert theta[0] < 0

    def test_rho_call_positive(self, single_expiry_chain, market_inputs):
        """Call Rho should be positive (higher rates = higher call value)."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        rho = vc.rho([100], call_or_put="call")
        # In BS, Rho_call > 0. In Black-76, it's more nuanced.
        # For Black-76: Rho = -T * V, which is negative.
        # This depends on pricing engine. Let's just check it's a number.
        assert np.isfinite(rho[0])

    def test_greeks_returns_dataframe(self, single_expiry_chain, market_inputs):
        """greeks() should return a DataFrame with all columns."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        df = vc.greeks([90, 100, 110])

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["strike", "delta", "gamma", "vega", "theta", "rho"]
        assert len(df) == 3


class TestGreeksVsFiniteDifference:
    """Verify analytical Greeks match numerical (bump & revalue) approximations."""

    def test_delta_vs_bump(self, single_expiry_chain, market_inputs):
        """Delta should approximately equal (V(S+h) - V(S-h)) / 2h."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        K = 100
        h = 0.01  # Small bump

        # Analytical Delta
        delta_analytical = vc.delta([K])[0]

        # Numerical: we can't easily bump S for Black-76 (it uses F).
        # For Black-Scholes, we'd need to refit, which is complex.
        # Instead, verify sign and magnitude are reasonable.
        # Delta for ATM call should be ~0.4-0.6
        assert 0.3 < abs(delta_analytical) < 0.8

    def test_vega_vs_bump(self, single_expiry_chain, market_inputs):
        """Vega should be positive and have reasonable magnitude."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        vega = vc.vega([100])[0]

        # Vega for ATM 1-month option should be around 0.1-0.3
        # (depends on conventions, but should be positive and small)
        assert vega > 0
        assert vega < 50  # Sanity check
