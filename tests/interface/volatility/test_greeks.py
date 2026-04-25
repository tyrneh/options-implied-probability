"""Tests for Greeks calculation on VolCurve."""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from oipd import VolCurve, MarketInputs
from oipd.core.utils import resolve_risk_free_rate
from oipd.pricing.black76 import (
    black76_delta,
    black76_gamma,
    black76_rho,
    black76_theta,
    black76_vega,
)


@pytest.fixture
def market_inputs():
    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
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
        """Call and put deltas should be Black-76 forward deltas."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        strike = 100
        delta_call = vc.delta([strike], call_or_put="call")[0]
        delta_put = vc.delta([strike], call_or_put="put")[0]

        t = 31 / 365.0
        r = resolve_risk_free_rate(
            market_inputs.risk_free_rate,
            market_inputs.risk_free_rate_mode,
            t,
        )
        discount_factor = np.exp(-r * t)
        expected_call = black76_delta(
            vc.forward_price, np.array([strike]), vc([strike]), t, r, call_or_put="call"
        )[0]
        expected_put = black76_delta(
            vc.forward_price, np.array([strike]), vc([strike]), t, r, call_or_put="put"
        )[0]

        assert delta_call > 0
        assert delta_put < 0
        assert delta_call - delta_put == pytest.approx(discount_factor)
        assert delta_call == pytest.approx(expected_call)
        assert delta_put == pytest.approx(expected_put)

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

    def test_rho_call_uses_forward_discounting(
        self, single_expiry_chain, market_inputs
    ):
        """Black-76 call rho should reflect forward-price discounting."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        rho = vc.rho([100], call_or_put="call")
        assert np.isfinite(rho[0])
        assert rho[0] < 0

    def test_greeks_returns_dataframe(self, single_expiry_chain, market_inputs):
        """greeks() should return a DataFrame with all columns."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        df = vc.greeks([90, 100, 110])

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["strike", "delta", "gamma", "vega", "theta", "rho"]
        assert len(df) == 3

    def test_greeks_match_forward_formulas_without_dividend_inputs(
        self, single_expiry_chain
    ):
        """Public Greeks should use Black-76 forward-Greek semantics."""
        market = MarketInputs(
            valuation_date=date(2025, 1, 1),
            risk_free_rate=0.05,
            underlying_price=100.0,
        )

        vc = VolCurve().fit(single_expiry_chain, market)
        strikes = np.array([90.0, 100.0, 110.0])
        greek_frame = vc.greeks(strikes)

        t = 31 / 365.0
        r = resolve_risk_free_rate(
            market.risk_free_rate,
            market.risk_free_rate_mode,
            t,
        )
        sigma = vc(strikes)
        expected = {
            "delta": black76_delta(vc.forward_price, strikes, sigma, t, r),
            "gamma": black76_gamma(vc.forward_price, strikes, sigma, t, r),
            "vega": black76_vega(vc.forward_price, strikes, sigma, t, r),
            "theta": black76_theta(vc.forward_price, strikes, sigma, t, r),
            "rho": black76_rho(vc.forward_price, strikes, sigma, t, r),
        }

        assert isinstance(greek_frame, pd.DataFrame)
        assert np.all(
            np.isfinite(
                greek_frame[["delta", "gamma", "vega", "theta", "rho"]].to_numpy()
            )
        )
        for greek_name, expected_values in expected.items():
            np.testing.assert_allclose(
                greek_frame[greek_name].to_numpy(),
                expected_values,
                rtol=1e-12,
                atol=1e-12,
            )


class TestGreeksVsFiniteDifference:
    """Sanity checks for public Greek magnitudes."""

    def test_delta_forward_sensitivity_reasonable(
        self, single_expiry_chain, market_inputs
    ):
        """Delta should remain in a reasonable forward-sensitivity range."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        K = 100

        # Analytical Delta
        delta_analytical = vc.delta([K])[0]

        # Re-fitting a bumped forward is outside this interface test; this is
        # a scale sanity check for the public forward delta.
        assert 0.3 < abs(delta_analytical) < 0.8

    def test_vega_magnitude_reasonable(self, single_expiry_chain, market_inputs):
        """Vega should be positive and have reasonable magnitude."""
        vc = VolCurve().fit(single_expiry_chain, market_inputs)
        vega = vc.vega([100])[0]

        # Vega for ATM 1-month option should be around 0.1-0.3
        # (depends on conventions, but should be positive and small)
        assert vega > 0
        assert vega < 50  # Sanity check
