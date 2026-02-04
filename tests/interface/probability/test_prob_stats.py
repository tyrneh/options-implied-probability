"""Tests for higher order statistics on ProbCurve."""

import numpy as np
import pandas as pd
import pytest
from datetime import date
from scipy.stats import norm

from oipd import VolCurve, MarketInputs, ProbCurve


@pytest.fixture
def market_inputs():
    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.0,  # Zero rate simplifies things (Forward = Spot)
        underlying_price=100.0,
        dividend_yield=0.0,
    )


@pytest.fixture
def flat_vol_chain():
    """Synthetic chain with perfectly flat volatility (Gaussian Distribution)."""
    strikes = np.linspace(50, 150, 21)
    expiry = pd.Timestamp("2025-02-01")  # ~30 days

    # Generate prices using Black-Scholes with constant vol
    # This implies the implied distribution should be Lognormal (approx Normal for short tenor)
    vol = 0.20
    S = 100.0
    r = 0.0
    T = 31 / 365.0

    calls = []
    puts = []

    for K in strikes:
        d1 = (np.log(S / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        c = S * norm.cdf(d1) - K * norm.cdf(d2)
        calls.append(c)
        puts.append(c - S + K)  # Parity for r=0

    df = pd.DataFrame(
        {
            "expiry": [expiry] * len(strikes),
            "strike": strikes,
            "last_price": calls,
            "option_type": ["call"] * len(strikes),
            "bid": calls,
            "ask": calls,
        }
    )

    df_puts = pd.DataFrame(
        {
            "expiry": [expiry] * len(strikes),
            "strike": strikes,
            "last_price": puts,
            "option_type": ["put"] * len(strikes),
            "bid": puts,
            "ask": puts,
        }
    )

    return pd.concat([df, df_puts], ignore_index=True)


class TestProbCurveStats:
    """Test standard moments and quantiles."""

    def test_skew_kurtosis_near_normal(self, flat_vol_chain, market_inputs):
        """A flat vol curve implies a Lognormal distribution.
        For short maturities and low vol, Lognormal is very close to Normal.
        Skew should be small positive (lognormal skew).
        Kurtosis should be small positive.
        """
        vc = VolCurve().fit(flat_vol_chain, market_inputs)
        pc = vc.implied_distribution()

        skew = pc.skew()
        kurt = pc.kurtosis()

        # Lognormal skew is approx 3*sigma*sqrt(T) + ... ~ small positive
        assert skew > 0
        assert abs(skew) < 1.0  # Should be small

        # Excess Kurtosis for Normal is 0. Lognormal is slightly positive.
        assert kurt > 0
        assert kurt < 1.0

    def test_quantile_median_approx_mean(self, flat_vol_chain, market_inputs):
        """For symmetric-ish distributions, Median (q=0.5) approx Mean."""
        vc = VolCurve().fit(flat_vol_chain, market_inputs)
        pc = vc.implied_distribution()

        median = pc.quantile(0.5)
        mean = pc.mean()

        # They should be very close for ATM 20% vol
        assert np.isclose(median, mean, rtol=0.01)

    def test_quantile_inverse_cdf(self, flat_vol_chain, market_inputs):
        """Quantile should range correctly based on probability."""
        vc = VolCurve().fit(flat_vol_chain, market_inputs)
        pc = vc.implied_distribution()

        p10 = pc.quantile(0.10)
        p90 = pc.quantile(0.90)

        assert p10 < pc.mean() < p90

        # Check round trip: CDF(Quantile(q)) approx q
        assert np.isclose(pc.prob_below(p10), 0.10, atol=0.02)
        assert np.isclose(pc.prob_below(p90), 0.90, atol=0.02)

    def test_quantile_bounds(self, flat_vol_chain, market_inputs):
        """Raises error for invalid quantiles."""
        vc = VolCurve().fit(flat_vol_chain, market_inputs)
        pc = vc.implied_distribution()

        with pytest.raises(ValueError):
            pc.quantile(-0.1)
        with pytest.raises(ValueError):
            pc.quantile(1.1)
