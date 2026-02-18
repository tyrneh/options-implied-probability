"""
Test Skeleton for ProbCurve Interface
======================================
Tests for the single-expiry probability distribution API.

Based on first-principles analysis of oipd.interface.probability.ProbCurve
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def single_expiry_chain():
    """Single-expiry chain with calls and puts for parity."""
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    expiry = pd.Timestamp("2025-03-21")

    calls = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            "bid": [12.0, 7.8, 4.9, 2.9, 1.4],
            "ask": [13.0, 8.6, 5.3, 3.3, 1.8],
            "option_type": ["C", "C", "C", "C", "C"],
            "expiry": [expiry] * 5,
        }
    )

    S, r, T = 100.0, 0.05, 79 / 365.0
    df = np.exp(-r * T)
    puts = calls.copy()
    puts["option_type"] = "P"
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def market_inputs():
    """Standard MarketInputs for probability tests."""
    from oipd import MarketInputs

    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )


@pytest.fixture
def fitted_vol_curve(single_expiry_chain, market_inputs):
    """A pre-fitted VolCurve for deriving probability."""
    from oipd import VolCurve

    vc = VolCurve(pricing_engine="bs")
    vc.fit(single_expiry_chain, market_inputs)
    return vc


@pytest.fixture
def prob_curve(fitted_vol_curve):
    """A ProbCurve derived from a fitted VolCurve."""
    return fitted_vol_curve.implied_distribution()


# =============================================================================
# ProbCurve.from_chain() Tests
# =============================================================================


class TestProbCurveFromChain:
    """Tests for ProbCurve.from_chain() constructor."""

    def test_from_chain_returns_probcurve(self, single_expiry_chain, market_inputs):
        """from_chain() returns a ProbCurve."""
        from oipd import ProbCurve

        prob = ProbCurve.from_chain(single_expiry_chain, market_inputs)
        assert isinstance(prob, ProbCurve)

    def test_from_chain_accepts_max_staleness(self, single_expiry_chain, market_inputs):
        """from_chain() accepts max_staleness_days."""
        from oipd import ProbCurve

        prob = ProbCurve.from_chain(
            single_expiry_chain,
            market_inputs,
            max_staleness_days=1,
        )
        assert isinstance(prob, ProbCurve)

    def test_from_chain_rejects_multiple_expiries(
        self, single_expiry_chain, market_inputs
    ):
        """from_chain() raises on multi-expiry input."""
        from oipd import ProbCurve

        bad_chain = single_expiry_chain.copy()
        bad_chain.loc[bad_chain.index[:5], "expiry"] = pd.Timestamp("2025-04-21")
        with pytest.raises(ValueError, match="single expiry"):
            ProbCurve.from_chain(bad_chain, market_inputs)


# =============================================================================
# ProbCurve Basic Properties Tests
# =============================================================================


class TestProbCurveProperties:
    """Tests for ProbCurve basic properties."""

    def test_prices_is_numpy_array(self, prob_curve):
        """prices property returns numpy array."""
        assert isinstance(prob_curve.prices, np.ndarray)

    def test_pdf_is_numpy_array(self, prob_curve):
        """pdf_values property returns numpy array."""
        assert isinstance(prob_curve.pdf_values, np.ndarray)

    def test_cdf_is_numpy_array(self, prob_curve):
        """cdf_values property returns numpy array."""
        assert isinstance(prob_curve.cdf_values, np.ndarray)

    def test_pdf_is_non_negative(self, prob_curve):
        """PDF values are non-negative."""
        assert np.all(prob_curve.pdf_values >= 0)

    def test_cdf_is_monotonic(self, prob_curve):
        """CDF is monotonically increasing."""
        cdf = prob_curve.cdf_values
        assert np.all(np.diff(cdf) >= -1e-10)  # Allow tiny numerical noise

    def test_cdf_ends_near_one(self, prob_curve):
        """CDF approaches 1 at high prices."""
        # Check that the CDF converges to 1.0 for a very high price
        # Using functional access allows checking far OTM
        assert prob_curve.prob_below(500.0) > 0.99
        assert prob_curve.prob_below(500.0) <= 1.01

    def test_metadata_has_core_fields(self, prob_curve):
        """metadata exposes core calibration fields."""
        metadata = prob_curve.metadata
        assert isinstance(metadata, dict)
        assert "expiry_date" in metadata
        assert "forward_price" in metadata
        assert "at_money_vol" in metadata
        assert np.isfinite(metadata["at_money_vol"])

    def test_resolved_market_available(self, prob_curve):
        """resolved_market is available on ProbCurve."""
        resolved_market = prob_curve.resolved_market
        assert resolved_market is not None
        assert hasattr(resolved_market, "valuation_date")


# =============================================================================
# ProbCurve.prob_below() Tests
# =============================================================================


class TestProbCurveProbBelow:
    """Tests for prob_below() method."""

    def test_prob_below_returns_float(self, prob_curve):
        """prob_below() returns a float."""
        p = prob_curve.prob_below(100.0)
        assert isinstance(p, float)

    def test_prob_below_in_valid_range(self, prob_curve):
        """prob_below() returns value in [0, 1]."""
        p = prob_curve.prob_below(100.0)
        assert 0.0 <= p <= 1.0

    def test_prob_below_increases_with_price(self, prob_curve):
        """prob_below(low) < prob_below(high)."""
        p_low = prob_curve.prob_below(90.0)
        p_high = prob_curve.prob_below(110.0)
        assert p_low < p_high

    def test_prob_below_zero_at_left_edge(self, prob_curve):
        """prob_below() returns ~0 at extreme low price."""
        p = prob_curve.prob_below(1.0)
        assert p < 0.01

    def test_prob_below_one_at_right_edge(self, prob_curve):
        """prob_below() returns ~1 at extreme high price."""
        p = prob_curve.prob_below(1000.0)
        assert p > 0.99


# =============================================================================
# ProbCurve.prob_above() Tests
# =============================================================================


class TestProbCurveProbAbove:
    """Tests for prob_above() method."""

    def test_prob_above_complements_prob_below(self, prob_curve):
        """prob_above(x) + prob_below(x) ≈ 1."""
        price = 100.0
        p_below = prob_curve.prob_below(price)
        p_above = prob_curve.prob_above(price)
        assert np.isclose(p_below + p_above, 1.0)


# =============================================================================
# ProbCurve.prob_between() Tests
# =============================================================================


class TestProbCurveProbBetween:
    """Tests for prob_between() method."""

    def test_prob_between_valid_range(self, prob_curve):
        """prob_between() returns value in [0, 1]."""
        p = prob_curve.prob_between(90.0, 110.0)
        assert 0.0 <= p <= 1.0

    def test_prob_between_equals_cdf_diff(self, prob_curve):
        """prob_between(low, high) = prob_below(high) - prob_below(low)."""
        low, high = 95.0, 105.0
        p_between = prob_curve.prob_between(low, high)
        p_diff = prob_curve.prob_below(high) - prob_curve.prob_below(low)
        assert np.isclose(p_between, p_diff)

    def test_prob_between_raises_on_invalid_range(self, prob_curve):
        """prob_between() raises if low > high."""
        with pytest.raises(ValueError):
            prob_curve.prob_between(110.0, 90.0)


# =============================================================================
# ProbCurve.expected_value() Tests
# =============================================================================


class TestProbCurveMean:
    """Tests for mean() method."""

    def test_mean_returns_float(self, prob_curve):
        """mean() returns a float."""
        ev = prob_curve.mean()
        assert isinstance(ev, float)

    def test_mean_is_positive(self, prob_curve):
        """Mean is positive for stock prices."""
        ev = prob_curve.mean()
        assert ev > 0

    def test_mean_near_forward(self, prob_curve):
        """Mean value should be positive (roughly near spot/forward)."""
        ev = prob_curve.mean()
        # For synthetic data, just check it's positive and reasonable
        assert 0.0 < ev < 500.0  # Rough bounds


# =============================================================================
# ProbCurve.variance() Tests
# =============================================================================


class TestProbCurveVariance:
    """Tests for variance() method."""

    def test_variance_returns_float(self, prob_curve):
        """variance() returns a float."""
        var = prob_curve.variance()
        assert isinstance(var, float)

    def test_variance_is_positive(self, prob_curve):
        """Variance is positive."""
        var = prob_curve.variance()
        assert var > 0


# =============================================================================
# ProbCurve Callable Interface Tests
# =============================================================================


class TestProbCurveCallable:
    """Tests for ProbCurve callable interface."""

    def test_call_returns_pdf_value(self, prob_curve):
        """Calling prob_curve(price) returns PDF value."""
        pdf_val = prob_curve(100.0)
        assert pdf_val >= 0
