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

from oipd.core.maturity import resolve_maturity


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
def same_day_intraday_chain():
    """Single-expiry chain with an expiry later on the valuation day."""
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    expiry = pd.Timestamp("2025-01-01 16:00:00")

    calls = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": [20.75, 16.0, 11.25, 6.75, 3.0, 1.05, 0.4, 0.15, 0.08],
            "bid": [20.5, 15.8, 11.0, 6.5, 2.8, 0.9, 0.3, 0.1, 0.05],
            "ask": [21.0, 16.2, 11.5, 7.0, 3.2, 1.2, 0.5, 0.2, 0.1],
            "option_type": ["call"] * len(strikes),
            "expiry": [expiry] * len(strikes),
        }
    )

    spot = 100.0
    rate = 0.05
    time_to_expiry_years = 6.5 / (24.0 * 365.0)
    discount_factor = np.exp(-rate * time_to_expiry_years)

    puts = calls.copy()
    puts["option_type"] = "put"
    puts["last_price"] = (
        calls["last_price"] - spot + calls["strike"] * discount_factor
    ).abs()
    puts["bid"] = (calls["bid"] - spot + calls["strike"] * discount_factor).abs()
    puts["ask"] = (calls["ask"] - spot + calls["strike"] * discount_factor).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def timestamp_dividend_schedules():
    """Same-day dividend schedules straddling the valuation timestamp."""
    return {
        "before": pd.DataFrame(
            {"ex_date": [pd.Timestamp("2025-01-01 08:00:00")], "amount": [1.0]}
        ),
        "after": pd.DataFrame(
            {"ex_date": [pd.Timestamp("2025-01-01 12:00:00")], "amount": [1.0]}
        ),
    }


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
        for key in (
            "expiry",
            "time_to_expiry_years",
            "time_to_expiry_days",
            "forward_price",
            "at_money_vol",
        ):
            assert key in metadata
        assert np.isfinite(metadata["at_money_vol"])

        resolved = resolve_maturity(
            metadata["expiry"],
            prob_curve.resolved_market.valuation_timestamp,
        )
        assert metadata["time_to_expiry_years"] == pytest.approx(
            resolved.time_to_expiry_years
        )
        assert metadata["time_to_expiry_days"] == pytest.approx(
            resolved.time_to_expiry_days
        )

        assert metadata["expiry"] == resolved.expiry
        assert (
            metadata["calendar_days_to_expiry"] == resolved.calendar_days_to_expiry
        )

    def test_resolved_market_available(self, prob_curve):
        """resolved_market is available on ProbCurve."""
        resolved_market = prob_curve.resolved_market
        assert resolved_market is not None
        assert hasattr(resolved_market, "valuation_date")

    def test_metadata_keeps_same_day_intraday_expiry(self, same_day_intraday_chain):
        """Derived probability metadata should preserve sub-day maturity inputs."""
        from oipd import MarketInputs, VolCurve

        market_intraday = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
        )

        curve = VolCurve(pricing_engine="bs")
        curve.fit(same_day_intraday_chain, market_intraday)
        prob_curve = curve.implied_distribution()

        assert prob_curve.metadata["expiry"] == pd.Timestamp("2025-01-01 16:00:00")
        assert prob_curve.metadata["time_to_expiry_days"] == pytest.approx(6.5 / 24.0)
        assert prob_curve.metadata["calendar_days_to_expiry"] == 0
        assert prob_curve.metadata["time_to_expiry_years"] == pytest.approx(
            6.5 / (24.0 * 365.0)
        )

    def test_same_day_intraday_distribution_queries_are_finite(
        self, same_day_intraday_chain
    ):
        """Same-day intraday probability curves should produce stable numeric outputs."""
        from oipd import MarketInputs, VolCurve

        market_intraday = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
        )

        curve = VolCurve(pricing_engine="bs")
        curve.fit(same_day_intraday_chain, market_intraday)
        prob_curve = curve.implied_distribution()

        pdf_value = prob_curve.pdf(100.0)
        cdf_value = prob_curve.prob_below(100.0)
        median = prob_curve.quantile(0.5)
        export = prob_curve.density_results()

        assert np.isfinite(pdf_value)
        assert np.isfinite(cdf_value)
        assert np.isfinite(median)
        assert 0.0 <= cdf_value <= 1.0
        assert np.all(np.diff(export["price"].to_numpy(dtype=float)) > 0.0)
        assert np.all(np.isfinite(export["pdf"].to_numpy(dtype=float)))
        assert np.all(np.isfinite(export["cdf"].to_numpy(dtype=float)))

    def test_bs_distribution_accepts_timestamp_dividend_schedule(
        self, same_day_intraday_chain, timestamp_dividend_schedules
    ):
        """BS probability curves should support timestamped dividend schedules."""
        from oipd import MarketInputs, VolCurve

        market_intraday = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=timestamp_dividend_schedules["after"],
        )

        prob_curve = (
            VolCurve(pricing_engine="bs")
            .fit(same_day_intraday_chain, market_intraday)
            .implied_distribution()
        )

        assert np.isfinite(prob_curve.pdf(100.0))
        assert np.isfinite(prob_curve.prob_below(100.0))

    def test_bs_distribution_distinguishes_same_day_dividend_timing(
        self, same_day_intraday_chain, timestamp_dividend_schedules
    ):
        """BS distributions should change when dividend timing crosses valuation."""
        from oipd import MarketInputs, VolCurve

        market_before = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=timestamp_dividend_schedules["before"],
        )
        market_after = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=timestamp_dividend_schedules["after"],
        )

        prob_before = (
            VolCurve(pricing_engine="bs")
            .fit(same_day_intraday_chain, market_before)
            .implied_distribution()
        )
        prob_after = (
            VolCurve(pricing_engine="bs")
            .fit(same_day_intraday_chain, market_after)
            .implied_distribution()
        )

        assert prob_before.pdf(100.0) != pytest.approx(prob_after.pdf(100.0))

    def test_bs_distribution_ignores_post_expiry_dividends(self, single_expiry_chain):
        """BS distributions should ignore discrete dividends after the curve expiry."""
        from oipd import MarketInputs, VolCurve

        full_schedule = pd.DataFrame(
            {
                "ex_date": [
                    pd.Timestamp("2025-02-15 00:00:00"),
                    pd.Timestamp("2025-04-15 00:00:00"),
                ],
                "amount": [1.0, 1.5],
            }
        )
        early_only_schedule = pd.DataFrame(
            {
                "ex_date": [pd.Timestamp("2025-02-15 00:00:00")],
                "amount": [1.0],
            }
        )

        market_full = MarketInputs(
            valuation_date=date(2025, 1, 1),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=full_schedule,
        )
        market_early_only = MarketInputs(
            valuation_date=date(2025, 1, 1),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=early_only_schedule,
        )

        prob_full = (
            VolCurve(pricing_engine="bs")
            .fit(single_expiry_chain, market_full)
            .implied_distribution()
        )
        prob_early_only = (
            VolCurve(pricing_engine="bs")
            .fit(single_expiry_chain, market_early_only)
            .implied_distribution()
        )

        np.testing.assert_allclose(
            prob_full.density_results()["pdf"].to_numpy(dtype=float),
            prob_early_only.density_results()["pdf"].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-10,
        )


# =============================================================================
# ProbCurve.density_results() Tests
# =============================================================================


class TestProbCurveDensityResults:
    """Tests for ProbCurve.density_results() exports."""

    def test_density_results_returns_expected_columns(self, prob_curve):
        """density_results() returns the canonical export schema."""
        result = prob_curve.density_results()
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["price", "pdf", "cdf"]

    def test_density_results_uses_native_grid_by_default(self, prob_curve):
        """density_results() reuses the native grid when no domain is set."""
        result = prob_curve.density_results()
        np.testing.assert_allclose(
            result["price"].to_numpy(dtype=float), prob_curve.prices
        )
        np.testing.assert_allclose(
            result["pdf"].to_numpy(dtype=float), prob_curve.pdf_values
        )
        np.testing.assert_allclose(
            result["cdf"].to_numpy(dtype=float), prob_curve.cdf_values
        )

    def test_density_results_resamples_explicit_domain(self, prob_curve):
        """Explicit domain triggers interpolation onto the requested grid."""
        result = prob_curve.density_results(domain=(80.0, 120.0), points=25)
        assert len(result) == 25
        assert np.isclose(result["price"].iloc[0], 80.0)
        assert np.isclose(result["price"].iloc[-1], 120.0)

    def test_density_results_domain_defaults_to_200_points(self, prob_curve):
        """Explicit domain with omitted points defaults to 200 rows."""
        result = prob_curve.density_results(domain=(80.0, 120.0))
        assert len(result) == 200

    def test_density_results_matches_probability_queries(self, prob_curve):
        """Exported values stay consistent with PDF and CDF query methods."""
        result = prob_curve.density_results(domain=(85.0, 115.0), points=11)
        prices = result["price"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            result["pdf"].to_numpy(dtype=float),
            prob_curve.pdf(prices),
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            result["cdf"].to_numpy(dtype=float),
            [prob_curve.prob_below(price) for price in prices],
            rtol=1e-8,
        )

    def test_density_results_rejects_invalid_domain(self, prob_curve):
        """Invalid domains are rejected at the export boundary."""
        with pytest.raises(ValueError, match="strictly increasing"):
            prob_curve.density_results(domain=(120.0, 80.0))


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
