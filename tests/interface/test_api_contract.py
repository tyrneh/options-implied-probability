"""
API Contract Tests
==================
Ensures every public method defined in the PRD exists and is callable.

These tests verify the "shape" of the API, not numerical correctness.
They should NEVER fail due to refactoring internal logic.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date
from matplotlib.figure import Figure


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def market_inputs():
    """Standard MarketInputs for all tests."""
    from oipd import MarketInputs

    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_yield=0.02,
    )


@pytest.fixture
def single_expiry_chain():
    """Minimal option chain for VolCurve."""
    strikes = [80, 90, 100, 110, 120]
    expiry = pd.Timestamp("2025-02-01")

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


@pytest.fixture
def multi_expiry_chain():
    """Option chain with 2 expiries for VolSurface."""
    strikes = [80, 90, 100, 110, 120]

    exp1 = pd.Timestamp("2025-01-31")
    calls1 = pd.DataFrame(
        {
            "expiry": [exp1] * len(strikes),
            "strike": strikes,
            "bid": [20.5, 11.0, 3.5, 0.8, 0.2],
            "ask": [21.0, 11.5, 4.0, 1.2, 0.4],
            "last_price": [20.75, 11.25, 3.75, 1.0, 0.3],
            "option_type": ["call"] * len(strikes),
        }
    )

    exp2 = pd.Timestamp("2025-04-01")
    calls2 = pd.DataFrame(
        {
            "expiry": [exp2] * len(strikes),
            "strike": strikes,
            "bid": [22.5, 13.0, 5.5, 1.8, 0.6],
            "ask": [23.0, 13.5, 6.0, 2.2, 0.8],
            "last_price": [22.75, 13.25, 5.75, 2.0, 0.7],
            "option_type": ["call"] * len(strikes),
        }
    )

    calls = pd.concat([calls1, calls2], ignore_index=True)

    # Generate puts
    S, r = 100.0, 0.05
    t_array = (calls["expiry"] - pd.Timestamp("2025-01-01")).dt.days / 365.0
    df_array = np.exp(-r * t_array)

    puts = calls.copy()
    puts["option_type"] = "put"
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df_array).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df_array).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df_array).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def fitted_vol_curve(single_expiry_chain, market_inputs):
    """Pre-fitted VolCurve."""
    from oipd import VolCurve

    return VolCurve().fit(single_expiry_chain, market_inputs)


@pytest.fixture
def fitted_vol_surface(multi_expiry_chain, market_inputs):
    """Pre-fitted VolSurface."""
    from oipd import VolSurface

    return VolSurface().fit(multi_expiry_chain, market_inputs)


@pytest.fixture
def prob_curve(fitted_vol_curve):
    """ProbCurve derived from VolCurve."""
    return fitted_vol_curve.implied_distribution()


@pytest.fixture
def prob_surface(fitted_vol_surface):
    """ProbSurface derived from VolSurface."""
    return fitted_vol_surface.implied_distribution()


# =============================================================================
# VolCurve Contract Tests
# =============================================================================


class TestVolCurveContract:
    """Verify all PRD-documented VolCurve methods exist and are callable."""

    def test_fit_exists_and_returns_self(self, single_expiry_chain, market_inputs):
        from oipd import VolCurve

        vc = VolCurve()
        result = vc.fit(single_expiry_chain, market_inputs)
        assert result is vc

    def test_implied_vol_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.implied_vol(100.0)
        assert isinstance(result, (float, np.floating, np.ndarray))

    def test_call_is_alias_for_implied_vol(self, fitted_vol_curve):
        via_call = fitted_vol_curve(100.0)
        via_method = fitted_vol_curve.implied_vol(100.0)
        assert np.allclose(via_call, via_method)

    def test_total_variance_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.total_variance(100.0)
        assert isinstance(result, (float, np.floating, np.ndarray))
        val = result if np.isscalar(result) else result[0]
        assert val > 0

    def test_atm_vol_property_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.atm_vol
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)
        assert result > 0

    def test_forward_price_property_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.forward_price
        assert isinstance(result, float)
        assert result > 0

    def test_params_property_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.params
        assert result is not None

    def test_diagnostics_property_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.diagnostics
        assert result is not None
        assert hasattr(result, "status")

    def test_expiries_property_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.expiries
        assert isinstance(result, tuple)

    def test_iv_results_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.iv_results()
        assert isinstance(result, pd.DataFrame)

    def test_greeks_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.greeks([100.0])
        assert isinstance(result, pd.DataFrame)
        assert "delta" in result.columns

    def test_delta_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.delta([100.0])
        assert isinstance(result, np.ndarray)

    def test_gamma_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.gamma([100.0])
        assert isinstance(result, np.ndarray)

    def test_vega_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.vega([100.0])
        assert isinstance(result, np.ndarray)

    def test_theta_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.theta([100.0])
        assert isinstance(result, np.ndarray)

    def test_rho_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.rho([100.0])
        assert isinstance(result, np.ndarray)

    def test_price_exists(self, fitted_vol_curve):
        result = fitted_vol_curve.price([100.0], call_or_put="call")
        assert isinstance(result, np.ndarray)

    def test_implied_distribution_exists(self, fitted_vol_curve):
        from oipd.interface.probability import ProbCurve

        result = fitted_vol_curve.implied_distribution()
        assert isinstance(result, ProbCurve)


# =============================================================================
# VolSurface Contract Tests
# =============================================================================


class TestVolSurfaceContract:
    """Verify all PRD-documented VolSurface methods exist and are callable."""

    def test_fit_exists_and_returns_self(self, multi_expiry_chain, market_inputs):
        from oipd import VolSurface

        vs = VolSurface()
        result = vs.fit(multi_expiry_chain, market_inputs)
        assert result is vs

    def test_slice_exists(self, fitted_vol_surface):
        from oipd import VolCurve

        first_exp = fitted_vol_surface.expiries[0]
        result = fitted_vol_surface.slice(first_exp)
        assert isinstance(result, VolCurve)

    def test_expiries_property_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.expiries
        assert isinstance(result, (list, tuple))
        assert len(result) >= 1

    def test_implied_vol_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.implied_vol(100.0, 0.1)
        assert isinstance(result, float)

    def test_call_is_alias_for_implied_vol(self, fitted_vol_surface):
        via_call = fitted_vol_surface(100.0, 0.1)
        via_method = fitted_vol_surface.implied_vol(100.0, 0.1)
        assert np.isclose(via_call, via_method)

    def test_total_variance_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.total_variance(100.0, 0.1)
        assert isinstance(result, float)
        assert result > 0

    def test_forward_price_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.forward_price(0.1)
        assert isinstance(result, float)
        assert result > 0

    def test_atm_vol_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.atm_vol(0.1)
        assert isinstance(result, float)
        assert result > 0

    def test_iv_results_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.iv_results()
        assert isinstance(result, pd.DataFrame)

    def test_params_property_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.params
        assert isinstance(result, dict)

    def test_greeks_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.greeks([100.0], t=0.1)
        assert isinstance(result, pd.DataFrame)
        assert "delta" in result.columns

    def test_delta_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.delta([100.0], t=0.1)
        assert isinstance(result, np.ndarray)

    def test_gamma_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.gamma([100.0], t=0.1)
        assert isinstance(result, np.ndarray)

    def test_vega_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.vega([100.0], t=0.1)
        assert isinstance(result, np.ndarray)

    def test_theta_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.theta([100.0], t=0.1)
        assert isinstance(result, np.ndarray)

    def test_rho_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.rho([100.0], t=0.1)
        assert isinstance(result, np.ndarray)

    def test_price_exists(self, fitted_vol_surface):
        result = fitted_vol_surface.price([100.0], t=0.1, call_or_put="call")
        assert isinstance(result, np.ndarray)

    def test_implied_distribution_exists(self, fitted_vol_surface):
        from oipd.interface.probability import ProbSurface

        result = fitted_vol_surface.implied_distribution()
        assert isinstance(result, ProbSurface)


# =============================================================================
# ProbCurve Contract Tests
# =============================================================================


class TestProbCurveContract:
    """Verify all PRD-documented ProbCurve methods exist and are callable."""

    def test_from_chain_exists(self, single_expiry_chain, market_inputs):
        from oipd import ProbCurve

        result = ProbCurve.from_chain(single_expiry_chain, market_inputs)
        assert isinstance(result, ProbCurve)

    def test_pdf_exists(self, prob_curve):
        result = prob_curve.pdf(100.0)
        assert isinstance(result, (float, np.floating))

    def test_call_is_alias_for_pdf(self, prob_curve):
        via_call = prob_curve(100.0)
        via_method = prob_curve.pdf(100.0)
        assert np.isclose(via_call, via_method)

    def test_prob_below_exists(self, prob_curve):
        result = prob_curve.prob_below(100.0)
        assert isinstance(result, (float, np.floating))
        assert 0 <= result <= 1

    def test_prob_above_exists(self, prob_curve):
        result = prob_curve.prob_above(100.0)
        assert isinstance(result, (float, np.floating))
        assert 0 <= result <= 1

    def test_prob_between_exists(self, prob_curve):
        result = prob_curve.prob_between(90.0, 110.0)
        assert isinstance(result, (float, np.floating))
        assert 0 <= result <= 1

    def test_mean_exists(self, prob_curve):
        result = prob_curve.mean()
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_variance_exists(self, prob_curve):
        result = prob_curve.variance()
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_skew_exists(self, prob_curve):
        result = prob_curve.skew()
        assert isinstance(result, (float, np.floating))

    def test_kurtosis_exists(self, prob_curve):
        result = prob_curve.kurtosis()
        assert isinstance(result, (float, np.floating))

    def test_quantile_exists(self, prob_curve):
        result = prob_curve.quantile(0.5)
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_prices_property_exists(self, prob_curve):
        result = prob_curve.prices
        assert isinstance(result, np.ndarray)

    def test_pdf_values_property_exists(self, prob_curve):
        result = prob_curve.pdf_values
        assert isinstance(result, np.ndarray)

    def test_cdf_values_property_exists(self, prob_curve):
        result = prob_curve.cdf_values
        assert isinstance(result, np.ndarray)

    def test_metadata_exists(self, prob_curve):
        metadata = prob_curve.metadata
        assert isinstance(metadata, dict)
        assert "expiry_date" in metadata
        assert "forward_price" in metadata
        assert "at_money_vol" in metadata

    def test_resolved_market_exists(self, prob_curve):
        resolved_market = prob_curve.resolved_market
        assert resolved_market is not None
        assert hasattr(resolved_market, "valuation_date")


# =============================================================================
# ProbSurface Contract Tests
# =============================================================================


class TestProbSurfaceContract:
    """Verify all PRD-documented ProbSurface methods exist and are callable."""

    def test_from_chain_exists(self, multi_expiry_chain, market_inputs):
        from oipd import ProbSurface

        result = ProbSurface.from_chain(multi_expiry_chain, market_inputs)
        assert isinstance(result, ProbSurface)

    def test_slice_exists(self, prob_surface):
        from oipd.interface.probability import ProbCurve

        first_exp = prob_surface.expiries[0]
        result = prob_surface.slice(first_exp)
        assert isinstance(result, ProbCurve)

    def test_expiries_property_exists(self, prob_surface):
        result = prob_surface.expiries
        assert isinstance(result, tuple)
        assert len(result) >= 1

    def test_pdf_exists(self, prob_surface):
        result = prob_surface.pdf(100.0, t=45 / 365.0)
        assert isinstance(result, np.ndarray)

    def test_cdf_exists(self, prob_surface):
        result = prob_surface.cdf(100.0, t=45 / 365.0)
        assert isinstance(result, np.ndarray)

    def test_quantile_exists(self, prob_surface):
        result = prob_surface.quantile(0.5, t=45 / 365.0)
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_call_alias_exists(self, prob_surface):
        result = prob_surface(100.0, t=45 / 365.0)
        assert isinstance(result, np.ndarray)
