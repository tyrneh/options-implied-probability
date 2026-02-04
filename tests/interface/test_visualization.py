"""
Visualization Smoke Tests
=========================
Ensures all plotting methods execute without exception and return Figure objects.

These tests do NOT verify visual correctness (that requires manual review).
They protect against runtime errors in plot generation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CI
import matplotlib.pyplot as plt
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
# VolCurve Visualization Tests
# =============================================================================


class TestVolCurveVisualization:
    """Smoke tests for VolCurve plotting methods."""

    def test_plot_returns_figure(self, fitted_vol_curve):
        """plot() should return a matplotlib Figure."""
        fig = fitted_vol_curve.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_with_options(self, fitted_vol_curve):
        """plot() should accept optional arguments without error."""
        fig = fitted_vol_curve.plot(
            include_observed=True,
            figsize=(12, 6),
            title="Custom Title",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


# =============================================================================
# VolSurface Visualization Tests
# =============================================================================


class TestVolSurfaceVisualization:
    """Smoke tests for VolSurface plotting methods."""

    def test_plot_returns_figure(self, fitted_vol_surface):
        """plot() should return a matplotlib Figure."""
        fig = fitted_vol_surface.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_3d_returns_figure(self, fitted_vol_surface):
        """plot_3d() should return a matplotlib Figure."""
        fig = fitted_vol_surface.plot_3d()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_3d_with_options(self, fitted_vol_surface):
        """plot_3d() should accept optional arguments without error."""
        fig = fitted_vol_surface.plot_3d(
            figsize=(12, 8),
            view_angle=(30, -45),
            cmap="viridis",
            projection_type="persp",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_term_structure_returns_figure(self, fitted_vol_surface):
        """plot_term_structure() should return a matplotlib Figure."""
        fig = fitted_vol_surface.plot_term_structure()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_term_structure_with_options(self, fitted_vol_surface):
        """plot_term_structure() should accept optional arguments without error."""
        fig = fitted_vol_surface.plot_term_structure(
            at_money="forward",
            figsize=(12, 6),
            title="Custom Term Structure",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


# =============================================================================
# ProbCurve Visualization Tests
# =============================================================================


class TestProbCurveVisualization:
    """Smoke tests for ProbCurve plotting methods."""

    def test_plot_returns_figure(self, prob_curve):
        """plot() should return a matplotlib Figure."""
        fig = prob_curve.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_pdf(self, prob_curve):
        """plot(kind='pdf') should work."""
        fig = prob_curve.plot(kind="pdf")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_cdf(self, prob_curve):
        """plot(kind='cdf') should work."""
        fig = prob_curve.plot(kind="cdf")
        assert isinstance(fig, Figure)
        plt.close(fig)


# =============================================================================
# ProbSurface Visualization Tests
# =============================================================================


class TestProbSurfaceVisualization:
    """Smoke tests for ProbSurface plotting methods."""

    def test_plot_fan_returns_figure(self, prob_surface):
        """plot_fan() should return a matplotlib Figure."""
        fig = prob_surface.plot_fan()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_fan_with_quantiles(self, prob_surface):
        """plot_fan() should accept custom percentile bounds."""
        fig = prob_surface.plot_fan(lower_percentile=10.0, upper_percentile=90.0)
        assert isinstance(fig, Figure)
        plt.close(fig)
