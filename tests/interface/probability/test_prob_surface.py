"""
Test Skeleton for ProbSurface Interface
========================================
Tests for the multi-expiry probability surface API.

Based on first-principles analysis of oipd.interface.probability.ProbSurface
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def multi_expiry_chain():
    """Option chain with multiple expiries (calls and puts)."""
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]

    exp1 = pd.Timestamp("2025-02-21")
    calls1 = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": [12.0, 8.0, 5.0, 3.0, 1.5],
            "bid": [11.5, 7.5, 4.5, 2.5, 1.2],
            "ask": [12.5, 8.5, 5.5, 3.5, 1.8],
            "option_type": ["C", "C", "C", "C", "C"],
            "expiry": [exp1] * 5,
        }
    )

    exp2 = pd.Timestamp("2025-05-21")
    calls2 = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": [14.0, 10.0, 7.0, 4.5, 2.5],
            "bid": [13.5, 9.5, 6.5, 4.0, 2.2],
            "ask": [14.5, 10.5, 7.5, 5.0, 2.8],
            "option_type": ["C", "C", "C", "C", "C"],
            "expiry": [exp2] * 5,
        }
    )

    calls = pd.concat([calls1, calls2], ignore_index=True)

    S, r = 100.0, 0.05
    t_array = (calls["expiry"] - pd.Timestamp("2025-01-01")).dt.days / 365.0
    df_array = np.exp(-r * t_array)

    puts = calls.copy()
    puts["option_type"] = "P"
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df_array).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df_array).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df_array).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def market_inputs():
    """Standard MarketInputs for probability surface tests."""
    from oipd import MarketInputs

    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
    )


@pytest.fixture
def fitted_vol_surface(multi_expiry_chain, market_inputs):
    """A pre-fitted VolSurface for deriving probability surface."""
    from oipd import VolSurface

    vs = VolSurface(pricing_engine="bs")
    vs.fit(multi_expiry_chain, market_inputs)
    return vs


@pytest.fixture
def prob_surface(fitted_vol_surface):
    """A ProbSurface derived from a fitted VolSurface."""
    return fitted_vol_surface.implied_distribution()


# =============================================================================
# ProbSurface.from_chain() Tests
# =============================================================================


class TestProbSurfaceFromChain:
    """Tests for ProbSurface.from_chain() constructor."""

    def test_from_chain_returns_probsurface(self, multi_expiry_chain, market_inputs):
        """from_chain() returns a ProbSurface."""
        from oipd import ProbSurface

        prob = ProbSurface.from_chain(multi_expiry_chain, market_inputs)
        assert isinstance(prob, ProbSurface)

    def test_from_chain_accepts_max_staleness(self, multi_expiry_chain, market_inputs):
        """from_chain() accepts max_staleness_days."""
        from oipd import ProbSurface

        prob = ProbSurface.from_chain(
            multi_expiry_chain,
            market_inputs,
            max_staleness_days=1,
        )
        assert isinstance(prob, ProbSurface)

    def test_from_chain_rejects_single_expiry(self, multi_expiry_chain, market_inputs):
        """from_chain() raises when only one expiry is provided."""
        from oipd import ProbSurface
        from oipd.core.errors import CalculationError

        single_expiry_chain = multi_expiry_chain[
            multi_expiry_chain["expiry"] == multi_expiry_chain["expiry"].iloc[0]
        ]
        with pytest.raises(CalculationError, match="at least two"):
            ProbSurface.from_chain(single_expiry_chain, market_inputs)


# =============================================================================
# ProbSurface Basic Properties Tests
# =============================================================================


class TestProbSurfaceProperties:
    """Tests for ProbSurface basic properties."""

    def test_expiries_returns_tuple(self, prob_surface):
        """expiries property returns a tuple."""
        assert isinstance(prob_surface.expiries, tuple)

    def test_expiries_matches_vol_surface(self, fitted_vol_surface, prob_surface):
        """ProbSurface has same expiries as source VolSurface."""
        assert len(prob_surface.expiries) == len(fitted_vol_surface.expiries)


# =============================================================================
# ProbSurface.slice() Tests
# =============================================================================


class TestProbSurfaceSlice:
    """Tests for ProbSurface.slice() method."""

    def test_slice_returns_probcurve(self, prob_surface):
        """slice(expiry) returns a ProbCurve object."""
        from oipd.interface.probability import ProbCurve

        first_exp = prob_surface.expiries[0]
        curve = prob_surface.slice(first_exp)
        assert isinstance(curve, ProbCurve)

    def test_slice_has_valid_pdf(self, prob_surface):
        """Sliced ProbCurve has valid PDF."""
        curve = prob_surface.slice(prob_surface.expiries[0])
        assert np.all(curve.pdf_values >= 0)

    def test_slice_metadata_and_market(self, prob_surface):
        """Sliced ProbCurve exposes metadata and resolved market."""
        curve = prob_surface.slice(prob_surface.expiries[0])
        metadata = curve.metadata
        assert isinstance(metadata, dict)
        assert "expiry_date" in metadata
        assert "forward_price" in metadata
        assert "at_money_vol" in metadata
        assert np.isfinite(metadata["at_money_vol"])
        resolved_market = curve.resolved_market
        assert resolved_market is not None
        assert hasattr(resolved_market, "valuation_date")

    def test_slice_invalid_expiry_raises(self, prob_surface):
        """slice() with invalid expiry raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            prob_surface.slice("2030-12-31")

    def test_slice_accepts_string_date(self, prob_surface):
        """slice() accepts string date format."""
        first_exp = prob_surface.expiries[0]
        exp_str = first_exp.strftime("%Y-%m-%d")
        curve = prob_surface.slice(exp_str)
        assert curve is not None


# =============================================================================
# ProbSurface.plot_fan() Tests
# =============================================================================


class TestProbSurfacePlotFan:
    """Tests for ProbSurface.plot_fan() visualization."""

    def test_plot_fan_does_not_crash(self, prob_surface):
        """plot_fan() executes without raising."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = prob_surface.plot_fan()
        assert fig is not None
        plt.close(fig)
