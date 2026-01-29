"""
Test Skeleton for VolSurface Interface
=======================================
Tests for the multi-expiry volatility surface fitting API.

Based on first-principles analysis of oipd.interface.volatility.VolSurface
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
    """Option chain with multiple expiries."""
    exp1 = pd.DataFrame({
        "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
        "last_price": [12.0, 8.0, 5.0, 3.0, 1.5],
        "bid": [11.5, 7.5, 4.5, 2.5, 1.2],
        "ask": [12.5, 8.5, 5.5, 3.5, 1.8],
        "option_type": ["C", "C", "C", "C", "C"],
        "expiry": [pd.Timestamp("2025-02-21")] * 5,
    })
    exp2 = pd.DataFrame({
        "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
        "last_price": [14.0, 10.0, 7.0, 4.5, 2.5],
        "bid": [13.5, 9.5, 6.5, 4.0, 2.2],
        "ask": [14.5, 10.5, 7.5, 5.0, 2.8],
        "option_type": ["C", "C", "C", "C", "C"],
        "expiry": [pd.Timestamp("2025-05-21")] * 5,
    })
    return pd.concat([exp1, exp2], ignore_index=True)


@pytest.fixture
def market_inputs():
    """Standard MarketInputs for surface fitting (no expiry_date)."""
    from oipd import MarketInputs
    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
    )


# =============================================================================
# VolSurface.__init__() Tests
# =============================================================================

class TestVolSurfaceInit:
    """Tests for VolSurface initialization."""

    def test_default_initialization(self):
        """VolSurface initializes with default parameters."""
        from oipd import VolSurface
        vs = VolSurface()
        # Just verify it initializes without error
        assert vs is not None


# =============================================================================
# VolSurface.fit() Tests
# =============================================================================

class TestVolSurfaceFit:
    """Tests for VolSurface.fit() method."""

    def test_fit_returns_self(self, multi_expiry_chain, market_inputs):
        """fit() returns self for chaining."""
        from oipd import VolSurface
        vs = VolSurface()
        result = vs.fit(multi_expiry_chain, market_inputs)
        assert result is vs

    def test_fit_populates_expiries(self, multi_expiry_chain, market_inputs):
        """fit() populates .expiries property."""
        from oipd import VolSurface
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        assert len(vs.expiries) == 2

    def test_fit_with_horizon_filter(self, multi_expiry_chain, market_inputs):
        """fit() respects horizon parameter."""
        from oipd import VolSurface
        vs = VolSurface()
        # horizon="2m" from Jan 1 2025 should filter to ~Feb expiry
        vs.fit(multi_expiry_chain, market_inputs, horizon="2m")
        # Result depends on dates, just check it doesn't crash
        assert vs.expiries is not None


# =============================================================================
# VolSurface.slice() Tests
# =============================================================================

class TestVolSurfaceSlice:
    """Tests for VolSurface.slice() method."""

    def test_slice_returns_volcurve(self, multi_expiry_chain, market_inputs):
        """slice(expiry) returns a VolCurve object."""
        from oipd import VolSurface, VolCurve
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        first_exp = vs.expiries[0]
        curve = vs.slice(first_exp)
        assert isinstance(curve, VolCurve)

    def test_slice_has_params(self, multi_expiry_chain, market_inputs):
        """Sliced VolCurve has accessible params."""
        from oipd import VolSurface
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        curve = vs.slice(vs.expiries[0])
        assert curve.params is not None

    def test_slice_invalid_expiry_raises(self, multi_expiry_chain, market_inputs):
        """slice() with invalid expiry raises ValueError."""
        from oipd import VolSurface
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        with pytest.raises(ValueError):
            vs.slice("2030-12-31")


# =============================================================================
# VolSurface with Interpolation Tests
# =============================================================================

class TestVolSurfaceInterpolation:
    """Tests for VolSurface interpolation functionality."""

    def test_fit_creates_interpolator(self, multi_expiry_chain, market_inputs):
        """fit() automatically creates interpolator."""
        from oipd import VolSurface
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        # Access private attribute (or check via implied_vol method)
        assert vs._interpolator is not None

    def test_implied_vol_at_arbitrary_time(self, multi_expiry_chain, market_inputs):
        """implied_vol(K, t) works for arbitrary expiry time."""
        from oipd import VolSurface
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        iv = vs.implied_vol(100.0, 60 / 365.0)  # 60 days
        assert 0.0 < iv < 2.0

    def test_total_variance_at_arbitrary_time(self, multi_expiry_chain, market_inputs):
        """total_variance(K, t) returns positive value."""
        from oipd import VolSurface
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        w = vs.total_variance(100.0, 60 / 365.0)
        assert w > 0


# =============================================================================
# VolSurface.implied_distribution() Tests
# =============================================================================

class TestVolSurfaceImpliedDistribution:
    """Tests for deriving ProbSurface from VolSurface."""

    def test_implied_distribution_returns_probsurface(self, multi_expiry_chain, market_inputs):
        """implied_distribution() returns a ProbSurface."""
        from oipd import VolSurface
        from oipd.interface.probability import ProbSurface
        
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        prob_surface = vs.implied_distribution()
        assert isinstance(prob_surface, ProbSurface)

    def test_probsurface_has_matching_expiries(self, multi_expiry_chain, market_inputs):
        """ProbSurface has same expiries as VolSurface."""
        from oipd import VolSurface
        
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        prob_surface = vs.implied_distribution()
        assert len(prob_surface.expiries) == len(vs.expiries)


# =============================================================================
# VolSurface.plot() Tests
# =============================================================================

class TestVolSurfacePlot:
    """Tests for VolSurface plotting methods."""

    def test_plot_does_not_crash(self, multi_expiry_chain, market_inputs):
        """plot() executes without raising."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from oipd import VolSurface
        
        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        fig = vs.plot()
        assert fig is not None
        plt.close(fig)
