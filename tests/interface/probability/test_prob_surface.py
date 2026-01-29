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
def fitted_vol_surface():
    """A pre-fitted VolSurface for deriving probability surface."""
    from oipd import VolSurface, MarketInputs
    
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
    chain = pd.concat([exp1, exp2], ignore_index=True)
    
    market = MarketInputs(
        valuation_date=date(2025, 1, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
    )
    
    vs = VolSurface()
    vs.fit(chain, market)
    return vs


@pytest.fixture
def prob_surface(fitted_vol_surface):
    """A ProbSurface derived from a fitted VolSurface."""
    return fitted_vol_surface.implied_distribution()


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
        assert np.all(curve.pdf >= 0)

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
