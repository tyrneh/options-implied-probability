"""
Test the Probability Interfaces (Distribution, DistributionSurface).
"""

import pandas as pd
import pytest
import numpy as np
from datetime import date

from oipd.interface.volatility import VolCurve, VolSurface
from oipd.interface.probability import Distribution, DistributionSurface
from oipd.pipelines.market_inputs import MarketInputs


@pytest.fixture
def sample_data():
    """Create a minimal option chain for testing."""
    strikes = [200, 225, 250, 275, 300]
    # Synthetic prices roughly around spot=250
    calls = [55.0, 35.0, 18.0, 8.0, 3.0]
    
    df = pd.DataFrame({
        "strike": strikes,
        "option_type": ["C"] * 5,
        "mid_price": calls,
        "bid": [c - 0.5 for c in calls],
        "ask": [c + 0.5 for c in calls],
        "expiry": [pd.Timestamp("2025-12-20")] * 5
    })
    return df


@pytest.fixture
def market_inputs():
    return MarketInputs(
        valuation_date=date(2025, 10, 1),
        expiry_date=date(2025, 12, 20),
        underlying_price=250.0,
        risk_free_rate=0.05,
        dividend_yield=0.0
    )


def test_distribution_creation(sample_data, market_inputs):
    """Test creating a Distribution via VolCurve."""
    # 1. Fit VolCurve first
    vc = VolCurve(method="svi")
    vc.fit(sample_data, market_inputs)
    
    # 2. Derive Distribution
    dist = vc.implied_distribution()
    
    assert isinstance(dist, Distribution)
    assert dist.prices is not None
    assert dist.pdf is not None
    assert dist.cdf is not None
    assert len(dist.prices) > 0
    assert len(dist.pdf) == len(dist.prices)
    
    # Check basic properties
    assert 0.0 <= dist.prob_below(250) <= 1.0
    assert np.isclose(dist.prob_below(1000), 1.0, atol=0.05)
    assert np.isclose(dist.prob_below(0), 0.0, atol=0.05)


def test_distribution_methods(sample_data, market_inputs):
    """Test probability query methods."""
    vc = VolCurve(method="svi")
    vc.fit(sample_data, market_inputs)
    dist = vc.implied_distribution()
    
    # Probabilities
    p_low = dist.prob_below(225)
    p_high = dist.prob_below(275)
    assert p_low < p_high
    
    p_between = dist.prob_between(225, 275)
    assert np.isclose(p_between, p_high - p_low)
    
    p_above = dist.prob_above(250)
    assert np.isclose(p_above + dist.prob_below(250), 1.0)
    
    # Moments
    ev = dist.expected_value()
    assert 240 < ev < 260  # Should be close to forward/spot
    
    var = dist.variance()
    assert var > 0


def test_distribution_surface(sample_data, market_inputs):
    """Test DistributionSurface creation and slicing."""
    # Create multi-expiry data
    df1 = sample_data.copy()
    df2 = sample_data.copy()
    df2["expiry"] = pd.Timestamp("2026-03-20")
    # Shift prices slightly for second expiry to make it different
    df2["mid_price"] += 5.0 
    
    full_chain = pd.concat([df1, df2])
    
    # Fit VolSurface
    vs = VolSurface()
    vs.fit(full_chain, market_inputs)
    
    # Derive DistributionSurface
    ds = vs.implied_distribution()
    
    assert isinstance(ds, DistributionSurface)
    assert len(ds.expiries) == 2
    
    # Slice
    d1 = ds.slice("2025-12-20")
    assert isinstance(d1, Distribution)
    assert d1.expected_value() > 0
    
    d2 = ds.slice("2026-03-20")
    assert isinstance(d2, Distribution)
    
    # Check that they are different
    assert not np.isclose(d1.expected_value(), d2.expected_value())
