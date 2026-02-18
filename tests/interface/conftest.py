"""Shared interface-level fixtures for expensive setup reuse."""

from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def market_inputs():
    """Build canonical market inputs shared by interface tests.

    Args:
        None.

    Returns:
        MarketInputs: Standardized market configuration.
    """
    from oipd import MarketInputs

    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_yield=0.02,
    )


@pytest.fixture(scope="module")
def single_expiry_chain():
    """Create a single-expiry options chain with calls and parity-derived puts.

    Args:
        None.

    Returns:
        pd.DataFrame: Single-expiry chain with required interface columns.
    """
    strikes = [80, 90, 100, 110, 120]
    expiry = pd.Timestamp("2025-02-01")

    calls = pd.DataFrame(
        {
            "expiry": [expiry] * len(strikes),
            "strike": strikes,
            "bid": [20.5, 11.0, 3.5, 0.8, 0.2],
            "ask": [21.0, 11.5, 4.0, 1.2, 0.4],
            "last_price": [20.75, 11.25, 3.75, 1.0, 0.3],
            "option_type": ["call"] * len(strikes),
        }
    )

    spot_price = 100.0
    risk_free_rate = 0.05
    maturity_years = 31.0 / 365.0
    discount_factor = np.exp(-risk_free_rate * maturity_years)

    puts = calls.copy()
    puts["option_type"] = "put"
    puts["last_price"] = (
        calls["last_price"] - spot_price + calls["strike"] * discount_factor
    ).abs()
    puts["bid"] = (calls["bid"] - spot_price + calls["strike"] * discount_factor).abs()
    puts["ask"] = (calls["ask"] - spot_price + calls["strike"] * discount_factor).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture(scope="module")
def multi_expiry_chain():
    """Create a two-expiry options chain with calls and parity-derived puts.

    Args:
        None.

    Returns:
        pd.DataFrame: Multi-expiry chain with required interface columns.
    """
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
    spot_price = 100.0
    risk_free_rate = 0.05
    t_array = (calls["expiry"] - pd.Timestamp("2025-01-01")).dt.days / 365.0
    discount_factor = np.exp(-risk_free_rate * t_array)

    puts = calls.copy()
    puts["option_type"] = "put"
    puts["last_price"] = (
        calls["last_price"] - spot_price + calls["strike"] * discount_factor
    ).abs()
    puts["bid"] = (calls["bid"] - spot_price + calls["strike"] * discount_factor).abs()
    puts["ask"] = (calls["ask"] - spot_price + calls["strike"] * discount_factor).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture(scope="module")
def fitted_vol_curve(single_expiry_chain, market_inputs):
    """Fit VolCurve once per module for interface smoke tests.

    Args:
        single_expiry_chain: Single-expiry option chain fixture.
        market_inputs: Standard market inputs fixture.

    Returns:
        VolCurve: Calibrated volatility curve.
    """
    from oipd import VolCurve

    return VolCurve().fit(single_expiry_chain, market_inputs)


@pytest.fixture(scope="module")
def fitted_vol_surface(multi_expiry_chain, market_inputs):
    """Fit VolSurface once per module for interface smoke tests.

    Args:
        multi_expiry_chain: Multi-expiry option chain fixture.
        market_inputs: Standard market inputs fixture.

    Returns:
        VolSurface: Calibrated volatility surface.
    """
    from oipd import VolSurface

    return VolSurface().fit(multi_expiry_chain, market_inputs)


@pytest.fixture(scope="module")
def prob_curve(fitted_vol_curve):
    """Derive ProbCurve once per module from a fitted volatility curve.

    Args:
        fitted_vol_curve: Pre-fitted volatility curve fixture.

    Returns:
        ProbCurve: Implied single-expiry distribution.
    """
    return fitted_vol_curve.implied_distribution()


@pytest.fixture(scope="module")
def prob_surface(fitted_vol_surface):
    """Derive ProbSurface once per module from a fitted volatility surface.

    Args:
        fitted_vol_surface: Pre-fitted volatility surface fixture.

    Returns:
        ProbSurface: Implied multi-expiry distribution surface.
    """
    return fitted_vol_surface.implied_distribution()
