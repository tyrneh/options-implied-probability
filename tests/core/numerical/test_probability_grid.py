"""Tests for uniform strike grid construction in probability pipelines."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from oipd.market_inputs import MarketInputs, resolve_market
from oipd.pipelines.probability.rnd_curve import _build_strike_grid


def _make_resolved_market() -> "ResolvedMarket":
    """Create a resolved market snapshot for grid tests.

    Returns:
        ResolvedMarket: Resolved market inputs with a stable valuation date.
    """
    inputs = MarketInputs(
        risk_free_rate=0.02,
        valuation_date=date(2026, 2, 4),
        underlying_price=100.0,
    )
    return resolve_market(inputs)


def test_build_strike_grid_uses_explicit_domain():
    """Builds a uniform grid that honors an explicit domain."""
    resolved_market = _make_resolved_market()
    vol_meta = {"expiry_date": date(2026, 3, 4), "at_money_vol": 0.2}

    grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=100.0,
        days_to_expiry=28,
        domain=(90.0, 110.0),
        points=11,
    )

    assert grid.shape[0] == 11
    np.testing.assert_allclose(grid[0], 90.0)
    np.testing.assert_allclose(grid[-1], 110.0)
    np.testing.assert_allclose(np.diff(grid), np.diff(grid)[0])


def test_build_strike_grid_uses_default_domain():
    """Uses metadata default_domain when no explicit domain is provided."""
    resolved_market = _make_resolved_market()
    vol_meta = {
        "default_domain": (80.0, 120.0),
        "expiry_date": date(2026, 3, 4),
        "at_money_vol": 0.2,
    }

    grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=100.0,
        days_to_expiry=28,
        domain=None,
        points=11,
    )

    np.testing.assert_allclose(grid[0], 80.0)
    np.testing.assert_allclose(grid[-1], 120.0)
    np.testing.assert_allclose(np.diff(grid), np.diff(grid)[0])


def test_build_strike_grid_uses_observed_strikes_for_bounds():
    """Infers bounds from observed strikes when no domain is provided."""
    resolved_market = _make_resolved_market()
    observed_iv = pd.DataFrame({"strike": [95.0, 105.0]})
    vol_meta = {
        "observed_iv": observed_iv,
        "expiry_date": date(2026, 3, 4),
        "at_money_vol": 0.2,
    }

    grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=100.0,
        days_to_expiry=28,
        domain=None,
        points=11,
    )

    np.testing.assert_allclose(grid[0], 94.5)
    np.testing.assert_allclose(grid[-1], 105.5)
    np.testing.assert_allclose(np.diff(grid), np.diff(grid)[0])


def test_build_strike_grid_falls_back_to_atm_based_bounds():
    """Falls back to ATM-vol-based bounds when no domain or observed strikes."""
    resolved_market = _make_resolved_market()
    vol_meta = {"expiry_date": date(2026, 3, 4), "at_money_vol": 0.25}

    grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=100.0,
        days_to_expiry=28,
        domain=None,
        points=11,
    )

    assert grid[0] > 0
    assert grid[-1] > grid[0]
    np.testing.assert_allclose(np.diff(grid), np.diff(grid)[0])
