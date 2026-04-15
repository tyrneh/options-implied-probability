"""Tests for strike-grid and PDF-domain construction in probability pipelines."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from oipd.market_inputs import MarketInputs, resolve_market
from oipd.pipelines.probability.rnd_curve import (
    DEFAULT_PDF_TAIL_MASS_TOLERANCE,
    _build_strike_grid,
    _build_grid_with_spacing,
    _grid_spacing_from_initial_domain,
    resolve_pdf_domain,
)


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
    vol_meta = {"expiry": date(2026, 3, 4), "at_money_vol": 0.2}

    grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=100.0,
        time_to_expiry_years=28 / 365.0,
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
        "expiry": date(2026, 3, 4),
        "at_money_vol": 0.2,
    }

    grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=100.0,
        time_to_expiry_years=28 / 365.0,
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
        "expiry": date(2026, 3, 4),
        "at_money_vol": 0.2,
    }

    grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=100.0,
        time_to_expiry_years=28 / 365.0,
        domain=None,
        points=11,
    )

    np.testing.assert_allclose(grid[0], 94.5)
    np.testing.assert_allclose(grid[-1], 105.5)
    np.testing.assert_allclose(np.diff(grid), np.diff(grid)[0])


def test_resolve_pdf_domain_starts_near_zero_when_observed_strikes_exist():
    """Native PDF domain starts near zero instead of observed strike min."""
    resolved_market = _make_resolved_market()
    vol_meta = {
        "observed_iv": pd.DataFrame({"strike": [95.0, 105.0]}),
        "raw_observed_domain": (90.0, 110.0),
        "expiry": date(2026, 3, 4),
        "at_money_vol": 0.2,
    }

    flat_vol = lambda strikes: np.full_like(np.asarray(strikes, dtype=float), 0.2)

    domain, metadata = resolve_pdf_domain(
        flat_vol,
        resolved_market,
        pricing_engine="bs",
        vol_metadata=vol_meta,
        pricing_underlying=100.0,
        effective_r=0.02,
        effective_dividend=0.0,
        years_to_expiry=28 / 365.0,
        points=51,
    )

    assert domain[0] == pytest.approx(0.01)
    assert domain[1] > 105.0
    assert metadata["observed_domain"] == (95.0, 105.0)
    assert metadata["raw_observed_domain"] == (90.0, 110.0)


def test_resolve_pdf_domain_widens_until_uncovered_mass_is_small():
    """Adaptive resolver expands the right bound when mass remains outside."""
    resolved_market = _make_resolved_market()
    vol_meta = {
        "observed_iv": pd.DataFrame({"strike": [80.0, 120.0]}),
        "expiry": date(2026, 3, 4),
        "at_money_vol": 0.25,
    }

    flat_vol = lambda strikes: np.full_like(np.asarray(strikes, dtype=float), 0.25)

    domain, metadata = resolve_pdf_domain(
        flat_vol,
        resolved_market,
        pricing_engine="bs",
        vol_metadata=vol_meta,
        pricing_underlying=100.0,
        effective_r=0.02,
        effective_dividend=0.0,
        years_to_expiry=28 / 365.0,
        points=41,
        initial_upper_multiplier=0.9,
        max_expansions=4,
    )

    assert domain[1] > 108.0
    assert metadata["domain_expansions"] >= 1
    assert metadata["added_mass_last_expansion"] <= DEFAULT_PDF_TAIL_MASS_TOLERANCE


def test_resolve_pdf_domain_treats_default_domain_as_seed_not_hard_bound():
    """Interpolated default_domain hints seed the search without forcing the final bound."""
    resolved_market = _make_resolved_market()
    vol_meta = {
        "default_domain": (80.0, 300.0),
        "expiry": date(2026, 3, 4),
        "at_money_vol": 0.10,
    }

    flat_vol = lambda strikes: np.full_like(np.asarray(strikes, dtype=float), 0.10)

    domain, metadata = resolve_pdf_domain(
        flat_vol,
        resolved_market,
        pricing_engine="bs",
        vol_metadata=vol_meta,
        pricing_underlying=100.0,
        effective_r=0.02,
        effective_dividend=0.0,
        years_to_expiry=28 / 365.0,
        points=41,
        initial_upper_multiplier=0.5,
        max_expansions=1,
    )

    assert domain[1] < 300.0
    assert metadata["raw_observed_domain"] is None
    assert metadata["observed_domain"] is None


def test_build_grid_with_spacing_preserves_spacing_when_domain_widens():
    """Expanded domains keep approximately constant spacing."""
    initial_domain = (0.01, 100.0)
    widened_domain = (0.01, 200.0)
    base_step = _grid_spacing_from_initial_domain(initial_domain, points=51)

    grid = _build_grid_with_spacing(
        pricing_underlying=100.0,
        domain=widened_domain,
        base_step=base_step,
    )

    diffs = np.diff(grid)
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)
    assert diffs[0] == pytest.approx(base_step, rel=0.05)


def test_build_strike_grid_falls_back_to_atm_based_bounds():
    """Falls back to ATM-vol-based bounds when no domain or observed strikes."""
    resolved_market = _make_resolved_market()
    vol_meta = {"expiry": date(2026, 3, 4), "at_money_vol": 0.25}

    grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=100.0,
        time_to_expiry_years=28 / 365.0,
        domain=None,
        points=11,
    )

    assert grid[0] > 0
    assert grid[-1] > grid[0]
    np.testing.assert_allclose(np.diff(grid), np.diff(grid)[0])
