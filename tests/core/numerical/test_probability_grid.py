"""Tests for strike-grid and PDF-domain construction in probability pipelines."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from oipd.market_inputs import MarketInputs, resolve_market
from oipd.pipelines.probability.rnd_curve import (
    DEFAULT_NATIVE_GRID_MAX_POINTS,
    DEFAULT_NATIVE_GRID_MIN_POINTS,
    DEFAULT_PDF_TAIL_MASS_TOLERANCE,
    DEFAULT_VIEW_DOMAIN_SOURCE,
    DEFAULT_VIEW_QUANTILES,
    _build_strike_grid,
    _build_grid_with_spacing,
    _grid_spacing_from_initial_domain,
    _resolve_native_grid_points,
    _resolve_default_view_domain,
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


def test_resolve_native_grid_points_small_domain_hits_min_cap() -> None:
    """Auto native grid policy should not go below the minimum point cap."""
    result = _resolve_native_grid_points(
        (99.0, 101.0),
        pricing_underlying=100.0,
    )

    assert result.policy == "auto"
    assert result.points == DEFAULT_NATIVE_GRID_MIN_POINTS
    assert result.hit_min_cap
    assert not result.hit_max_cap
    assert result.actual_step < 0.01


def test_resolve_native_grid_points_wide_spy_domain_exceeds_legacy_default() -> None:
    """Wide SPY-like domains should receive more than 200 native points."""
    result = _resolve_native_grid_points(
        (0.01, 1635.0),
        pricing_underlying=500.0,
        observed_strikes=pd.DataFrame({"strike": np.arange(450.0, 551.0, 5.0)}),
    )

    assert result.policy == "auto"
    assert result.points > 200
    assert result.points > DEFAULT_NATIVE_GRID_MIN_POINTS
    assert result.observed_gap == pytest.approx(5.0)
    assert result.target_step == pytest.approx(2.5)
    assert result.actual_step <= result.target_step


def test_resolve_native_grid_points_btc_domain_does_not_exceed_max_cap() -> None:
    """BTC-like wide domains should be capped at the native-grid maximum."""
    result = _resolve_native_grid_points(
        (0.01, 2_000_000.0),
        pricing_underlying=100_000.0,
    )

    assert result.policy == "auto"
    assert result.points == DEFAULT_NATIVE_GRID_MAX_POINTS
    assert result.hit_max_cap
    assert not result.hit_min_cap


def test_resolve_native_grid_points_fixed_grid_points_override_auto() -> None:
    """Explicit fixed grid_points should resolve exactly to the requested size."""
    result = _resolve_native_grid_points(
        (0.01, 1635.0),
        pricing_underlying=500.0,
        grid_points=400,
        observed_strikes=pd.DataFrame({"strike": [450.0, 455.0, 460.0]}),
    )

    assert result.policy == "fixed"
    assert result.points == 400
    assert result.target_step is None
    assert not result.hit_min_cap
    assert not result.hit_max_cap
    assert result.observed_gap == pytest.approx(5.0)


def test_resolve_default_view_domain_includes_observed_domain_and_anchors() -> None:
    """Default view domain should stay native-bounded but include key anchors."""
    prices = np.linspace(0.01, 200.0, 1001)
    cdf_values = np.linspace(0.0, 1.0, prices.size)

    metadata = _resolve_default_view_domain(
        prices,
        cdf_values,
        observed_domain=(80.0, 120.0),
        pricing_underlying=100.0,
        spot_price=95.0,
        forward_price=125.0,
    )

    view_low, view_high = metadata["default_view_domain"]
    assert metadata["default_view_domain_source"] == DEFAULT_VIEW_DOMAIN_SOURCE
    assert metadata["default_view_quantiles"] == DEFAULT_VIEW_QUANTILES
    assert prices[0] <= view_low < view_high <= prices[-1]
    assert view_low <= 80.0
    assert view_high >= 120.0
    assert view_low <= 95.0 <= view_high
    assert view_low <= 100.0 <= view_high
    assert view_low <= 125.0 <= view_high


def test_resolve_default_view_domain_excludes_btc_like_far_tail() -> None:
    """Compact view domain should avoid showing irrelevant wide right tails."""
    prices = np.linspace(0.01, 2_000_000.0, 2500)
    logistic_argument = np.clip((prices - 100_000.0) / 10_000.0, -50.0, 50.0)
    cdf_values = 1.0 / (1.0 + np.exp(-logistic_argument))
    cdf_values[0] = 0.0
    cdf_values[-1] = 1.0

    metadata = _resolve_default_view_domain(
        prices,
        cdf_values,
        observed_domain=(90_000.0, 110_000.0),
        pricing_underlying=100_000.0,
        spot_price=100_000.0,
        forward_price=101_000.0,
    )

    view_low, view_high = metadata["default_view_domain"]
    assert prices[0] <= view_low < view_high <= prices[-1]
    assert view_low <= 90_000.0
    assert view_high >= 110_000.0
    assert view_low <= 101_000.0 <= view_high
    assert view_high < 500_000.0
    assert view_high < prices[-1] * 0.5


@pytest.mark.parametrize("bad_grid_points", [True, False, 4, 0, -1])
def test_resolve_native_grid_points_rejects_bool_and_too_few_points(
    bad_grid_points,
) -> None:
    """Explicit native grid_points should reject booleans and values below 5."""
    with pytest.raises(ValueError, match="grid_points"):
        _resolve_native_grid_points(
            (0.01, 100.0),
            pricing_underlying=100.0,
            grid_points=bad_grid_points,
        )


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
