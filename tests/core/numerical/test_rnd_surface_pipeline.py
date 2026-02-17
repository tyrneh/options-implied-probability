"""Tests for the stateless probability-surface pipeline helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from oipd import MarketInputs, VolSurface
from oipd.core.probability_density_conversion.finite_diff import (
    finite_diff_first_derivative,
)
from oipd.core.utils import calculate_time_to_expiry, resolve_risk_free_rate
from oipd.pipelines.probability.rnd_surface import (
    build_daily_fan_density_frame,
    build_global_log_moneyness_grid,
    derive_surface_distribution_at_t,
    resolve_surface_query_time,
)
from oipd.pricing import black76_call_price


def _build_multi_expiry_chain() -> pd.DataFrame:
    """Create a small two-expiry option chain fixture.

    Returns:
        pd.DataFrame: Multi-expiry options with calls and parity-derived puts.
    """
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
    valuation_date = pd.Timestamp("2025-01-01")
    t_array = (calls["expiry"] - valuation_date).dt.days / 365.0
    discount_factor = np.exp(-0.05 * t_array)

    puts = calls.copy()
    puts["option_type"] = "P"
    puts["last_price"] = (
        calls["last_price"] - 100.0 + calls["strike"] * discount_factor
    ).abs()
    puts["bid"] = (calls["bid"] - 100.0 + calls["strike"] * discount_factor).abs()
    puts["ask"] = (calls["ask"] - 100.0 + calls["strike"] * discount_factor).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def fitted_surface() -> VolSurface:
    """Build a fitted ``VolSurface`` used by pipeline tests.

    Returns:
        VolSurface: Fitted volatility surface.
    """
    market = MarketInputs(
        valuation_date=date(2025, 1, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
    )
    surface = VolSurface(pricing_engine="bs")
    surface.fit(_build_multi_expiry_chain(), market)
    return surface


def test_build_global_log_moneyness_grid_matches_legacy_logic(
    fitted_surface: VolSurface,
) -> None:
    """Grid builder matches the previous inlined ``ProbSurface`` logic."""
    actual = build_global_log_moneyness_grid(fitted_surface, points=241)

    k_min = np.inf
    k_max = -np.inf
    for expiry_timestamp in fitted_surface.expiries:
        slice_data = fitted_surface._model.get_slice(expiry_timestamp)
        metadata = slice_data.get("metadata", {})
        chain = slice_data.get("chain")
        forward_price = metadata.get("forward_price")
        if (
            chain is None
            or "strike" not in chain.columns
            or forward_price is None
            or float(forward_price) <= 0.0
        ):
            continue

        strikes = chain["strike"].to_numpy(dtype=float)
        valid_mask = np.isfinite(strikes) & (strikes > 0.0)
        if not np.any(valid_mask):
            continue

        log_moneyness = np.log(strikes[valid_mask] / float(forward_price))
        k_min = min(k_min, float(np.nanmin(log_moneyness)))
        k_max = max(k_max, float(np.nanmax(log_moneyness)))

    if not np.isfinite(k_min) or not np.isfinite(k_max):
        k_min, k_max = -1.25, 1.25
    elif np.isclose(k_min, k_max):
        k_min -= 0.25
        k_max += 0.25
    expected = np.linspace(
        k_min - 0.05 * (k_max - k_min), k_max + 0.05 * (k_max - k_min), 241
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_resolve_surface_query_time_matches_legacy_rules(
    fitted_surface: VolSurface,
) -> None:
    """Time resolver preserves strict domain checks and conversions."""
    expiry_ts, t_years = resolve_surface_query_time(fitted_surface, "2025-03-21")
    assert expiry_ts == pd.Timestamp("2025-03-21")
    np.testing.assert_allclose(t_years, 79.0 / 365.0)

    expiry_float, t_float = resolve_surface_query_time(fitted_surface, 79.0 / 365.0)
    assert expiry_float == pd.Timestamp("2025-03-21")
    np.testing.assert_allclose(t_float, 79.0 / 365.0)

    with pytest.raises(ValueError, match="strictly positive"):
        resolve_surface_query_time(fitted_surface, 0.0)
    with pytest.raises(ValueError, match="last fitted pillar"):
        resolve_surface_query_time(fitted_surface, "2030-12-31")


def test_derive_surface_distribution_at_t_matches_legacy_block(
    fitted_surface: VolSurface,
) -> None:
    """Slice derivation matches the previous inlined implementation exactly."""
    k_grid = build_global_log_moneyness_grid(fitted_surface, points=241)
    t_years = calculate_time_to_expiry(
        fitted_surface.expiries[0], fitted_surface._market.valuation_date
    )

    actual_strikes, actual_pdf, actual_cdf = derive_surface_distribution_at_t(
        fitted_surface,
        t_years,
        log_moneyness_grid=k_grid,
    )

    forward_price = float(fitted_surface.forward_price(t_years))
    expected_strikes = forward_price * np.exp(k_grid)
    effective_rate = resolve_risk_free_rate(
        fitted_surface._market.risk_free_rate,
        fitted_surface._market.risk_free_rate_mode,
        t_years,
    )
    implied_vols = np.asarray(
        fitted_surface._interpolator.implied_vol(expected_strikes, t_years),
        dtype=float,
    )
    call_prices = np.asarray(
        black76_call_price(
            forward_price,
            expected_strikes,
            implied_vols,
            t_years,
            effective_rate,
        ),
        dtype=float,
    )

    dcall_dk = np.asarray(
        finite_diff_first_derivative(call_prices, k_grid), dtype=float
    )
    dcall_dstrike = dcall_dk / expected_strikes
    expected_cdf = 1.0 + np.exp(effective_rate * t_years) * dcall_dstrike
    expected_cdf = np.clip(expected_cdf, 0.0, 1.0)
    expected_cdf = np.maximum.accumulate(expected_cdf)
    cdf_span = float(expected_cdf[-1] - expected_cdf[0])
    if cdf_span > 1e-10:
        expected_cdf = (expected_cdf - expected_cdf[0]) / cdf_span
    else:
        expected_cdf = np.linspace(0.0, 1.0, expected_cdf.size)
    expected_cdf = np.maximum.accumulate(np.clip(expected_cdf, 0.0, 1.0))

    dcdf_dk = finite_diff_first_derivative(expected_cdf, k_grid)
    expected_pdf = np.maximum(np.asarray(dcdf_dk, dtype=float) / expected_strikes, 0.0)
    expected_pdf = expected_pdf / float(np.trapz(expected_pdf, expected_strikes))

    increments = (
        0.5
        * (expected_pdf[1:] + expected_pdf[:-1])
        * (expected_strikes[1:] - expected_strikes[:-1])
    )
    expected_cdf = np.concatenate(([0.0], np.cumsum(increments)))
    expected_cdf = expected_cdf / float(expected_cdf[-1])
    expected_cdf = np.maximum.accumulate(np.clip(expected_cdf, 0.0, 1.0))

    np.testing.assert_allclose(actual_strikes, expected_strikes, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_pdf, expected_pdf, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_cdf, expected_cdf, rtol=0.0, atol=0.0)


def test_build_daily_fan_density_frame_uses_daily_sampling(
    fitted_surface: VolSurface,
) -> None:
    """Daily fan dataframe includes one maturity per calendar day."""
    k_grid = build_global_log_moneyness_grid(fitted_surface, points=241)
    frame = build_daily_fan_density_frame(
        fitted_surface,
        log_moneyness_grid=k_grid,
    )

    first_expiry = min(fitted_surface.expiries)
    last_expiry = max(fitted_surface.expiries)
    expected_days = (last_expiry - first_expiry).days + 1
    unique_days = frame["expiry_date"].nunique()

    assert set(frame.columns) == {"expiry_date", "strike", "cdf"}
    assert unique_days == expected_days
