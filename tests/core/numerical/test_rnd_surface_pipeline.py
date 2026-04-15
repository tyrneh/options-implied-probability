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
from oipd.core.maturity import calculate_time_to_expiry
from oipd.core.utils import resolve_risk_free_rate
from oipd.pipelines.probability.rnd_surface import (
    build_fan_quantile_summary_frame,
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


def test_build_global_log_moneyness_grid_spans_surface_observations(
    fitted_surface: VolSurface,
) -> None:
    """Grid builder should be finite, increasing, and cover fitted observations."""
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

    assert actual.shape == (241,)
    assert np.all(np.isfinite(actual))
    assert np.all(np.diff(actual) > 0.0)
    assert actual[0] <= k_min
    assert actual[-1] >= k_max


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


def test_derive_surface_distribution_at_t_returns_valid_distribution(
    fitted_surface: VolSurface,
) -> None:
    """Surface derivation should return a valid density slice, not a legacy grid."""
    k_grid = build_global_log_moneyness_grid(fitted_surface, points=241)
    t_years = calculate_time_to_expiry(
        fitted_surface.expiries[0], fitted_surface._market.valuation_date
    )

    actual_strikes, actual_pdf, actual_cdf = derive_surface_distribution_at_t(
        fitted_surface,
        t_years,
        log_moneyness_grid=k_grid,
    )

    assert actual_strikes.shape == k_grid.shape
    assert actual_pdf.shape == k_grid.shape
    assert actual_cdf.shape == k_grid.shape
    assert np.all(np.isfinite(actual_strikes))
    assert np.all(np.isfinite(actual_pdf))
    assert np.all(np.isfinite(actual_cdf))
    assert np.all(actual_strikes > 0.0)
    assert np.all(np.diff(actual_strikes) > 0.0)
    assert np.all(actual_pdf >= 0.0)
    assert np.all(np.diff(actual_cdf) >= -1e-10)
    assert actual_cdf[0] >= -1e-10
    assert actual_cdf[-1] <= 1.0 + 1e-10
    assert np.trapezoid(actual_pdf, actual_strikes) == pytest.approx(1.0, rel=1e-5)


def test_build_fan_quantile_summary_frame_uses_daily_sampling(
    fitted_surface: VolSurface,
) -> None:
    """Fan summary frame preserves the daily sampling and quantile contract."""
    prob_surface = fitted_surface.implied_distribution()
    frame = build_fan_quantile_summary_frame(prob_surface)

    first_expiry = min(fitted_surface.expiries)
    last_expiry = max(fitted_surface.expiries)
    expected_days = (last_expiry - first_expiry).days + 1
    unique_expiries = pd.to_datetime(frame["expiry"])

    assert list(frame.columns) == [
        "expiry",
        "is_pillar",
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
    ]
    assert len(frame) == expected_days
    assert unique_expiries.nunique() == expected_days
    assert unique_expiries.min() == first_expiry
    assert unique_expiries.max() == last_expiry
    assert frame["is_pillar"].sum() == len(fitted_surface.expiries)
    pd.testing.assert_index_equal(
        pd.Index(unique_expiries[frame["is_pillar"]].tolist()),
        pd.Index(list(fitted_surface.expiries)),
    )

    quantile_columns = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]
    quantile_matrix = frame[quantile_columns].to_numpy(dtype=float)
    assert np.all(np.diff(quantile_matrix, axis=1) >= -1e-8)


def test_build_fan_quantile_summary_frame_skips_invalid_slice_and_warns(
    fitted_surface: VolSurface,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid sampled slices are skipped with one aggregate warning."""
    prob_surface = fitted_surface.implied_distribution()
    from oipd.interface.probability import ProbCurve

    original_slice = prob_surface.slice
    invalid_expiry = min(prob_surface.expiries) + pd.Timedelta(days=1)

    def _slice_with_invalid_curve(
        expiry_timestamp: str | date | pd.Timestamp,
    ) -> ProbCurve:
        """Return a malformed ProbCurve for one sampled non-pillar expiry."""
        curve = original_slice(expiry_timestamp)
        if pd.Timestamp(expiry_timestamp) == invalid_expiry:
            invalid_cdf = np.full_like(curve.cdf_values, np.nan, dtype=float)
            return ProbCurve.from_arrays(
                resolved_market=curve.resolved_market,
                metadata=dict(curve.metadata),
                prices=curve.prices,
                pdf_values=curve.pdf_values,
                cdf_values=invalid_cdf,
            )
        return curve

    monkeypatch.setattr(
        prob_surface,
        "slice",
        _slice_with_invalid_curve,
    )

    with pytest.warns(UserWarning, match="Skipped 1 sampled expiry"):
        frame = build_fan_quantile_summary_frame(prob_surface)

    unique_expiries = pd.to_datetime(frame["expiry"])
    first_expiry = min(fitted_surface.expiries)
    last_expiry = max(fitted_surface.expiries)
    expected_days = (last_expiry - first_expiry).days + 1

    assert len(frame) == expected_days - 1
    assert invalid_expiry not in set(unique_expiries)
    assert frame["is_pillar"].sum() == len(fitted_surface.expiries)

    quantile_columns = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]
    quantile_matrix = frame[quantile_columns].to_numpy(dtype=float)
    assert np.all(np.diff(quantile_matrix, axis=1) >= -1e-8)


def test_build_fan_quantile_summary_frame_raises_when_all_slices_invalid(
    fitted_surface: VolSurface,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summary generation still fails when every sampled slice is unusable."""
    prob_surface = fitted_surface.implied_distribution()
    from oipd.interface.probability import ProbCurve

    original_slice = prob_surface.slice

    def _slice_with_invalid_curves(
        expiry_timestamp: str | date | pd.Timestamp,
    ) -> ProbCurve:
        """Return a malformed ProbCurve for every sampled expiry."""
        curve = original_slice(expiry_timestamp)
        invalid_cdf = np.full_like(curve.cdf_values, np.nan, dtype=float)
        return ProbCurve.from_arrays(
            resolved_market=curve.resolved_market,
            metadata=dict(curve.metadata),
            prices=curve.prices,
            pdf_values=curve.pdf_values,
            cdf_values=invalid_cdf,
        )

    monkeypatch.setattr(
        prob_surface,
        "slice",
        _slice_with_invalid_curves,
    )

    with pytest.raises(
        ValueError,
        match="No valid probability slices available for fan summary generation",
    ):
        build_fan_quantile_summary_frame(prob_surface)
