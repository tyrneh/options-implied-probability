"""Tests for the stateless probability-surface pipeline helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from oipd import MarketInputs, VolSurface
from oipd.pipelines.probability.rnd_surface import (
    build_fan_quantile_summary_frame,
    resolve_surface_query_time,
)


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
