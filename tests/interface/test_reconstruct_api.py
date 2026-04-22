"""Public-contract tests for reconstruction helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from oipd import rebuild_slice_from_svi, rebuild_surface_from_ssvi

SVI_PARAMS = {
    "a": 0.01,
    "b": 0.12,
    "rho": -0.25,
    "m": 0.0,
    "sigma": 0.2,
}


def test_rebuild_slice_accepts_time_to_expiry_days():
    """Rebuild helpers should expose continuous day-equivalent maturity."""
    rebuilt = rebuild_slice_from_svi(
        SVI_PARAMS,
        forward_price=100.0,
        time_to_expiry_days=30.4,
        risk_free_rate=0.01,
    )

    assert not rebuilt.data.empty
    assert "maturity" not in rebuilt.data.columns
    assert rebuilt.data["time_to_expiry_days"].iloc[0] == pytest.approx(30.4)
    assert rebuilt.data["time_to_expiry_years"].iloc[0] == pytest.approx(30.4 / 365.0)
    assert rebuilt.data["calendar_days_to_expiry"].iloc[0] == 30
    assert rebuilt.data["cdf_violation_policy"].iloc[0] == "warn"
    assert "cdf_monotonicity_repair_applied" in rebuilt.data.columns
    assert "raw_cdf_total_negative_variation" in rebuilt.data.columns


def test_rebuild_slice_accepts_raise_cdf_violation_policy():
    """Rebuild helper should expose strict CDF policy for research workflows."""
    rebuilt = rebuild_slice_from_svi(
        SVI_PARAMS,
        forward_price=100.0,
        time_to_expiry_days=30.4,
        risk_free_rate=0.01,
        cdf_violation_policy="raise",
    )

    assert rebuilt.data["cdf_violation_policy"].iloc[0] == "raise"


def test_rebuild_slice_rejects_days_to_expiry_kwarg():
    """Rebuild helpers should hard-fail on removed day-count kwargs."""
    with pytest.raises(TypeError, match="days_to_expiry"):
        rebuild_slice_from_svi(
            SVI_PARAMS,
            forward_price=100.0,
            days_to_expiry=30,
            risk_free_rate=0.01,
        )


def test_rebuild_slice_subday_alias_bucket_stays_zero():
    """Deprecated day-count alias should not force sub-day maturities up to 1 day."""
    rebuilt = rebuild_slice_from_svi(
        SVI_PARAMS,
        forward_price=100.0,
        time_to_expiry_days=0.25,
        risk_free_rate=0.01,
    )

    assert rebuilt.data["time_to_expiry_days"].iloc[0] == pytest.approx(0.25)
    assert rebuilt.data["calendar_days_to_expiry"].iloc[0] == 0


def test_rebuild_surface_accepts_time_to_expiry_days_column():
    """Surface rebuild should accept the new reporting field in snapshots."""
    ssvi_params = pd.DataFrame(
        {
            "time_to_expiry_years": [30.0 / 365.0, 60.0 / 365.0],
            "theta": [0.04, 0.06],
            "time_to_expiry_days": [30.0, 60.0],
            "rho": [-0.2, -0.2],
            "eta": [0.5, 0.5],
            "gamma": [0.4, 0.4],
            "alpha": [0.0, 0.0],
        }
    )
    surface = rebuild_surface_from_ssvi(
        ssvi_params,
        forward_prices={30.0 / 365.0: 100.0, 60.0 / 365.0: 101.0},
        risk_free_rate=0.01,
    )

    assert surface.available_time_to_expiry_days() == (30.0, 60.0)

    rebuilt = surface.slice(time_to_expiry_days=45.0)
    assert not rebuilt.data.empty
    assert rebuilt.data["time_to_expiry_days"].iloc[0] == pytest.approx(45.0)
    assert rebuilt.data["time_to_expiry_years"].iloc[0] == pytest.approx(45.0 / 365.0)
    assert rebuilt.data["cdf_violation_policy"].iloc[0] == "warn"
    assert "cdf_monotonicity_repair_applied" in rebuilt.data.columns


def test_rebuilt_surface_slice_rejects_days_to_expiry_kwarg():
    """RebuiltSurface.slice should hard-fail on the removed day-count kwarg."""
    ssvi_params = pd.DataFrame(
        {
            "time_to_expiry_years": [30.0 / 365.0, 60.0 / 365.0],
            "theta": [0.04, 0.06],
            "time_to_expiry_days": [30.0, 60.0],
            "rho": [-0.2, -0.2],
            "eta": [0.5, 0.5],
            "gamma": [0.4, 0.4],
            "alpha": [0.0, 0.0],
        }
    )
    surface = rebuild_surface_from_ssvi(
        ssvi_params,
        forward_prices={30.0 / 365.0: 100.0, 60.0 / 365.0: 101.0},
        risk_free_rate=0.01,
    )

    with pytest.raises(TypeError, match="days_to_expiry"):
        surface.slice(days_to_expiry=45)


def test_rebuild_surface_accepts_time_to_expiry_years_column_without_day_alias():
    """Surface rebuild should derive reporting days from the canonical years column."""
    ssvi_params = pd.DataFrame(
        {
            "time_to_expiry_years": [30.0 / 365.0, 60.0 / 365.0],
            "theta": [0.04, 0.06],
            "rho": [-0.2, -0.2],
            "eta": [0.5, 0.5],
            "gamma": [0.4, 0.4],
            "alpha": [0.0, 0.0],
        }
    )
    surface = rebuild_surface_from_ssvi(
        ssvi_params,
        forward_prices={30.0 / 365.0: 100.0, 60.0 / 365.0: 101.0},
        risk_free_rate=0.01,
    )

    assert surface.available_time_to_expiry_days() == pytest.approx((30.0, 60.0))


def test_rebuild_surface_accepts_raise_cdf_violation_policy():
    """Surface rebuild should carry strict CDF policy into cached slices."""
    ssvi_params = pd.DataFrame(
        {
            "time_to_expiry_years": [30.0 / 365.0, 60.0 / 365.0],
            "theta": [0.04, 0.06],
            "rho": [-0.2, -0.2],
            "eta": [0.5, 0.5],
            "gamma": [0.4, 0.4],
            "alpha": [0.0, 0.0],
        }
    )
    surface = rebuild_surface_from_ssvi(
        ssvi_params,
        forward_prices={30.0 / 365.0: 100.0, 60.0 / 365.0: 101.0},
        risk_free_rate=0.01,
        cdf_violation_policy="raise",
    )

    rebuilt = surface.slice(time_to_expiry_years=30.0 / 365.0)
    assert rebuilt.data["cdf_violation_policy"].iloc[0] == "raise"


def test_rebuild_surface_rejects_legacy_maturity_snapshot_column():
    """Surface rebuild should hard-fail on the removed years alias column."""
    ssvi_params = pd.DataFrame(
        {
            "maturity": [30.0 / 365.0, 60.0 / 365.0],
            "theta": [0.04, 0.06],
            "rho": [-0.2, -0.2],
            "eta": [0.5, 0.5],
            "gamma": [0.4, 0.4],
            "alpha": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="rename it to 'time_to_expiry_years'"):
        rebuild_surface_from_ssvi(
            ssvi_params,
            forward_prices={30.0 / 365.0: 100.0, 60.0 / 365.0: 101.0},
            risk_free_rate=0.01,
        )


def test_rebuild_surface_rejects_days_to_expiry_snapshot_column():
    """Surface rebuild should hard-fail on the removed snapshot day-count column."""
    ssvi_params = pd.DataFrame(
        {
            "time_to_expiry_years": [30.0 / 365.0, 60.0 / 365.0],
            "days_to_expiry": [30, 60],
            "theta": [0.04, 0.06],
            "rho": [-0.2, -0.2],
            "eta": [0.5, 0.5],
            "gamma": [0.4, 0.4],
            "alpha": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="rename it to 'time_to_expiry_days'"):
        rebuild_surface_from_ssvi(
            ssvi_params,
            forward_prices={30.0 / 365.0: 100.0, 60.0 / 365.0: 101.0},
            risk_free_rate=0.01,
        )


def test_rebuilt_surface_no_longer_exposes_available_days():
    """RebuiltSurface should expose only the canonical day-equivalent accessor."""
    ssvi_params = pd.DataFrame(
        {
            "time_to_expiry_years": [30.0 / 365.0, 60.0 / 365.0],
            "theta": [0.04, 0.06],
            "rho": [-0.2, -0.2],
            "eta": [0.5, 0.5],
            "gamma": [0.4, 0.4],
            "alpha": [0.0, 0.0],
        }
    )
    surface = rebuild_surface_from_ssvi(
        ssvi_params,
        forward_prices={30.0 / 365.0: 100.0, 60.0 / 365.0: 101.0},
        risk_free_rate=0.01,
    )

    assert not hasattr(surface, "available_days")
