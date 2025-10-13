from __future__ import annotations

import numpy as np
import pytest

from oipd.core.svi import (
    SVICalibrationDiagnostics,
    SVIParameters,
    from_total_variance,
    raw_to_jw,
    svi_total_variance,
)
from oipd.core.surface_fitting import (
    AVAILABLE_SURFACE_FITS,
    available_surface_fits,
    fit_surface,
)


def test_available_surface_fits_matches_constant():
    assert available_surface_fits() == AVAILABLE_SURFACE_FITS


def test_fit_surface_bspline_smoke():
    strikes = np.linspace(90.0, 110.0, 6)
    iv = np.linspace(0.2, 0.25, 6)
    vol_curve = fit_surface("bspline", strikes=strikes, iv=iv)
    eval_points = np.array([95.0, 105.0])
    result = vol_curve(eval_points)
    assert result.shape == eval_points.shape
    assert np.all(np.isfinite(result))


def test_fit_surface_svi_requires_forward_and_maturity():
    strikes = np.linspace(90.0, 110.0, 6)
    iv = np.linspace(0.2, 0.25, 6)

    with pytest.raises(ValueError):
        fit_surface("svi", strikes=strikes, iv=iv, maturity_years=0.5)

    with pytest.raises(ValueError):
        fit_surface("svi", strikes=strikes, iv=iv, forward=100.0)


def test_fit_surface_svi_unknown_option_rejected():
    strikes = np.linspace(90.0, 110.0, 6)
    iv = np.linspace(0.2, 0.25, 6)
    with pytest.raises(TypeError):
        fit_surface(
            "svi",
            strikes=strikes,
            iv=iv,
            forward=100.0,
            maturity_years=0.5,
            unknown_option=123,
        )


def test_fit_surface_unknown_method():
    strikes = np.array([1.0, 2.0, 3.0, 4.0])
    iv = np.array([0.1, 0.12, 0.13, 0.15])
    with pytest.raises(ValueError):
        fit_surface("unknown", strikes=strikes, iv=iv)


def test_fit_surface_svi_exposes_jw_and_diagnostics():
    params = SVIParameters(a=0.03, b=0.2, rho=-0.3, m=-0.05, sigma=0.25)
    maturity = 0.5
    forward = 100.0
    k = np.linspace(-0.4, 0.4, 11)
    strikes = forward * np.exp(k)
    total_var = svi_total_variance(k, params)
    iv = from_total_variance(total_var, maturity)

    vol_curve = fit_surface(
        "svi",
        strikes=strikes,
        iv=iv,
        forward=forward,
        maturity_years=maturity,
    )

    assert hasattr(vol_curve, "params_jw")
    fitted_params = SVIParameters(
        a=vol_curve.params["a"],
        b=vol_curve.params["b"],
        rho=vol_curve.params["rho"],
        m=vol_curve.params["m"],
        sigma=vol_curve.params["sigma"],
    )
    expected_jw = raw_to_jw(fitted_params)
    np.testing.assert_allclose(vol_curve.params_jw, expected_jw, atol=1e-9)

    assert hasattr(vol_curve, "diagnostics")
    diagnostics = vol_curve.diagnostics
    assert isinstance(diagnostics, SVICalibrationDiagnostics)
    assert hasattr(diagnostics, "rmse_unweighted")
    assert diagnostics.rmse_unweighted < 1e-2
