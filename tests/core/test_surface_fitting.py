from __future__ import annotations

import numpy as np
import pytest

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
