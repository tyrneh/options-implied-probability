from __future__ import annotations

import numpy as np
import pytest

from oipd.core.surface_fitting import (
    available_surface_fits,
    fit_surface,
    SVIFitDiagnostics,
    SVIConfig,
    SurfaceConfig,
    SmileSlice,
)


def test_surface_config_defaults():
    config = SurfaceConfig()
    kwargs = config.kwargs()
    assert config.name == "svi"
    assert kwargs == SVIConfig().to_dict()


def test_surface_config_dict_passthrough():
    payload = {"foo": 1, "bar": 2}
    config = SurfaceConfig(name="bspline", params=payload)
    kwargs = config.kwargs()
    assert kwargs == payload
    assert kwargs is not payload  # ensure copy


def test_surface_config_svi_override():
    svi_cfg = SVIConfig(max_iter=100, tol=1e-6)
    config = SurfaceConfig(name="svi", params=svi_cfg)
    kwargs = config.kwargs()
    assert kwargs["max_iter"] == 100
    assert kwargs["tol"] == 1e-6


def test_surface_config_invalid_params():
    class Dummy:
        pass

    config = SurfaceConfig(name="svi", params=Dummy())
    with pytest.raises(TypeError):
        config.kwargs()


def test_smile_slice_basic():
    k = np.linspace(-0.2, 0.2, 5)
    tv = np.linspace(0.03, 0.05, 5)
    slice_ = SmileSlice(k=k, total_variance=tv, maturity_years=0.5, forward=100.0)
    np.testing.assert_allclose(slice_.k, k)
    np.testing.assert_allclose(slice_.total_variance, tv)


def test_smile_slice_requires_positive_inputs():
    k = np.linspace(-0.1, 0.1, 5)
    tv = np.linspace(0.02, 0.03, 5)

    with pytest.raises(ValueError):
        SmileSlice(k=k, total_variance=tv, maturity_years=0.0, forward=100.0)
    with pytest.raises(ValueError):
        SmileSlice(k=k, total_variance=tv, maturity_years=0.5, forward=-10.0)


def test_smile_slice_shape_and_length_checks():
    k = np.linspace(-0.1, 0.1, 5)
    tv = np.linspace(0.02, 0.03, 4)
    with pytest.raises(ValueError):
        SmileSlice(k=k, total_variance=tv, maturity_years=0.5, forward=100.0)

    k2 = np.random.randn(2, 3)
    tv2 = np.random.rand(2, 3)
    with pytest.raises(ValueError):
        SmileSlice(k=k2, total_variance=tv2, maturity_years=0.5, forward=100.0)

    k3 = np.linspace(-0.1, 0.1, 4)
    tv3 = np.linspace(0.02, 0.03, 4)
    with pytest.raises(ValueError):
        SmileSlice(k=k3, total_variance=tv3, maturity_years=0.5, forward=100.0)


def test_smile_slice_rejects_negative_total_variance():
    k = np.linspace(-0.1, 0.1, 5)
    tv = np.array([0.02, 0.025, -0.01, 0.03, 0.035])
    with pytest.raises(ValueError):
        SmileSlice(k=k, total_variance=tv, maturity_years=0.5, forward=100.0)


def test_svi_fit_diagnostics_defaults():
    diag = SVIFitDiagnostics()
    assert diag.status == "not_run"
    assert diag.objective is None
    assert diag.extras == {}


def test_registry_lists_default_fits():
    fits = available_surface_fits()
    assert "bspline" in fits
    assert "svi" in fits


def test_fit_surface_bspline_smoke():
    strikes = np.linspace(90.0, 110.0, 6)
    iv = np.linspace(0.2, 0.25, 6)
    config = SurfaceConfig(name="bspline")
    vol_curve = fit_surface(config, strikes=strikes, iv=iv)
    eval_points = np.array([95.0, 105.0])
    result = vol_curve(eval_points)
    assert result.shape == eval_points.shape
    assert np.all(np.isfinite(result))


def test_fit_surface_unknown_name():
    config = SurfaceConfig(name="bspline")
    object.__setattr__(config, "name", "unknown")  # type: ignore[misc]
    with pytest.raises(ValueError):
        fit_surface(config, strikes=np.array([1, 2, 3]), iv=np.array([0.1, 0.2, 0.3]))
