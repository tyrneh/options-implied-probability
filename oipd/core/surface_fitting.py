"""Simple helpers for fitting single-expiry implied-volatility smiles."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping

import numpy as np
from scipy import interpolate

from oipd.core.errors import CalculationError
from oipd.core.svi import (
    DEFAULT_SVI_OPTIONS,
    calibrate_svi_parameters,
    from_total_variance,
    log_moneyness,
    merge_svi_options,
    svi_total_variance,
    to_total_variance,
)

VolCurve = Callable[[Iterable[float] | np.ndarray], np.ndarray]

AVAILABLE_SURFACE_FITS: tuple[str, ...] = ("bspline", "svi")
DEFAULT_BSPLINE_OPTIONS: dict[str, Any] = {
    "smoothing_factor": 10.0,
    "degree": 3,
    "dx": 0.1,
}


def available_surface_fits() -> tuple[str, ...]:
    """Return the supported surface fitting methods."""

    return AVAILABLE_SURFACE_FITS


def fit_surface(
    method: str,
    *,
    strikes: np.ndarray,
    iv: np.ndarray,
    forward: float | None = None,
    maturity_years: float | None = None,
    options: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> VolCurve:
    """Fit an implied-volatility smile using the requested method."""

    method_name = method.lower()
    if method_name not in AVAILABLE_SURFACE_FITS:
        raise ValueError(
            f"Unknown surface fit '{method}'. Available: {AVAILABLE_SURFACE_FITS}"
        )

    strikes_arr = np.asarray(strikes, dtype=float)
    iv_arr = np.asarray(iv, dtype=float)
    if strikes_arr.shape != iv_arr.shape:
        raise ValueError("Strike and IV arrays must have identical shapes")

    method_options: dict[str, Any] = {}
    if options:
        method_options.update(dict(options))
    if overrides:
        method_options.update(overrides)

    if method_name == "svi":
        return _fit_svi(
            strikes_arr,
            iv_arr,
            forward=forward,
            maturity_years=maturity_years,
            **method_options,
        )

    if forward is not None or maturity_years is not None:
        raise ValueError(
            "forward and maturity_years are only valid for the 'svi' method"
        )

    return _fit_bspline(strikes_arr, iv_arr, **method_options)


def _build_svi_options(overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    if overrides is None:
        return dict(DEFAULT_SVI_OPTIONS)
    return merge_svi_options(overrides)


def _fit_svi(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    forward: float | None,
    maturity_years: float | None,
    **config_overrides: Any,
) -> VolCurve:
    if forward is None or forward <= 0:
        raise ValueError("forward must be a positive number for SVI fitting")
    if maturity_years is None or maturity_years <= 0:
        raise ValueError("maturity_years must be positive for SVI fitting")

    config = _build_svi_options(config_overrides or None)

    k = log_moneyness(strikes, forward)
    total_variance = to_total_variance(iv, maturity_years)

    params, _ = calibrate_svi_parameters(
        k,
        total_variance,
        maturity_years,
        config,
    )

    fitted_total_variance = svi_total_variance(k, params)
    fitted_iv = from_total_variance(fitted_total_variance, maturity_years)

    def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        eval_array = np.asarray(eval_strikes, dtype=float)
        eval_k = log_moneyness(eval_array, forward)
        total_var_eval = svi_total_variance(eval_k, params)
        return from_total_variance(total_var_eval, maturity_years)

    vol_curve.grid = (strikes, fitted_iv)  # type: ignore[attr-defined]
    vol_curve.params = {
        "method": "svi",
        "a": params.a,
        "b": params.b,
        "rho": params.rho,
        "m": params.m,
        "sigma": params.sigma,
        "forward": forward,
        "maturity_years": maturity_years,
    }  # type: ignore[attr-defined]
    return vol_curve
# TODO: WHY IS VOL CURVE RETURNED AS A FUNCTION?

def _fit_bspline(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    smoothing_factor: float = 10.0,
    degree: int = 3,
    dx: float = 0.1,
) -> VolCurve:
    if strikes.size < 4:
        raise CalculationError(
            "Insufficient data for B-spline fitting: need at least 4 points"
        )

    try:
        tck = interpolate.splrep(strikes, iv, s=smoothing_factor, k=degree)
    except Exception as exc:  # pragma: no cover - scipy error propagation
        raise CalculationError(
            f"Failed to fit B-spline to implied volatility data: {exc}"
        ) from exc

    span = strikes.max() - strikes.min()
    steps = max(2, int(np.ceil(span / max(dx, 1e-6))))
    grid_x = np.linspace(strikes.min(), strikes.max(), steps)
    spline = interpolate.BSpline(*tck)
    grid_y = spline(grid_x)

    def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        eval_array = np.asarray(eval_strikes, dtype=float)
        return spline(eval_array)

    vol_curve.grid = (grid_x, grid_y)  # type: ignore[attr-defined]
    vol_curve.params = {
        "method": "bspline",
        "smoothing_factor": smoothing_factor,
        "degree": degree,
        "dx": dx,
    }  # type: ignore[attr-defined]
    return vol_curve


__all__ = [
    "AVAILABLE_SURFACE_FITS",
    "DEFAULT_BSPLINE_OPTIONS",
    "DEFAULT_SVI_OPTIONS",
    "VolCurve",
    "available_surface_fits",
    "fit_surface",
]
