"""Simple helpers for fitting single-expiry implied-volatility smiles."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy import interpolate

from oipd.core.errors import CalculationError
from oipd.core.svi import (
    DEFAULT_SVI_OPTIONS,
    calibrate_svi_parameters,
    from_total_variance,
    log_moneyness,
    merge_svi_options,
    raw_to_jw,
    svi_total_variance,
    to_total_variance,
)
from oipd.core.svi_types import SVICalibrationOptions

VolCurve = Callable[[Iterable[float] | np.ndarray], np.ndarray]

AVAILABLE_SURFACE_FITS: tuple[str, ...] = ("bspline", "svi")
DEFAULT_BSPLINE_OPTIONS: dict[str, Any] = {
    "smoothing_factor": 10.0,
    "degree": 3,
    "dx": 0.1,
}


def available_surface_fits() -> tuple[str, ...]:
    """List the supported volatility surface fitting methods.

    Returns:
        A tuple containing the canonical names of all supported surface fit
        implementations in the order they should be displayed to users.
    """

    return AVAILABLE_SURFACE_FITS


def fit_surface(
    method: str,
    *,
    strikes: np.ndarray,
    iv: np.ndarray,
    forward: float | None = None,
    maturity_years: float | None = None,
    options: SVICalibrationOptions | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> VolCurve:
    """Fit an implied-volatility smile using the requested method.

    Args:
        method: Name of the surface fitting approach to use.
        strikes: Observed strikes expressed in absolute terms.
        iv: Implied volatilities corresponding to ``strikes``.
        forward: Forward price for the underlying asset when SVI is requested.
        maturity_years: Time to expiry in years for SVI calibration.
        options: Optional SVI calibration configuration when ``method`` is ``"svi"``.
        **overrides: Keyword-only overrides applied on top of ``options``.

    Returns:
        A callable that produces implied volatility values when evaluated at
        arbitrary strike inputs.

    Raises:
        ValueError: If the method is unknown, if the strike and volatility
            arrays have mismatched shapes, or if arguments specific to SVI are
            supplied for other methods.
    """

    method_name = method.lower()
    if method_name not in AVAILABLE_SURFACE_FITS:
        raise ValueError(
            f"Unknown surface fit '{method}'. Available: {AVAILABLE_SURFACE_FITS}"
        )

    strikes_arr = np.asarray(strikes, dtype=float)
    iv_arr = np.asarray(iv, dtype=float)
    if strikes_arr.shape != iv_arr.shape:
        raise ValueError("Strike and IV arrays must have identical shapes")

    if options is not None and not isinstance(options, (SVICalibrationOptions, Mapping)):
        raise TypeError(
            "options must be a mapping of SVI overrides or an SVICalibrationOptions instance"
        )
    method_overrides = dict(overrides) if overrides else {}

    if method_name == "svi":
        bid_iv = method_overrides.pop("bid_iv", None)
        ask_iv = method_overrides.pop("ask_iv", None)
        volumes = method_overrides.pop("volumes", None)

        return _fit_svi(
            strikes_arr,
            iv_arr,
            forward=forward,
            maturity_years=maturity_years,
            config=options,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
            volumes=volumes,
            **method_overrides,
        )

    if forward is not None or maturity_years is not None:
        raise ValueError(
            "forward and maturity_years are only valid for the 'svi' method"
        )

    return _fit_bspline(strikes_arr, iv_arr, **method_overrides)


def _fit_svi(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    forward: float | None,
    maturity_years: float | None,
    config: SVICalibrationOptions | None = None,
    bid_iv: np.ndarray | None = None,
    ask_iv: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    **config_overrides: Any,
) -> VolCurve:
    """Calibrate an SVI smile and return an evaluator over strike space.

    Args:
        strikes: Observed strike values for the maturity of interest.
        iv: Implied volatilities aligned with ``strikes``.
        forward: Forward price corresponding to the maturity.
        maturity_years: Time to expiry in years used for calibration.
        config: Optional base calibration options.
        bid_iv: Optional bid implied volatility quotes aligning with ``strikes``.
        ask_iv: Optional ask implied volatility quotes aligning with ``strikes``.
        volumes: Optional trade volumes aligning with ``strikes``.
        **config_overrides: Additional SVI option overrides applied on top of
            ``config``.

    Returns:
        A callable that evaluates the calibrated SVI smile at arbitrary
        strike levels.

    Raises:
        ValueError: If the forward or maturity inputs are missing or invalid.
        CalculationError: If SVI calibration fails to converge.
    """

    if forward is None or forward <= 0:
        raise ValueError("forward must be a positive number for SVI fitting")
    if maturity_years is None or maturity_years <= 0:
        raise ValueError("maturity_years must be positive for SVI fitting")

    base_config = merge_svi_options(config)

    if config_overrides:
        unknown = set(config_overrides) - SVICalibrationOptions.field_names()
        if unknown:
            raise TypeError(f"Unknown SVI option(s): {sorted(unknown)}")
        merged = base_config.to_mapping()
        merged.update(config_overrides)
        config = merge_svi_options(merged)
    else:
        config = base_config

    k = log_moneyness(strikes, forward)
    total_variance = to_total_variance(iv, maturity_years)

    params, diagnostics = calibrate_svi_parameters(
        k,
        total_variance,
        maturity_years,
        config,
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    fitted_total_variance = svi_total_variance(k, params)
    fitted_iv = from_total_variance(fitted_total_variance, maturity_years)

    def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        """Evaluate the calibrated SVI smile at the provided strike inputs.

        Args:
            eval_strikes: Strikes at which to compute implied volatility.

        Returns:
            Implied volatilities implied by the calibrated SVI smile.
        """

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
    vol_curve.params_raw = (params.a, params.b, params.rho, params.m, params.sigma)  # type: ignore[attr-defined]
    vol_curve.params_jw = raw_to_jw(params)  # type: ignore[attr-defined]
    vol_curve.diagnostics = diagnostics  # type: ignore[attr-defined]
    return vol_curve


def _fit_bspline(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    smoothing_factor: float = 10.0,
    degree: int = 3,
    dx: float = 0.1,
) -> VolCurve:
    """Fit a smoothing B-spline through an implied-volatility smile.

    Args:
        strikes: Observed strike values for the maturity of interest.
        iv: Implied volatilities aligned with ``strikes``.
        smoothing_factor: Smoothing factor passed to ``scipy.splrep``.
        degree: Degree of the B-spline basis.
        dx: Target spacing for the stored evaluation grid.

    Returns:
        A callable that evaluates the fitted B-spline at arbitrary strikes.

    Raises:
        CalculationError: If too few strikes are provided or if SciPy fails to
            produce a spline representation.
    """

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
        """Evaluate the fitted B-spline at the requested strike locations.

        Args:
            eval_strikes: Strikes at which to interpolate implied volatility.

        Returns:
            Implied volatilities interpolated from the fitted spline.
        """

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
