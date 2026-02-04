"""SVI single-slice calibration implementation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from oipd.core.errors import CalculationError
from oipd.core.vol_surface_fitting.shared.svi import (
    DEFAULT_SVI_OPTIONS,
    calibrate_svi_parameters,
    from_total_variance,
    log_moneyness,
    merge_svi_options,
    raw_to_jw,
    svi_total_variance,
    to_total_variance,
)
from oipd.core.vol_surface_fitting.shared.svi_types import SVICalibrationOptions


def fit_svi_slice(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    forward: float | None,
    maturity_years: float | None,
    options: SVICalibrationOptions | Mapping[str, Any] | None = None,
    bid_iv: np.ndarray | None = None,
    ask_iv: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    **overrides: Any,
):
    """Calibrate an SVI smile and return a callable volatility curve.

    Args:
        strikes: Observed strike values for the maturity of interest.
        iv: Implied volatilities aligned with ``strikes``.
        forward: Forward price corresponding to the maturity.
        maturity_years: Time to expiry in years used for calibration.
        options: Optional base calibration options.
        bid_iv: Optional bid implied volatility quotes aligning with ``strikes``.
        ask_iv: Optional ask implied volatility quotes aligning with ``strikes``.
        volumes: Optional trade volumes aligning with ``strikes``.
        **overrides: Additional SVI option overrides applied on top of
            ``options``.

    Returns:
        Tuple of the calibrated callable ``VolCurve`` and a metadata dict
        containing calibration diagnostics and parameters.

    Raises:
        ValueError: If the forward or maturity inputs are missing or invalid.
        CalculationError: If SVI calibration fails to converge.
    """

    if forward is None or forward <= 0:
        raise ValueError("forward must be a positive number for SVI fitting")
    if maturity_years is None or maturity_years <= 0:
        raise ValueError("maturity_years must be positive for SVI fitting")

    if options is None:
        base_config = merge_svi_options(DEFAULT_SVI_OPTIONS)
    elif isinstance(options, SVICalibrationOptions):
        base_config = merge_svi_options(options)
    elif isinstance(options, Mapping):
        base_config = merge_svi_options(dict(options))
    else:
        raise TypeError(
            "options must be an SVICalibrationOptions instance or mapping for SVI"
        )

    if overrides:
        unknown = set(overrides) - SVICalibrationOptions.field_names()
        if unknown:
            raise TypeError(f"Unknown SVI option(s): {sorted(unknown)}")
        merged = base_config.to_mapping()
        merged.update(overrides)
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
        eval_array = np.asarray(eval_strikes, dtype=float)
        eval_k = log_moneyness(eval_array, forward)
        total_var_eval = svi_total_variance(eval_k, params)
        return from_total_variance(total_var_eval, maturity_years)

    vol_curve.grid = (strikes, fitted_iv)  # type: ignore[attr-defined]
    vol_curve.params = {  # type: ignore[attr-defined]
        "method": "svi",
        "a": params.a,
        "b": params.b,
        "rho": params.rho,
        "m": params.m,
        "sigma": params.sigma,
        "forward": forward,
        "maturity_years": maturity_years,
    }
    vol_curve.params_raw = (  # type: ignore[attr-defined]
        params.a,
        params.b,
        params.rho,
        params.m,
        params.sigma,
    )
    vol_curve.params_jw = raw_to_jw(params)  # type: ignore[attr-defined]
    vol_curve.diagnostics = diagnostics  # type: ignore[attr-defined]

    metadata = {
        "params": params,
        "diagnostics": diagnostics,
        "config": config,
    }
    return vol_curve, metadata


__all__ = ["fit_svi_slice"]
