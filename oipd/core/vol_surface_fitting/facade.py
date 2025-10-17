"""Facade exposing volatility fitting APIs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from oipd.core.vol_surface_fitting.shared.svi_types import SVICalibrationOptions

from .algorithms import bspline as _bspline  # noqa: F401  - ensure registration
from .algorithms import svi as _svi  # noqa: F401  - ensure registration
from .algorithms.bspline import DEFAULT_BSPLINE_OPTIONS
from .registry import get_slice_fitter


AVAILABLE_SURFACE_FITS: tuple[str, ...] = ("bspline", "svi")
_ALIASES: dict[str, str] = {"svi-jw": "svi"}


def available_surface_fits() -> tuple[str, ...]:
    """Return the available slice fitting implementations."""

    return AVAILABLE_SURFACE_FITS


def _canonical_method(method: str) -> str:
    lower = method.lower()
    return _ALIASES.get(lower, lower)


def fit_slice(
    method: str,
    strikes: Iterable[float] | np.ndarray,
    iv: Iterable[float] | np.ndarray,
    *,
    forward: float | None = None,
    maturity_years: float | None = None,
    options: SVICalibrationOptions | Mapping[str, Any] | None = None,
    **overrides: Any,
):
    """Fit a single-expiry smile and return the callable curve along with metadata."""

    method_name = _canonical_method(method)
    if method_name not in AVAILABLE_SURFACE_FITS:
        raise ValueError(
            f"Unknown surface fit '{method}'. Available: {AVAILABLE_SURFACE_FITS}"
        )

    strikes_arr = np.asarray(strikes, dtype=float)
    iv_arr = np.asarray(iv, dtype=float)
    if strikes_arr.shape != iv_arr.shape:
        raise ValueError("Strike and IV arrays must have identical shapes")

    fitter = get_slice_fitter(method_name)

    if method_name == "svi":
        method_kwargs = dict(overrides)
        method_kwargs.update(
            {
                "forward": forward,
                "maturity_years": maturity_years,
                "options": options,
            }
        )
        vol_curve, metadata = fitter.fit(strikes_arr, iv_arr, **method_kwargs)
        return vol_curve, metadata

    # bspline
    if forward is not None or maturity_years is not None:
        raise ValueError(
            "forward and maturity_years are only valid for the 'svi' method"
        )

    method_kwargs = dict(overrides)
    if options is not None:
        if isinstance(options, Mapping):
            method_kwargs.update(dict(options))
        else:
            raise TypeError("options must be a mapping for non-SVI surface fits")

    vol_curve, metadata = fitter.fit(strikes_arr, iv_arr, **method_kwargs)
    return vol_curve, metadata


def fit_surface(
    method: str,
    *,
    strikes: Iterable[float] | np.ndarray,
    iv: Iterable[float] | np.ndarray,
    forward: float | None = None,
    maturity_years: float | None = None,
    options: SVICalibrationOptions | Mapping[str, Any] | None = None,
    **overrides: Any,
):
    """Fit an implied-volatility smile and return the callable curve."""

    vol_curve, _metadata = fit_slice(
        method,
        strikes,
        iv,
        forward=forward,
        maturity_years=maturity_years,
        options=options,
        **overrides,
    )
    return vol_curve


__all__ = [
    "AVAILABLE_SURFACE_FITS",
    "DEFAULT_BSPLINE_OPTIONS",
    "available_surface_fits",
    "fit_slice",
    "fit_surface",
]
