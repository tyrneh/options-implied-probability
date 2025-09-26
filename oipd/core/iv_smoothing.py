from __future__ import annotations

"""Volatility smile smoothing utilities for the IV pipeline."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

import numpy as np
from scipy import interpolate

from oipd.core.errors import CalculationError


VolCurve = Callable[[Iterable[float] | np.ndarray], np.ndarray]


@dataclass
class VolCurveAttributes:
    grid: tuple[np.ndarray, np.ndarray]
    params: Dict[str, Any]
    diagnostics: Dict[str, Any]


def _attach_attributes(func: VolCurve, attributes: VolCurveAttributes) -> VolCurve:
    """Attach metadata attributes to a callable volatility curve."""

    setattr(func, "grid", attributes.grid)
    setattr(func, "params", attributes.params)
    setattr(func, "diagnostics", attributes.diagnostics)
    return func


def _fit_bspline_iv(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    smoothing_factor: float = 10.0,
    degree: int = 3,
    dx: float = 0.1,
) -> VolCurve:
    """Fit a B-spline to the observed IV curve and return a callable smile."""

    if len(strikes) < 4:
        raise CalculationError(
            f"Insufficient data for B-spline fitting: need at least 4 points, got {len(strikes)}"
        )

    try:
        tck = interpolate.splrep(strikes, iv, s=smoothing_factor, k=degree)
    except Exception as exc:  # pragma: no cover - scipy error propagation
        raise CalculationError(
            f"Failed to fit B-spline to implied volatility data: {exc}"
        ) from exc

    domain = max(2, int(np.ceil((strikes.max() - strikes.min()) / max(dx, 1e-6))))
    grid_x = np.linspace(strikes.min(), strikes.max(), domain)
    grid_y = interpolate.BSpline(*tck)(grid_x)

    def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        eval_array = np.asarray(eval_strikes, dtype=float)
        return interpolate.BSpline(*tck)(eval_array)

    attributes = VolCurveAttributes(
        grid=(grid_x, grid_y),
        params={
            "method": "bspline",
            "smoothing_factor": smoothing_factor,
            "degree": degree,
            "dx": dx,
        },
        diagnostics={"points_used": len(strikes)},
    )

    return _attach_attributes(vol_curve, attributes)


_SMOOTHERS: Dict[str, Callable[..., VolCurve]] = {
    "bspline": _fit_bspline_iv,
}


def available_smoothers() -> list[str]:
    """Return the list of registered smoothing methods."""

    return sorted(_SMOOTHERS.keys())


def get_smoother(name: str) -> Callable[..., VolCurve]:
    """Retrieve a smoothing callable by name."""

    try:
        return _SMOOTHERS[name]
    except KeyError as exc:  # pragma: no cover - guard path
        raise ValueError(
            f"Unknown IV smoother '{name}'. Available: {available_smoothers()}"
        ) from exc


def smooth_iv(
    method: str,
    strikes: np.ndarray,
    iv: np.ndarray,
    **kwargs: Any,
) -> VolCurve:
    """Fit a volatility smile using the requested smoothing method."""

    strikes_arr = np.asarray(strikes, dtype=float)
    iv_arr = np.asarray(iv, dtype=float)
    if strikes_arr.size != iv_arr.size:
        raise ValueError(
            f"Strike and IV arrays must have the same length. Got {strikes_arr.size} and {iv_arr.size}."
        )

    smoother = get_smoother(method)
    return smoother(strikes_arr, iv_arr, **kwargs)


__all__ = ["VolCurve", "available_smoothers", "get_smoother", "smooth_iv"]
