"""B-spline smile fitting implementation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import interpolate

from oipd.core.errors import CalculationError


DEFAULT_BSPLINE_OPTIONS: dict[str, float | int] = {
    "smoothing_factor": 10.0,
    "degree": 3,
    "dx": 0.1,
}


def fit_bspline_slice(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    smoothing_factor: float = 10.0,
    degree: int = 3,
    dx: float = 0.1,
):
    """Fit a smoothing B-spline through an implied-volatility smile.

    Args:
        strikes: Observed strike values for the maturity of interest.
        iv: Implied volatilities aligned with ``strikes``.
        smoothing_factor: Smoothing factor passed to ``scipy.splrep``.
        degree: Degree of the B-spline basis.
        dx: Target spacing for the stored evaluation grid.

    Returns:
        Tuple of the spline-backed callable ``VolCurve`` and metadata describing
        the fitted parameters.

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
    except Exception as exc:  # pragma: no cover - SciPy error propagation
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
    vol_curve.params = {  # type: ignore[attr-defined]
        "method": "bspline",
        "smoothing_factor": smoothing_factor,
        "degree": degree,
        "dx": dx,
    }

    metadata = {
        "spline": spline,
        "grid": (grid_x, grid_y),
    }
    return vol_curve, metadata


__all__ = ["DEFAULT_BSPLINE_OPTIONS", "fit_bspline_slice"]

