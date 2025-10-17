"""Finite-difference utilities for probability density conversion."""

from __future__ import annotations

import warnings
import numpy as np


def finite_diff_second_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Stable five-point stencil second derivative with non-uniform fallback."""

    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length. Got x: {len(x)}, y: {len(y)}")
    if len(x) < 5:
        raise ValueError(f"Need at least 5 points for 5-point stencil. Got {len(x)}")

    h = np.diff(x)
    if not np.allclose(h, h[0], rtol=1e-6):
        warnings.warn(
            "Non-uniform grid detected. Using np.gradient fallback which may be less stable. "
            "Consider interpolating to a uniform grid first.",
            UserWarning,
        )
        return np.gradient(np.gradient(y, x), x)

    step = h[0]
    d2y = np.zeros_like(y)
    for i in range(2, len(y) - 2):
        d2y[i] = (-y[i - 2] + 16 * y[i - 1] - 30 * y[i] + 16 * y[i + 1] - y[i + 2]) / (
            12 * step**2
        )

    d2y[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / step**2
    d2y[1] = (y[0] - 2 * y[1] + y[2]) / step**2
    d2y[-2] = (y[-3] - 2 * y[-2] + y[-1]) / step**2
    d2y[-1] = (2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / step**2
    return d2y


__all__ = ["finite_diff_second_derivative"]
