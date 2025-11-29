"""B-spline smile fitting implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...registry import register_slice_fitter
from .fitter import DEFAULT_BSPLINE_OPTIONS, fit_bspline_slice


class _BsplineSliceFitter:
    """Adapter that conforms to :class:`SliceFitter`."""

    def fit(self, strikes: np.ndarray, iv: np.ndarray, **kwargs: Any):
        return fit_bspline_slice(strikes, iv, **kwargs)


register_slice_fitter("bspline", _BsplineSliceFitter())

__all__ = ["fit_bspline_slice", "DEFAULT_BSPLINE_OPTIONS"]
