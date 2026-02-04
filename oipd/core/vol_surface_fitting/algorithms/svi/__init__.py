"""SVI slice fitting implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...registry import register_slice_fitter
from .fitter import fit_svi_slice


class _SviSliceFitter:
    """Adapter that conforms to :class:`SliceFitter`."""

    def fit(self, strikes: np.ndarray, iv: np.ndarray, **kwargs: Any):
        return fit_svi_slice(strikes, iv, **kwargs)


_SVI_FITTER = _SviSliceFitter()
register_slice_fitter("svi", _SVI_FITTER)
register_slice_fitter("svi-jw", _SVI_FITTER)

__all__ = ["fit_svi_slice"]
