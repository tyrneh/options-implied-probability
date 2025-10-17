"""Core protocols and types for volatility surface fitting."""

from __future__ import annotations

from typing import Protocol, Callable, Iterable, Mapping, Any

import numpy as np


class VolCurve(Protocol):
    """Typed callable representing a single-expiry implied volatility curve."""

    grid: tuple[np.ndarray, np.ndarray] | None
    meta: Mapping[str, Any]

    def __call__(self, strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        ...


class IVSurface(Protocol):
    """Protocol describing a calibrated implied-volatility surface."""

    expiries: np.ndarray
    strikes: np.ndarray
    meta: Mapping[str, Any]

    def __call__(self, expiry_index: int, strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        ...


class SliceFitter(Protocol):
    """Protocol for fitting a single-expiry smile."""

    def fit(self, strikes: np.ndarray, iv: np.ndarray, **kwargs: Any) -> tuple[VolCurve, Mapping[str, Any]]:
        ...


class SurfaceFitter(Protocol):
    """Protocol for fitting a multi-expiry implied-volatility surface."""

    def fit(
        self,
        expiries: np.ndarray,
        strikes: np.ndarray,
        iv: np.ndarray,
        **kwargs: Any,
    ) -> tuple[IVSurface, Mapping[str, Any]]:
        ...


__all__ = ["VolCurve", "IVSurface", "SliceFitter", "SurfaceFitter"]

