"""Registry for volatility fitting implementations."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from .api import SliceFitter, SurfaceFitter


_SLICE_FITTERS: Dict[str, SliceFitter] = {}
_SURFACE_FITTERS: Dict[str, SurfaceFitter] = {}


def register_slice_fitter(name: str, fitter: SliceFitter) -> None:
    """Register a single-expiry calibration implementation."""

    _SLICE_FITTERS[name] = fitter


def register_surface_fitter(name: str, fitter: SurfaceFitter) -> None:
    """Register a multi-expiry calibration implementation."""

    _SURFACE_FITTERS[name] = fitter


def get_slice_fitter(name: str) -> SliceFitter:
    """Return a registered single-expiry fitter."""

    try:
        return _SLICE_FITTERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown slice fitter '{name}'") from exc


def get_surface_fitter(name: str) -> SurfaceFitter:
    """Return a registered multi-expiry fitter."""

    try:
        return _SURFACE_FITTERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown surface fitter '{name}'") from exc


def available_fitters() -> (
    Tuple[Mapping[str, SliceFitter], Mapping[str, SurfaceFitter]]
):
    """Return the registered slice and surface fitters."""

    return _SLICE_FITTERS, _SURFACE_FITTERS


__all__ = [
    "register_slice_fitter",
    "register_surface_fitter",
    "get_slice_fitter",
    "get_surface_fitter",
    "available_fitters",
]
