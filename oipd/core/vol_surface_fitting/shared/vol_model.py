"""Volatility model selectors shared across slices and surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


VolMethod = Literal["svi", "svi-jw", "bspline", "ssvi"]


@dataclass(frozen=True)
class VolModel:
    """Configuration selector for volatility model calibration.

    Args:
        method: Requested calibration approach. ``None`` lets the caller decide
            the appropriate default (``"svi"`` for single-expiry smiles and
            ``"ssvi"`` for term structures). Slice-compatible methods are
            ``"svi"``, ``"svi-jw"``, and ``"bspline"``; surface-compatible
            methods are ``"ssvi"``.
        strict_no_arbitrage: Whether to enforce the strongest available
            no-arbitrage checks for the selected method.
    """

    method: Optional[VolMethod] = None
    strict_no_arbitrage: bool = True


SLICE_METHODS: tuple[str, ...] = ("svi", "svi-jw", "bspline")
SURFACE_METHODS: tuple[str, ...] = ("ssvi",)
