"""Pipeline helper to build surface interpolator from fitted slices."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from oipd.core.vol_surface_fitting.forward_interpolator import ForwardInterpolator
from oipd.core.vol_surface_fitting.variance_interpolator import (
    TotalVarianceInterpolator,
)
from oipd.core.utils import calculate_days_to_expiry
from oipd.pipelines.vol_surface.models import FittedSurface


def build_surface_interpolator(
    slices: Mapping[float, Any],
    forwards: Mapping[float, float],
    check_arbitrage: bool = False,
) -> TotalVarianceInterpolator:
    """Build a TotalVarianceInterpolator from fitted slices.

    Args:
        slices: Dict mapping time (years) to VolCurve objects.
            Each VolCurve must be callable: curve(K) -> sigma.
        forwards: Dict mapping time (years) to forward prices.
        check_arbitrage: If True, clamps variance to prevent negative increments.

    Returns:
        TotalVarianceInterpolator instance ready for use.
    """
    if not slices:
        raise ValueError("At least one slice is required to build interpolator")

    # Build forward interpolator
    forward_pillars = sorted([(float(t), float(f)) for t, f in forwards.items()])
    forward_interp = ForwardInterpolator(forward_pillars)

    # Build variance pillars
    def make_variance_curve(t: float, vol_curve: Any):
        """Create a callable k -> w from a VolCurve."""
        F_t = forwards.get(t)
        if F_t is None:
            # Fallback: use the forward interpolator
            F_t = forward_interp(t)

        def variance_curve(k: float | np.ndarray) -> float | np.ndarray:
            # k = ln(K/F), so K = F * exp(k)
            K = F_t * np.exp(k)
            sigma = vol_curve(K)
            return sigma**2 * t

        return variance_curve

    variance_pillars = []
    for t in sorted(slices.keys()):
        vol_curve = slices[t]
        variance_curve = make_variance_curve(float(t), vol_curve)
        variance_pillars.append((float(t), variance_curve))

    return TotalVarianceInterpolator(
        pillars=variance_pillars,
        forward_interp=forward_interp,
        check_arbitrage=check_arbitrage,
    )


def build_interpolator_from_fitted_surface(
    fitted_surface: FittedSurface,
    check_arbitrage: bool = False,
) -> TotalVarianceInterpolator:
    """Build interpolate directly from a fitted surface model.

    Args:
        fitted_surface: The fitted surface results containing slices.
        check_arbitrage: Whether to enforce no-arbitrage checks.

    Returns:
        TotalVarianceInterpolator.
    """
    # Extract slices and forwards from the fitted model
    slices = {}
    forwards = {}
    for expiry_ts in fitted_surface.expiries:
        slice_data = fitted_surface.get_slice(expiry_ts)
        # Time in years
        # ResolvedMarket must be present in the slice data from fit_surface
        resolved_market = slice_data.get("resolved_market")
        if not resolved_market:
            # Should not happen if coming from fit_surface
            raise ValueError(f"ResolvedMarket missing for expiry {expiry_ts}")

        # Calculate T locally
        days = calculate_days_to_expiry(expiry_ts, resolved_market.valuation_date)
        t = days / 365.0
        slices[t] = slice_data["curve"]

        forward = slice_data["metadata"].get("forward_price")
        if forward is None:
            raise ValueError(
                f"Forward price missing for expiry slice t={t:.4f}. Cannot build interpolator."
            )
        forwards[t] = forward

    return build_surface_interpolator(slices, forwards, check_arbitrage=check_arbitrage)
