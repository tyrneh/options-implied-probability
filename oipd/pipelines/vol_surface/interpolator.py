"""Pipeline helper to build surface interpolator from fitted slices."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from oipd.core.vol_surface_fitting.forward_interpolator import ForwardInterpolator
from oipd.core.vol_surface_fitting.variance_interpolator import TotalVarianceInterpolator


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
    # Each VolCurve is callable: curve(K) -> sigma
    # We need callable: k -> w = sigma(k)^2 * t
    # But VolCurve takes K, not k. We need to convert.
    # Actually, the TotalVarianceInterpolator expects curve(k) -> w.
    # We need to wrap the VolCurve to accept k and return w.

    def make_variance_curve(t: float, vol_curve: Any):
        """Create a callable k -> w from a VolCurve."""
        F_t = forwards.get(t)
        if F_t is None:
            # Fallback: use the forward interpolator
            F_t = forward_interp(t)

        def variance_curve(k: float) -> float:
            # k = ln(K/F), so K = F * exp(k)
            K = F_t * float(__import__("numpy").exp(k))
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
