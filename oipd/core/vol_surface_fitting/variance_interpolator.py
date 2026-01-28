"""Linear interpolation of total variance across maturities."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple, Protocol, Union
import numpy as np

from oipd.core.vol_surface_fitting.forward_interpolator import ForwardInterpolator


class VarianceCurve(Protocol):
    """Protocol for objects that can return total variance at log-moneyness.
    
    This is the mathematical dual of the user-facing ``VolCurve``, but defined
    in the coordinate system best suited for interpolation:
    - Input: log-moneyness $k = \ln(K/F)$ (instead of Strike K)
    - Output: Total Variance $w = \sigma^2 T$ (instead of Volatility $\sigma$)
    """

    def __call__(self, k: float) -> float:
        """Return total variance at log-moneyness k."""
        ...


class TotalVarianceInterpolator:
    """Linear interpolator for total variance in (log-moneyness, time) space.

    Interpolates total variance ($w = \\sigma^2 T$) linearly between fitted
    pillar slices. Handles edge cases for short-end and long-end extrapolation.

    Args:
        pillars: Sequence of (time, variance_curve) tuples sorted by time.
            "Pillars" are the fixed, calibrated expiries that support the surface.
            We interpolate between these known slices.
            Each variance_curve is a callable(k) -> w.
        forward_interp: ForwardInterpolator instance for forward prices.
        check_arbitrage: If True, clamps variance to prevent negative increments.

    Example:
        >>> interp = TotalVarianceInterpolator(pillars, forward_interp)
        >>> w = interp(K=100.0, t=0.3)  # Total variance at K=100, t=0.3y
    """

    def __init__(
        self,
        pillars: Sequence[Tuple[float, Callable[[float], float]]],
        forward_interp: ForwardInterpolator,
        check_arbitrage: bool = False,
    ) -> None:
        if not pillars:
            raise ValueError("At least one pillar is required")

        # Sort by time
        sorted_pillars = sorted(pillars, key=lambda x: x[0])
        self._times = np.array([p[0] for p in sorted_pillars], dtype=float)
        self._curves = [p[1] for p in sorted_pillars]
        self._forward_interp = forward_interp
        self._check_arbitrage = check_arbitrage

    def __call__(self, K: float, t: float) -> float:
        """Return the interpolated total variance at strike K and time t.

        Args:
            K: Strike price.
            t: Time to maturity in years.

        Returns:
            Total variance w(K, t).
        """
        if t <= 0:
            return 0.0

        # Get forward at time t
        F_t = self._forward_interp(t)
        if F_t <= 0:
            raise ValueError(f"Invalid forward price {F_t} at t={t}")

        # Compute log-moneyness
        k = np.log(K / F_t)

        # Edge case: short-end extrapolation (t < T_first)
        if t <= self._times[0]:
            # Linear from (0, 0) to (T_1, w_1)
            T_1 = self._times[0]
            w_1 = self._curves[0](k)
            return w_1 * (t / T_1)

        # Edge case: long-end extrapolation (t > T_last)
        if t >= self._times[-1]:
            # Constant volatility extrapolation
            T_last = self._times[-1]
            w_last = self._curves[-1](k)
            sigma_last_sq = w_last / T_last if T_last > 0 else 0.0
            return sigma_last_sq * t

        # Find bounding pillars
        idx = np.searchsorted(self._times, t)
        T_1 = self._times[idx - 1]
        T_2 = self._times[idx]
        w_1 = self._curves[idx - 1](k)
        w_2 = self._curves[idx](k)

        # Arbitrage check
        if self._check_arbitrage and w_2 < w_1:
            w_2 = w_1  # Clamp to prevent negative forward variance

        # Linear interpolation in total variance
        alpha = (T_2 - t) / (T_2 - T_1)
        return alpha * w_1 + (1 - alpha) * w_2

    def implied_vol(self, K: float, t: float) -> float:
        """Return the interpolated implied volatility at strike K and time t.

        Args:
            K: Strike price.
            t: Time to maturity in years.

        Returns:
            Implied volatility sigma(K, t).
        """
        w = self(K, t)
        if t <= 0 or w < 0:
            return 0.0
        return np.sqrt(w / t)

    @property
    def pillar_times(self) -> list[float]:
        """Return the pillar times."""
        return self._times.tolist()
