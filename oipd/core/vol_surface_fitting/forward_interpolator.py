"""Linear interpolation for forward prices across maturities."""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np


class ForwardInterpolator:
    """Linear interpolator for forward prices.

    Provides forward price estimates at arbitrary maturities using linear
    interpolation between observed pillar forwards. Extrapolates flat
    outside the observed range.

    Args:
        pillars: Sequence of (time, forward) tuples sorted by time.
            Time is in years, forward is the forward price.

    Example:
        >>> interp = ForwardInterpolator([(0.1, 100.0), (0.5, 105.0)])
        >>> interp(0.3)  # Midpoint
        102.5
    """

    def __init__(self, pillars: Sequence[Tuple[float, float]]) -> None:
        if not pillars:
            raise ValueError("At least one pillar is required")

        # Sort by time
        sorted_pillars = sorted(pillars, key=lambda x: x[0])
        self._times = np.array([p[0] for p in sorted_pillars], dtype=float)
        self._forwards = np.array([p[1] for p in sorted_pillars], dtype=float)

        if len(self._times) < 1:
            raise ValueError("At least one pillar is required")

    def __call__(self, t: float) -> float:
        """Return the interpolated forward price at time t.

        Args:
            t: Time to maturity in years.

        Returns:
            Interpolated forward price.
        """
        # Edge cases: extrapolation
        if t <= self._times[0]:
            return float(self._forwards[0])
        if t >= self._times[-1]:
            return float(self._forwards[-1])

        # Linear interpolation
        return float(np.interp(t, self._times, self._forwards))

    @property
    def pillars(self) -> list[Tuple[float, float]]:
        """Return the pillar (time, forward) pairs."""
        return list(zip(self._times.tolist(), self._forwards.tolist()))
