"""Generate option price curves from implied volatility surfaces."""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Tuple

import numpy as np

from oipd.core.errors import InvalidInputError
from oipd.core.utils import convert_days_to_years
from oipd.core.vol_surface_fitting import VolCurve
from oipd.pricing import get_pricer


def price_curve_from_iv(
    vol_curve: VolCurve,
    underlying_price: float,
    *,
    strike_grid: np.ndarray | None = None,
    days_to_expiry: Optional[int] = None,
    time_to_expiry_years: Optional[float] = None,
    risk_free_rate: float,
    pricing_engine: Literal["black76", "bs"],
    dividend_yield: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate call prices on a strike grid from a smoothed IV curve.

    Args:
        vol_curve: Callable volatility curve evaluated at strikes.
        underlying_price: Forward/spot input for the selected pricing engine.
        strike_grid: Optional strike grid. If omitted, attempts to use
            ``vol_curve.grid`` when available.
        days_to_expiry: Optional days-to-expiry input. Used when
            ``time_to_expiry_years`` is not provided.
        time_to_expiry_years: Optional explicit maturity in year fractions.
            Takes precedence over ``days_to_expiry`` when provided.
        risk_free_rate: Continuously compounded risk-free rate.
        pricing_engine: Pricing engine identifier (``"black76"`` or ``"bs"``).
        dividend_yield: Continuous dividend yield used by Black-Scholes.

    Returns:
        Tuple containing strike grid and call prices.

    Raises:
        InvalidInputError: If strike grid dimensionality is invalid or no
            maturity input is provided.
    """

    if strike_grid is None:
        if hasattr(vol_curve, "grid"):
            strike_grid = getattr(vol_curve, "grid")[0]
        else:
            raise InvalidInputError(
                "strike_grid must be provided when smoother grid is unavailable"
            )

    strikes = np.asarray(strike_grid, dtype=float)
    if strikes.ndim != 1:
        raise InvalidInputError("strike_grid must be one-dimensional")

    sigma = vol_curve(strikes)
    if time_to_expiry_years is not None:
        years = float(time_to_expiry_years)
    elif days_to_expiry is not None:
        years = float(convert_days_to_years(days_to_expiry))
    else:
        raise InvalidInputError(
            "Either days_to_expiry or time_to_expiry_years must be provided."
        )

    pricer = get_pricer(pricing_engine)
    q = dividend_yield or 0.0
    call_prices = pricer(underlying_price, strikes, sigma, years, risk_free_rate, q)
    return strikes, np.asarray(call_prices, dtype=float)


__all__ = ["price_curve_from_iv"]
