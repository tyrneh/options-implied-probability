"""Generate option price curves from implied volatility surfaces."""

from __future__ import annotations

from typing import Iterable, Literal, Tuple

import numpy as np

from oipd.core.errors import InvalidInputError
from oipd.core.vol_surface_fitting import VolCurve
from oipd.pricing import get_pricer


def price_curve_from_iv(
    vol_curve: VolCurve,
    underlying_price: float,
    *,
    strike_grid: np.ndarray | None = None,
    days_to_expiry: int,
    risk_free_rate: float,
    pricing_engine: Literal["black76", "bs"],
    dividend_yield: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate call prices on a strike grid from a smoothed IV curve."""

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
    years = days_to_expiry / 365.0
    pricer = get_pricer(pricing_engine)
    q = dividend_yield or 0.0
    call_prices = pricer(underlying_price, strikes, sigma, years, risk_free_rate, q)
    return strikes, np.asarray(call_prices, dtype=float)


__all__ = ["price_curve_from_iv"]
