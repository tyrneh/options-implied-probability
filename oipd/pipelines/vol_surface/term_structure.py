"""Pipeline helpers for interpolated volatility term structures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from oipd.core.utils import calculate_days_to_expiry, convert_days_to_years


def build_atm_term_structure(
    expiries: Sequence[pd.Timestamp],
    valuation_date: date,
    implied_vol: Callable[[float, float], float],
    forward_price: Callable[[float], float],
    spot_price: float,
    *,
    at_money: Literal["forward", "spot"] = "forward",
    num_points: int = 100,
) -> pd.DataFrame:
    """Build an interpolated ATM term structure from a fitted surface.

    Args:
        expiries: Sequence of fitted expiry timestamps.
        valuation_date: Valuation anchor date for day-count calculations.
        implied_vol: Callable returning implied volatility for (K, t_years).
        forward_price: Callable returning forward price F(t_years).
        spot_price: Current spot price used for "spot" ATM.
        at_money: Definition of ATM ("forward" uses K=F(t), "spot" uses K=S0).
        num_points: Number of interpolation points across the expiry range.

    Returns:
        DataFrame with columns:
            - days_to_expiry
            - time_to_expiry_years
            - atm_strike
            - atm_iv

    Raises:
        ValueError: If no expiries are provided or num_points < 1.
    """
    if not expiries:
        raise ValueError("At least one expiry is required to build term structure.")
    if num_points < 1:
        raise ValueError("num_points must be at least 1.")
    if at_money not in ("forward", "spot"):
        raise ValueError("at_money must be 'forward' or 'spot'.")

    expiries_sorted = sorted(pd.to_datetime(expiries).tz_localize(None))
    t_min_days = calculate_days_to_expiry(expiries_sorted[0], valuation_date)
    t_max_days = calculate_days_to_expiry(expiries_sorted[-1], valuation_date)

    t_min_days = max(t_min_days, 1)
    t_max_days = max(t_max_days, t_min_days)

    t_grid_days = np.linspace(t_min_days, t_max_days, num_points)
    t_grid_years = convert_days_to_years(t_grid_days)

    days_list: list[float] = []
    t_years_list: list[float] = []
    strikes_list: list[float] = []
    ivs_list: list[float] = []

    for days, t_years in zip(t_grid_days, t_grid_years):
        if at_money == "forward":
            K_atm = float(forward_price(float(t_years)))
        else:
            K_atm = float(spot_price)

        if not np.isfinite(K_atm) or K_atm <= 0:
            continue

        iv = float(implied_vol(K_atm, float(t_years)))
        if not np.isfinite(iv):
            continue

        days_list.append(float(days))
        t_years_list.append(float(t_years))
        strikes_list.append(float(K_atm))
        ivs_list.append(float(iv))

    if not days_list:
        raise ValueError("No valid term structure points could be computed.")

    return pd.DataFrame(
        {
            "days_to_expiry": np.array(days_list, dtype=float),
            "time_to_expiry_years": np.array(t_years_list, dtype=float),
            "atm_strike": np.array(strikes_list, dtype=float),
            "atm_iv": np.array(ivs_list, dtype=float),
        }
    )


__all__ = ["build_atm_term_structure"]
