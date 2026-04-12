from __future__ import annotations

"""Utility helpers shared across pricing modules."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["prepare_dividends", "implied_dividend_yield_from_forward"]


from oipd.core.maturity import (
    DateTimeLike,
    calculate_time_to_expiry,
    normalize_datetime_like,
)


def _present_value(
    schedule: pd.DataFrame,
    r: float,
    valuation_date: DateTimeLike,
    expiry: Optional[DateTimeLike] = None,
) -> float:
    """Compute PV of all cash dividends still live at the valuation timestamp.

    Args:
        schedule: Dividend schedule containing ``ex_date`` and ``amount``.
        r: Continuously compounded discount rate.
        valuation_date: Valuation anchor date/time.
        expiry: Optional option expiry used to exclude post-expiry cashflows.

    Returns:
        float: Present value of dividends with ex-date not earlier than the
        valuation timestamp and, when provided, not later than the expiry
        timestamp.
    """
    valuation_timestamp = normalize_datetime_like(valuation_date)
    expiry_timestamp = normalize_datetime_like(expiry) if expiry is not None else None
    pv = 0.0
    for ex_date, cash in schedule[["ex_date", "amount"]].itertuples(index=False):
        ex_timestamp = normalize_datetime_like(ex_date)
        if ex_timestamp < valuation_timestamp:
            continue
        if expiry_timestamp is not None and ex_timestamp > expiry_timestamp:
            continue

        tau = calculate_time_to_expiry(ex_timestamp, valuation_timestamp)
        pv += cash * np.exp(-r * tau)
    return pv


def prepare_dividends(
    underlying: float,
    *,
    dividend_schedule: Optional[pd.DataFrame] = None,
    dividend_yield: Optional[float] = None,
    r: float,
    valuation_date: DateTimeLike,
    expiry: Optional[DateTimeLike] = None,
) -> Tuple[float, float]:
    """Return ``(adjusted_underlying, effective_q)`` according to the user's inputs.

    Rules
    -----
    1. **Discrete schedule only** → subtract PV → ``(S*, 0.0)``
    2. **Continuous yield only**  → no price change → ``(S, q)``
    3. **Both provided**          → *error* (ambiguous)
    4. **Neither provided**       → ``(S, 0.0)``

    Args:
        underlying: Spot input before dividend adjustment.
        dividend_schedule: Optional discrete cash dividend schedule.
        dividend_yield: Optional continuous dividend yield.
        r: Continuously compounded risk-free rate.
        valuation_date: Valuation anchor date/time for dividend inclusion and
            discounting.
        expiry: Optional option expiry used to filter discrete dividends to the
            live option window only.

    Returns:
        Tuple[float, float]: Adjusted underlying and effective dividend yield.
    """
    if dividend_schedule is not None and dividend_yield not in (None, 0, 0.0):
        raise ValueError(
            "Provide *either* dividend_schedule *or* dividend_yield, not both."
        )

    if dividend_schedule is not None:
        if not {"ex_date", "amount"}.issubset(dividend_schedule.columns):
            raise ValueError("schedule must contain 'ex_date' and 'amount' columns")
        pv = _present_value(dividend_schedule, r, valuation_date, expiry=expiry)
        return underlying - pv, 0.0

    # Continuous yield path ----------------------------------------------
    return underlying, float(dividend_yield or 0.0)


def implied_dividend_yield_from_forward(
    underlying: float,
    forward: float,
    r: float,
    T_years: float,
) -> float:
    """Compute implied continuous dividend yield from forward.

    q = r - ln(F / S) / T

    Parameters
    ----------
    underlying : float
        Current underlying price S
    forward : float
        Forward price F implied from put-call parity
    r : float
        Continuously compounded risk-free rate
    T_years : float
        Time to expiry in years

    Returns
    -------
    float
        Implied continuous dividend yield q
    """
    if underlying <= 0:
        raise ValueError("Invalid underlying for implied yield calculation.")
    if forward <= 0:
        raise ValueError("Invalid forward for implied yield calculation.")
    if T_years <= 0:
        raise ValueError("Non-positive time to expiry.")
    return float(r - np.log(forward / underlying) / T_years)
