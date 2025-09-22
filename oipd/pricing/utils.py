from __future__ import annotations

"""Utility helpers shared across pricing modules."""

from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["prepare_dividends", "implied_dividend_yield_from_forward"]


def _present_value(
    schedule: pd.DataFrame, r: float, valuation_date: date
) -> float:
    """Compute PV of all cash dividends with *ex_date* ≥ valuation_date."""
    pv = 0.0
    for ex_date, cash in schedule[["ex_date", "amount"]].itertuples(
        index=False
    ):
        if ex_date >= valuation_date:
            tau = (ex_date - valuation_date).days / 365.0
            pv += cash * np.exp(-r * tau)
    return pv


def prepare_dividends(
    underlying: float,
    *,
    dividend_schedule: Optional[pd.DataFrame] = None,
    dividend_yield: Optional[float] = None,
    r: float,
    valuation_date: date,
) -> Tuple[float, float]:
    """Return ``(adjusted_underlying, effective_q)`` according to the user's inputs.

    Rules
    -----
    1. **Discrete schedule only** → subtract PV → ``(S*, 0.0)``
    2. **Continuous yield only**  → no price change → ``(S, q)``
    3. **Both provided**          → *error* (ambiguous)
    4. **Neither provided**       → ``(S, 0.0)``
    """
    if dividend_schedule is not None and dividend_yield not in (None, 0, 0.0):
        raise ValueError(
            "Provide *either* dividend_schedule *or* dividend_yield, not both."
        )

    if dividend_schedule is not None:
        if not {"ex_date", "amount"}.issubset(dividend_schedule.columns):
            raise ValueError(
                "schedule must contain 'ex_date' and 'amount' columns"
            )
        pv = _present_value(dividend_schedule, r, valuation_date)
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
