from __future__ import annotations

"""Utility helpers shared across pricing modules."""

from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "prepare_dividends",
]


def _present_value(schedule: pd.DataFrame, r: float, valuation_date: date) -> float:
    """Compute PV of all cash dividends with *ex_date* ≥ valuation_date."""
    pv = 0.0
    for ex_date, cash in schedule[["ex_date", "amount"]].itertuples(index=False):
        if ex_date >= valuation_date:
            tau = (ex_date - valuation_date).days / 365.0
            pv += cash * np.exp(-r * tau)
    return pv


def prepare_dividends(
    spot: float,
    *,
    dividend_schedule: Optional[pd.DataFrame] = None,
    dividend_yield: Optional[float] = None,
    r: float,
    valuation_date: date,
) -> Tuple[float, float]:
    """Return *(adjusted_spot, effective_q)* according to the user's inputs.

    Rules
    -----
    1. **Discrete schedule only** → subtract PV → ``(S*, 0.0)``
    2. **Continuous yield only**  → no spot change → ``(S, q)``
    3. **Both provided**          → *error* (ambiguous)
    4. **Neither provided**       → ``(S, 0.0)``
    """
    if dividend_schedule is not None and dividend_yield not in (None, 0, 0.0):
        raise ValueError(
            "Provide *either* dividend_schedule *or* dividend_yield, not both."
        )

    if dividend_schedule is not None:
        if not {"ex_date", "amount"}.issubset(dividend_schedule.columns):
            raise ValueError("schedule must contain 'ex_date' and 'amount' columns")
        pv = _present_value(dividend_schedule, r, valuation_date)
        return spot - pv, 0.0

    # Continuous yield path ----------------------------------------------
    return spot, float(dividend_yield or 0.0)


# ----------------------------------------------------------------------
# Back-compat temporary alias (will be removed in a future major release)
# ----------------------------------------------------------------------


def adjust_spot_for_dividends(  # pragma: no cover – deprecated shim
    spot: float,
    schedule: Optional[pd.DataFrame],
    r: float,
    valuation_date: date,
):
    """Deprecated – use *prepare_dividends()* instead."""
    import warnings

    warnings.warn(
        "'adjust_spot_for_dividends' is deprecated; use 'prepare_dividends' which "
        "also returns the effective q.",
        DeprecationWarning,
        stacklevel=2,
    )
    adj_spot, _ = prepare_dividends(
        spot,
        dividend_schedule=schedule,
        dividend_yield=None,
        r=r,
        valuation_date=valuation_date,
    )
    return adj_spot
