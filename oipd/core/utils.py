import pandas as pd
from datetime import date
from typing import Union

def resolve_horizon(horizon: Union[str, date, pd.Timestamp], valuation_date: Union[date, pd.Timestamp]) -> pd.Timestamp:
    """
    Resolves a horizon input into a concrete cutoff datetime.
    
    Args:
        horizon: The horizon to resolve. Can be:
            - A specific date/timestamp (e.g. date(2025, 1, 1), "2025-01-01")
            - A relative string (e.g. "30d", "3m", "2y")
        valuation_date: The reference date for relative horizons.
        
    Returns:
        pd.Timestamp: The calculated cutoff datetime (tz-naive).
        
    Raises:
        ValueError: If the horizon format is invalid.
        TypeError: If the input types are incorrect.
    """
    if not isinstance(horizon, (date, pd.Timestamp, str)):
        raise TypeError(f"horizon must be a string, date, or Timestamp, got {type(horizon)}")

    # validation date normalization
    val_date = pd.to_datetime(valuation_date).tz_localize(None)

    # 1. Try to parse as an absolute date/timestamp string first
    try:
        # Note: We rely on pandas to fail for things like "30d" which look like durations.
        # However, "3m" might be tricky if pandas thinks it's a date format.
        # Generally pd.to_datetime("3m") raises ParserError (which is a subclass of ValueError).
        return pd.to_datetime(horizon).tz_localize(None)
    except (ValueError, TypeError):
        pass

    # 2. Logic for relative horizons
    if isinstance(horizon, str):
        # Custom handling for "m" (months) and "y" (years)
        # We handle this manually because pd.to_timedelta("1m") is 1 minute, not 1 month.
        if horizon[-1].lower() in ("m", "y"):
            try:
                amount = int(horizon[:-1])
            except ValueError:
                # If we can't parse the number but it ends in m/y, it's definitely invalid
                raise ValueError(f"Could not parse horizon string: {horizon}")

            unit = horizon[-1].lower()
            if unit == "m":
                return val_date + pd.DateOffset(months=amount)
            elif unit == "y":
                return val_date + pd.DateOffset(years=amount)
        
        # Fallback to pandas timedelta (supports "d", "w", etc.)
        try:
            delta = pd.to_timedelta(horizon)
            return val_date + delta
        except Exception:
             # If both date parsing and timedelta parsing fail, it's invalid
             pass

    raise ValueError(f"Could not parse horizon: {horizon}")


def calculate_days_to_expiry(
    expiry: Union[str, date, pd.Timestamp],
    valuation_date: date,
) -> int:
    """Calculate calendar days from valuation_date to expiry.

    Args:
        expiry: Expiry date (date, Timestamp, or ISO string).
        valuation_date: Pricing/valuation anchor date.

    Returns:
        Days to expiry (floored at 0).
    """
    if isinstance(expiry, str):
        expiry = pd.to_datetime(expiry)
    if isinstance(expiry, pd.Timestamp):
        # Handle timezone-aware timestamps
        if expiry.tz is not None:
            expiry = expiry.tz_localize(None)
        expiry = expiry.date()

    delta = expiry - valuation_date
    return max(0, delta.days)

