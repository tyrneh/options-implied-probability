"""Canonical maturity/date-time helpers for pricing pipelines.

This module centralizes timestamp normalization and maturity resolution so
calibration/pricing code can rely on one consistent time convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Union

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24.0 * 60.0 * 60.0

DateTimeLike = Union[str, date, datetime, pd.Timestamp]

RELATIVE_MONTH_YEAR_PATTERN = re.compile(
    r"^\s*(?P<amount>\d+)\s*(?P<unit>[my])\s*$", re.I
)
RELATIVE_TIMEDELTA_PATTERN = re.compile(
    r"^\s*[-+]?\d+(?:\.\d+)?\s*"
    r"(?:"
    r"d|day|days|"
    r"w|week|weeks|"
    r"h|hour|hours|"
    r"min|minute|minutes|"
    r"s|sec|second|seconds|"
    r"ms|millisecond|milliseconds|"
    r"us|microsecond|microseconds|"
    r"ns|nanosecond|nanoseconds"
    r")\s*$",
    re.I,
)


def normalize_datetime_like(value: DateTimeLike) -> pd.Timestamp:
    """Normalize a date/datetime-like value to a timezone-naive Timestamp.

    Policy:
        - Plain dates remain calendar dates at midnight.
        - Explicit datetimes retain intraday precision.
        - Timezone-aware inputs are converted to UTC and then made tz-naive.

    Args:
        value: Date-like or datetime-like input.

    Returns:
        pd.Timestamp: Normalized timezone-naive timestamp.

    Raises:
        TypeError: If ``value`` is not date/datetime-like.
        ValueError: If ``value`` cannot be parsed into a timestamp.
    """
    if not isinstance(value, (str, date, datetime, pd.Timestamp)):
        raise TypeError(
            "value must be str, datetime.date, datetime.datetime, or pandas.Timestamp, "
            f"got {type(value)}"
        )

    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        raise ValueError(f"Could not parse datetime-like value: {value!r}") from exc

    if pd.isna(timestamp):
        raise ValueError(f"Could not parse datetime-like value: {value!r}")

    if timestamp.tzinfo is not None:
        # Deterministic timezone policy: convert aware inputs to UTC first.
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)

    return timestamp


def convert_days_to_years(
    days: Union[int, float, np.ndarray], days_per_year: float = 365.0
) -> Union[float, np.ndarray]:
    """Convert day counts to year fractions using a fixed day-count convention.

    Args:
        days: Day count input.
        days_per_year: Day-count denominator. Defaults to ACT/365.

    Returns:
        Year-fraction representation of ``days``.
    """
    return days / days_per_year


def resolve_horizon(
    horizon: DateTimeLike,
    valuation_date: DateTimeLike,
) -> pd.Timestamp:
    """Resolve a horizon input into a concrete cutoff timestamp.

    Args:
        horizon: Specific date/timestamp or relative horizon string such as
            ``"30d"``, ``"2m"``, or ``"1y"``.
        valuation_date: Reference valuation date/time for relative horizons.

    Returns:
        pd.Timestamp: Concrete timezone-naive cutoff timestamp.

    Raises:
        TypeError: If ``horizon`` is not date/datetime-like or a string.
        ValueError: If ``horizon`` cannot be parsed.
    """
    if not isinstance(horizon, (date, datetime, pd.Timestamp, str)):
        raise TypeError(
            f"horizon must be a string, date, or Timestamp, got {type(horizon)}"
        )

    valuation_timestamp = normalize_datetime_like(valuation_date)

    if isinstance(horizon, (date, datetime, pd.Timestamp)) and not isinstance(
        horizon, str
    ):
        return normalize_datetime_like(horizon)

    stripped_horizon = horizon.strip()
    month_year_match = RELATIVE_MONTH_YEAR_PATTERN.match(stripped_horizon)
    timedelta_match = RELATIVE_TIMEDELTA_PATTERN.match(stripped_horizon)

    if not month_year_match and not timedelta_match:
        try:
            return normalize_datetime_like(stripped_horizon)
        except (ValueError, TypeError):
            pass

    if month_year_match:
        try:
            amount = int(month_year_match.group("amount"))
        except ValueError as exc:
            raise ValueError(f"Could not parse horizon string: {horizon}") from exc

        unit = month_year_match.group("unit").lower()
        if unit == "m":
            return valuation_timestamp + pd.DateOffset(months=amount)
        return valuation_timestamp + pd.DateOffset(years=amount)

    if timedelta_match:
        try:
            delta = pd.to_timedelta(stripped_horizon)
            return valuation_timestamp + delta
        except Exception:
            pass

    raise ValueError(f"Could not parse horizon: {horizon}")


@dataclass(frozen=True)
class ResolvedMaturity:
    """Resolved maturity information for one valuation/expiry pair.

    Args:
        valuation_timestamp: Normalized valuation timestamp.
        expiry: Normalized expiry timestamp.
        time_to_expiry_years: Precise year fraction between valuation and expiry.
    """

    valuation_timestamp: pd.Timestamp
    expiry: pd.Timestamp
    time_to_expiry_years: float

    @property
    def time_to_expiry_days(self) -> float:
        """Return continuous ACT/365 day-equivalent maturity for reporting."""
        return float(self.time_to_expiry_years * 365.0)

    @property
    def calendar_days_to_expiry(self) -> int:
        """Return non-negative calendar-day difference for compatibility only."""
        valuation_day = self.valuation_timestamp.normalize()
        expiry_day = self.expiry.normalize()
        return max(0, int((expiry_day - valuation_day).days))


def resolve_maturity(
    expiry: DateTimeLike,
    valuation_date: DateTimeLike,
    *,
    days_per_year: float = 365.0,
    floor_at_zero: bool = False,
) -> ResolvedMaturity:
    """Resolve valuation/expiry inputs into a precise maturity representation.

    Args:
        expiry: Expiry identifier.
        valuation_date: Valuation anchor (kept as ``valuation_date`` for API
            compatibility, but resolved internally as a full timestamp).
        days_per_year: Day-count denominator for year fraction conversion.
        floor_at_zero: Whether to floor negative maturities at zero. Defaults
            to ``False`` so canonical callers can preserve signed maturities.

    Returns:
        ResolvedMaturity: Canonical maturity object.
    """
    valuation_timestamp = normalize_datetime_like(valuation_date)
    expiry_timestamp = normalize_datetime_like(expiry)

    seconds_to_expiry = float((expiry_timestamp - valuation_timestamp).total_seconds())
    time_to_expiry_years = seconds_to_expiry / (days_per_year * SECONDS_PER_DAY)
    if floor_at_zero:
        time_to_expiry_years = max(0.0, time_to_expiry_years)

    return ResolvedMaturity(
        valuation_timestamp=valuation_timestamp,
        expiry=expiry_timestamp,
        time_to_expiry_years=float(time_to_expiry_years),
    )


def calculate_calendar_days_to_expiry(
    expiry: DateTimeLike,
    valuation_date: DateTimeLike,
) -> int:
    """Return non-negative calendar-day maturity bucket for reporting.

    Args:
        expiry: Expiry identifier.
        valuation_date: Valuation anchor.

    Returns:
        int: Calendar-day difference, floored at zero.
    """
    return resolve_maturity(expiry, valuation_date).calendar_days_to_expiry


def calculate_time_to_expiry_days(
    expiry: DateTimeLike,
    valuation_date: DateTimeLike,
    days_per_year: float = 365.0,
) -> float:
    """Return continuous ACT/365 day-equivalent maturity for reporting.

    Args:
        expiry: Expiry identifier.
        valuation_date: Valuation anchor.
        days_per_year: Day-count denominator.

    Returns:
        float: Continuous day-equivalent maturity, floored at zero.
    """
    years = calculate_time_to_expiry(expiry, valuation_date, days_per_year)
    return float(years * days_per_year)


def calculate_time_to_expiry(
    expiry: DateTimeLike,
    valuation_date: DateTimeLike,
    days_per_year: float = 365.0,
) -> float:
    """Compatibility helper returning floored year fraction to expiry.

    Args:
        expiry: Expiry identifier.
        valuation_date: Valuation anchor.
        days_per_year: Day-count denominator.

    Returns:
        float: Time-to-expiry in years, floored at zero.
    """
    return resolve_maturity(
        expiry,
        valuation_date,
        days_per_year=days_per_year,
        floor_at_zero=True,
    ).time_to_expiry_years


def build_maturity_metadata(
    resolved_maturity: ResolvedMaturity,
) -> dict[str, object]:
    """Build a standardized maturity metadata payload.

    Args:
        resolved_maturity: Canonical resolved maturity object.

    Returns:
        dict[str, object]: Metadata led by canonical expiry and precise time
        fields.
    """
    return {
        "expiry": resolved_maturity.expiry,
        "time_to_expiry_years": resolved_maturity.time_to_expiry_years,
        "time_to_expiry_days": resolved_maturity.time_to_expiry_days,
        "calendar_days_to_expiry": resolved_maturity.calendar_days_to_expiry,
    }


def format_time_to_expiry_days_label(time_to_expiry_days: float) -> str:
    """Format continuous day-equivalent maturity for user-facing labels.

    Args:
        time_to_expiry_days: Continuous ACT/365 day-equivalent maturity.

    Returns:
        str: Compact maturity label that preserves sub-day precision when
        needed while avoiding noisy decimals for exact whole-day values.
    """
    rounded_days = round(float(time_to_expiry_days))
    if np.isclose(time_to_expiry_days, rounded_days, atol=1e-9):
        return f"{int(rounded_days)}d"
    if time_to_expiry_days >= 10.0:
        return f"{time_to_expiry_days:.1f}d"
    return f"{time_to_expiry_days:.2f}d"


def format_timestamp_for_display(value: DateTimeLike) -> str:
    """Format a date/time value for user-facing labels.

    Policy:
        - Midnight timestamps render as date-only labels.
        - Intraday timestamps render with time-of-day.
        - Seconds are shown when present so explicit second-level precision is visible.

    Args:
        value: Date-like or datetime-like value to format.

    Returns:
        str: Human-readable label preserving intraday precision when relevant.
    """
    timestamp = normalize_datetime_like(value)
    if timestamp.time() == datetime.min.time():
        return timestamp.strftime("%b %d, %Y")

    if timestamp.second != 0 or timestamp.microsecond != 0 or timestamp.nanosecond != 0:
        return timestamp.strftime("%b %d, %Y %H:%M:%S")

    return timestamp.strftime("%b %d, %Y %H:%M")
