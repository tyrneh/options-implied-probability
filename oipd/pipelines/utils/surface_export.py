"""Shared helpers for surface export date grids and validation."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def validate_export_domain(
    domain: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Validate an explicit export domain.

    Args:
        domain: Optional lower and upper bound tuple.

    Returns:
        Normalized domain tuple or ``None`` when no explicit domain is supplied.

    Raises:
        ValueError: If the domain is malformed or non-increasing.
    """
    if domain is None:
        return None
    if len(domain) != 2:
        raise ValueError("domain must be a tuple of (min_value, max_value).")

    lower = float(domain[0])
    upper = float(domain[1])
    if lower >= upper:
        raise ValueError("domain must be strictly increasing: (min_value, max_value).")

    return (lower, upper)


def resolve_surface_export_expiries(
    surface: Any,
    *,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    step_days: int | None = None,
) -> tuple[pd.Timestamp, ...]:
    """Resolve the expiry grid used by surface export methods.

    Args:
        surface: Fitted surface-like interface exposing ``expiries`` and
            ``_resolve_query_time``.
        start: Optional lower date bound.
        end: Optional upper date bound.
        step_days: Optional sampling step in calendar days.

    Returns:
        Sorted, deduplicated export expiries.

    Raises:
        ValueError: If bounds or step configuration are invalid.
    """
    expiries = tuple(_normalize_timestamp(expiry) for expiry in surface.expiries)
    if not expiries:
        return ()

    start_expiry = _resolve_export_bound(surface, bound=start)
    end_expiry = _resolve_export_bound(surface, bound=end)

    if start_expiry is None:
        start_expiry = min(expiries)
    if end_expiry is None:
        end_expiry = max(expiries)

    if start_expiry > end_expiry:
        raise ValueError("start must be less than or equal to end.")

    in_window_pillars = tuple(
        expiry for expiry in expiries if start_expiry <= expiry <= end_expiry
    )

    if step_days is None:
        return in_window_pillars

    if isinstance(step_days, bool) or not isinstance(step_days, int) or step_days <= 0:
        raise ValueError("step_days must be None or a strictly positive integer.")

    sampled_expiries = tuple(
        _normalize_timestamp(expiry)
        for expiry in pd.date_range(
            start=start_expiry, end=end_expiry, freq=f"{step_days}D"
        )
    )
    return tuple(sorted(set(sampled_expiries).union(in_window_pillars)))


def _resolve_export_bound(
    surface: Any,
    *,
    bound: str | date | pd.Timestamp | None,
) -> pd.Timestamp | None:
    """Resolve and validate a single export bound against surface rules.

    Args:
        surface: Surface-like interface exposing ``_resolve_query_time``.
        bound: Optional date-like bound.

    Returns:
        Resolved bound timestamp when provided, otherwise ``None``.

    Raises:
        ValueError: If ``bound`` is not date-like or violates surface rules.
    """
    if bound is None:
        return None
    if not isinstance(bound, (str, date, pd.Timestamp)):
        raise ValueError(
            "Surface export bounds must be date-like "
            "(str, datetime.date, or pandas.Timestamp)."
        )

    resolved_bound, _ = surface._resolve_query_time(bound)
    return _normalize_timestamp(resolved_bound)


def _normalize_timestamp(value: Any) -> pd.Timestamp:
    """Convert a timestamp-like value to a timezone-naive ``Timestamp``.

    Args:
        value: Timestamp-like input value.

    Returns:
        Timezone-naive timestamp.
    """
    timestamp = pd.to_datetime(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp
