"""Stateless DataFrame export helpers for volatility surfaces."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from oipd.pipelines.utils.surface_export import (
    resolve_surface_export_expiries,
    validate_export_domain,
)
from oipd.pipelines.vol_curve import compute_fitted_smile


def build_surface_iv_results_frame(
    vol_surface: Any,
    *,
    domain: tuple[float, float] | None = None,
    points: int = 200,
    include_observed: bool = True,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    step_days: int | None = None,
) -> pd.DataFrame:
    """Build a long-format implied-volatility export for a fitted surface.

    Args:
        vol_surface: Fitted volatility surface interface object.
        domain: Optional strike domain as ``(min_strike, max_strike)``.
        points: Number of curve evaluation points per expiry.
        include_observed: Whether to include observed market IV columns.
        start: Optional lower expiry bound.
        end: Optional upper expiry bound.
        step_days: Optional calendar-day sampling interval.

    Returns:
        Long DataFrame with an ``expiry`` column.
    """
    validate_export_domain(domain)
    export_expiries = resolve_surface_export_expiries(
        vol_surface,
        start=start,
        end=end,
        step_days=step_days,
    )
    if not export_expiries:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for expiry_timestamp in export_expiries:
        curve = vol_surface.slice(expiry_timestamp)
        frame = compute_fitted_smile(
            vol_curve=curve,
            metadata=curve._metadata,
            domain=domain,
            points=points,
            include_observed=include_observed,
        )
        frame.insert(0, "expiry", pd.Timestamp(expiry_timestamp))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
