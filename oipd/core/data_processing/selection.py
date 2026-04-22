"""Helpers for filtering option quotes and selecting price columns."""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import warnings

from oipd.core.errors import CalculationError


def filter_stale_options(
    options_data: pd.DataFrame,
    valuation_date,
    max_staleness_days: Optional[int],
    *,
    emit_warning: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter stale strikes based on last trade date metadata.

    Args:
        options_data: Raw option quotes, expected to include a ``last_trade_date``
            column when staleness filtering is desired.
        valuation_date: Valuation timestamp used to compute quote ages. Accepts
            any pandas-compatible timestamp.
        max_staleness_days: Maximum allowed age for quotes. ``None`` disables the
            filter or when ``last_trade_date`` is absent.
        emit_warning: Whether to emit a summary warning when rows are removed.

    Returns:
        Tuple containing:
        - DataFrame restricted to rows that satisfy the staleness constraint.
        - Dictionary with filtering statistics (removed count, age range, etc.)
    """

    if max_staleness_days is None or "last_trade_date" not in options_data.columns:
        return options_data, {}

    # Normalize timezone to avoid tz-aware vs tz-naive subtraction; Golden Master
    # data comes in with UTC stamps while valuation_date is naive.
    last_trade_datetimes = pd.to_datetime(options_data["last_trade_date"], utc=True)
    last_trade_datetimes = last_trade_datetimes.dt.tz_localize(None)
    if last_trade_datetimes.isna().any():
        return options_data, {}

    options_data = options_data.copy()
    valuation_ts = pd.Timestamp(valuation_date).tz_localize(None)
    days_old = (
        valuation_ts - last_trade_datetimes.dt.normalize()
    ).dt.days  # TODO check how timezone normalization works
    fresh_mask = days_old <= max_staleness_days

    stale_rows = options_data[~fresh_mask]
    shared_strike_mask = np.zeros(len(options_data), dtype=bool)
    unique_stale_strikes = 0

    if "strike" in options_data.columns and not stale_rows.empty:
        unique_strikes = stale_rows["strike"].unique()
        unique_stale_strikes = len(unique_strikes)
        shared_strike_mask = options_data["strike"].isin(unique_strikes)

    combined_stale_mask = (~fresh_mask) | shared_strike_mask
    removed_count = int(np.sum(combined_stale_mask))

    stats: Dict[str, Any] = {}
    if removed_count > 0:
        removed_days = days_old[combined_stale_mask]
        min_age = int(removed_days.min()) if not removed_days.empty else "N/A"
        max_age = int(removed_days.max()) if not removed_days.empty else "N/A"
        strike_desc = unique_stale_strikes if unique_stale_strikes else "N/A"

        stats = {
            "removed_count": removed_count,
            "max_staleness_days": max_staleness_days,
            "min_age": min_age,
            "max_age": max_age,
            "strike_desc": strike_desc,
        }

        if emit_warning:
            warnings.warn(
                f"Filtered {removed_count} option rows (covering {strike_desc} strikes) "
                f"older than {max_staleness_days} days "
                f"(most recent: {min_age} days old, oldest: {max_age} days old)",
                UserWarning,
            )

    filtered = options_data[~combined_stale_mask].reset_index(drop=True)
    return filtered, stats


def select_price_column(
    options_data: pd.DataFrame,
    price_method: Literal["last", "mid"],
    *,
    emit_warning: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Select the appropriate option price column based on user preference.

    Args:
        options_data: Quote DataFrame containing price columns such as ``bid``,
            ``ask``, ``mid``, and ``last_price``.
        price_method: Desired pricing convention. ``"mid"`` uses mid quotes when
            available, falling back to last traded price. ``"last"`` always uses
            the last traded price.

    Returns:
        tuple[DataFrame, int]: A tuple containing:
            - DataFrame with a ``price`` column populated according to the requested
              pricing convention and filtered for positive prices.
            - Count of rows whose missing mid/bid/ask price was filled with
              ``last_price``.

    Raises:
        CalculationError: When the requested pricing convention cannot be
            satisfied given the available columns.
    """

    data = options_data.copy()
    fallback_candidate_mask = pd.Series(False, index=data.index)

    if price_method == "mid":
        if "mid" in data.columns:
            data["price"] = data["mid"]
            fallback_candidate_mask = data["price"].isna()
        elif "bid" in data.columns and "ask" in data.columns:
            mid = (data["bid"] + data["ask"]) / 2
            quoted_mid_mask = data["bid"].notna() & data["ask"].notna()
            fallback_candidate_mask = ~quoted_mid_mask
            if quoted_mid_mask.any():
                data["price"] = mid
            else:
                data["price"] = np.nan
        else:
            raise CalculationError(
                "Requested price_method='mid' but bid/ask data not available. "
                "Provide bid/ask columns or a precomputed mid price."
            )
    else:
        data["price"] = data["last_price"]

    if "price" not in data.columns:
        raise CalculationError("Failed to determine option price column")

    successful_fallback_mask = pd.Series(False, index=data.index)
    if price_method == "mid" and fallback_candidate_mask.any():
        if "last_price" not in data.columns:
            raise CalculationError(
                "Requested price_method='mid' but fallback last_price is unavailable."
            )
        last_price_values = pd.to_numeric(data["last_price"], errors="coerce")
        successful_fallback_mask = (
            fallback_candidate_mask
            & np.isfinite(last_price_values)
            & (last_price_values > 0)
        )
        data.loc[successful_fallback_mask, "price"] = data.loc[
            successful_fallback_mask, "last_price"
        ]

    survivor_mask = pd.to_numeric(data["price"], errors="coerce") > 0
    filled_count = int((successful_fallback_mask & survivor_mask).sum())

    if filled_count > 0 and emit_warning:
        warnings.warn(
            f"Filled {filled_count} missing mid prices with last_price due to unavailable bid/ask",
            UserWarning,
        )

    data = data[survivor_mask].copy()
    return data, filled_count


__all__ = [
    "filter_stale_options",
    "select_price_column",
]
