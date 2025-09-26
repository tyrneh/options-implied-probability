from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from oipd.core.errors import CalculationError
from oipd.core.iv import compute_iv as _compute_iv
from oipd.core.parity import preprocess_with_parity
from oipd.market_inputs import ResolvedMarket


def apply_put_call_parity(
    options_data: pd.DataFrame,
    spot: float,
    resolved_market: ResolvedMarket,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """Apply put-call parity preprocessing and attempt to infer forward price."""

    discount_factor = float(
        np.exp(-resolved_market.risk_free_rate * resolved_market.days_to_expiry / 365.0)
    )
    processed = preprocess_with_parity(options_data, spot, discount_factor)
    forward_price = None
    if "F_used" in processed.columns:
        try:
            forward_price = float(processed["F_used"].iloc[0])
        except Exception:
            forward_price = None
    return processed, forward_price


def filter_stale_options(
    options_data: pd.DataFrame,
    valuation_date,
    max_staleness_days: Optional[int],
    *,
    emit_warning: bool = True,
) -> pd.DataFrame:
    """Filter stale strikes based on last_trade_date column."""

    if max_staleness_days is None or "last_trade_date" not in options_data.columns:
        return options_data

    last_trade_datetimes = pd.to_datetime(options_data["last_trade_date"])
    if last_trade_datetimes.isna().any():
        return options_data

    options_data = options_data.copy()
    last_trade_dates = last_trade_datetimes.dt.date
    days_old = pd.Series((valuation_date - last_trade_dates).days)
    fresh_mask = days_old <= max_staleness_days
    stale_count = (~fresh_mask).sum()
    if stale_count > 0 and emit_warning:
        warnings.warn(
            f"Filtered {stale_count} strikes older than {max_staleness_days} days "
            f"(most recent: {days_old.min()} days old, oldest: {days_old.max()} days old)",
            UserWarning,
        )
    return options_data[fresh_mask].reset_index(drop=True)


def select_price_column(
    options_data: pd.DataFrame, price_method: Literal["last", "mid"]
) -> pd.DataFrame:
    """Select the appropriate option price column based on user preference."""

    data = options_data.copy()

    if price_method == "mid":
        if "mid" in data.columns:
            data["price"] = data["mid"]
        elif "bid" in data.columns and "ask" in data.columns:
            mid = (data["bid"] + data["ask"]) / 2
            mask = data["bid"].notna() & data["ask"].notna()
            if mask.any():
                data["price"] = np.where(mask, mid, data["last_price"])
                if not mask.all():
                    warnings.warn(
                        "Using last_price for rows with missing bid/ask",
                        UserWarning,
                    )
            else:
                warnings.warn(
                    "Requested price_method='mid' but bid/ask data not available. "
                    "Falling back to price_method='last'",
                    UserWarning,
                )
                data["price"] = data["last_price"]
        else:
            raise CalculationError(
                "Requested price_method='mid' but bid/ask data not available. "
                "Provide bid/ask columns or a precomputed mid price."
            )
    else:
        data["price"] = data["last_price"]

    if "price" not in data.columns:
        raise CalculationError("Failed to determine option price column")

    if data["price"].isna().any() and "last_price" in data.columns:
        missing_mask = data["price"].isna()
        if missing_mask.any():
            data.loc[missing_mask, "price"] = data.loc[missing_mask, "last_price"]
            if missing_mask.any():
                warnings.warn(
                    "Filled missing mid prices with last_price due to unavailable bid/ask",
                    UserWarning,
                )

    data = data[data["price"] > 0].copy()
    return data


def compute_iv(
    options_data_priced: pd.DataFrame,
    underlying: Optional[float],
    resolved_market: ResolvedMarket,
    solver: Literal["brent", "newton"],
    pricing_engine: Literal["black76", "bs"],
    dividend_yield: Optional[float],
) -> pd.DataFrame:
    """Vectorized implied volatility extraction."""

    if underlying is None:
        raise ValueError(
            "Effective underlying/forward price is required for IV extraction"
        )

    return _compute_iv(
        options_data_priced,
        underlying,
        days_to_expiry=resolved_market.days_to_expiry,
        risk_free_rate=resolved_market.risk_free_rate,
        solver_method=solver,
        pricing_engine=pricing_engine,
        dividend_yield=dividend_yield,
    )


__all__ = [
    "apply_put_call_parity",
    "filter_stale_options",
    "select_price_column",
    "compute_iv",
]
