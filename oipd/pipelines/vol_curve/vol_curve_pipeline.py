"""Stateless pipeline helpers for volatility estimation."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Mapping

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.core.data_processing import (
    apply_put_call_parity,
    compute_iv,
    filter_stale_options,
    select_price_column,
)
from oipd.core.vol_surface_fitting import fit_slice
from oipd.core.vol_surface_fitting.shared.svi_types import SVICalibrationOptions
from oipd.data_access.readers import DataFrameReader
from oipd.pricing.utils import prepare_dividends
from oipd.market_inputs import ResolvedMarket
from oipd.core.utils import calculate_days_to_expiry, convert_days_to_years


def fit_vol_curve_internal(
    options_data: pd.DataFrame,
    resolved_market: ResolvedMarket,
    *,
    pricing_engine: Literal["black76", "bs"] = "black76",
    price_method: Literal["mid", "last", "bid", "ask"] = "mid",
    max_staleness_days: int = 3,
    solver: Literal["brent", "newton"] = "brent",
    method: str = "svi",
    method_options: Optional[Mapping[str, Any] | SVICalibrationOptions] = None,
    suppress_price_warning: bool = False,
    suppress_staleness_warning: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit a volatility curve to a single slice of options data.

    This function performs vol-only fitting without computing the risk-neutral
    distribution (RND). RND computation is deferred to ``implied_distribution()``.

    Args:
        options_data: DataFrame containing option quotes.
        resolved_market: Market inputs (rates, spot, etc.).
        pricing_engine: 'black76' or 'bs'.
        price_method: Column to use for pricing ('mid', 'last', etc.).
        max_staleness_days: Filter out quotes older than this.
        max_staleness_days: Filter out quotes older than this.
        solver: IV solver method.
        method: Volatility fitting method (e.g., 'svi').
        method_options: Options for the fitting method.
        suppress_price_warning: If True, suppress warning when filling missing mid prices.
        suppress_staleness_warning: If True, suppress warning when filtering stale quotes.

    Returns:
        Tuple containing:
        - The fitted volatility curve object (callable).
        - A dictionary of metadata (residuals, parameters, etc.).
    """
    valuation_date = resolved_market.valuation_date

    # 1. Prepare Dividends / Spot for Black-Scholes
    if pricing_engine == "bs":
        effective_spot, effective_dividend = prepare_dividends(
            underlying=resolved_market.underlying_price,
            dividend_schedule=resolved_market.dividend_schedule,
            dividend_yield=resolved_market.dividend_yield,
            r=resolved_market.risk_free_rate,
            valuation_date=valuation_date,
        )
    else:
        effective_spot = resolved_market.underlying_price
        effective_dividend = None

    # 2. Clean and normalize input
    reader = DataFrameReader()
    cleaned_options = reader.read(options_data.copy())

    # 3. Apply put-call parity to get synthetic calls + infer forward price
    parity_adjusted, forward_price = apply_put_call_parity(
        cleaned_options, effective_spot, resolved_market
    )

    # For Black-76, use forward price; for BS, use spot
    if pricing_engine == "black76":
        if forward_price is None:
            raise CalculationError(
                "Black-76 requires parity-implied forward but put quotes are missing."
            )
        else:
            underlying_for_iv = forward_price
    else:
        underlying_for_iv = effective_spot

    # 4. Filter stale quotes
    filtered_options, staleness_stats = filter_stale_options(
        parity_adjusted,
        valuation_date,
        max_staleness_days,
        emit_warning=not suppress_staleness_warning,
    )

    # 5. Select price column (mid, last, bid, ask)
    priced_options, mid_price_filled = select_price_column(
        filtered_options, price_method, emit_warning=not suppress_price_warning
    )
    if priced_options.empty:
        raise CalculationError("No valid options data after price selection")

    # Calculate time to expiry for this slice
    # priced_options came from cleaned_options which has 'expiry'
    if "expiry" not in priced_options.columns:
        raise CalculationError("Options data missing 'expiry' column.")

    expiry_val = priced_options["expiry"].iloc[0]
    expiry_date = (
        expiry_val.date()
        if isinstance(expiry_val, pd.Timestamp)
        else pd.to_datetime(expiry_val).date()
    )
    days_to_expiry_slice = calculate_days_to_expiry(expiry_date, valuation_date)

    # 6. Compute implied volatilities
    options_with_iv = compute_iv(
        priced_options,
        underlying_for_iv,
        days_to_expiry=days_to_expiry_slice,
        risk_free_rate=resolved_market.risk_free_rate,
        solver_method=solver,
        pricing_engine=pricing_engine,
        dividend_yield=effective_dividend,
    )

    # 7. Fit the volatility smile (SVI or bspline)
    strikes = options_with_iv["strike"].to_numpy(dtype=float)
    ivs = options_with_iv["iv"].to_numpy(dtype=float)

    # Extract volume array if available
    volume_array: np.ndarray | None = None
    if "volume" in options_with_iv.columns:
        volume_array = options_with_iv["volume"].to_numpy(dtype=float)
        if not np.isfinite(volume_array).any() or np.all(volume_array <= 0):
            volume_array = None

    # 8. Compute observed bid/ask/last IVs for plotting AND for SVI calibration
    observed_bid_iv = _compute_observed_iv(
        priced_options,
        "bid",
        underlying_for_iv,
        resolved_market,
        solver,
        pricing_engine,
        effective_dividend,
        days_to_expiry=days_to_expiry_slice,
    )
    observed_ask_iv = _compute_observed_iv(
        priced_options,
        "ask",
        underlying_for_iv,
        resolved_market,
        solver,
        pricing_engine,
        effective_dividend,
        days_to_expiry=days_to_expiry_slice,
    )
    observed_last_iv = _compute_observed_iv(
        priced_options,
        "last_price",
        underlying_for_iv,
        resolved_market,
        solver,
        pricing_engine,
        effective_dividend,
        days_to_expiry=days_to_expiry_slice,
    )

    # Align bid/ask IVs with the strike array for SVI calibration
    fit_kwargs: Dict[str, Any] = {}
    if method == "svi":

        def _align_iv_series(iv_df: Optional[pd.DataFrame]) -> np.ndarray | None:
            if iv_df is None or iv_df.empty:
                return None
            joined = (
                pd.DataFrame({"strike": strikes})
                .merge(iv_df, on="strike", how="left")
                .sort_index()
            )
            iv_values = joined["iv"].to_numpy(dtype=float)
            if np.all(np.isnan(iv_values)):
                return None
            return iv_values

        aligned_bid_iv = _align_iv_series(observed_bid_iv)
        aligned_ask_iv = _align_iv_series(observed_ask_iv)

        if aligned_bid_iv is not None or aligned_ask_iv is not None:
            fit_kwargs["bid_iv"] = aligned_bid_iv
            fit_kwargs["ask_iv"] = aligned_ask_iv
        if volume_array is not None:
            fit_kwargs["volumes"] = volume_array

    vol_curve, fit_metadata = fit_slice(
        method,
        strikes,
        ivs,
        forward=underlying_for_iv,
        maturity_years=convert_days_to_years(days_to_expiry_slice),
        options=method_options,
        **fit_kwargs,
    )

    # Calculate At-The-Money (ATM) Volatility
    # Defined as IV at strike = forward
    try:
        atm_vol = float(vol_curve(underlying_for_iv))
    except (TypeError, ValueError):
        atm_vol = None

    # 9. Return vol curve + metadata (NO RND computation!)
    metadata = {
        "at_money_vol": atm_vol,
        "forward_price": underlying_for_iv,
        "pricing_engine": pricing_engine,
        "method": method,
        "observed_iv": options_with_iv,
        "observed_iv_bid": observed_bid_iv,
        "observed_iv_ask": observed_ask_iv,
        "observed_iv_last": observed_last_iv,
        "mid_price_filled": mid_price_filled,
        "staleness_report": staleness_stats,
        "expiry_date": expiry_date,
        **fit_metadata,
    }

    return vol_curve, metadata


def _compute_observed_iv(
    source_df: pd.DataFrame,
    price_column: str,
    underlying_price: float,
    resolved_market: ResolvedMarket,
    solver: str,
    pricing_engine: str,
    dividend_yield: Optional[float],
    days_to_expiry: int,
) -> Optional[pd.DataFrame]:
    """Compute implied volatility for an alternate observed price column."""

    if price_column not in source_df.columns:
        return None

    priced = source_df.loc[
        source_df[price_column].notna() & (source_df[price_column] > 0)
    ].copy()
    if priced.empty:
        return None

    priced["price"] = priced[price_column]
    try:
        iv_df = compute_iv(
            priced,
            underlying_price,
            days_to_expiry=days_to_expiry,
            risk_free_rate=resolved_market.risk_free_rate,
            solver_method=solver,
            pricing_engine=pricing_engine,
            dividend_yield=dividend_yield,
        )
    except Exception:
        return None

    columns = ["strike", "iv"]
    if "option_type" in iv_df.columns:
        columns.append("option_type")
    return iv_df.loc[:, columns]


def compute_fitted_smile(
    vol_curve: Any,
    metadata: Dict[str, Any],
    domain: Optional[Tuple[float, float]] = None,
    points: int = 200,
    include_observed: bool = True,
) -> pd.DataFrame:
    """
    Generate a DataFrame representing the fitted smile and observed data.

    The evaluation grid is the union of a smooth linspace and all observed strikes to preserve market data points.

    Args:
        vol_curve: The callable volatility curve.
        metadata: Metadata dictionary containing observed IVs.
        domain: Optional (min, max) strike range.
        points: Number of points in the grid.
        include_observed: Whether to include market observed IVs (mid, bid, ask).

    Returns:
        DataFrame with columns: strike, fitted_iv, [market_iv, market_bid_iv, ...]
    """
    observed_iv = metadata.get("observed_iv")

    # Determine grid
    if domain is None:
        if observed_iv is None or observed_iv.empty:
            # Fallback: check for default_domain in metadata (interpolated slices)
            default_domain = metadata.get("default_domain")
            if default_domain:
                min_strike, max_strike = default_domain
                # Add 5% padding for interpolated slices too
                padding = 0.05 * (max_strike - min_strike)
                strike_grid = np.linspace(
                    max(0.01, min_strike - padding),
                    max_strike + padding,
                    points,
                )
            else:
                raise ValueError(
                    "No observed data found to infer grid domain. "
                    "Please provide an explicit `domain=(min, max)` argument."
                )
        else:
            min_strike = observed_iv["strike"].min()
            max_strike = observed_iv["strike"].max()
            if np.isclose(min_strike, max_strike):
                strike_grid = np.array([min_strike])
            else:
                # Add 5% padding
                padding = 0.05 * (max_strike - min_strike)
                lower_bound = min_strike - padding

                # Prevent lower bound from dropping too close to zero or negative
                # If padding pushes below zero, we clamp to 50% of the minimum strike
                # to maintain a reasonable visualization range.
                if lower_bound <= 0:
                    lower_bound = min_strike * 0.5

                strike_grid = np.linspace(
                    max(0.01, lower_bound),
                    max_strike + padding,
                    points,
                )
    else:
        strike_grid = np.linspace(domain[0], domain[1], points)

    # Ensure observed strikes are included in the grid so we can merge them later
    if observed_iv is not None and not observed_iv.empty:
        observed_strikes = observed_iv["strike"].to_numpy(dtype=float)
        # Union and sort
        strike_grid = np.unique(np.concatenate((strike_grid, observed_strikes)))
        strike_grid.sort()

    # Evaluate curve
    fitted_values = vol_curve(strike_grid)

    smile_df = pd.DataFrame(
        {
            "strike": strike_grid,
            "fitted_iv": fitted_values,
        }
    )

    if not include_observed:
        return smile_df

    # Merge observed data if available
    if observed_iv is not None and not observed_iv.empty:
        # Mid IV
        mid_subset = observed_iv[["strike", "iv"]].rename(columns={"iv": "market_iv"})
        smile_df = pd.merge(smile_df, mid_subset, on="strike", how="left")

    # Bid/Ask/Last IVs from metadata
    for key, col_name in [
        ("observed_iv_bid", "market_bid_iv"),
        ("observed_iv_ask", "market_ask_iv"),
        ("observed_iv_last", "market_last_iv"),
    ]:
        obs_df = metadata.get(key)
        if isinstance(obs_df, pd.DataFrame) and not obs_df.empty:
            subset = obs_df[["strike", "iv"]].rename(columns={"iv": col_name})
            # Ensure strike is float for merging
            subset["strike"] = subset["strike"].astype(float)
            smile_df = pd.merge(smile_df, subset, on="strike", how="left")

    return smile_df
