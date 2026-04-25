"""Stateless pipeline helpers for volatility estimation."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Mapping

import numpy as np
import pandas as pd

from oipd.core.maturity import build_maturity_metadata, resolve_maturity
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
from oipd.core.utils import (
    resolve_risk_free_rate,
)

PARITY_FORWARD_DATA_REQUIREMENT_MESSAGE = (
    "Public volatility fitting requires usable same-strike call/put pairs for "
    "parity-forward inference. Provide both call and put quotes at matching "
    "strikes with usable prices for each expiry."
)


def _extract_strike_domain(options_data: pd.DataFrame) -> tuple[float, float] | None:
    """Return the finite strike domain for one options table when available.

    Args:
        options_data: Option rows that may contain a ``strike`` column.

    Returns:
        tuple[float, float] | None: Minimum and maximum strike when available.
    """
    if "strike" not in options_data.columns or options_data.empty:
        return None

    strike_values = options_data["strike"].to_numpy(dtype=float)
    strike_values = strike_values[np.isfinite(strike_values)]
    if strike_values.size == 0:
        return None

    return float(np.min(strike_values)), float(np.max(strike_values))


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
    """Fit a volatility curve to a single slice of options data.

    This function performs vol-only fitting without computing the risk-neutral
    distribution (RND). RND computation is deferred to ``implied_distribution()``.

    Args:
        options_data: DataFrame containing option quotes.
        resolved_market: Market inputs (rates, spot, etc.).
        pricing_engine: 'black76' or 'bs'.
        price_method: Column to use for pricing ('mid', 'last', etc.).
        max_staleness_days: Filter out quotes older than this.
        solver: IV solver method.
        method: Volatility fitting method (e.g., 'svi').
        method_options: Options for the fitting method.
        suppress_price_warning: If True, suppress warning when filling missing mid prices.
        suppress_staleness_warning: If True, suppress warning when filtering stale quotes.

    Returns:
        Tuple containing the fitted volatility curve object and metadata.

    Raises:
        CalculationError: If maturity, parity forward inference, price selection,
            or implied-volatility extraction fails.
        ValueError: If input cleaning, parity preprocessing, or numerical inputs are
            structurally invalid.
    """
    valuation_timestamp = resolved_market.valuation_timestamp

    # 1. Clean and normalize input
    reader = DataFrameReader()
    cleaned_options = reader.read(options_data.copy())

    # Derive expiry for this slice (required for time to expiry and rate conversion)
    if "expiry" not in cleaned_options.columns:
        raise CalculationError("Options data missing 'expiry' column.")

    expiry_val = cleaned_options["expiry"].iloc[0]
    resolved_maturity = resolve_maturity(
        expiry_val,
        valuation_timestamp,
        floor_at_zero=False,
    )
    years_to_expiry = resolved_maturity.time_to_expiry_years
    if years_to_expiry <= 0:
        raise CalculationError(
            "Expiry must be strictly after valuation_date for volatility calibration."
        )

    rate_mode = resolved_market.source_meta["risk_free_rate_mode"]
    effective_r = resolve_risk_free_rate(
        resolved_market.risk_free_rate, rate_mode, years_to_expiry
    )

    # 2. Prepare Dividends / Spot for Black-Scholes
    if pricing_engine == "bs":
        effective_spot, effective_dividend = prepare_dividends(
            underlying=resolved_market.underlying_price,
            dividend_schedule=resolved_market.dividend_schedule,
            dividend_yield=resolved_market.dividend_yield,
            r=effective_r,
            valuation_date=valuation_timestamp,
            expiry=resolved_maturity.expiry,
        )
    else:
        effective_spot = resolved_market.underlying_price
        effective_dividend = None

    # 3. Apply put-call parity to get synthetic calls + infer forward price
    parity_adjusted, forward_price = apply_put_call_parity(
        cleaned_options, effective_spot, resolved_market
    )
    parity_report = parity_adjusted.attrs.get("parity_report")
    forward_price_source = None

    # For Black-76, use forward price; for BS, use spot
    if pricing_engine == "black76":
        if forward_price is None:
            raise CalculationError(PARITY_FORWARD_DATA_REQUIREMENT_MESSAGE)
        else:
            underlying_for_iv = forward_price
            forward_price_source = "put_call_parity"
    else:
        underlying_for_iv = effective_spot

    # 4. Filter stale quotes
    filtered_options, staleness_stats = filter_stale_options(
        parity_adjusted,
        valuation_timestamp,
        max_staleness_days,
        emit_warning=not suppress_staleness_warning,
    )

    # 5. Select price column (mid, last, bid, ask)
    priced_options, mid_price_filled = select_price_column(
        filtered_options, price_method, emit_warning=not suppress_price_warning
    )
    if priced_options.empty:
        raise CalculationError("No valid options data after price selection")

    # 6. Compute implied volatilities
    options_with_iv = compute_iv(
        priced_options,
        underlying_for_iv,
        time_to_expiry_years=years_to_expiry,
        risk_free_rate=effective_r,
        solver_method=solver,
        pricing_engine=pricing_engine,
        dividend_yield=effective_dividend,
    )

    # 7. Fit the volatility smile (SVI or bspline)
    strikes = options_with_iv["strike"].to_numpy(dtype=float)
    ivs = options_with_iv["iv"].to_numpy(dtype=float)

    volume_array = _extract_volume_array(options_with_iv)

    # 8. Compute observed bid/ask/last IVs for plotting AND for SVI calibration
    observed_bid_iv = _compute_observed_iv(
        priced_options,
        "bid",
        underlying_for_iv,
        effective_r,
        solver,
        pricing_engine,
        effective_dividend,
        time_to_expiry_years=years_to_expiry,
    )
    observed_ask_iv = _compute_observed_iv(
        priced_options,
        "ask",
        underlying_for_iv,
        effective_r,
        solver,
        pricing_engine,
        effective_dividend,
        time_to_expiry_years=years_to_expiry,
    )
    observed_last_iv = _compute_observed_iv(
        priced_options,
        "last_price",
        underlying_for_iv,
        effective_r,
        solver,
        pricing_engine,
        effective_dividend,
        time_to_expiry_years=years_to_expiry,
    )

    # Align bid/ask IVs with the strike array for SVI calibration
    fit_kwargs: Dict[str, Any] = {}
    if method == "svi":

        def _align_iv_series(iv_df: Optional[pd.DataFrame]) -> np.ndarray | None:
            """Align observed IV values to the fitted strike order.

            Args:
                iv_df: Observed IV table with ``strike`` and ``iv`` columns.

            Returns:
                IV array in fitted strike order, or ``None`` when no finite IVs
                are available.
            """
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
        maturity_years=years_to_expiry,
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
        "raw_observed_domain": _extract_strike_domain(cleaned_options),
        "post_iv_survival_domain": _extract_strike_domain(options_with_iv),
        "observed_iv": options_with_iv,
        "observed_iv_bid": observed_bid_iv,
        "observed_iv_ask": observed_ask_iv,
        "observed_iv_last": observed_last_iv,
        "mid_price_filled": mid_price_filled,
        "staleness_report": staleness_stats,
        **build_maturity_metadata(resolved_maturity),
        "risk_free_rate_continuous": effective_r,
        **fit_metadata,
    }
    if parity_report is not None:
        metadata["parity_report"] = parity_report
    if forward_price_source is not None:
        metadata["forward_price_source"] = forward_price_source

    return vol_curve, metadata


def _compute_observed_iv(
    source_df: pd.DataFrame,
    price_column: str,
    underlying_price: float,
    risk_free_rate: float,
    solver: str,
    pricing_engine: str,
    dividend_yield: Optional[float],
    time_to_expiry_years: float,
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
            time_to_expiry_years=time_to_expiry_years,
            risk_free_rate=risk_free_rate,
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


def _extract_volume_array(options_with_iv: pd.DataFrame) -> Optional[np.ndarray]:
    """Return valid per-strike volume values for SVI weighting.

    Args:
        options_with_iv: IV-ready options data after parity preprocessing and
            price selection.

    Returns:
        Optional volume array aligned with ``options_with_iv`` rows. Returns
        ``None`` when no positive finite volume information is available.
    """

    if "volume" not in options_with_iv.columns:
        return None

    volume_array = pd.to_numeric(options_with_iv["volume"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(volume_array).any() or np.all(volume_array <= 0):
        return None
    return volume_array


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
