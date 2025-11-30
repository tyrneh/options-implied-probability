"""Stateless pipeline helpers for volatility estimation."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple, Mapping

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.pipelines._legacy.estimator import ModelParams, _estimate
from oipd.data_access.readers import DataFrameReader
from oipd.pricing.utils import prepare_dividends
from oipd.market_inputs import ResolvedMarket


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
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit a volatility curve to a single slice of options data.

    Args:
        options_data: DataFrame containing option quotes.
        resolved_market: Market inputs (rates, spot, etc.).
        pricing_engine: 'black76' or 'bs'.
        price_method: Column to use for pricing ('mid', 'last', etc.).
        max_staleness_days: Filter out quotes older than this.
        solver: IV solver method.
        method: Volatility fitting method (e.g., 'svi').
        method_options: Options for the fitting method.

    Returns:
        Tuple containing:
        - The fitted volatility curve object (callable).
        - A dictionary of metadata (residuals, parameters, etc.).
    """
    valuation_date = resolved_market.valuation_date

    # 1. Prepare Dividends / Spot
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

    # Clean and normalize input similarly to DataFrameSource used in legacy RND
    reader = DataFrameReader()
    cleaned_options = reader.read(options_data.copy())

    # Mirror legacy pipeline via ModelParams/_estimate to ensure bitwise parity
    model_params = ModelParams(
        solver=solver,
        pricing_engine=pricing_engine,
        price_method=price_method,
        max_staleness_days=max_staleness_days,
        surface_method=method,
        surface_options=method_options,
    )

    # _estimate returns pdf/cdf we ignore; meta carries the fitted vol curve
    _, _, _, meta = _estimate(cleaned_options, resolved_market, model_params)
    if "vol_curve" not in meta:
        raise CalculationError("Expected vol_curve in metadata from _estimate")

    vol_curve = meta["vol_curve"]
    metadata = {
        "forward_price": meta.get("forward_price", resolved_market.underlying_price),
        "pricing_engine": pricing_engine,
        "method": meta.get("surface_fit", method),
        "fit_diagnostics": meta.get("observed_iv"),
        **{k: v for k, v in meta.items() if k.startswith("observed_iv")},
    }

    return vol_curve, metadata


def compute_fitted_smile(
    vol_curve: Any,
    metadata: Dict[str, Any],
    domain: Optional[Tuple[float, float]] = None,
    points: int = 200,
    include_observed: bool = True,
) -> pd.DataFrame:
    """
    Generate a DataFrame representing the fitted smile and observed data.

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
                # Add 20% padding
                padding = 0.2 * (max_strike - min_strike)
                strike_grid = np.linspace(
                    max(0.01, min_strike - padding),
                    max_strike + padding,
                    points,
                )
    else:
        strike_grid = np.linspace(domain[0], domain[1], points)

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
