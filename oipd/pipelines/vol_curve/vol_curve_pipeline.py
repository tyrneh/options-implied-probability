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
