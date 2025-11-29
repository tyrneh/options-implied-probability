"""Pipeline for fitting independent volatility slices."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from oipd.core.errors import CalculationError
from oipd.pipelines.market_inputs import (
    FillMode,
    MarketInputs,
    VendorSnapshot,
    resolve_market,
)
from oipd.pipelines.vol_estimation import fit_vol_curve_internal
from oipd.pipelines.vol_surface.models import DiscreteSurface


def fit_independent_slices(
    chain: pd.DataFrame,
    market: MarketInputs,
    *,
    vendor: Optional[VendorSnapshot],
    fill_mode: FillMode,
    column_mapping: Optional[Mapping[str, str]],
    method_options: Optional[Mapping[str, Any]],
    pricing_engine: str,
    price_method: str,
    max_staleness_days: int,
    solver: str,
    method: str,
) -> DiscreteSurface:
    """Fit a volatility curve independently for each expiry in the chain.

    Args:
        chain: Option chain DataFrame containing multiple expiries.
        market: Base market inputs (valuation date, rates, underlying price).
        vendor: Optional vendor snapshot to fill missing market fields.
        fill_mode: How to combine user/vendor inputs.
        column_mapping: Optional mapping from user column names to OIPD standard names.
        method_options: Options for the fitting method.
        pricing_engine: Pricing engine ("black76" or "bs").
        price_method: Price selection strategy (``"mid"`` or ``"last"``).
        max_staleness_days: Maximum allowed quote age before filtering.
        solver: Implied-vol solver to use.
        method: Smile fitting method (e.g., ``"svi"``).

    Returns:
        SliceCollection: A collection of fitted slices.
    """
    chain_input = chain.copy()
    if column_mapping:
        chain_input = chain_input.rename(columns=column_mapping)

    if "expiry" not in chain_input.columns:
        raise CalculationError(
            "Expiry column 'expiry' not found in input data"
        )

    expiries = pd.to_datetime(chain_input["expiry"], errors="coerce")
    if expiries.isna().any():
        raise CalculationError("Invalid expiry values encountered during parsing")
    expiries = expiries.dt.tz_localize(None)
    chain_input["expiry"] = expiries

    unique_expiries = sorted(expiries.unique())
    
    slices: dict[pd.Timestamp, dict[str, Any]] = {}
    resolved_markets: dict[pd.Timestamp, Any] = {}
    slice_chains: dict[pd.Timestamp, pd.DataFrame] = {}

    for expiry_timestamp in unique_expiries:
        expiry_date = expiry_timestamp.date()
        slice_df = chain_input[chain_input["expiry"] == expiry_timestamp].copy()

        slice_market = MarketInputs(
            risk_free_rate=market.risk_free_rate,
            valuation_date=market.valuation_date,
            risk_free_rate_mode=market.risk_free_rate_mode,
            underlying_price=market.underlying_price,
            dividend_yield=market.dividend_yield,
            dividend_schedule=market.dividend_schedule,
            expiry_date=expiry_date,
        )

        resolved = resolve_market(slice_market, vendor, mode=fill_mode)
        vol_curve, metadata = fit_vol_curve_internal(
            slice_df,
            resolved,
            pricing_engine=pricing_engine,
            price_method=price_method,
            max_staleness_days=max_staleness_days,
            solver=solver,
            method=method,
            method_options=method_options,
        )

        slices[expiry_timestamp] = {
            "curve": vol_curve,
            "metadata": metadata,
        }
        resolved_markets[expiry_timestamp] = resolved
        slice_chains[expiry_timestamp] = slice_df

    return DiscreteSurface(slices, resolved_markets, slice_chains)
