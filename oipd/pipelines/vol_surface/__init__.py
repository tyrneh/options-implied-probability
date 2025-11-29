"""Surface fitting pipeline dispatcher."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from oipd.pipelines.market_inputs import FillMode, MarketInputs, VendorSnapshot
from oipd.pipelines.vol_surface.svi_slices import fit_independent_slices
from oipd.pipelines.vol_surface.models import FittedSurface


def fit_surface(
    chain: pd.DataFrame,
    market: MarketInputs,
    *,
    vendor: Optional[VendorSnapshot] = None,
    fill_mode: FillMode = "strict",
    column_mapping: Optional[Mapping[str, str]] = None,
    method_options: Optional[Mapping[str, Any]] = None,
    pricing_engine: str = "black76",
    price_method: str = "mid",
    max_staleness_days: int = 3,
    solver: str = "brent",
    method: str = "svi",
) -> FittedSurface:
    """Fit a volatility surface using the specified method.

    Args:
        chain: Option chain DataFrame.
        market: Market inputs.
        vendor: Vendor snapshot.
        fill_mode: Fill mode.
        column_mapping: Column mapping.
        method_options: Options for the fitting method.
        pricing_engine: Pricing engine ("black76" or "bs").
        price_method: Price method.
        max_staleness_days: Max staleness days.
        solver: Solver.
        method: Fitting method (e.g., "svi").

    Returns:
        FittedSurface: Fitted surface model.
    """
    # Dispatch based on method.
    # Currently only "svi" (independent slices) is supported.
    # In the future, "ssvi", "local_vol", etc. can be added here.

    if method == "svi":
        return fit_independent_slices(
            chain=chain,
            market=market,
            vendor=vendor,
            fill_mode=fill_mode,
            column_mapping=column_mapping,
            method_options=method_options,
            pricing_engine=pricing_engine,
            price_method=price_method,
            max_staleness_days=max_staleness_days,
            solver=solver,
            method=method,
        )

    # Fallback or error for unknown methods
    # For now, we assume "svi" is the default and only option if not specified otherwise,
    # but strictly we should raise if unknown.
    # However, fit_independent_slices handles the "method" arg for the curve fitting itself (e.g. "svi" vs "sabr" for curves).
    # This is a slight ambiguity: is "method" the surface method or the curve method?
    # In the current design, "svi" implies "independent slices using SVI curves".

    # If the user passes something else, we might want to try fitting independent slices with that method?
    # e.g. method="sabr" -> independent slices of SABR curves.

    # Let's assume for now that if it's not a known *surface* method (like "ssvi"),
    # we default to independent slices and pass the method down to the curve fitter.
    return fit_independent_slices(
        chain=chain,
        market=market,
        vendor=vendor,
        fill_mode=fill_mode,
        column_mapping=column_mapping,
        method_options=method_options,
        pricing_engine=pricing_engine,
        price_method=price_method,
        max_staleness_days=max_staleness_days,
        solver=solver,
        method=method,
    )
