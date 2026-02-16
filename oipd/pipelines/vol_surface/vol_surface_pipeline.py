"""Surface fitting pipeline dispatcher."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

import pandas as pd

from oipd.market_inputs import MarketInputs
from oipd.pipelines.vol_surface.svi_slices import fit_independent_slices
from oipd.pipelines.vol_surface.models import FittedSurface


def fit_surface(
    chain: pd.DataFrame,
    market: MarketInputs,
    *,
    column_mapping: Optional[Mapping[str, str]] = None,
    method_options: Optional[Mapping[str, Any]] = None,
    pricing_engine: str = "black76",
    price_method: str = "mid",
    max_staleness_days: int = 3,
    solver: str = "brent",
    method: str = "svi",
    failure_policy: Literal["raise", "skip_warn"] = "skip_warn",
) -> FittedSurface:
    """Fit a volatility surface using the specified method.

    Args:
        chain: Option chain DataFrame.
        market: Market inputs.
        column_mapping: Column mapping.
        method_options: Options for the fitting method.
        pricing_engine: Pricing engine ("black76" or "bs").
        price_method: Price method.
        max_staleness_days: Max staleness days.
        solver: Solver.
        method: Fitting method (e.g., "svi").
        failure_policy: Handling policy when an expiry slice fails. Use ``"raise"``
            for strict mode or ``"skip_warn"`` for best-effort mode.

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
            column_mapping=column_mapping,
            method_options=method_options,
            pricing_engine=pricing_engine,
            price_method=price_method,
            max_staleness_days=max_staleness_days,
            solver=solver,
            method=method,
            failure_policy=failure_policy,
        )

    # If the user passes something else, we should strictly raise.
    # We do NOT silently fallback to independent slices.
    raise ValueError(f"Unknown surface fitting method: {method}")
