"""Stateless helpers for risk-neutral probability estimation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.pipelines.market_inputs import ResolvedMarket
from oipd.pipelines.vol_estimation import fit_vol_curve_internal
from oipd.core.probability_density_conversion import (
    price_curve_from_iv,
    pdf_from_price_curve,
    calculate_cdf_from_pdf,
)
from oipd.pricing.utils import prepare_dividends


def derive_distribution_internal(
    options_data: pd.DataFrame,
    resolved_market: ResolvedMarket,
    *,
    solver: str = "brent",
    pricing_engine: str = "black76",
    price_method: str = "mid",
    max_staleness_days: int = 3,
    method: str = "svi",
    method_options: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute PDF/CDF from option quotes using the stateless pipeline.

    Args:
        options_data: Raw option quotes as a DataFrame.
        resolved_market: Fully resolved market inputs.
        solver: Implied-vol solver (``"brent"`` or ``"newton"``).
        pricing_engine: Pricing engine (``"black76"`` or ``"bs"``).
        price_method: Price selection strategy (``"mid"`` or ``"last"``).
        max_staleness_days: Maximum allowed quote age; stale strikes are dropped.
        method: Volatility fitting method (``"svi"`` or ``"bspline"``).
        method_options: Method-specific overrides (e.g., ``{"random_seed": 42}``).

    Returns:
        Tuple of ``(prices, pdf, cdf, metadata)`` where prices/pdf/cdf are
        numpy arrays, and metadata includes the fitted volatility curve and diagnostics.

    Raises:
        CalculationError: If the pipeline cannot produce a valid fit.
    """
    
    # 1. Fit Volatility Curve
    # This handles data cleaning, parity, staleness, and fitting
    vol_curve, vol_meta = fit_vol_curve_internal(
        options_data,
        resolved_market,
        pricing_engine=pricing_engine,
        price_method=price_method,
        max_staleness_days=max_staleness_days,
        solver=solver,
        method=method,
        method_options=method_options,
    )
    
    # 2. Prepare Pricing Inputs
    # We need to replicate the dividend logic for the pricing step
    valuation_date = resolved_market.valuation_date
    
    if pricing_engine == "bs":
        effective_spot, effective_dividend = prepare_dividends(
            underlying=resolved_market.underlying_price,
            dividend_schedule=resolved_market.dividend_schedule,
            dividend_yield=resolved_market.dividend_yield,
            r=resolved_market.risk_free_rate,
            valuation_date=valuation_date,
        )
        pricing_underlying = effective_spot
    else:
        # For Black76, we use the forward price inferred during vol fitting
        # or fallback to spot if not available (though fit_vol_curve_internal should ensure it)
        pricing_underlying = vol_meta.get("forward_price", resolved_market.underlying_price)
        effective_dividend = None

    # 3. Generate Price Curve from Vol
    # This creates a dense grid of call prices
    pricing_strike_grid, pricing_call_prices = price_curve_from_iv(
        vol_curve,
        pricing_underlying,
        days_to_expiry=resolved_market.days_to_expiry,
        risk_free_rate=resolved_market.risk_free_rate,
        pricing_engine=pricing_engine,
        dividend_yield=effective_dividend,
    )

    # 4. Determine Observation Bounds
    # We clip the PDF generation to the observed strike range to avoid extrapolation artifacts
    observed_iv = vol_meta.get("observed_iv")
    if observed_iv is not None and not observed_iv.empty:
        observed_min_strike = float(observed_iv["strike"].min())
        observed_max_strike = float(observed_iv["strike"].max())
    else:
        # Fallback if no observed IVs (unlikely if fit succeeded)
        observed_min_strike = float(pricing_strike_grid.min())
        observed_max_strike = float(pricing_strike_grid.max())

    # 5. Derive PDF (Breeden-Litzenberger)
    pdf_prices, pdf_values = pdf_from_price_curve(
        pricing_strike_grid,
        pricing_call_prices,
        risk_free_rate=resolved_market.risk_free_rate,
        days_to_expiry=resolved_market.days_to_expiry,
        min_strike=observed_min_strike,
        max_strike=observed_max_strike,
    )

    # 6. Derive CDF
    try:
        _, cdf_values = calculate_cdf_from_pdf(pdf_prices, pdf_values)
    except Exception as exc:
        raise CalculationError(f"Failed to compute CDF: {exc}") from exc

    # 7. Assemble Metadata
    # We merge the vol metadata with any additional info
    metadata = vol_meta.copy()
    # Add back model params for completeness if needed by consumers, 
    # though strictly they are inputs.
    
    return pdf_prices, pdf_values, cdf_values, metadata
