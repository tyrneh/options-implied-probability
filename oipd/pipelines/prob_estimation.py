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

    # 2. Derive Distribution from Fitted Curve
    # We delegate the rest of the process to the dedicated pipeline function
    return derive_distribution_from_curve(
        vol_curve,
        resolved_market,
        pricing_engine=pricing_engine,
        vol_metadata=vol_meta,
    )


def derive_distribution_from_curve(
    vol_curve: Any,
    resolved_market: ResolvedMarket,
    *,
    pricing_engine: str = "black76",
    vol_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Derive PDF/CDF from a pre-fitted volatility curve.

    Args:
        vol_curve: Fitted volatility curve object (callable).
        resolved_market: Fully resolved market inputs.
        pricing_engine: Pricing engine (``"black76"`` or ``"bs"``).
        vol_metadata: Optional metadata from the vol fit (for diagnostics).

    Returns:
        Tuple of ``(prices, pdf, cdf, metadata)``.
    """
    vol_meta = vol_metadata or {}
    valuation_date = resolved_market.valuation_date

    # 1. Prepare Pricing Inputs
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
        # For Black76, use forward price from vol fit or fallback to spot
        pricing_underlying = vol_meta.get(
            "forward_price", resolved_market.underlying_price
        )
        effective_dividend = None

    # 2. Generate Price Curve from Vol
    pricing_strike_grid, pricing_call_prices = price_curve_from_iv(
        vol_curve,
        pricing_underlying,
        days_to_expiry=resolved_market.days_to_expiry,
        risk_free_rate=resolved_market.risk_free_rate,
        pricing_engine=pricing_engine,
        dividend_yield=effective_dividend,
    )

    # 3. Determine Observation Bounds
    observed_iv = vol_meta.get("observed_iv")
    if observed_iv is not None and not observed_iv.empty:
        observed_min_strike = float(observed_iv["strike"].min())
        observed_max_strike = float(observed_iv["strike"].max())
    else:
        observed_min_strike = float(pricing_strike_grid.min())
        observed_max_strike = float(pricing_strike_grid.max())

    # 4. Derive PDF
    pdf_prices, pdf_values = pdf_from_price_curve(
        pricing_strike_grid,
        pricing_call_prices,
        risk_free_rate=resolved_market.risk_free_rate,
        days_to_expiry=resolved_market.days_to_expiry,
        min_strike=observed_min_strike,
        max_strike=observed_max_strike,
    )

    # 5. Derive CDF
    try:
        _, cdf_values = calculate_cdf_from_pdf(pdf_prices, pdf_values)
    except Exception as exc:
        raise CalculationError(f"Failed to compute CDF: {exc}") from exc

    # 6. Assemble Metadata
    metadata = vol_meta.copy()

    return pdf_prices, pdf_values, cdf_values, metadata
