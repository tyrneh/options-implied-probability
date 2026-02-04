"""Stateless helpers for risk-neutral probability estimation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.core.utils import calculate_days_to_expiry, convert_days_to_years
from oipd.market_inputs import ResolvedMarket
from oipd.pipelines.vol_curve import fit_vol_curve_internal
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
    domain: Optional[Tuple[float, float]] = None,
    points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Derive PDF/CDF from a pre-fitted volatility curve.

    Args:
        vol_curve: Fitted volatility curve object (callable).
        resolved_market: Fully resolved market inputs.
        pricing_engine: Pricing engine (``"black76"`` or ``"bs"``).
        vol_metadata: Optional metadata from the vol fit (for diagnostics).

    Returns:
        Tuple of ``(prices, pdf, cdf, metadata)``. When ``domain`` is not
        provided and observed strikes are available in the metadata, the
        distribution is evaluated on the observed strike grid to preserve
        market-aligned points.
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

    # 2a. Determine Days to Expiry
    expiry_date = vol_meta.get("expiry_date")
    if expiry_date is None:
        raise CalculationError(
            "Volatility metadata missing 'expiry_date'. Cannot derive distribution."
        )

    days_to_expiry = calculate_days_to_expiry(expiry_date, valuation_date)

    # Determine strike grid - interpolated slices store default_domain in metadata
    strike_grid = None
    target_domain = domain or vol_meta.get("default_domain")

    observed_strikes: np.ndarray | None = None
    observed_iv = vol_meta.get("observed_iv")
    if observed_iv is not None:
        try:
            if not observed_iv.empty and "strike" in observed_iv.columns:
                observed_strikes = observed_iv["strike"].to_numpy(dtype=float)
        except Exception:
            observed_strikes = None

    if target_domain:
        strike_grid = np.linspace(target_domain[0], target_domain[1], points)
    elif observed_strikes is not None and observed_strikes.size > 0:
        strike_grid = np.unique(observed_strikes)
        strike_grid.sort()
    else:
        # Fallback: Create a reasonable grid based on ATM vol and T
        # Assume roughly log-normal distribution width ~ sigma * sqrt(T)
        T = convert_days_to_years(days_to_expiry)
        atm_vol = vol_meta.get("at_money_vol")
        if atm_vol is None:
            raise CalculationError(
                "Cannot determine default grid: 'at_money_vol' missing in metadata."
            )

        # Center around forward
        F = pricing_underlying

        # 5 standard deviations covers >99.99% of mass
        sigma_root_t = atm_vol * np.sqrt(T)
        width = 5.0 * sigma_root_t

        # Grid range in log-moneyness then back to price
        low_K = F * np.exp(-width - 0.5 * sigma_root_t**2)
        high_K = F * np.exp(width - 0.5 * sigma_root_t**2)

        # Ensure positive
        low_K = max(low_K, 0.01)

        strike_grid = np.linspace(low_K, high_K, points)

    # 2. Generate Price Curve from Vol
    pricing_strike_grid, pricing_call_prices = price_curve_from_iv(
        vol_curve,
        pricing_underlying,
        strike_grid=strike_grid,
        days_to_expiry=days_to_expiry,
        risk_free_rate=resolved_market.risk_free_rate,
        pricing_engine=pricing_engine,
        dividend_yield=effective_dividend,
    )

    # 3. Determine Observation Bounds
    # We use the full grid range to allow the distribution to reflect the extrapolated tail
    observed_min_strike = float(pricing_strike_grid.min())
    observed_max_strike = float(pricing_strike_grid.max())

    # 4. Derive PDF
    pdf_prices, pdf_values = pdf_from_price_curve(
        pricing_strike_grid,
        pricing_call_prices,
        risk_free_rate=resolved_market.risk_free_rate,
        days_to_expiry=days_to_expiry,
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
