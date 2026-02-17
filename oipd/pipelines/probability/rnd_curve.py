"""Stateless helpers for risk-neutral probability estimation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.core.utils import (
    calculate_days_to_expiry,
    convert_days_to_years,
    resolve_risk_free_rate,
)
from oipd.market_inputs import ResolvedMarket
from oipd.pipelines.vol_curve import fit_vol_curve_internal
from oipd.core.probability_density_conversion import (
    price_curve_from_iv,
    pdf_from_price_curve,
    calculate_cdf_from_pdf,
)
from oipd.pricing.utils import prepare_dividends


def _build_strike_grid(
    resolved_market: ResolvedMarket,
    vol_meta: Mapping[str, Any],
    *,
    pricing_underlying: float,
    days_to_expiry: Optional[int] = None,
    time_to_expiry_years: Optional[float] = None,
    domain: Optional[Tuple[float, float]] = None,
    points: int = 200,
) -> np.ndarray:
    """Build a uniform strike grid for probability estimation.

    Args:
        resolved_market: Fully resolved market inputs.
        vol_meta: Metadata from the volatility calibration.
        pricing_underlying: Forward/spot used for pricing calls.
        days_to_expiry: Optional days-to-expiry input for backwards compatibility.
        time_to_expiry_years: Optional explicit time to expiry in years.
        domain: Optional strike domain as (min, max).
        points: Number of grid points to generate.

    Returns:
        np.ndarray: Uniformly spaced strike grid.

    Raises:
        CalculationError: If a valid strike domain cannot be inferred.
    """
    target_domain = domain or vol_meta.get("default_domain")
    if target_domain is not None:
        min_strike, max_strike = target_domain
        min_strike = max(0.01, float(min_strike))
        max_strike = float(max_strike)
        if min_strike >= max_strike:
            raise CalculationError(
                "Strike domain must satisfy min_strike < max_strike."
            )
        return np.linspace(min_strike, max_strike, points)

    observed_iv = vol_meta.get("observed_iv")
    if observed_iv is not None:
        try:
            if not observed_iv.empty and "strike" in observed_iv.columns:
                min_strike = float(observed_iv["strike"].min())
                max_strike = float(observed_iv["strike"].max())
                if np.isclose(min_strike, max_strike):
                    padding = max(0.01, 0.05 * max(abs(min_strike), 1.0))
                    lower_bound = max(0.01, min_strike - padding)
                    upper_bound = max_strike + padding
                else:
                    padding = 0.05 * (max_strike - min_strike)
                    lower_bound = min_strike - padding
                    if lower_bound <= 0:
                        lower_bound = min_strike * 0.5
                    lower_bound = max(0.01, lower_bound)
                    upper_bound = max_strike + padding

                if lower_bound >= upper_bound:
                    upper_bound = lower_bound + max(
                        0.01, 0.05 * max(abs(max_strike), 1.0)
                    )
                return np.linspace(lower_bound, upper_bound, points)
        except Exception:
            pass

    if time_to_expiry_years is not None:
        T = float(time_to_expiry_years)
    elif days_to_expiry is not None:
        T = float(convert_days_to_years(days_to_expiry))
    else:
        raise CalculationError(
            "Either days_to_expiry or time_to_expiry_years must be provided."
        )

    if T <= 0:
        raise CalculationError(
            "Time to expiry must be positive to build a strike grid."
        )

    atm_vol = vol_meta.get("at_money_vol")
    if atm_vol is None:
        raise CalculationError(
            "Cannot determine default grid: 'at_money_vol' missing in metadata."
        )

    sigma_root_t = float(atm_vol) * np.sqrt(T)
    width = 5.0 * sigma_root_t
    forward = float(pricing_underlying)

    low_strike = forward * np.exp(-width - 0.5 * sigma_root_t**2)
    high_strike = forward * np.exp(width - 0.5 * sigma_root_t**2)
    low_strike = max(low_strike, 0.01)
    if low_strike >= high_strike:
        high_strike = low_strike + max(0.01, 0.05 * max(abs(forward), 1.0))

    return np.linspace(low_strike, high_strike, points)


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
    time_to_expiry_years: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Derive PDF/CDF from a pre-fitted volatility curve.

    Args:
        vol_curve: Fitted volatility curve object (callable).
        resolved_market: Fully resolved market inputs.
        pricing_engine: Pricing engine (``"black76"`` or ``"bs"``).
        vol_metadata: Optional metadata from the vol fit (for diagnostics).
        domain: Optional strike domain as ``(min_strike, max_strike)``.
        points: Number of strike grid points.
        time_to_expiry_years: Optional explicit maturity in years. When
            provided, this takes precedence over metadata-derived expiry.

    Returns:
        Tuple of ``(prices, pdf, cdf, metadata)``. When ``domain`` is not
        provided, a uniform strike grid is constructed from the best available
        domain estimate (metadata, observed strikes, or ATM-based fallback).
    """
    vol_meta = vol_metadata or {}
    valuation_date = resolved_market.valuation_date

    # 1. Determine maturity and resolve rate convention
    if time_to_expiry_years is not None:
        years_to_expiry = float(time_to_expiry_years)
        if years_to_expiry <= 0:
            raise CalculationError(
                "time_to_expiry_years must be positive to derive distribution."
            )
    else:
        expiry_date = vol_meta.get("expiry_date")
        if expiry_date is None:
            raise CalculationError(
                "Volatility metadata missing 'expiry_date'. Cannot derive distribution."
            )
        days_to_expiry = calculate_days_to_expiry(expiry_date, valuation_date)
        years_to_expiry = convert_days_to_years(days_to_expiry)

    rate_mode = resolved_market.source_meta["risk_free_rate_mode"]
    effective_r = resolve_risk_free_rate(
        resolved_market.risk_free_rate, rate_mode, years_to_expiry
    )

    # 2. Prepare Pricing Inputs
    if pricing_engine == "bs":
        effective_spot, effective_dividend = prepare_dividends(
            underlying=resolved_market.underlying_price,
            dividend_schedule=resolved_market.dividend_schedule,
            dividend_yield=resolved_market.dividend_yield,
            r=effective_r,
            valuation_date=valuation_date,
        )
        pricing_underlying = effective_spot
    else:
        # For Black76, use forward price from vol fit or fallback to spot
        pricing_underlying = vol_meta.get(
            "forward_price", resolved_market.underlying_price
        )
        effective_dividend = None

    strike_grid = _build_strike_grid(
        resolved_market,
        vol_meta,
        pricing_underlying=pricing_underlying,
        time_to_expiry_years=years_to_expiry,
        domain=domain,
        points=points,
    )

    # 2. Generate Price Curve from Vol
    pricing_strike_grid, pricing_call_prices = price_curve_from_iv(
        vol_curve,
        pricing_underlying,
        strike_grid=strike_grid,
        time_to_expiry_years=years_to_expiry,
        risk_free_rate=effective_r,
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
        risk_free_rate=effective_r,
        time_to_expiry_years=years_to_expiry,
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
    metadata["time_to_expiry_years"] = years_to_expiry

    return pdf_prices, pdf_values, cdf_values, metadata
