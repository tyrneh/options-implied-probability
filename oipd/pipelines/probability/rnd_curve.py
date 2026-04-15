"""Stateless helpers for risk-neutral probability estimation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.core.maturity import build_maturity_metadata, resolve_maturity
from oipd.core.utils import (
    resolve_risk_free_rate,
)
from oipd.market_inputs import ResolvedMarket
from oipd.pipelines.vol_curve import fit_vol_curve_internal
from oipd.pipelines.utils.surface_export import validate_export_domain
from oipd.core.probability_density_conversion import (
    price_curve_from_iv,
    pdf_from_price_curve,
    calculate_cdf_from_pdf,
)
from oipd.core.probability_density_conversion.finite_diff import (
    finite_diff_first_derivative,
)
from oipd.pricing.utils import prepare_dividends


DEFAULT_PDF_MIN_STRIKE = 0.01
DEFAULT_PDF_POINTS = 200
DEFAULT_PDF_TAIL_MASS_TOLERANCE = 1e-4
DEFAULT_PDF_INITIAL_UPPER_MULTIPLIER = 3.0
DEFAULT_PDF_MAX_EXPANSIONS = 6
DEFAULT_PDF_EXPANSION_FACTOR = 2.0


def _build_strike_grid(
    resolved_market: ResolvedMarket,
    vol_meta: Mapping[str, Any],
    *,
    pricing_underlying: float,
    time_to_expiry_years: Optional[float] = None,
    domain: Optional[Tuple[float, float]] = None,
    points: int = 200,
) -> np.ndarray:
    """Build a uniform strike grid for probability estimation.

    Args:
        resolved_market: Fully resolved market inputs.
        vol_meta: Metadata from the volatility calibration.
        pricing_underlying: Forward/spot used for pricing calls.
        time_to_expiry_years: Explicit time to expiry in years.
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

    if time_to_expiry_years is None:
        raise CalculationError(
            "time_to_expiry_years is required to build a default strike grid."
        )

    T = float(time_to_expiry_years)

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


def _extract_strike_domain(
    strike_frame: Any,
) -> tuple[float, float] | None:
    """Return the finite strike domain from a frame-like payload when available.

    Args:
        strike_frame: Object that may contain a ``strike`` column.

    Returns:
        tuple[float, float] | None: Minimum and maximum strike when available.
    """
    if not isinstance(strike_frame, pd.DataFrame):
        return None
    if strike_frame.empty or "strike" not in strike_frame.columns:
        return None

    strike_values = strike_frame["strike"].to_numpy(dtype=float)
    strike_values = strike_values[np.isfinite(strike_values)]
    if strike_values.size == 0:
        return None

    return float(np.min(strike_values)), float(np.max(strike_values))


def _initial_pdf_upper_bound(
    resolved_market: ResolvedMarket,
    vol_metadata: Mapping[str, Any],
    *,
    pricing_underlying: float,
    years_to_expiry: float,
    points: int,
    initial_upper_multiplier: float,
) -> tuple[float, float | None]:
    """Resolve the initial right boundary for PDF-domain iteration.

    Args:
        resolved_market: Fully resolved market inputs.
        vol_metadata: Metadata captured during the volatility fit.
        pricing_underlying: Spot or forward used for pricing.
        years_to_expiry: Time to expiry in years.
        points: Baseline grid resolution.
        initial_upper_multiplier: Initial multiple of the observed max strike.

    Returns:
        tuple[float, float | None]: Initial upper bound and optional fallback
        lower bound when only the ATM-vol heuristic is available.
    """
    observed_domain = _extract_strike_domain(vol_metadata.get("observed_iv"))
    raw_observed_domain = vol_metadata.get("raw_observed_domain")
    default_domain = vol_metadata.get("default_domain")

    if raw_observed_domain is not None:
        return float(raw_observed_domain[1]) * float(initial_upper_multiplier), None
    if observed_domain is not None:
        return float(observed_domain[1]) * float(initial_upper_multiplier), None
    if default_domain is not None:
        return float(default_domain[1]) * float(initial_upper_multiplier), None

    fallback_grid = _build_strike_grid(
        resolved_market,
        vol_metadata,
        pricing_underlying=pricing_underlying,
        time_to_expiry_years=years_to_expiry,
        domain=None,
        points=points,
    )
    return float(fallback_grid[-1]), float(fallback_grid[0])


def _grid_spacing_from_initial_domain(
    domain: tuple[float, float],
    points: int,
) -> float:
    """Return the target grid spacing preserved across domain expansions.

    Args:
        domain: Initial domain bounds.
        points: Baseline number of grid points.

    Returns:
        float: Target spacing for future provisional grids.
    """
    initial_width = float(domain[1] - domain[0])
    base_step = initial_width / max(points - 1, 1)
    if base_step <= 0:
        raise CalculationError("PDF domain resolution requires a positive grid width.")
    return base_step


def _build_grid_with_spacing(
    *,
    pricing_underlying: float,
    domain: tuple[float, float],
    base_step: float,
) -> np.ndarray:
    """Build a uniform provisional grid while preserving target spacing.

    Args:
        pricing_underlying: Spot or forward used for pricing.
        domain: Provisional price domain.
        base_step: Target grid spacing preserved across expansions.

    Returns:
        np.ndarray: Uniform strike grid.
    """
    width = float(domain[1] - domain[0])
    grid_points = max(5, int(np.ceil(width / base_step)) + 1)
    return _build_strike_grid(
        resolved_market=None,  # type: ignore[arg-type]
        vol_meta={},
        pricing_underlying=pricing_underlying,
        domain=domain,
        points=grid_points,
    )


def _evaluate_distribution_on_grid(
    vol_curve: Any,
    *,
    pricing_underlying: float,
    pricing_engine: str,
    effective_r: float,
    effective_dividend: float | None,
    years_to_expiry: float,
    strike_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Price, differentiate, and measure covered mass on one provisional grid.

    Args:
        vol_curve: Fitted volatility curve callable.
        pricing_underlying: Spot or forward used for pricing.
        pricing_engine: Pricing engine identifier.
        effective_r: Continuously compounded rate used for pricing.
        effective_dividend: Dividend yield for Black-Scholes pricing.
        years_to_expiry: Time to expiry in years.
        strike_grid: Provisional strike grid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float, float]: Price grid, PDF,
        CDF, covered mass estimate from the PDF integral, and right-tail
        survival probability inferred from the call-price slope.
    """
    pricing_grid, call_prices = price_curve_from_iv(
        vol_curve,
        pricing_underlying,
        strike_grid=strike_grid,
        time_to_expiry_years=years_to_expiry,
        risk_free_rate=effective_r,
        pricing_engine=pricing_engine,
        dividend_yield=effective_dividend,
    )
    pdf_prices, pdf_values = pdf_from_price_curve(
        pricing_grid,
        call_prices,
        risk_free_rate=effective_r,
        time_to_expiry_years=years_to_expiry,
        min_strike=float(pricing_grid.min()),
        max_strike=float(pricing_grid.max()),
    )
    covered_mass = float(np.trapezoid(pdf_values, pdf_prices))
    _, cdf_values = calculate_cdf_from_pdf(pdf_prices, pdf_values)
    call_slope = finite_diff_first_derivative(call_prices, pricing_grid)
    right_tail_survival = float(
        np.clip(
            np.exp(effective_r * years_to_expiry) * max(-call_slope[-1], 0.0),
            0.0,
            1.0,
        )
    )
    return pdf_prices, pdf_values, cdf_values, covered_mass, right_tail_survival


def resolve_pdf_domain(
    vol_curve: Any,
    resolved_market: ResolvedMarket,
    *,
    pricing_engine: str,
    vol_metadata: Mapping[str, Any],
    pricing_underlying: float,
    effective_r: float,
    effective_dividend: float | None,
    years_to_expiry: float,
    points: int = DEFAULT_PDF_POINTS,
    min_strike: float = DEFAULT_PDF_MIN_STRIKE,
    initial_upper_multiplier: float = DEFAULT_PDF_INITIAL_UPPER_MULTIPLIER,
    tail_mass_tolerance: float = DEFAULT_PDF_TAIL_MASS_TOLERANCE,
    max_expansions: int = DEFAULT_PDF_MAX_EXPANSIONS,
    expansion_factor: float = DEFAULT_PDF_EXPANSION_FACTOR,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Resolve the native full-domain support for a single-expiry PDF.

    Args:
        vol_curve: Fitted volatility curve callable.
        resolved_market: Fully resolved market inputs.
        pricing_engine: Pricing engine identifier.
        vol_metadata: Metadata captured during the volatility fit.
        pricing_underlying: Spot or forward used for pricing.
        effective_r: Continuously compounded rate used for pricing.
        effective_dividend: Dividend yield for Black-Scholes pricing.
        years_to_expiry: Time to expiry in years.
        points: Baseline grid resolution for the initial domain.
        min_strike: Numerical lower bound used instead of literal zero.
        initial_upper_multiplier: Initial multiple of observed max strike.
        tail_mass_tolerance: Maximum tolerated uncovered right-tail mass.
        max_expansions: Maximum widening attempts.
        expansion_factor: Factor applied to the upper bound when widening.

    Returns:
        tuple[tuple[float, float], dict[str, Any]]: Resolved domain and
        diagnostics describing how it was chosen.

    Raises:
        CalculationError: If a valid positive domain cannot be resolved.
    """
    observed_domain = _extract_strike_domain(vol_metadata.get("observed_iv"))
    raw_observed_domain = vol_metadata.get("raw_observed_domain")
    if raw_observed_domain is not None:
        raw_observed_domain = (
            float(raw_observed_domain[0]),
            float(raw_observed_domain[1]),
        )
    post_iv_survival_domain = vol_metadata.get("post_iv_survival_domain")
    if post_iv_survival_domain is not None:
        post_iv_survival_domain = (
            float(post_iv_survival_domain[0]),
            float(post_iv_survival_domain[1]),
        )
    default_domain = vol_metadata.get("default_domain")
    minimum_required_upper = max(
        float(domain_upper)
        for domain_upper in (
            raw_observed_domain[1] if raw_observed_domain is not None else None,
            observed_domain[1] if observed_domain is not None else None,
        )
        if domain_upper is not None
    ) if any(
        domain_candidate is not None
        for domain_candidate in (
            raw_observed_domain,
            observed_domain,
        )
    ) else None
    initial_upper, default_lower = _initial_pdf_upper_bound(
        resolved_market,
        vol_metadata,
        pricing_underlying=pricing_underlying,
        years_to_expiry=years_to_expiry,
        points=points,
        initial_upper_multiplier=initial_upper_multiplier,
    )

    lower_bound = max(float(min_strike), 1e-8)
    if default_lower is not None:
        lower_bound = max(lower_bound, float(default_lower))
    if minimum_required_upper is not None:
        initial_upper = max(initial_upper, float(minimum_required_upper))

    if not np.isfinite(initial_upper) or initial_upper <= lower_bound:
        raise CalculationError("Could not resolve a valid initial PDF domain.")

    resolved_domain = (lower_bound, initial_upper)
    base_step = _grid_spacing_from_initial_domain(resolved_domain, points)
    previous_upper: float | None = None
    hit_max_expansions = False
    expansion_count = 0
    added_mass_last_expansion = float("inf")
    tail_mass_beyond_upper = float("inf")

    while True:
        strike_grid = _build_grid_with_spacing(
            pricing_underlying=pricing_underlying,
            domain=resolved_domain,
            base_step=base_step,
        )
        (
            pdf_prices,
            pdf_values,
            _cdf_values,
            _covered_mass,
            tail_mass_beyond_upper,
        ) = _evaluate_distribution_on_grid(
            vol_curve,
            pricing_underlying=pricing_underlying,
            pricing_engine=pricing_engine,
            effective_r=effective_r,
            effective_dividend=effective_dividend,
            years_to_expiry=years_to_expiry,
            strike_grid=strike_grid,
        )
        if previous_upper is not None:
            added_mask = pdf_prices > previous_upper
            if np.count_nonzero(added_mask) >= 2:
                added_mass_last_expansion = float(
                    np.trapezoid(pdf_values[added_mask], pdf_prices[added_mask])
                )
            else:
                added_mass_last_expansion = 0.0
        meets_required_upper = (
            minimum_required_upper is None
            or resolved_domain[1] >= float(minimum_required_upper)
        )
        if previous_upper is not None:
            if (
                tail_mass_beyond_upper <= tail_mass_tolerance
                and added_mass_last_expansion <= tail_mass_tolerance
                and meets_required_upper
            ):
                break
        elif tail_mass_beyond_upper <= tail_mass_tolerance and meets_required_upper:
            added_mass_last_expansion = tail_mass_beyond_upper
            break

        if expansion_count >= max_expansions:
            hit_max_expansions = True
            break

        previous_upper = resolved_domain[1]
        resolved_domain = (
            resolved_domain[0],
            resolved_domain[1] * float(expansion_factor),
        )
        expansion_count += 1

    diagnostics = {
        "resolved_domain": resolved_domain,
        "raw_observed_domain": raw_observed_domain,
        "post_iv_survival_domain": post_iv_survival_domain or observed_domain,
        "observed_domain": observed_domain,
        "tail_mass_beyond_upper": tail_mass_beyond_upper,
        "added_mass_last_expansion": added_mass_last_expansion,
        "domain_expansions": expansion_count,
        "domain_hit_max_expansions": hit_max_expansions,
        "domain_grid_spacing": base_step,
        "domain_source": "resolved_pdf_domain",
        "domain_pricing_engine": pricing_engine,
    }
    return resolved_domain, diagnostics


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
    valuation_timestamp = resolved_market.valuation_timestamp
    resolved_maturity = None

    # 1. Determine maturity and resolve rate convention
    if time_to_expiry_years is not None:
        years_to_expiry = float(time_to_expiry_years)
        if years_to_expiry <= 0:
            raise CalculationError(
                "time_to_expiry_years must be positive to derive distribution."
            )
    else:
        expiry = vol_meta.get("expiry")
        if expiry is None:
            raise CalculationError(
                "Volatility metadata missing canonical 'expiry'. Cannot derive "
                "distribution."
            )
        resolved_maturity = resolve_maturity(
            expiry,
            valuation_timestamp,
            floor_at_zero=False,
        )
        years_to_expiry = float(resolved_maturity.time_to_expiry_years)
        if years_to_expiry <= 0:
            raise CalculationError(
                "Expiry must be strictly after valuation_date to derive distribution."
            )

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
            valuation_date=resolved_market.valuation_timestamp,
            expiry=resolved_maturity.expiry,
        )
        pricing_underlying = effective_spot
    else:
        # For Black76, use forward price from vol fit or fallback to spot
        pricing_underlying = vol_meta.get(
            "forward_price", resolved_market.underlying_price
        )
        effective_dividend = None

    if domain is None:
        resolved_domain, domain_metadata = resolve_pdf_domain(
            vol_curve,
            resolved_market,
            pricing_engine=pricing_engine,
            vol_metadata=vol_meta,
            pricing_underlying=float(pricing_underlying),
            effective_r=effective_r,
            effective_dividend=effective_dividend,
            years_to_expiry=years_to_expiry,
            points=points,
        )
    else:
        validated_domain = validate_export_domain(domain)
        if validated_domain is None:
            raise CalculationError("Explicit PDF domain could not be validated.")
        resolved_domain = validated_domain
        domain_metadata = {
            "resolved_domain": resolved_domain,
            "raw_observed_domain": vol_meta.get("raw_observed_domain"),
            "post_iv_survival_domain": vol_meta.get("post_iv_survival_domain"),
            "observed_domain": _extract_strike_domain(vol_meta.get("observed_iv")),
            "tail_mass_beyond_upper": None,
            "added_mass_last_expansion": None,
            "domain_expansions": 0,
            "domain_hit_max_expansions": False,
            "domain_grid_spacing": None,
            "domain_source": "explicit",
            "domain_pricing_engine": pricing_engine,
        }

    if domain is None and domain_metadata.get("domain_grid_spacing") is not None:
        strike_grid = _build_grid_with_spacing(
            pricing_underlying=pricing_underlying,
            domain=resolved_domain,
            base_step=float(domain_metadata["domain_grid_spacing"]),
        )
    else:
        strike_grid = _build_strike_grid(
            resolved_market,
            vol_meta,
            pricing_underlying=pricing_underlying,
            time_to_expiry_years=years_to_expiry,
            domain=resolved_domain,
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
    if resolved_maturity is not None:
        metadata.update(build_maturity_metadata(resolved_maturity))
    else:
        metadata["time_to_expiry_years"] = years_to_expiry
        metadata["time_to_expiry_days"] = years_to_expiry * 365.0
    metadata.update(domain_metadata)

    return pdf_prices, pdf_values, cdf_values, metadata


def build_density_results_frame(
    prices: np.ndarray,
    pdf_values: np.ndarray,
    cdf_values: np.ndarray,
    *,
    domain: Optional[Tuple[float, float]] = None,
    points: int = 200,
) -> pd.DataFrame:
    """Build a density export DataFrame from aligned probability arrays.

    Args:
        prices: Price grid for the fitted probability slice.
        pdf_values: PDF values aligned with ``prices``.
        cdf_values: CDF values aligned with ``prices``.
        domain: Optional explicit export/view domain as ``(min_price, max_price)``.
            When provided, this function resamples the already-computed native
            distribution arrays onto the requested range. It does not decide the
            canonical upstream PDF support.
        points: Number of points when resampling to ``domain``. Ignored when
            ``domain`` is omitted and the native grid is returned unchanged.

    Returns:
        DataFrame with columns ``price``, ``pdf``, and ``cdf``.

    Raises:
        ValueError: If the export domain or resampling grid is invalid.
    """
    validated_domain = validate_export_domain(domain)

    prices_array = np.asarray(prices, dtype=float)
    pdf_array = np.asarray(pdf_values, dtype=float)
    cdf_array = np.asarray(cdf_values, dtype=float)

    if validated_domain is None:
        return pd.DataFrame(
            {
                "price": prices_array,
                "pdf": pdf_array,
                "cdf": cdf_array,
            }
        )

    if isinstance(points, bool) or not isinstance(points, int) or points <= 0:
        raise ValueError("points must be a strictly positive integer.")
    grid_points = points

    grid_prices = np.linspace(validated_domain[0], validated_domain[1], grid_points)
    cdf_monotone = np.maximum.accumulate(cdf_array)

    return pd.DataFrame(
        {
            "price": grid_prices,
            "pdf": np.interp(grid_prices, prices_array, pdf_array, left=0.0, right=0.0),
            "cdf": np.interp(
                grid_prices,
                prices_array,
                cdf_monotone,
                left=0.0,
                right=1.0,
            ),
        }
    )
