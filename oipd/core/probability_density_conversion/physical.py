"""Physical-probability conversion utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq

from oipd.core.errors import CalculationError, InvalidInputError


def _cdf_from_pdf(price_grid: np.ndarray, pdf_values: np.ndarray) -> np.ndarray:
    """Build a monotone CDF from PDF values on a strike grid.

    This helper is intentionally separate from ``calculate_cdf_from_pdf`` in
    ``rnd.py``. Here, the input PDF is already normalized on the finite grid and
    we want a strict left-to-right cumulative integral with monotonic cleanup.

    Args:
        price_grid: Strictly increasing evaluation grid.
        pdf_values: Non-negative density values aligned with ``price_grid``.

    Returns:
        np.ndarray: Monotone cumulative distribution in ``[0, 1]``.

    Raises:
        CalculationError: If the PDF integrates to a non-positive mass.
    """
    trapezoids = 0.5 * (pdf_values[1:] + pdf_values[:-1]) * np.diff(price_grid)
    cdf_values = np.concatenate(([0.0], np.cumsum(trapezoids)))
    total_mass = float(cdf_values[-1])
    if total_mass <= 0.0:
        raise CalculationError("Failed to derive CDF: PDF has non-positive mass.")
    cdf_values = cdf_values / total_mass
    cdf_values = np.maximum.accumulate(cdf_values)
    return np.clip(cdf_values, 0.0, 1.0)


def _validate_inputs(
    prices: np.ndarray,
    rn_pdf: np.ndarray,
    forward_price: float,
    time_to_expiry_years: float,
    erp: float,
) -> None:
    """Validate inputs used for risk-neutral to physical conversion.

    Args:
        prices: Price grid used for integration.
        rn_pdf: Risk-neutral PDF values on ``prices``.
        forward_price: Forward price used as the tilt reference point.
        time_to_expiry_years: Time to expiry in years.
        erp: Annualized equity risk premium in decimal form.

    Returns:
        None.

    Raises:
        InvalidInputError: If arrays are malformed or scalar inputs are invalid.
    """
    if prices.ndim != 1 or rn_pdf.ndim != 1:
        raise InvalidInputError("prices and rn_pdf must be 1D arrays.")
    if prices.shape != rn_pdf.shape:
        raise InvalidInputError("prices and rn_pdf must have matching shapes.")
    if prices.size < 5:
        raise InvalidInputError(
            "At least 5 grid points are required for physical conversion."
        )
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(rn_pdf)):
        raise InvalidInputError("prices and rn_pdf must contain only finite values.")
    if np.any(prices <= 0.0):
        raise InvalidInputError("All prices must be strictly positive.")
    if np.any(np.diff(prices) <= 0.0):
        raise InvalidInputError("prices must be strictly increasing.")
    if not np.isfinite(forward_price) or forward_price <= 0.0:
        raise InvalidInputError("forward_price must be a positive finite number.")
    if not np.isfinite(time_to_expiry_years) or time_to_expiry_years <= 0.0:
        raise InvalidInputError(
            "time_to_expiry_years must be a positive finite number."
        )
    if not np.isfinite(erp):
        raise InvalidInputError("erp must be finite.")


def _tilted_pdf(
    normalized_rn_pdf: np.ndarray,
    prices: np.ndarray,
    log_price_ratio: np.ndarray,
    tilt_parameter: float,
) -> np.ndarray:
    """Apply an exponential tilt to the normalized risk-neutral density.

    Exponential tilt means we reweight each state by
    ``exp(lambda * log(S / F)) = (S / F)^lambda``:
    positive ``lambda`` shifts probability mass toward higher terminal prices,
    negative ``lambda`` shifts mass toward lower prices.

    Args:
        normalized_rn_pdf: Risk-neutral PDF normalized to integrate to one.
        prices: Price grid for integration.
        log_price_ratio: ``log(prices / forward_price)`` values.
        tilt_parameter: Exponential tilt parameter.

    Returns:
        np.ndarray: Tilted and normalized physical PDF.

    Raises:
        CalculationError: If tilted density cannot be normalized.
    """
    exponent = tilt_parameter * log_price_ratio
    exponent = exponent - float(np.max(exponent))
    unnormalized_pdf = normalized_rn_pdf * np.exp(exponent)
    partition_function = float(np.trapz(unnormalized_pdf, prices))
    if partition_function <= 0.0 or not np.isfinite(partition_function):
        raise CalculationError("Failed to normalize tilted physical density.")
    return unnormalized_pdf / partition_function


def physical_from_rn_exponential_tilt(
    prices: np.ndarray,
    rn_pdf: np.ndarray,
    forward_price: float,
    time_to_expiry_years: float,
    erp: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert a risk-neutral density to a physical density via exponential tilt.

    The conversion uses:
    ``p_lambda(S) ∝ q(S) * exp(lambda * log(S / F))``
    where ``lambda`` is calibrated so that:
    ``E_p[S_T] = F * exp(erp * T)``.

    Args:
        prices: Strictly increasing price grid.
        rn_pdf: Risk-neutral PDF values aligned with ``prices``.
        forward_price: Forward level used as the tilt anchor.
        time_to_expiry_years: Time-to-expiry in years.
        erp: Annualized equity risk premium in decimal form (e.g. ``0.0423``).

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, Any]]: Physical PDF, physical CDF,
            and diagnostics including calibrated tilt parameter and moment match.

    Raises:
        InvalidInputError: If inputs are malformed.
        CalculationError: If moment matching fails or conversion is not feasible.
    """
    prices_arr = np.asarray(prices, dtype=float)
    rn_pdf_arr = np.asarray(rn_pdf, dtype=float)
    _validate_inputs(prices_arr, rn_pdf_arr, forward_price, time_to_expiry_years, erp)

    rn_mass = float(np.trapz(rn_pdf_arr, prices_arr))
    if rn_mass <= 0.0 or not np.isfinite(rn_mass):
        raise CalculationError("Risk-neutral PDF must have positive finite mass.")
    normalized_rn_pdf = rn_pdf_arr / rn_mass

    target_mean = float(forward_price * np.exp(erp * time_to_expiry_years))
    min_price = float(np.min(prices_arr))
    max_price = float(np.max(prices_arr))
    if target_mean <= min_price or target_mean >= max_price:
        raise CalculationError(
            "Target physical mean lies outside the available price grid. "
            "Widen the strike grid or reduce ERP assumptions."
        )

    log_price_ratio = np.log(prices_arr / float(forward_price))

    def moment_gap(tilt_parameter: float) -> float:
        tilted_pdf = _tilted_pdf(
            normalized_rn_pdf, prices_arr, log_price_ratio, tilt_parameter
        )
        implied_mean = float(np.trapz(prices_arr * tilted_pdf, prices_arr))
        return implied_mean - target_mean

    lower, upper = -1.0, 1.0
    gap_lower = moment_gap(lower)
    gap_upper = moment_gap(upper)

    bracket_found = gap_lower == 0.0 or gap_upper == 0.0 or (gap_lower * gap_upper < 0)
    expansion_steps = 0
    while not bracket_found and expansion_steps < 20:
        lower *= 2.0
        upper *= 2.0
        gap_lower = moment_gap(lower)
        gap_upper = moment_gap(upper)
        bracket_found = (
            gap_lower == 0.0 or gap_upper == 0.0 or (gap_lower * gap_upper < 0)
        )
        expansion_steps += 1

    if not bracket_found:
        raise CalculationError(
            "Unable to bracket a valid tilt parameter for physical conversion. "
            "Check ERP assumptions or grid coverage."
        )

    if gap_lower == 0.0:
        tilt_parameter = lower
    elif gap_upper == 0.0:
        tilt_parameter = upper
    else:
        tilt_parameter = float(brentq(moment_gap, lower, upper, maxiter=200))

    physical_pdf = _tilted_pdf(
        normalized_rn_pdf,
        prices_arr,
        log_price_ratio,
        tilt_parameter,
    )
    physical_cdf = _cdf_from_pdf(prices_arr, physical_pdf)
    realized_mean = float(np.trapz(prices_arr * physical_pdf, prices_arr))
    rn_mean = float(np.trapz(prices_arr * normalized_rn_pdf, prices_arr))

    diagnostics: dict[str, Any] = {
        "transform": "exponential_tilt",
        "tilt_parameter": float(tilt_parameter),
        "target_mean": target_mean,
        "realized_mean": realized_mean,
        "mean_error": realized_mean - target_mean,
        "rn_mean": rn_mean,
        "erp": float(erp),
        "time_to_expiry_years": float(time_to_expiry_years),
    }
    return physical_pdf, physical_cdf, diagnostics


__all__ = ["physical_from_rn_exponential_tilt"]
