"""Numerical helpers for surface-level probability conversion."""

from __future__ import annotations

import numpy as np

from oipd.core.errors import CalculationError
from oipd.core.probability_density_conversion.finite_diff import (
    finite_diff_first_derivative,
)


def pdf_from_local_call_second_derivative(
    call_price_up: np.ndarray | float,
    call_price_mid: np.ndarray | float,
    call_price_down: np.ndarray | float,
    step: np.ndarray | float,
    discount_growth_factor: float,
) -> np.ndarray:
    """Compute local PDF values from call-price second derivatives.

    Args:
        call_price_up: Call prices at ``K + h``.
        call_price_mid: Call prices at ``K``.
        call_price_down: Call prices at ``K - h``.
        step: Finite-difference step ``h``.
        discount_growth_factor: ``exp(r * T)`` scaling term.

    Returns:
        np.ndarray: Pointwise PDF estimates.
    """
    call_up = np.asarray(call_price_up, dtype=float)
    call_mid = np.asarray(call_price_mid, dtype=float)
    call_down = np.asarray(call_price_down, dtype=float)
    finite_diff_step = np.asarray(step, dtype=float)

    second_derivative = (call_up - 2.0 * call_mid + call_down) / (finite_diff_step**2)
    return np.asarray(discount_growth_factor * second_derivative, dtype=float)


def cdf_from_local_call_first_derivative(
    call_price_up: np.ndarray | float,
    call_price_down: np.ndarray | float,
    step: np.ndarray | float,
    discount_growth_factor: float,
) -> np.ndarray:
    """Compute local CDF values from call-price first derivatives.

    Args:
        call_price_up: Call prices at ``K + h``.
        call_price_down: Call prices at ``K - h``.
        step: Finite-difference step ``h``.
        discount_growth_factor: ``exp(r * T)`` scaling term.

    Returns:
        np.ndarray: Pointwise CDF estimates.
    """
    call_up = np.asarray(call_price_up, dtype=float)
    call_down = np.asarray(call_price_down, dtype=float)
    finite_diff_step = np.asarray(step, dtype=float)

    first_derivative = (call_up - call_down) / (2.0 * finite_diff_step)
    return np.asarray(1.0 + discount_growth_factor * first_derivative, dtype=float)


def normalized_cdf_from_call_curve(
    call_prices: np.ndarray,
    strike_grid: np.ndarray,
    log_moneyness_grid: np.ndarray,
    *,
    effective_rate: float,
    time_to_expiry_years: float,
) -> np.ndarray:
    """Compute a clipped, monotone, normalized CDF on a strike grid.

    Args:
        call_prices: Call prices evaluated on ``strike_grid``.
        strike_grid: Strike grid in price space.
        log_moneyness_grid: Matching log-moneyness grid.
        effective_rate: Effective continuous risk-free rate.
        time_to_expiry_years: Maturity in years.

    Returns:
        np.ndarray: Monotone CDF values in ``[0, 1]``.
    """
    dcall_dk = finite_diff_first_derivative(call_prices, log_moneyness_grid)
    dcall_dk = np.asarray(dcall_dk, dtype=float)
    dcall_dstrike = dcall_dk / strike_grid

    cdf_values = 1.0 + np.exp(effective_rate * time_to_expiry_years) * dcall_dstrike
    cdf_values = np.clip(cdf_values, 0.0, 1.0)
    cdf_values = np.maximum.accumulate(cdf_values)

    cdf_span = float(cdf_values[-1] - cdf_values[0])
    if cdf_span > 1e-10:
        cdf_values = (cdf_values - cdf_values[0]) / cdf_span
    else:
        cdf_values = np.linspace(0.0, 1.0, cdf_values.size)

    return np.maximum.accumulate(np.clip(cdf_values, 0.0, 1.0))


def pdf_and_cdf_from_normalized_cdf(
    cdf_values: np.ndarray,
    strike_grid: np.ndarray,
    log_moneyness_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive normalized PDF and rebuilt CDF from an input normalized CDF.

    Args:
        cdf_values: Monotone CDF values on ``strike_grid``.
        strike_grid: Strike grid in price space.
        log_moneyness_grid: Matching log-moneyness grid.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(pdf_values, rebuilt_cdf_values)``.

    Raises:
        CalculationError: If the PDF has non-positive integral.
    """
    dcdf_dk = finite_diff_first_derivative(cdf_values, log_moneyness_grid)
    pdf_values = np.maximum(np.asarray(dcdf_dk, dtype=float) / strike_grid, 0.0)

    area = float(np.trapz(pdf_values, strike_grid))
    if area <= 0.0:
        raise CalculationError("Failed to derive a positive probability density.")
    pdf_values = pdf_values / area

    increments = 0.5 * (pdf_values[1:] + pdf_values[:-1]) * (
        strike_grid[1:] - strike_grid[:-1]
    )
    rebuilt_cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf_total = float(rebuilt_cdf[-1])
    if cdf_total > 0.0:
        rebuilt_cdf = rebuilt_cdf / cdf_total
    rebuilt_cdf = np.maximum.accumulate(np.clip(rebuilt_cdf, 0.0, 1.0))

    return pdf_values, rebuilt_cdf


__all__ = [
    "cdf_from_local_call_first_derivative",
    "normalized_cdf_from_call_curve",
    "pdf_and_cdf_from_normalized_cdf",
    "pdf_from_local_call_second_derivative",
]
