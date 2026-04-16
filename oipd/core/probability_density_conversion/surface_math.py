"""Numerical helpers for surface-level probability conversion."""

from __future__ import annotations

import numpy as np


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


__all__ = [
    "cdf_from_local_call_first_derivative",
    "pdf_from_local_call_second_derivative",
]
