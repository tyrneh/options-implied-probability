"""Tests for extracted probability surface numerical helpers."""

from __future__ import annotations

import numpy as np

from oipd.core.probability_density_conversion import (
    cdf_from_local_call_first_derivative,
    normalized_cdf_from_call_curve,
    pdf_and_cdf_from_normalized_cdf,
    pdf_from_local_call_second_derivative,
)
from oipd.core.probability_density_conversion.finite_diff import (
    finite_diff_first_derivative,
)
from oipd.pricing import black76_call_price


def _sample_surface_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build a smooth synthetic surface slice for helper tests.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
            ``(k_grid, strike_grid, call_prices, effective_rate, t_years)``.
    """
    k_grid = np.linspace(-0.4, 0.4, 11)
    forward_price = 100.0
    strike_grid = forward_price * np.exp(k_grid)
    t_years = 0.5
    effective_rate = 0.03
    implied_vols = 0.2 + 0.08 * (k_grid**2)
    call_prices = black76_call_price(
        forward_price,
        strike_grid,
        implied_vols,
        t_years,
        effective_rate,
    )
    return k_grid, strike_grid, np.asarray(call_prices, dtype=float), effective_rate, t_years


def test_pdf_from_local_call_second_derivative_matches_legacy_formula() -> None:
    """Matches the inlined local second-derivative PDF formula."""
    prices = np.array([90.0, 100.0, 110.0])
    step = np.maximum(1e-4 * prices, 1e-4)
    call_up = np.array([11.2, 6.4, 3.2])
    call_mid = np.array([11.0, 6.2, 3.1])
    call_down = np.array([10.8, 6.0, 3.0])
    factor = 1.01

    expected = factor * ((call_up - 2.0 * call_mid + call_down) / (step**2))
    actual = pdf_from_local_call_second_derivative(
        call_up,
        call_mid,
        call_down,
        step,
        factor,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_cdf_from_local_call_first_derivative_matches_legacy_formula() -> None:
    """Matches the inlined local first-derivative CDF formula."""
    step = np.array([0.01, 0.02, 0.03])
    call_up = np.array([7.0, 6.0, 5.0])
    call_down = np.array([7.2, 6.1, 5.2])
    factor = 1.05

    expected = 1.0 + factor * ((call_up - call_down) / (2.0 * step))
    actual = cdf_from_local_call_first_derivative(call_up, call_down, step, factor)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_normalized_cdf_from_call_curve_matches_legacy_block() -> None:
    """Matches the previous inlined CDF normalization block for surface slices."""
    k_grid, strike_grid, call_prices, effective_rate, t_years = _sample_surface_arrays()

    actual = normalized_cdf_from_call_curve(
        call_prices,
        strike_grid,
        k_grid,
        effective_rate=effective_rate,
        time_to_expiry_years=t_years,
    )

    dcall_dk = finite_diff_first_derivative(call_prices, k_grid)
    dcall_dstrike = np.asarray(dcall_dk, dtype=float) / strike_grid
    expected = 1.0 + np.exp(effective_rate * t_years) * dcall_dstrike
    expected = np.clip(expected, 0.0, 1.0)
    expected = np.maximum.accumulate(expected)
    cdf_span = float(expected[-1] - expected[0])
    if cdf_span > 1e-10:
        expected = (expected - expected[0]) / cdf_span
    else:
        expected = np.linspace(0.0, 1.0, expected.size)
    expected = np.maximum.accumulate(np.clip(expected, 0.0, 1.0))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_pdf_and_cdf_from_normalized_cdf_matches_legacy_block() -> None:
    """Matches previous inlined PDF/CDF rebuild logic for surface slices."""
    k_grid, strike_grid, call_prices, effective_rate, t_years = _sample_surface_arrays()
    cdf_values = normalized_cdf_from_call_curve(
        call_prices,
        strike_grid,
        k_grid,
        effective_rate=effective_rate,
        time_to_expiry_years=t_years,
    )

    actual_pdf, actual_cdf = pdf_and_cdf_from_normalized_cdf(
        cdf_values,
        strike_grid,
        k_grid,
    )

    dcdf_dk = finite_diff_first_derivative(cdf_values, k_grid)
    expected_pdf = np.maximum(np.asarray(dcdf_dk, dtype=float) / strike_grid, 0.0)
    expected_pdf = expected_pdf / float(np.trapz(expected_pdf, strike_grid))
    increments = 0.5 * (expected_pdf[1:] + expected_pdf[:-1]) * (
        strike_grid[1:] - strike_grid[:-1]
    )
    expected_cdf = np.concatenate(([0.0], np.cumsum(increments)))
    expected_cdf = expected_cdf / float(expected_cdf[-1])
    expected_cdf = np.maximum.accumulate(np.clip(expected_cdf, 0.0, 1.0))

    np.testing.assert_allclose(actual_pdf, expected_pdf, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_cdf, expected_cdf, rtol=0.0, atol=0.0)
