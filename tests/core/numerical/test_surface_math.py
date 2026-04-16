"""Tests for extracted probability surface numerical helpers."""

from __future__ import annotations

import numpy as np

from oipd.core.probability_density_conversion import (
    cdf_from_local_call_first_derivative,
    pdf_from_local_call_second_derivative,
)


def test_pdf_from_local_call_second_derivative_matches_formula() -> None:
    """Pointwise local PDF helper should match the second-derivative formula."""
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


def test_cdf_from_local_call_first_derivative_matches_formula() -> None:
    """Pointwise local CDF helper should match the first-derivative formula."""
    step = np.array([0.01, 0.02, 0.03])
    call_up = np.array([7.0, 6.0, 5.0])
    call_down = np.array([7.2, 6.1, 5.2])
    factor = 1.05

    expected = 1.0 + factor * ((call_up - call_down) / (2.0 * step))
    actual = cdf_from_local_call_first_derivative(call_up, call_down, step, factor)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
