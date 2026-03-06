"""Tests for CRRA-based physical probability conversion."""

from __future__ import annotations

import numpy as np

from oipd.core.probability_density_conversion import physical_from_rn_crra


def _build_rn_pdf(center: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth synthetic risk-neutral PDF on a positive grid.

    Args:
        center: Center of the synthetic density.

    Returns:
        tuple[np.ndarray, np.ndarray]: Price grid and normalized RN PDF.
    """
    prices = np.linspace(40.0, 180.0, 401)
    sigma = 0.20
    rn_pdf = np.exp(-0.5 * ((np.log(prices / center) / sigma) ** 2))
    rn_pdf /= np.trapz(rn_pdf, prices)
    return prices, rn_pdf


def test_risk_aversion_zero_returns_risk_neutral_density() -> None:
    """Gamma=0 should leave the RN density unchanged."""
    prices, rn_pdf = _build_rn_pdf()
    physical_pdf, physical_cdf = physical_from_rn_crra(prices, rn_pdf, 0.0)

    np.testing.assert_allclose(physical_pdf, rn_pdf, atol=1e-10)
    np.testing.assert_allclose(physical_cdf[-1], 1.0, atol=1e-10)


def test_physical_pdf_integrates_to_one() -> None:
    """Converted physical PDF remains normalized."""
    prices, rn_pdf = _build_rn_pdf()
    physical_pdf, _ = physical_from_rn_crra(prices, rn_pdf, 3.0)

    assert np.isclose(np.trapz(physical_pdf, prices), 1.0, atol=1e-8)


def test_physical_cdf_is_monotone_and_ends_at_one() -> None:
    """Converted physical CDF is usable for quantile and probability queries."""
    prices, rn_pdf = _build_rn_pdf()
    _, physical_cdf = physical_from_rn_crra(prices, rn_pdf, 3.0)

    assert np.all(np.diff(physical_cdf) >= -1e-12)
    assert np.isclose(float(physical_cdf[-1]), 1.0, atol=1e-10)


def test_higher_risk_aversion_shifts_mass_to_higher_prices() -> None:
    """Larger CRRA coefficients should increase the physical mean price."""
    prices, rn_pdf = _build_rn_pdf()
    pdf_low, _ = physical_from_rn_crra(prices, rn_pdf, 1.0)
    pdf_high, _ = physical_from_rn_crra(prices, rn_pdf, 5.0)

    mean_low = float(np.trapz(prices * pdf_low, prices))
    mean_high = float(np.trapz(prices * pdf_high, prices))
    assert mean_high > mean_low
