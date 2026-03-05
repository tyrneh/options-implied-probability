"""Tests for risk-neutral to physical probability conversion."""

from __future__ import annotations

import numpy as np
import pytest

from oipd.core.errors import InvalidInputError
from oipd.core.probability_density_conversion import (
    physical_from_rn_exponential_tilt,
)


def _build_rn_pdf(forward: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth synthetic risk-neutral PDF on a positive grid.

    Args:
        forward: Forward level for the synthetic density center.

    Returns:
        tuple[np.ndarray, np.ndarray]: Price grid and normalized RN PDF.
    """
    prices = np.linspace(40.0, 180.0, 401)
    sigma = 0.20
    rn_pdf = np.exp(-0.5 * ((np.log(prices / forward) / sigma) ** 2))
    rn_pdf /= np.trapz(rn_pdf, prices)
    return prices, rn_pdf


def test_physical_conversion_preserves_pdf_properties() -> None:
    """Physical PDF is non-negative, finite, and normalized."""
    prices, rn_pdf = _build_rn_pdf()
    physical_pdf, physical_cdf, _ = physical_from_rn_exponential_tilt(
        prices,
        rn_pdf,
        forward_price=100.0,
        time_to_expiry_years=0.5,
        erp=0.0423,
    )

    assert np.all(np.isfinite(physical_pdf))
    assert np.all(physical_pdf >= 0.0)
    assert np.isclose(np.trapz(physical_pdf, prices), 1.0, atol=1e-4)

    assert np.all(np.isfinite(physical_cdf))
    assert np.all(np.diff(physical_cdf) >= -1e-10)
    assert 0.0 <= float(np.min(physical_cdf)) <= 1.0
    assert 0.0 <= float(np.max(physical_cdf)) <= 1.0
    assert np.isclose(float(physical_cdf[-1]), 1.0, atol=1e-4)


def test_physical_conversion_matches_target_mean() -> None:
    """Calibrated physical density matches ERP-implied target mean."""
    prices, rn_pdf = _build_rn_pdf()
    erp = 0.05
    maturity = 0.75
    target_mean = 100.0 * np.exp(erp * maturity)

    physical_pdf, _, diagnostics = physical_from_rn_exponential_tilt(
        prices,
        rn_pdf,
        forward_price=100.0,
        time_to_expiry_years=maturity,
        erp=erp,
    )
    realized_mean = float(np.trapz(prices * physical_pdf, prices))
    assert np.isclose(realized_mean, target_mean, atol=1e-4)
    assert np.isclose(diagnostics["realized_mean"], target_mean, atol=1e-4)


def test_physical_conversion_with_zero_erp_targets_forward_mean() -> None:
    """Zero ERP enforces a physical mean equal to the forward."""
    prices, rn_pdf = _build_rn_pdf()
    target_mean = 100.0
    rn_mean = float(np.trapz(prices * rn_pdf, prices))
    physical_pdf, _, _ = physical_from_rn_exponential_tilt(
        prices,
        rn_pdf,
        forward_price=100.0,
        time_to_expiry_years=1.0,
        erp=0.0,
    )
    physical_mean = float(np.trapz(prices * physical_pdf, prices))
    assert abs(physical_mean - target_mean) <= abs(rn_mean - target_mean) + 1e-8


def test_physical_conversion_rejects_invalid_inputs() -> None:
    """Invalid scalar and array inputs are rejected."""
    prices, rn_pdf = _build_rn_pdf()
    with pytest.raises(InvalidInputError):
        physical_from_rn_exponential_tilt(
            prices,
            rn_pdf[:-1],
            forward_price=100.0,
            time_to_expiry_years=0.5,
            erp=0.0423,
        )

    with pytest.raises(InvalidInputError):
        physical_from_rn_exponential_tilt(
            prices,
            rn_pdf,
            forward_price=-1.0,
            time_to_expiry_years=0.5,
            erp=0.0423,
        )

    with pytest.raises(InvalidInputError):
        physical_from_rn_exponential_tilt(
            prices,
            rn_pdf,
            forward_price=100.0,
            time_to_expiry_years=0.0,
            erp=0.0423,
        )
