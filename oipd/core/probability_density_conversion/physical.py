"""Physical-probability conversion utilities."""

from __future__ import annotations

import numpy as np


def _cdf_from_pdf(price_grid: np.ndarray, pdf_values: np.ndarray) -> np.ndarray:
    """Build a cumulative distribution from PDF values on a price grid.

    Args:
        price_grid: Evaluation grid used for integration.
        pdf_values: Density values aligned with ``price_grid``.

    Returns:
        np.ndarray: Cumulative distribution values on ``price_grid``.
    """
    trapezoids = 0.5 * (pdf_values[1:] + pdf_values[:-1]) * np.diff(price_grid)
    cdf_values = np.concatenate(([0.0], np.cumsum(trapezoids)))
    return cdf_values / cdf_values[-1]


def physical_from_rn_crra(
    prices: np.ndarray,
    rn_pdf: np.ndarray,
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a risk-neutral density to a physical density under CRRA utility.

    This implements the power-utility transform implied by the pricing-kernel
    relation discussed in Bliss and Panigirtzoglou, *Option-Implied Risk
    Aversion Estimates*:

    ``q(S_T) ∝ p(S_T) U'(S_T)``

    With CRRA utility, ``U'(S_T) = S_T^{-gamma}``, so the physical density is
    recovered by reweighting the risk-neutral density as:

    ``p_gamma(S_T) ∝ q(S_T) * S_T**gamma``

    Args:
        prices: Positive price grid used for integration.
        rn_pdf: Risk-neutral PDF values aligned with ``prices``.
        risk_aversion: CRRA coefficient ``gamma``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Physical PDF and physical CDF.
    """
    prices_arr = np.asarray(prices, dtype=float)
    rn_pdf_arr = np.asarray(rn_pdf, dtype=float)

    if np.isclose(float(risk_aversion), 0.0):
        return rn_pdf_arr, _cdf_from_pdf(prices_arr, rn_pdf_arr)

    normalized_rn_pdf = rn_pdf_arr / np.trapz(rn_pdf_arr, prices_arr)
    physical_pdf = normalized_rn_pdf * np.power(prices_arr, float(risk_aversion))
    physical_pdf = physical_pdf / np.trapz(physical_pdf, prices_arr)
    physical_cdf = _cdf_from_pdf(prices_arr, physical_pdf)
    return physical_pdf, physical_cdf


__all__ = ["physical_from_rn_crra"]
