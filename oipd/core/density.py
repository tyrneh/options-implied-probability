from __future__ import annotations

"""Helpers for transforming option prices into risk-neutral densities."""

from typing import Iterable, Literal, Tuple

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from oipd.core.errors import InvalidInputError
from oipd.core.iv_smoothing import VolCurve
from oipd.pricing import get_pricer


def finite_diff_second_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Stable five-point stencil second derivative with non-uniform fallback."""

    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length. Got x: {len(x)}, y: {len(y)}")
    if len(x) < 5:
        raise ValueError(f"Need at least 5 points for 5-point stencil. Got {len(x)}")

    h = np.diff(x)
    if not np.allclose(h, h[0], rtol=1e-6):
        import warnings

        warnings.warn(
            "Non-uniform grid detected. Using np.gradient fallback which may be less stable. "
            "Consider interpolating to a uniform grid first.",
            UserWarning,
        )
        return np.gradient(np.gradient(y, x), x)

    step = h[0]
    d2y = np.zeros_like(y)
    for i in range(2, len(y) - 2):
        d2y[i] = (-y[i - 2] + 16 * y[i - 1] - 30 * y[i] + 16 * y[i + 1] - y[i + 2]) / (
            12 * step**2
        )

    d2y[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / step**2
    d2y[1] = (y[0] - 2 * y[1] + y[2]) / step**2
    d2y[-2] = (y[-3] - 2 * y[-2] + y[-1]) / step**2
    d2y[-1] = (2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / step**2
    return d2y


def price_curve_from_iv(
    vol_curve: VolCurve,
    underlying_price: float,
    *,
    strike_grid: np.ndarray | None = None,
    days_to_expiry: int,
    risk_free_rate: float,
    pricing_engine: Literal["black76", "bs"],
    dividend_yield: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate call prices on a strike grid from a smoothed IV curve."""

    if strike_grid is None:
        if hasattr(vol_curve, "grid"):
            strike_grid = getattr(vol_curve, "grid")[0]
        else:
            raise InvalidInputError(
                "strike_grid must be provided when smoother grid is unavailable"
            )

    strikes = np.asarray(strike_grid, dtype=float)
    if strikes.ndim != 1:
        raise InvalidInputError("strike_grid must be one-dimensional")

    sigma = vol_curve(strikes)
    years = days_to_expiry / 365.0
    pricer = get_pricer(pricing_engine)
    q = dividend_yield or 0.0
    call_prices = pricer(underlying_price, strikes, sigma, years, risk_free_rate, q)
    return strikes, np.asarray(call_prices, dtype=float)


def pdf_from_price_curve(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    *,
    risk_free_rate: float,
    days_to_expiry: int,
    min_strike: float | None = None,
    max_strike: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Breeden-Litzenberger to obtain a PDF from call prices."""

    strikes_arr = np.asarray(strikes, dtype=float)
    prices_arr = np.asarray(call_prices, dtype=float)
    if strikes_arr.shape != prices_arr.shape:
        raise InvalidInputError("Strikes and prices must have the same shape")

    second_derivative = finite_diff_second_derivative(prices_arr, strikes_arr)
    years = days_to_expiry / 365.0
    pdf = np.exp(risk_free_rate * years) * second_derivative
    pdf = np.maximum(pdf, 0.0)

    if min_strike is not None or max_strike is not None:
        left = 0
        right = len(strikes_arr) - 1
        if min_strike is not None:
            while left < len(strikes_arr) and strikes_arr[left] < min_strike:
                left += 1
        if max_strike is not None:
            while right >= 0 and strikes_arr[right] > max_strike:
                right -= 1
        strikes_arr = strikes_arr[left : right + 1]
        pdf = pdf[left : right + 1]

    return strikes_arr, pdf


def calculate_cdf_from_pdf(
    x_array: np.ndarray, pdf_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the PDF numerically to recover the CDF."""

    if len(x_array) == 0:
        raise InvalidInputError("Input arrays cannot be empty")
    if len(x_array) != len(pdf_array):
        raise InvalidInputError("Price and PDF arrays must have same length")

    cdf = []
    total_area = simpson(y=pdf_array, x=x_array)
    remaining_area = 1 - total_area
    for idx in range(len(x_array)):
        if idx == 0:
            integral = remaining_area / 2
        else:
            integral = (
                simpson(y=pdf_array[idx - 1 : idx + 1], x=x_array[idx - 1 : idx + 1])
                + cdf[-1]
            )
        cdf.append(integral)
    return x_array, np.array(cdf)


def calculate_quartiles(
    cdf_point_arrays: Tuple[np.ndarray, np.ndarray],
) -> dict[float, float]:
    """Compute quartiles from a CDF curve."""

    x_array, cdf_values = cdf_point_arrays
    cdf_interpolated = interp1d(x_array, cdf_values)
    x_start, x_end = x_array[0], x_array[-1]
    return {
        0.25: brentq(lambda x: cdf_interpolated(x) - 0.25, x_start, x_end),
        0.5: brentq(lambda x: cdf_interpolated(x) - 0.5, x_start, x_end),
        0.75: brentq(lambda x: cdf_interpolated(x) - 0.75, x_start, x_end),
    }


__all__ = [
    "finite_diff_second_derivative",
    "price_curve_from_iv",
    "pdf_from_price_curve",
    "calculate_cdf_from_pdf",
    "calculate_quartiles",
]
