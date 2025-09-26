from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from typing import Tuple


def finite_diff_second_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length. Got x: {len(x)}, y: {len(y)}")
    if len(x) < 5:
        raise ValueError(f"Need at least 5 points for 5-point stencil. Got {len(x)}")
    h = np.diff(x)
    if not np.allclose(h, h[0], rtol=1e-6):
        return np.gradient(np.gradient(y, x), x)
    h = h[0]
    d2y = np.zeros_like(y)
    for i in range(2, len(y) - 2):
        d2y[i] = (-y[i - 2] + 16 * y[i - 1] - 30 * y[i] + 16 * y[i + 1] - y[i + 2]) / (
            12 * h**2
        )
    d2y[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / h**2
    d2y[1] = (y[0] - 2 * y[1] + y[2]) / h**2
    d2y[-2] = (y[-3] - 2 * y[-2] + y[-1]) / h**2
    d2y[-1] = (2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / h**2
    return d2y


def calculate_cdf_from_pdf(
    x_array: np.ndarray, pdf_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if len(x_array) == 0:
        raise ValueError("Input arrays cannot be empty")
    if len(x_array) != len(pdf_array):
        raise ValueError("Price and PDF arrays must have same length")

    cdf = []
    n = len(x_array)
    total_area = simpson(y=pdf_array[0:n], x=x_array)
    remaining_area = 1 - total_area
    for i in range(n):
        if i == 0:
            integral = 0.0 + remaining_area / 2
        else:
            integral = (
                simpson(y=pdf_array[i - 1 : i + 1], x=x_array[i - 1 : i + 1]) + cdf[-1]
            )
        cdf.append(integral)
    return x_array, np.array(cdf)


__all__ = ["finite_diff_second_derivative", "calculate_cdf_from_pdf"]
