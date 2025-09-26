from __future__ import annotations

from typing import Tuple

import numpy as np
from pandas import DataFrame
from scipy import interpolate
from scipy.optimize import minimize

from oipd.core.pdf import CalculationError, InvalidInputError


def fit_bspline_smile(options_data: DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a B-spline to IV observations and return (K_grid, iv_grid)."""
    if options_data is None or options_data.empty:
        raise InvalidInputError("options_data cannot be empty")

    x = options_data["strike"]
    y = options_data["iv"]

    if len(x) < 4:
        raise CalculationError(
            f"Insufficient data for B-spline fitting: need at least 4 points, got {len(x)}"
        )

    try:
        tck = interpolate.splrep(x, y, s=10, k=3)
    except Exception as e:
        raise CalculationError(
            f"Failed to fit B-spline to implied volatility data: {str(e)}"
        )

    dx = 0.1
    domain = int((max(x) - min(x)) / dx)
    K_grid = np.linspace(min(x), max(x), domain)
    iv_grid = interpolate.BSpline(*tck)(K_grid)

    return K_grid, iv_grid


def fit_svi_smile(
    options_data: DataFrame,
    effective_underlying: float,
    days_to_expiry: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit SVI total variance smile; return (K_grid, iv_grid)."""
    if effective_underlying <= 0:
        raise CalculationError(
            "Effective underlying must be positive for SVI calibration"
        )

    x_strikes = options_data["strike"].to_numpy(dtype=float)
    iv_obs = options_data["iv"].to_numpy(dtype=float)

    mask = np.isfinite(iv_obs) & (iv_obs > 0)
    x_strikes = x_strikes[mask]
    iv_obs = iv_obs[mask]
    if len(iv_obs) < 5:
        raise CalculationError(
            f"Insufficient data for SVI fitting: need at least 5 IV points, got {len(iv_obs)}"
        )

    T = max(float(days_to_expiry) / 365.0, 1e-8)
    X = float(effective_underlying)
    k_obs = np.log(x_strikes / X)
    w_obs = (iv_obs**2) * T

    a0 = max(1e-6, 0.5 * np.nanmin(w_obs))
    b0 = max(1e-6, (np.nanmax(w_obs) - np.nanmin(w_obs)) / 2)
    rho0 = 0.0
    m0 = 0.0
    sigma0 = 0.1
    x0 = np.array([a0, b0, rho0, m0, sigma0], dtype=float)

    bounds = [
        (1e-8, 4.0),
        (1e-8, 10.0),
        (-0.999, 0.999),
        (-3.0, 3.0),
        (1e-6, 5.0),
    ]

    def svi_w(params, k):
        a, b, rho, m, sigma = params
        km = k - m
        return a + b * (rho * km + np.sqrt(km * km + sigma * sigma))

    def objective(params):
        w_model = svi_w(params, k_obs)
        penalty = np.where(w_model <= 0, 1e6 * (1 - np.maximum(w_model, 0)), 0.0)
        iv_model = np.sqrt(np.maximum(w_model, 1e-12) / T)
        resid = iv_model - iv_obs
        return float(np.sum(resid * resid) + np.sum(penalty))

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    if (not res.success) or (not np.isfinite(res.fun)):
        raise CalculationError(
            f"SVI optimizer failed: {res.message if hasattr(res,'message') else 'unknown error'}"
        )

    theta = res.x
    k_min = float(np.min(k_obs))
    k_max = float(np.max(k_obs))
    k_grid = np.linspace(k_min, k_max, max(200, int((k_max - k_min) / 0.01) + 1))
    w_grid = np.maximum(svi_w(theta, k_grid), 1e-12)
    iv_grid = np.sqrt(w_grid / T)
    K_grid = X * np.exp(k_grid)
    return K_grid, iv_grid


__all__ = ["fit_bspline_smile", "fit_svi_smile"]
