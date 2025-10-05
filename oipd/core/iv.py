from __future__ import annotations

"""Implied volatility solvers and smoothing dispatch helpers."""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from oipd.core.errors import CalculationError
from oipd.core.surface_fitting import VolCurve, SurfaceConfig, fit_surface
from oipd.pricing.black76 import black76_call_price as _b76_price
from oipd.pricing.black_scholes import (
    black_scholes_call_price as _bs_price,
    black_scholes_call_vega as _bs_vega,
)


def compute_iv(
    options_data: pd.DataFrame,
    underlying_price: float,
    *,
    days_to_expiry: int,
    risk_free_rate: float,
    solver_method: Literal["newton", "brent"],
    pricing_engine: Literal["black76", "bs"],
    dividend_yield: Optional[float] = None,
) -> pd.DataFrame:
    """Vectorised implied volatility solver.

    Returns a copy of ``options_data`` with an added ``iv`` column and rows where
    IV could not be solved (NaN) removed.
    """

    if underlying_price is None:
        raise ValueError(
            "Effective underlying/forward price is required for IV extraction"
        )

    years_to_expiry = days_to_expiry / 365.0
    prices_arr = options_data["price"].to_numpy(dtype=float)
    strikes_arr = options_data["strike"].to_numpy(dtype=float)

    if pricing_engine == "black76":
        iv_values = np.fromiter(
            (
                black76_iv_brent_method(
                    p, underlying_price, k, years_to_expiry, risk_free_rate
                )
                for p, k in zip(prices_arr, strikes_arr)
            ),
            dtype=float,
        )
    else:
        if solver_method == "newton":
            solver = bs_iv_newton_method
        elif solver_method == "brent":
            solver = bs_iv_brent_method
        else:  # pragma: no cover - guarded by caller
            raise ValueError(
                "Invalid solver_method. Choose either 'newton' or 'brent'."
            )

        q = dividend_yield or 0.0
        iv_values = np.fromiter(
            (
                solver(
                    p,
                    underlying_price,
                    k,
                    years_to_expiry,
                    risk_free_rate,
                    q=q,
                )
                for p, k in zip(prices_arr, strikes_arr)
            ),
            dtype=float,
        )

    result = options_data.copy()
    result["iv"] = iv_values
    result = result.dropna(subset=["iv"])

    if result.empty:
        raise CalculationError("Failed to calculate implied volatility for any options")

    return result


def smooth_iv(
    method: str,
    strikes: np.ndarray,
    iv: np.ndarray,
    **kwargs,
) -> VolCurve:
    """Delegate to the configured IV smoothing method."""

    config = SurfaceConfig(name=method, params=kwargs or None)
    return fit_surface(config, strikes=strikes, iv=iv)


def bs_iv_brent_method(
    price: float,
    S: float,
    K: float,
    t: float,
    r: float,
    *,
    q: float = 0.0,
) -> float:
    """Compute the BS implied volatility using Brent's method."""

    if t <= 0:
        return np.nan

    try:
        return brentq(lambda iv: _bs_price(S, K, iv, t, r, q) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan


def black76_iv_brent_method(
    price: float, F: float, K: float, t: float, r: float
) -> float:
    """Compute the Black-76 implied volatility using Brent's method."""

    if t <= 0:
        return np.nan

    df = np.exp(-r * t)
    lower = df * max(F - K, 0.0)
    upper = df * F
    if price < lower or price > upper:
        return np.nan

    try:
        return brentq(lambda iv: _b76_price(F, K, iv, t, r, 0.0) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan


def bs_iv_newton_method(
    price: float,
    S: float,
    K: float,
    t: float,
    r: float,
    *,
    q: float = 0.0,
    precision: float = 1e-4,
    initial_guess: Optional[float] = None,
    max_iter: int = 1000,
    verbose: bool = False,
) -> float:
    """Compute BS implied volatility using Newton-Raphson."""

    if t <= 0:
        return np.nan

    if initial_guess is None:
        initial_guess = 0.2 if abs(S - K) < 0.1 * S else 0.5

    iv = float(initial_guess)

    for _ in range(max_iter):
        price_model = _bs_price(S, K, iv, t, r, q)
        diff = price - price_model
        if abs(diff) < precision:
            return iv

        vega = _bs_vega(S, K, iv, t, r, q)
        if abs(vega) < 1e-6:
            return np.nan

        iv += diff / vega
        if iv < 1e-6 or iv > 5.0:
            return np.nan

    if verbose:  # pragma: no cover - debugging aid
        print("Newton solver did not converge within max_iter")

    return np.nan


__all__ = [
    "compute_iv",
    "smooth_iv",
    "bs_iv_brent_method",
    "bs_iv_newton_method",
    "black76_iv_brent_method",
]
