from __future__ import annotations

"""Black-Scholes European call option pricer (with flat dividend yield).*"""

import numpy as np
from scipy.stats import norm
from typing import Union

ArrayLike = Union[float, np.ndarray]


def black_scholes_call_price(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    q: float = 0.0,
):
    """Vectorised risk-neutral price of a European **call**.

    All inputs can be scalars or numpy arrays of identical shape (NumPy
    broadcasting rules apply).  *q* is the continuous dividend yield.  When the
    caller already subtracted the PV of discrete dividends from *S*, they can
    leave *q* = 0.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    # Prevent divide-by-zero when sigma or t are zero in broadcast positions
    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    call = S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    return call


def black_scholes_call_vega(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    q: float = 0.0,
):
    """Analytical Vega (∂Price/∂σ) for Black-Scholes call."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
