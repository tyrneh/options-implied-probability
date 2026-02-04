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


def black_scholes_vega(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    q: float = 0.0,
):
    """Analytical Vega (∂Price/∂σ) for Black-Scholes. Same for Call and Put."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)


def black_scholes_delta(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    q: float = 0.0,
    call_or_put: str = "call",
) -> np.ndarray:
    """Analytical Delta (∂V/∂S) for Black-Scholes."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

    # Call Delta = e^{-qT} N(d1)
    # Put Delta = e^{-qT} (N(d1) - 1) = -e^{-qT} N(-d1)
    delta_call = np.exp(-q * t) * norm.cdf(d1)

    if call_or_put == "put":
        return delta_call - np.exp(-q * t)
    return delta_call


def black_scholes_gamma(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    q: float = 0.0,
) -> np.ndarray:
    """Analytical Gamma (∂²V/∂S²) for Black-Scholes. Same for Call and Put."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

    # Gamma = e^{-qT} * n(d1) / (S * sigma * sqrt(T))
    return np.exp(-q * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))


def black_scholes_theta(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    q: float = 0.0,
    call_or_put: str = "call",
) -> np.ndarray:
    """Analytical Theta (∂V/∂t) for Black-Scholes (per year, negative = time decay)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    # Theta_call = - (S * e^{-qT} * n(d1) * sigma) / (2 * sqrt(T))
    #              - r * K * e^{-rT} * N(d2)
    #              + q * S * e^{-qT} * N(d1)
    term1 = -(S * np.exp(-q * t) * norm.pdf(d1) * sigma) / (2 * np.sqrt(t))
    term2 = -r * K * np.exp(-r * t) * norm.cdf(d2)
    term3 = q * S * np.exp(-q * t) * norm.cdf(d1)
    theta_call = term1 + term2 + term3

    if call_or_put == "put":
        # Theta_put = Theta_call + r * K * e^{-rT} - q * S * e^{-qT}
        # Derived from Put-Call Parity: P = C - S*e^{-qT} + K*e^{-rT}
        theta_put = theta_call + r * K * np.exp(-r * t) - q * S * np.exp(-q * t)
        return theta_put

    return theta_call


def black_scholes_rho(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    q: float = 0.0,
    call_or_put: str = "call",
) -> np.ndarray:
    """Analytical Rho (∂V/∂r) for Black-Scholes."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    # Rho_call = K * t * e^{-rT} * N(d2)
    rho_call = K * t * np.exp(-r * t) * norm.cdf(d2)

    if call_or_put == "put":
        # Rho_put = -K * t * e^{-rT} * N(-d2)
        return -K * t * np.exp(-r * t) * norm.cdf(-d2)

    return rho_call
