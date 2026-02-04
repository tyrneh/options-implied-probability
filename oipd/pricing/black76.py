from __future__ import annotations

"""Black-76 forward-based European call option pricer."""

import numpy as np
from scipy.stats import norm
from typing import Union

ArrayLike = Union[float, np.ndarray]


def black76_call_price(
    F: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    q: float = 0.0,
):
    """Risk-neutral price of a European call in forward space.

    Parameters
    ----------
    F : array-like
        Forward price of the underlying asset.
    K : array-like
        Strike price.
    sigma : array-like
        Volatility.
    t : array-like
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    q : float, optional
        Present for API compatibility; ignored.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    df = np.exp(-r * t)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return df * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_delta(
    F: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    call_or_put: str = "call",
) -> np.ndarray:
    """Analytical Delta (∂V/∂F) for Black-76."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    df = np.exp(-r * t)

    # Call Delta = e^{-rT} N(d1)
    # Put Delta = e^{-rT} (N(d1) - 1)
    delta_call = df * norm.cdf(d1)

    if call_or_put == "put":
        return delta_call - df
    return delta_call


def black76_gamma(
    F: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
) -> np.ndarray:
    """Analytical Gamma (∂²V/∂F²) for Black-76. Same for Call and Put."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    df = np.exp(-r * t)

    # Gamma = e^{-rT} * pdf(d1) / (F * sigma * sqrt(T))
    return df * norm.pdf(d1) / (F * sigma * np.sqrt(t))


def black76_vega(
    F: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
) -> np.ndarray:
    """Analytical Vega (∂V/∂σ) for Black-76. Same for Call and Put."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    df = np.exp(-r * t)

    # Vega = F * e^{-rT} * pdf(d1) * sqrt(T)
    return F * df * norm.pdf(d1) * np.sqrt(t)


def black76_theta(
    F: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    call_or_put: str = "call",
) -> np.ndarray:
    """Analytical Theta (∂V/∂t) for Black-76.

    Represents time decay; usually negative for long options.
    Note: Standard Black-76 Theta definition often assumes F is constant.
    This implementation calculates ∂V/∂t where t is time-to-expiry, so
    it effectively returns -∂V/∂T.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    df = np.exp(-r * t)

    call_price = df * (F * norm.cdf(d1) - K * norm.cdf(d1 - sigma * np.sqrt(t)))

    # Theta_call = - (F * e^{-rT} * n(d1) * sigma) / (2 * sqrt(T)) - r * C
    theta_call = -(F * df * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) - r * call_price

    if call_or_put == "put":
        # Theta_put = Theta_call + r * e^{-rT} * (F - K)
        # Derived from Put-Call Parity: P = C - e^{-rT}(F - K)
        return theta_call + r * df * (F - K)

    return theta_call


def black76_rho(
    F: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    t: ArrayLike,
    r: float,
    call_or_put: str = "call",
) -> np.ndarray:
    """Analytical Rho (∂V/∂r) for Black-76.

    Sensitivity to the risk-free rate. For Black-76 (futures/forwards),
    rates only affect the discount factor e^{-rT}.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    eps = 1e-12
    sigma = np.where(sigma < eps, eps, sigma)
    t = np.where(t < eps, eps, t)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    df = np.exp(-r * t)

    # Call Rho = -T * CallPrice
    # Since V(F, t) = e^{-rT} * Black(F, K, ...), and r only appears in term 1.

    call_price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    rho_call = -t * call_price

    if call_or_put == "put":
        put_price = call_price - df * (F - K)
        return -t * put_price

    return rho_call
