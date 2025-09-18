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
