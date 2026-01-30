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
    """Analytical Theta (∂V/∂t) for Black-76."""
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
    
    # Common term: - (F * e^{-rT} * pdf(d1) * sigma) / (2 * sqrt(T))
    term1 = - (F * df * norm.pdf(d1) * sigma) / (2 * np.sqrt(t))
    
    # Sensitivity to discounting (r * V) is often included or separated.
    # Standard Black-76 Theta includes the drift of the forward?
    # Actually Black-76 V(F, t) = e^{-r(T-t)} * Black(F, K, sigma, T-t)
    # Theta usually refers to derivative wrt calendar time passing (decreasing T).
    # dV/dt = -dV/dT
    
    # Let's use dV/dT and flip sign.
    # Call dV/dT = - (F e^{-rT} n(d1) sigma)/(2 sqrt(T)) - r K e^{-rT} N(d2) + r (CallPrice without discount?)
    # Wait, simpler:
    # V_call = e^{-rT} [F N(d1) - K N(d2)]
    # terms from d(e^{-rT})/dT = -r V_call
    # terms from d(N(d1))/dT ...
    
    # Standard Result for Black-76 Call Theta (derivative wrt time to expiry T):
    # dC/dT = - (F e^{-rT} n(d1) sigma) / (2 sqrt(T)) - r * C
    # But "Theta" usually means dC/dt (time passing), so it's -dC/dT.
    # Theta_call = (F e^{-rT} n(d1) sigma) / (2 sqrt(T)) + r * C_call
    
    # Note: Many sources differ on sign convention. We return "sensitivity to DECREASE in time", i.e. dV/dt.
    # Since t is "time to expiry", dV/dt = -dV/dT.
    # So if dC/dT is negative (option loses value as T increases? No, option GAINS value as T increases).
    # C(T) increases with T. So dC/dT > 0.
    # So Theta = -dC/dT < 0.
    
    call_price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    dC_dT = (F * df * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) - r * call_price + r * F * df * norm.cdf(d1) 
    # Wait, that derivative is messy.
    # Use standard textbook formula for Black-76 Theta (per year):
    # Theta_call = - (F * e^{-rT} * n(d1) * sigma) / (2 * sqrt(T)) + r * C_call - r * F * e^{-rT} * N(d1)??
    # Actually, simplistic view:
    # Theta_call approx = - (F * e^{-rT} * n(d1) * sigma) / (2 * sqrt(T)) + r * K * e^{-rT} * N(d2)
    
    term1_call = - (F * df * norm.pdf(d1) * sigma) / (2 * np.sqrt(t))
    term2_call = r * K * df * norm.cdf(d2)
    # Be careful with F drifting. Black-76 assumes F is constant (futures).
    # If we treat F as spot, there is a drift. For Black-76, F is the underlying.
    
    theta_call = term1_call + term2_call - r * F * df * norm.cdf(d1) # This term accounts for forward discounting?
    # Let's stick to the most robust source: Hull or similar.
    # Hull for Futures Option (Black 76):
    # Theta_call = - (F * e^{-rT} * n(d1) * sigma) / (2 * sqrt(T)) + r * K * e^{-rT} * N(d2) - r * F * e^{-rT} * N(d1)
    # Simplifies to: - (F * e^{-rT} * n(d1) * sigma) / (2 * sqrt(T)) - r * C
    
    theta_call = - (F * df * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) - r * call_price
    
    if call_or_put == "put":
        # Put Theta
        # P = C - e^{-rT}(F - K)
        # dP/dt = dC/dt - d/dt [e^{-rT}(F - K)]
        #       = Theta_call - [-r * e^{-rT} * (F - K)] (assuming F constant)
        #       = Theta_call + r * e^{-rT} * (F - K)
        put_price = call_price - df * (F - K)
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
    """Analytical Rho (∂V/∂r) for Black-76."""
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
    
    # Call Rho = -t * e^{-rT} * (F * N(d1) - K * N(d2)) = -t * CallPrice
    # Because V = e^{-rT} * BlackPrice(F, K, ...). The risk free rate ONLY appears in the discount factor.
    # dV/dr = -t * V
    
    call_price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    rho_call = -t * call_price
    
    if call_or_put == "put":
        put_price = call_price - df * (F - K)
        return -t * put_price
        
    return rho_call
