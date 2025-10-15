"""Utilities for the Surface SVI (SSVI) volatility parameterisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import interpolate


RHO_BOUND: float = 0.999


def phi_eta_gamma(theta: np.ndarray | float, eta: float, gamma: float) -> np.ndarray:
    """Evaluate Gatheral's ``phi(θ) = η θ^{-γ} (1 + θ)^{γ-1}`` form.

    Args:
        theta: Total variance level ``θ``. Must be strictly positive.
        eta: Positive scale parameter ``η``.
        gamma: Shape parameter ``γ`` in ``[0, 1]``.

    Returns:
        Array containing ``phi(θ)`` for each ``θ``.

    Raises:
        ValueError: If any ``θ`` is non-positive or if parameters are out of
        bounds.
    """

    theta_arr = np.asarray(theta, dtype=float)
    if np.any(theta_arr <= 0.0):  # pragma: no cover - guarded by callers
        raise ValueError("theta must be strictly positive for phi evaluation")
    if eta <= 0.0 or not (0.0 <= gamma <= 1.0):  # pragma: no cover - caller bug
        raise ValueError("Invalid SSVI parameters for phi evaluation")

    return eta * np.power(theta_arr, -gamma) * np.power(1.0 + theta_arr, gamma - 1.0)


def ssvi_total_variance(
    k: np.ndarray | Iterable[float],
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Compute total variance ``w(k)`` for the canonical SSVI surface.

    The formula follows Gatheral & Jacquier (2012):

    ``w(k, θ) = θ / 2 * [1 + ρ φ(θ) k + sqrt((φ(θ)k + ρ)^2 + (1 - ρ^2))]``.

    Args:
        k: Log-moneyness grid.
        theta: Total variance level ``θ`` for the maturity.
        rho: Correlation-like parameter in ``(-1, 1)``.
        eta: Positive scale parameter ``η``.
        gamma: Shape parameter ``γ`` in ``[0, 1]``.

    Returns:
        Total variance values for each ``k``.
    """

    k_arr = np.asarray(k, dtype=float)
    phi = phi_eta_gamma(theta, eta, gamma)
    core = phi * k_arr + rho
    sqrt_term = np.sqrt(np.maximum(0.0, core * core + (1.0 - rho * rho)))
    return 0.5 * theta * (1.0 + rho * phi * k_arr + sqrt_term)


def theta_phi(theta: float, eta: float, gamma: float) -> float:
    """Return ``θ φ(θ)`` for the canonical ``φ`` form."""

    phi = phi_eta_gamma(theta, eta, gamma)
    return float(theta * phi)


def theta_phi_squared(theta: float, eta: float, gamma: float) -> float:
    """Return ``θ φ(θ)^2`` for the canonical ``φ`` form."""

    phi = phi_eta_gamma(theta, eta, gamma)
    return float(theta * phi * phi)


def theta_phi_derivative(theta: float, eta: float, gamma: float) -> float:
    """Compute the derivative ``∂_θ(θ φ(θ))`` for the canonical ``φ`` form."""

    theta_val = float(theta)
    phi_val = float(phi_eta_gamma(theta_val, eta, gamma))
    factor = (1.0 - gamma) / theta_val + (gamma - 1.0) / (1.0 + theta_val)
    return theta_val * phi_val * factor


def ssvi_calendar_margin(
    theta_early: float,
    theta_late: float,
    rho: float,
    eta: float,
    gamma: float,
    k_grid: np.ndarray,
) -> float:
    """Return the minimum calendar spread margin ``w_late(k) - w_early(k)``."""

    w_early = ssvi_total_variance(k_grid, theta_early, rho, eta, gamma)
    w_late = ssvi_total_variance(k_grid, theta_late, rho, eta, gamma)
    return float(np.min(w_late - w_early))


@dataclass(frozen=True)
class SSVISurfaceParams:
    """Immutable bundle describing a calibrated SSVI surface."""

    maturities: np.ndarray
    theta: np.ndarray
    rho: float
    eta: float
    gamma: float
    alpha: float = 0.0

    def interpolator(self) -> interpolate.PchipInterpolator:
        """Return a monotone interpolator for ``θ(t)``."""

        return interpolate.PchipInterpolator(
            self.maturities,
            self.theta,
            extrapolate=True,
        )


def compute_ssvi_margins(
    theta: Sequence[float],
    rho: float,
    eta: float,
    gamma: float,
) -> dict[str, float]:
    """Evaluate SSVI inequality margins for diagnostics."""

    theta_arr = np.asarray(theta, dtype=float)
    margins1 = []
    margins2 = []
    margins3 = []

    for theta_val in theta_arr:
        phi = float(phi_eta_gamma(theta_val, eta, gamma))
        theta_phi_val = theta_val * phi
        theta_phi_sq_val = theta_val * phi * phi
        deriv = theta_phi_derivative(theta_val, eta, gamma)
        margin1 = 4.0 - theta_phi_val * (1.0 + abs(rho))
        margin2 = 4.0 - theta_phi_sq_val * (1.0 + abs(rho))
        if abs(rho) < 1e-6:
            upper = 2.0 * phi
        else:
            upper = (1.0 / (rho * rho)) * (1.0 + np.sqrt(max(0.0, 1.0 - rho * rho))) * phi
        margin3 = upper - deriv
        margins1.append(margin1)
        margins2.append(margin2)
        margins3.append(margin3)

    return {
        "min_theta_phi_margin": float(np.min(margins1)),
        "min_theta_phi_sq_margin": float(np.min(margins2)),
        "min_derivative_margin": float(np.min(margins3)),
    }

