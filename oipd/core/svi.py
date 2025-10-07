"""Analytical helpers for Stochastic Volatility Inspired (SVI) smiles.

This module contains parameter containers and closed-form utilities for the
single-expiry SVI model following Gatheral & Jacquier (2012).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import optimize

from oipd.core.errors import CalculationError

DEFAULT_SVI_OPTIONS: Dict[str, float] = {
    "max_iter": 200,
    "tol": 1e-8,
    "regularisation": 1e-4,
    "rho_bound": 0.999,
    "sigma_min": 1e-4,
}

_SVI_OPTION_KEYS = set(DEFAULT_SVI_OPTIONS.keys())

@dataclass(frozen=True)
class SVIParameters:
    """Raw SVI parameters ``(a, b, rho, m, sigma)``.

    The total implied variance is defined as::

        w(k) = a + b * (rho * (k - m) + sqrt((k - m)**2 + sigma**2))

    with constraints ``b >= 0``, ``|rho| < 1`` and ``sigma > 0``.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:  # pragma: no cover - dataclass guard
        if self.b < 0:
            raise ValueError("SVI parameter 'b' must be non-negative")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError("SVI parameter 'rho' must lie in (-1, 1)")
        if self.sigma <= 0.0:
            raise ValueError("SVI parameter 'sigma' must be positive")


def log_moneyness(strikes: np.ndarray | Iterable[float], forward: float) -> np.ndarray:
    """Return log-moneyness k = ln(K / F)."""

    if forward <= 0:
        raise ValueError("Forward must be positive for log-moneyness conversion")
    strikes_arr = np.asarray(strikes, dtype=float)
    if np.any(strikes_arr <= 0):
        raise ValueError("Strikes must be strictly positive for log-moneyness")
    return np.log(strikes_arr / forward)


def to_total_variance(iv: np.ndarray | Iterable[float], maturity_years: float) -> np.ndarray:
    """Convert implied volatilities to total variance w = sigma^2 * T."""

    if maturity_years <= 0.0:
        raise ValueError("Maturity in years must be positive")
    iv_arr = np.asarray(iv, dtype=float)
    total_var = np.square(iv_arr) * maturity_years
    return total_var


def from_total_variance(total_variance: np.ndarray | Iterable[float], maturity_years: float) -> np.ndarray:
    """Convert total variance back to implied volatilities."""

    if maturity_years <= 0.0:
        raise ValueError("Maturity in years must be positive")
    tv_arr = np.asarray(total_variance, dtype=float)
    if np.any(tv_arr < 0):
        raise ValueError("Total variance must be non-negative")
    return np.sqrt(tv_arr / maturity_years)


def svi_total_variance(k: np.ndarray | Iterable[float], params: SVIParameters) -> np.ndarray:
    """Evaluate raw SVI total variance at the given log-moneyness values."""

    k_arr = np.asarray(k, dtype=float)
    diff = k_arr - params.m
    root = np.sqrt(np.square(diff) + params.sigma**2)
    return params.a + params.b * (params.rho * diff + root)


def svi_first_derivative(k: np.ndarray | Iterable[float], params: SVIParameters) -> np.ndarray:
    """First derivative of total variance with respect to log-moneyness."""

    k_arr = np.asarray(k, dtype=float)
    diff = k_arr - params.m
    denom = np.sqrt(np.square(diff) + params.sigma**2)
    return params.b * (params.rho + diff / denom)


def svi_second_derivative(k: np.ndarray | Iterable[float], params: SVIParameters) -> np.ndarray:
    """Second derivative of total variance with respect to log-moneyness."""

    k_arr = np.asarray(k, dtype=float)
    diff = k_arr - params.m
    denom = np.sqrt(np.square(diff) + params.sigma**2)
    return params.b * params.sigma**2 / np.power(denom, 3)


def g_function(k: np.ndarray | Iterable[float], params: SVIParameters) -> np.ndarray:
    """Gatheral–Jacquier butterfly condition diagnostic ``g(k)``.

    The density is non-negative iff ``g(k) >= 0`` for all k.
    """

    k_arr = np.asarray(k, dtype=float)
    w = svi_total_variance(k_arr, params)
    wp = svi_first_derivative(k_arr, params)
    wpp = svi_second_derivative(k_arr, params)

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = 1 - (k_arr * wp) / (2 * w)
        term2 = 1 - wp / 2
        term3 = (wp**2) * (1 / (4 * w) + 1 / 16)
        g = term1 * term2 - term3 + 0.5 * wpp

    g = np.where(np.isnan(g), -np.inf, g)
    return g


def min_g_on_grid(params: SVIParameters, k_grid: np.ndarray | Iterable[float]) -> float:
    """Return the minimum ``g(k)`` over the supplied grid."""

    values = g_function(k_grid, params)
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return float("nan")
    return float(finite_vals.min())


def svi_minimum_location(params: SVIParameters) -> float:
    """Location ``k*`` where total variance attains its minimum."""

    sqrt_term = np.sqrt(1 - params.rho**2)
    return params.m - params.rho * params.sigma / sqrt_term


def svi_minimum_variance(params: SVIParameters) -> float:
    """Minimum total variance achieved by the SVI smile."""

    sqrt_term = np.sqrt(1 - params.rho**2)
    return params.a + params.b * params.sigma * sqrt_term


def _initial_guess(k: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    idx_min = int(np.argmin(total_variance))
    a0 = float(max(1e-6, total_variance[idx_min]))
    m0 = float(k[idx_min])
    b0 = 0.1
    sigma0 = max(0.1, float(np.std(k)) if np.std(k) > 0 else 0.2)
    rho0 = 0.0
    return np.array([a0, b0, rho0, m0, sigma0], dtype=float)


def svi_options(**overrides: Any) -> Dict[str, float]:
    """Return a copy of the default SVI options with overrides applied."""

    return merge_svi_options(overrides)


def merge_svi_options(overrides: Mapping[str, Any] | None) -> Dict[str, float]:
    if overrides is None:
        return dict(DEFAULT_SVI_OPTIONS)

    unknown = set(overrides) - _SVI_OPTION_KEYS
    if unknown:
        raise TypeError(f"Unknown SVI option(s): {sorted(unknown)}")

    merged = dict(DEFAULT_SVI_OPTIONS)
    merged.update({key: overrides[key] for key in overrides})
    return merged


def _build_bounds(k: np.ndarray, config: Mapping[str, float]) -> Sequence[Tuple[float, float]]:
    span = max(1.0, float(np.max(k) - np.min(k)))
    m_lo = float(np.min(k) - span)
    m_hi = float(np.max(k) + span)
    return [
        (-1.0, 4.0),
        (0.0, 5.0),
        (-config["rho_bound"], config["rho_bound"]),
        (m_lo, m_hi),
        (config["sigma_min"], 5.0),
    ]


def _penalty_terms(params_vec: np.ndarray, config: Mapping[str, float]) -> float:
    a, b, rho, _, sigma = params_vec
    if b < 0 or sigma <= 0 or abs(rho) >= 1:
        return 1e6
    min_var = a + b * sigma * np.sqrt(max(1e-12, 1 - rho**2))
    penalty = 1e5 * max(0.0, -min_var)
    penalty += config["regularisation"] * (b**2)
    return penalty


def calibrate_svi_parameters(
    k: np.ndarray,
    total_variance: np.ndarray,
    maturity_years: float,
    config: Mapping[str, float] | None,
) -> Tuple[SVIParameters, Dict[str, Any]]:
    """Calibrate raw SVI parameters to observed total variance data."""

    options = merge_svi_options(config)

    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if k.shape != total_variance.shape:
        raise ValueError("k and total_variance must have the same shape")
    if k.ndim != 1 or k.size < 5:
        raise ValueError("Need at least 5 strikes for SVI calibration")

    guess = _initial_guess(k, total_variance)
    bounds = _build_bounds(k, options)

    def objective(vec: np.ndarray) -> float:
        params = SVIParameters(*vec)
        model = svi_total_variance(k, params)
        residual = model - total_variance
        base = float(np.sum(residual**2))
        penalty = _penalty_terms(vec, options)
        return base + penalty

    result = optimize.minimize(
        objective,
        guess,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(options["max_iter"]), "ftol": float(options["tol"])}
    )

    diagnostics: Dict[str, Any] = {
        "status": "success" if result.success else "failure",
        "objective": float(result.fun),
        "iterations": result.nit,
        "message": result.message,
    }

    if not result.success:
        raise CalculationError(f"SVI calibration failed: {result.message}")

    params_vec = result.x
    params = SVIParameters(*params_vec)

    k_min, k_max = float(np.min(k)), float(np.max(k))
    k_grid = np.linspace(k_min - 0.5, k_max + 0.5, 201)
    min_g = min_g_on_grid(params, k_grid)
    diagnostics["min_g"] = min_g
    if min_g < -1e-6:
        diagnostics["status"] = "failure"
        raise CalculationError(
            f"SVI calibration violates butterfly condition: min_g={min_g:.3e}"
        )

    diagnostics["status"] = "success"
    return params, diagnostics


__all__ = [
    "DEFAULT_SVI_OPTIONS",
    "merge_svi_options",
    "svi_options",
    "SVIParameters",
    "log_moneyness",
    "to_total_variance",
    "from_total_variance",
    "svi_total_variance",
    "svi_first_derivative",
    "svi_second_derivative",
    "g_function",
    "min_g_on_grid",
    "svi_minimum_location",
    "svi_minimum_variance",
    "calibrate_svi_parameters",
]
