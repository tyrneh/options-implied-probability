"""Calibration routines for the SSVI volatility surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import optimize, interpolate

from oipd.core.errors import CalculationError
from oipd.core.ssvi import (
    RHO_BOUND,
    SSVISurfaceParams,
    compute_ssvi_margins,
    phi_eta_gamma,
    ssvi_calendar_margin,
    ssvi_total_variance,
    theta_phi,
    theta_phi_derivative,
    theta_phi_squared,
)
from oipd.core.vol_model import VolModel


@dataclass(frozen=True)
class SSVISliceObservations:
    """Observed total variance for a single maturity.

    Attributes:
        maturity: Time to expiry in year fractions.
        log_moneyness: Array of log-moneyness coordinates.
        total_variance: Observed total variance on ``log_moneyness`` grid.
        weights: Non-negative weights for each observation.
    """

    maturity: float
    log_moneyness: np.ndarray
    total_variance: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class SSVISurfaceFit:
    """Calibrated SSVI parameters and diagnostics."""

    params: SSVISurfaceParams
    theta_interpolator: interpolate.PchipInterpolator
    objective: float
    calendar_margin: float
    inequality_margins: dict[str, float]


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(y))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _atanh(x: float) -> float:
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


def _transform_parameters(
    vector: np.ndarray,
    num_slices: int,
) -> tuple[np.ndarray, float, float, float]:
    """Map unconstrained parameters to SSVI variables."""

    theta_raw = vector[:num_slices]
    rho_raw, eta_raw, gamma_raw = vector[num_slices : num_slices + 3]

    theta = _softplus(theta_raw)
    rho = np.tanh(rho_raw) * RHO_BOUND
    eta = float(_softplus(np.array([eta_raw]))[0])
    gamma = float(_sigmoid(np.array([gamma_raw]))[0])
    return theta, rho, eta, gamma


def _initial_guess(observations: Sequence[SSVISliceObservations]) -> np.ndarray:
    theta_guess = []
    for obs in observations:
        positive = obs.total_variance[obs.total_variance > 0]
        if positive.size == 0:
            theta_guess.append(0.05)
        else:
            theta_guess.append(float(np.median(positive)))

    theta_guess = np.maximum(theta_guess, 1e-4)
    theta_raw = _softplus_inv(np.asarray(theta_guess, dtype=float))
    rho_raw = _atanh(-0.3 / RHO_BOUND)
    eta_raw = _softplus_inv(np.asarray([1.0]))[0]
    gamma_raw = _logit(0.5)
    return np.concatenate([theta_raw, [rho_raw, eta_raw, gamma_raw]])


def _residual_sum(
    observations: Sequence[SSVISliceObservations],
    theta: np.ndarray,
    rho: float,
    eta: float,
    gamma: float,
) -> float:
    total = 0.0
    for obs, theta_val in zip(observations, theta):
        model_w = ssvi_total_variance(obs.log_moneyness, theta_val, rho, eta, gamma)
        diff = model_w - obs.total_variance
        total += float(np.sum(obs.weights * diff * diff))
    return total


def _constraint_penalty(
    observations: Sequence[SSVISliceObservations],
    theta: np.ndarray,
    rho: float,
    eta: float,
    gamma: float,
    *,
    strict: bool,
) -> float:
    penalty = 0.0

    diffs = np.diff(theta)
    penalty += np.sum(np.square(np.minimum(0.0, diffs))) * 1e4

    for theta_val in theta:
        phi = float(phi_eta_gamma(theta_val, eta, gamma))
        margin1 = 4.0 - theta_phi(theta_val, eta, gamma) * (1.0 + abs(rho))
        margin2 = 4.0 - theta_phi_squared(theta_val, eta, gamma) * (1.0 + abs(rho))
        if abs(rho) < 1e-6:
            upper = 2.0 * phi
        else:
            upper = (1.0 / (rho * rho)) * (1.0 + np.sqrt(max(0.0, 1.0 - rho * rho))) * phi
        margin3 = upper - theta_phi_derivative(theta_val, eta, gamma)
        penalty += np.square(np.minimum(0.0, margin1)) * 1e4
        penalty += np.square(np.minimum(0.0, margin2)) * 1e4
        penalty += np.square(np.minimum(0.0, margin3)) * 1e3

    if strict and len(theta) > 1:
        k_grid = np.linspace(-2.5, 2.5, 41)
        for (obs_a, theta_a), (obs_b, theta_b) in zip(
            zip(observations[:-1], theta[:-1]),
            zip(observations[1:], theta[1:]),
        ):
            margin = ssvi_calendar_margin(theta_a, theta_b, rho, eta, gamma, k_grid)
            penalty += np.square(np.minimum(0.0, margin)) * 1e4

    return float(penalty)


def calibrate_ssvi_surface(
    observations: Sequence[SSVISliceObservations],
    vol_model: VolModel,
    *,
    max_iter: int = 500,
) -> SSVISurfaceFit:
    """Calibrate SSVI parameters from slice observations.

    Args:
        observations: Sequence of per-maturity total variance observations.
        vol_model: Volatility model selector (only method ``"ssvi"`` is valid).
        max_iter: Maximum iterations for the optimiser.

    Returns:
        Calibrated surface fit with diagnostics.

    Raises:
        CalculationError: If calibration fails or converges to invalid values.
    """

    if not observations:
        raise CalculationError("SSVI calibration requires at least one maturity")
    if vol_model.method not in {"ssvi", None}:  # pragma: no cover - guarded earlier
        raise CalculationError("VolModel.method must be 'ssvi' for SSVI calibration")

    observations = tuple(sorted(observations, key=lambda obs: obs.maturity))
    initial = _initial_guess(observations)
    num_slices = len(observations)

    def objective(vec: np.ndarray) -> float:
        theta, rho, eta, gamma = _transform_parameters(vec, num_slices)
        residual = _residual_sum(observations, theta, rho, eta, gamma)
        penalty = _constraint_penalty(
            observations,
            theta,
            rho,
            eta,
            gamma,
            strict=vol_model.strict_no_arbitrage,
        )
        return residual + penalty

    result = optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-9},
    )

    if not result.success:
        raise CalculationError(f"SSVI calibration failed: {result.message}")

    theta, rho, eta, gamma = _transform_parameters(result.x, num_slices)
    maturities = np.array([obs.maturity for obs in observations], dtype=float)

    interpolator = interpolate.PchipInterpolator(maturities, theta, extrapolate=True)
    params = SSVISurfaceParams(
        maturities=maturities,
        theta=np.asarray(theta, dtype=float),
        rho=float(rho),
        eta=float(eta),
        gamma=float(gamma),
        alpha=0.0,
    )

    k_grid = np.linspace(-2.5, 2.5, 61)
    calendar_margin = 0.0
    if len(theta) > 1:
        margins = [
            ssvi_calendar_margin(t0, t1, params.rho, params.eta, params.gamma, k_grid)
            for t0, t1 in zip(theta[:-1], theta[1:])
        ]
        calendar_margin = float(np.min(margins))

    inequality_margins = compute_ssvi_margins(theta, params.rho, params.eta, params.gamma)

    return SSVISurfaceFit(
        params=params,
        theta_interpolator=interpolator,
        objective=float(result.fun),
        calendar_margin=calendar_margin,
        inequality_margins=inequality_margins,
    )

