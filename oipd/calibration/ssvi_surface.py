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

GAMMA_MIN = 5e-3
GAMMA_MAX = 0.5
THETA_INCREMENT_EPS = 1e-6


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
    calendar_margins: list[float]
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


def _eta_upper_bound(theta: np.ndarray, rho: float, gamma: float) -> float:
    """Return the largest admissible ``η`` satisfying SSVI slice inequalities."""

    bounds: list[float] = []
    abs_rho = abs(rho)
    sqrt_term = np.sqrt(max(1e-12, 1.0 - rho * rho))

    for theta_val in theta:
        theta_val = float(max(theta_val, 1e-8))
        power1 = theta_val ** (1.0 - gamma) * (1.0 + theta_val) ** (gamma - 1.0)
        if power1 > 0.0:
            bounds.append(4.0 / ((1.0 + abs_rho) * power1))

        power2 = theta_val ** (gamma - 0.5) * (1.0 + theta_val) ** (1.0 - gamma)
        if power2 > 0.0:
            bounds.append(2.0 * sqrt_term * power2)

    if not bounds:
        return 1.0

    upper = min(bounds)
    if not np.isfinite(upper) or upper <= 0.0:
        return 1.0
    return float(upper)


def _transform_parameters(
    vector: np.ndarray,
    num_slices: int,
) -> tuple[np.ndarray, float, float, float]:
    """Map unconstrained parameters to SSVI variables."""

    theta_raw = vector[:num_slices]
    rho_raw, eta_raw, gamma_raw = vector[num_slices : num_slices + 3]

    increments = _softplus(theta_raw) + THETA_INCREMENT_EPS
    theta = np.cumsum(increments)

    rho = float(np.tanh(rho_raw) * RHO_BOUND)
    gamma_unit = float(_sigmoid(np.array([gamma_raw]))[0])
    gamma = float(GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * gamma_unit)

    eta_cap = _eta_upper_bound(theta, rho, gamma)
    sigma_eta = float(_sigmoid(np.array([eta_raw]))[0])
    eta_candidate = sigma_eta * eta_cap
    if eta_cap > 0.0:
        eta_max = np.nextafter(eta_cap, 0.0)
        eta = float(np.clip(eta_candidate, 0.0, eta_max))
    else:
        eta = 0.0

    return theta, rho, eta, gamma


def _initial_guess(observations: Sequence[SSVISliceObservations]) -> np.ndarray:
    theta_guess = []
    for obs in observations:
        positive = obs.total_variance[obs.total_variance > 0]
        if positive.size == 0:
            theta_guess.append(0.05)
        else:
            theta_guess.append(float(np.median(positive)))

    theta_guess = np.maximum.accumulate(np.maximum(np.asarray(theta_guess, dtype=float), 5e-4))

    desired_increments = []
    cumulative = 0.0
    for value in theta_guess:
        if value <= cumulative + THETA_INCREMENT_EPS:
            increment = THETA_INCREMENT_EPS * 2.0
        else:
            increment = value - cumulative
        increment = max(increment, THETA_INCREMENT_EPS * 2.0)
        desired_increments.append(increment)
        cumulative += increment

    adjusted = np.maximum(np.asarray(desired_increments, dtype=float) - THETA_INCREMENT_EPS, 1e-8)
    theta_raw = _softplus_inv(adjusted)

    rho_raw = _atanh(-0.3 / RHO_BOUND)

    gamma_target = 0.25
    gamma_unit = (gamma_target - GAMMA_MIN) / (GAMMA_MAX - GAMMA_MIN)
    gamma_unit = float(np.clip(gamma_unit, 1e-3, 1.0 - 1e-3))
    gamma_raw = _logit(gamma_unit)

    eta_raw = 0.0
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

    for theta_val in theta:
        phi = float(phi_eta_gamma(theta_val, eta, gamma))
        margin1 = 4.0 - theta_phi(theta_val, eta, gamma) * (1.0 + abs(rho))
        margin2 = 4.0 * (1.0 - rho * rho) - theta_phi_squared(theta_val, eta, gamma)
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
    calendar_margins: list[float] = []
    if len(theta) > 1:
        margins = [
            ssvi_calendar_margin(t0, t1, params.rho, params.eta, params.gamma, k_grid)
            for t0, t1 in zip(theta[:-1], theta[1:])
        ]
        calendar_margin = float(np.min(margins))
        calendar_margins = [float(m) for m in margins]

    inequality_margins = compute_ssvi_margins(theta, params.rho, params.eta, params.gamma)

    if vol_model.strict_no_arbitrage:
        tol = 1e-6
        if calendar_margin < -tol:
            raise CalculationError(
                f"SSVI calibration violated calendar monotonicity (min margin {calendar_margin:.3e})"
            )
        for key, margin in inequality_margins.items():
            if isinstance(margin, (list, tuple, np.ndarray)):
                min_margin = float(np.min(margin)) if margin else float("inf")
                if min_margin < -tol:
                    raise CalculationError(
                        f"SSVI calibration violated {key} (min margin {min_margin:.3e})"
                    )
            else:
                if float(margin) < -tol:
                    raise CalculationError(
                        f"SSVI calibration violated {key} (margin {float(margin):.3e})"
                    )

    return SSVISurfaceFit(
        params=params,
        theta_interpolator=interpolator,
        objective=float(result.fun),
        calendar_margin=calendar_margin,
        calendar_margins=calendar_margins,
        inequality_margins=inequality_margins,
    )
