"""Analytical helpers for Stochastic Volatility Inspired (SVI) smiles.

This module contains parameter containers and closed-form utilities for the
single-expiry SVI model following Gatheral & Jacquier (2012).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple
import warnings

import numpy as np
from scipy import optimize, special

from oipd.core.errors import CalculationError
from oipd.logging import get_logger
from oipd.pricing.black76 import black76_call_price

from .svi_types import (
    SVICalibrationDiagnostics,
    SVICalibrationOptions,
    SVITrialRecord,
)

DEFAULT_SVI_OPTIONS: SVICalibrationOptions = SVICalibrationOptions()

RawSVI = Tuple[float, float, float, float, float]
JWParams = Tuple[float, float, float, float, float]

JW_SEED_DIRECTIONS: Tuple[Tuple[float, float, float, float, float], ...] = (
    (0.0, -0.2, -0.2, 0.3, 0.0),  # flatter wings
    (0.0, 0.3, 0.3, -0.2, 0.0),  # steeper wings
    (0.1, 0.0, 0.0, 0.0, 0.1),  # higher ATM level
    (-0.1, 0.0, 0.0, 0.0, -0.1),  # lower ATM level
    (0.0, 0.0, 0.2, -0.3, 0.0),  # skew heavier
)

_SVI_OPTION_KEYS = SVICalibrationOptions.field_names()
logger = get_logger("oipd.svi")


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
    """Convert strike quotes into log-moneyness relative to the forward.

    Args:
        strikes: Strike prices quoted in absolute terms.
        forward: Forward price for the underlying asset.

    Returns:
        An array containing log-moneyness values ``k = ln(K / F)``.

    Raises:
        ValueError: If the forward or any strike is non-positive.
    """

    if forward <= 0:
        raise ValueError("Forward must be positive for log-moneyness conversion")
    strikes_arr = np.asarray(strikes, dtype=float)
    if np.any(strikes_arr <= 0):
        raise ValueError("Strikes must be strictly positive for log-moneyness")
    return np.log(strikes_arr / forward)


def raw_to_jw(raw: SVIParameters | RawSVI) -> JWParams:
    """Map raw SVI parameters ``(a, b, rho, m, sigma)`` to JW representation."""

    if isinstance(raw, SVIParameters):
        a, b, rho, m, sigma = raw.a, raw.b, raw.rho, raw.m, raw.sigma
    else:
        a, b, rho, m, sigma = raw

    s = float(np.hypot(m, sigma))
    if s <= 0.0:
        raise ValueError("Invalid SVI parameters: sqrt(m^2 + sigma^2) must be positive")

    v = a + b * (s - rho * m)
    psi = b * (rho - m / s)
    p = b * (1 - rho)
    c = b * (1 + rho)
    vmin = a + b * sigma * np.sqrt(max(1e-12, 1 - rho * rho))
    return (v, psi, p, c, vmin)


def jw_to_raw(jw: JWParams) -> SVIParameters:
    """Map JW parameters ``(v, psi, p, c, v_min)`` back to raw SVI."""

    v, psi, p, c, vmin = jw
    b = 0.5 * (p + c)
    if b <= 0.0:
        raise ValueError("JW parameters imply non-positive b")

    rho = (c - p) / (p + c)
    rho = float(np.clip(rho, -0.999999, 0.999999))

    k = rho - psi / b
    k = float(np.clip(k, -0.999999, 0.999999))

    rho_sq = rho * rho
    one_minus_k_sq = max(1e-12, 1 - k * k)
    root_term = np.sqrt(max(1e-12, (1 - rho_sq) * one_minus_k_sq))

    denom = -(1 - rho * k) + root_term
    if abs(denom) < 1e-12:
        raise ValueError("JW parameters yield ill-conditioned mapping back to raw SVI")

    s = (vmin - v) / (b * denom)
    if s <= 0.0:
        raise ValueError("Computed auxiliary variable s must be positive")

    m = s * k
    sigma = s * np.sqrt(one_minus_k_sq)
    if sigma <= 0.0:
        raise ValueError("Recovered sigma must be positive")

    a = v - b * (s - rho * m)
    return SVIParameters(a=a, b=b, rho=rho, m=m, sigma=sigma)


def to_total_variance(
    iv: np.ndarray | Iterable[float], maturity_years: float
) -> np.ndarray:
    """Convert implied volatility quotes to total variance values.

    Args:
        iv: Implied volatility observations.
        maturity_years: Time to expiry in year fractions.

    Returns:
        Total variance values computed as ``sigma^2 * T``.

    Raises:
        ValueError: If ``maturity_years`` is not strictly positive.
    """

    if maturity_years <= 0.0:
        raise ValueError("Maturity in years must be positive")
    iv_arr = np.asarray(iv, dtype=float)
    total_var = np.square(iv_arr) * maturity_years
    return total_var


def from_total_variance(
    total_variance: np.ndarray | Iterable[float], maturity_years: float
) -> np.ndarray:
    """Convert total variance observations back into implied volatilities.

    Args:
        total_variance: Total variance values, typically produced by
            :func:`to_total_variance`.
        maturity_years: Time to expiry in year fractions.

    Returns:
        Implied volatilities derived from ``total_variance``.

    Raises:
        ValueError: If ``maturity_years`` is not strictly positive or if any
            total variance entry is negative.
    """

    if maturity_years <= 0.0:
        raise ValueError("Maturity in years must be positive")
    tv_arr = np.asarray(total_variance, dtype=float)
    if np.any(tv_arr < 0):
        raise ValueError("Total variance must be non-negative")
    return np.sqrt(tv_arr / maturity_years)


def _svi_core_terms(
    k: np.ndarray | Iterable[float], params: SVIParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common arrays used by SVI analytic formulas.

    Args:
        k: Log-moneyness coordinates where the smile is evaluated.
        params: Candidate SVI parameters.

    Returns:
        Tuple containing ``(k_array, diff, root)`` where ``diff = k - m`` and
        ``root = hypot(diff, sigma)``.
    """

    k_arr = np.asarray(k, dtype=float)
    diff = k_arr - params.m
    root = np.hypot(diff, params.sigma)
    root_clipped = np.clip(root, 1e-12, None)
    return k_arr, diff, root_clipped


def _svi_total_variance_with_derivatives(
    k: np.ndarray | Iterable[float], params: SVIParameters
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return total variance and its first/second derivatives for SVI.

    Args:
        k: Log-moneyness coordinates where the smile is evaluated.
        params: Candidate SVI parameters.

    Returns:
        Tuple containing ``(k_arr, w, dw, w_prime, dw_prime, w_second, dw_second)``
        where ``dw`` is an ``(n, 5)`` array storing derivatives of ``w`` with
        respect to ``(a, b, rho, m, sigma)`` in column order. ``dw_prime`` and
        ``dw_second`` follow the same convention for the first and second
        derivatives of total variance with respect to log-moneyness.
    """

    k_arr, diff, root = _svi_core_terms(k, params)
    inv_root = 1.0 / root
    diff_over_root = diff * inv_root
    sigma = params.sigma
    sigma_sq = sigma * sigma
    inv_root_sq = inv_root * inv_root
    inv_root_cu = inv_root_sq * inv_root
    inv_root_fifth = inv_root_cu * inv_root_sq

    w = params.a + params.b * (params.rho * diff + root)

    dw = np.empty((k_arr.shape[0], 5), dtype=float)
    dw[:, 0] = 1.0
    dw[:, 1] = params.rho * diff + root
    dw[:, 2] = params.b * diff
    dw[:, 3] = -params.b * (params.rho + diff_over_root)
    dw[:, 4] = params.b * (sigma * inv_root)

    term_f = params.rho + diff_over_root

    w_prime = params.b * term_f
    dw_prime = np.empty_like(dw)
    dw_prime[:, 0] = 0.0
    dw_prime[:, 1] = term_f
    dw_prime[:, 2] = params.b
    dw_prime[:, 3] = -params.b * sigma_sq * inv_root_cu
    dw_prime[:, 4] = -params.b * diff * sigma * inv_root_cu

    w_second = params.b * sigma_sq * inv_root_cu
    dw_second = np.empty_like(dw)
    dw_second[:, 0] = 0.0
    dw_second[:, 1] = sigma_sq * inv_root_cu
    dw_second[:, 2] = 0.0
    dw_second[:, 3] = 3.0 * params.b * sigma_sq * diff * inv_root_fifth
    dw_second[:, 4] = params.b * (
        (2.0 * sigma * inv_root_cu) - (3.0 * sigma**3 * inv_root_fifth)
    )

    return k_arr, w, dw, w_prime, dw_prime, w_second, dw_second


def svi_total_variance(
    k: np.ndarray | Iterable[float], params: SVIParameters
) -> np.ndarray:
    """Evaluate raw SVI total variance at the given log-moneyness values.

    Args:
        k: Log-moneyness coordinates where the smile is evaluated.
        params: Calibrated SVI parameters.

    Returns:
        Total variance values implied by the supplied SVI parameters.
    """

    _, diff, root = _svi_core_terms(k, params)
    return params.a + params.b * (params.rho * diff + root)


def svi_first_derivative(
    k: np.ndarray | Iterable[float], params: SVIParameters
) -> np.ndarray:
    """Compute the first derivative of total variance with respect to k.

    Args:
        k: Log-moneyness coordinates for the evaluation.
        params: Calibrated SVI parameters.

    Returns:
        First derivative of the SVI total variance curve.
    """

    _, diff, root = _svi_core_terms(k, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        return params.b * (params.rho + diff / root)


def svi_second_derivative(
    k: np.ndarray | Iterable[float], params: SVIParameters
) -> np.ndarray:
    """Compute the second derivative of total variance with respect to k.

    Args:
        k: Log-moneyness coordinates for the evaluation.
        params: Calibrated SVI parameters.

    Returns:
        Second derivative of the SVI total variance curve.
    """

    _, diff, root = _svi_core_terms(k, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        return params.b * params.sigma**2 / np.power(root, 3)


def g_function(k: np.ndarray | Iterable[float], params: SVIParameters) -> np.ndarray:
    """Evaluate the Gatheral–Jacquier butterfly condition diagnostic ``g(k)``.

    The density is non-negative iff ``g(k) >= 0`` for all values of ``k``.

    Args:
        k: Log-moneyness coordinates for the diagnostic evaluation.
        params: Calibrated SVI parameters.

    Returns:
        Diagnostic values capturing the no-arbitrage butterfly constraint.
    """

    k_arr, diff, root = _svi_core_terms(k, params)
    w = params.a + params.b * (params.rho * diff + root)
    with np.errstate(divide="ignore", invalid="ignore"):
        wp = params.b * (params.rho + diff / root)
        wpp = params.b * params.sigma**2 / np.power(root, 3)

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = 1 - (k_arr * wp) / (2 * w)
        term1_sq = np.square(term1)
        term3 = (wp**2) * (1 / (4 * w) + 1 / 16)
        g = term1_sq - term3 + 0.5 * wpp

    g = np.where(np.isnan(g), -np.inf, g)
    return g


def min_g_on_grid(params: SVIParameters, k_grid: np.ndarray | Iterable[float]) -> float:
    """Return the minimum value of ``g(k)`` over the supplied grid.

    Args:
        params: Calibrated SVI parameters used for evaluation.
        k_grid: Log-moneyness coordinates forming the diagnostic grid.

    Returns:
        The minimum finite value of ``g(k)`` over ``k_grid`` or ``nan`` if no
        finite diagnostic values are available.
    """

    values = g_function(k_grid, params)
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return float("nan")
    return float(finite_vals.min())


def check_butterfly(
    params: SVIParameters,
    k_grid: np.ndarray | Iterable[float],
) -> dict[str, float | np.ndarray]:
    """Evaluate butterfly arbitrage diagnostics for a raw SVI slice.

    Args:
        params: Raw SVI parameters for the slice.
        k_grid: Log-moneyness grid used to evaluate ``g(k)``.

    Returns:
        Dictionary containing the evaluation grid, the diagnostic values, and
        the minimum finite margin observed across the grid.
    """

    grid = np.asarray(k_grid, dtype=float)
    values = g_function(grid, params)
    finite_mask = np.isfinite(values)
    min_margin = float(np.min(values[finite_mask])) if finite_mask.any() else float("nan")
    return {
        "grid": grid,
        "g_values": values,
        "min_margin": min_margin,
    }


def svi_minimum_location(params: SVIParameters) -> float:
    """Return the log-moneyness where total variance is minimised.

    Args:
        params: Calibrated SVI parameters.

    Returns:
        The location ``k*`` where the SVI total variance reaches its minimum.
    """

    sqrt_term = np.sqrt(1 - params.rho**2)
    return params.m - params.rho * params.sigma / sqrt_term


def svi_minimum_variance(params: SVIParameters) -> float:
    """Return the minimum total variance achieved by the SVI smile.

    Args:
        params: Calibrated SVI parameters.

    Returns:
        The minimum total variance value implied by the SVI smile.
    """

    sqrt_term = np.sqrt(1 - params.rho**2)
    return params.a + params.b * params.sigma * sqrt_term


def _call_prices_from_params(
    k: np.ndarray, params: SVIParameters, maturity_years: float
) -> np.ndarray:
    """Evaluate forward call prices implied by SVI on the supplied grid.

    Args:
        k: Log-moneyness grid where call prices are computed.
        params: Candidate SVI parameters.
        maturity_years: Time to expiry in year fractions.

    Returns:
        Forward-measure Black-76 call prices for the provided log-moneyness grid.
    """

    k_arr, diff, root = _svi_core_terms(k, params)
    total_variance = params.a + params.b * (params.rho * diff + root)
    implied_variance = np.maximum(total_variance / maturity_years, 0.0)
    implied_vol = np.sqrt(np.maximum(implied_variance, 1e-12))
    strikes = np.exp(k_arr)
    return black76_call_price(1.0, strikes, implied_vol, maturity_years, 0.0)


def _butterfly_penalty(
    params: SVIParameters,
    k_grid: np.ndarray,
    weight: float,
    *,
    return_grad: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Return butterfly penalty and optionally its gradient."""

    if weight <= 0.0:
        if return_grad:
            return 0.0, np.zeros(5, dtype=float)
        return 0.0

    (
        _,
        w,
        dw,
        w_prime,
        dw_prime,
        w_second,
        dw_second,
    ) = _svi_total_variance_with_derivatives(k_grid, params)

    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        penalty = weight * 1e6
        if return_grad:
            return penalty, np.zeros(5, dtype=float)
        return penalty

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_w = 1.0 / w
        inv_w_sq = inv_w * inv_w
        A = (k_grid * w_prime) * 0.5 * inv_w

    term1 = 1.0 - A
    B = 0.25 * inv_w + 0.0625
    term3 = w_prime * w_prime * B
    g_vals = term1 * term1 - term3 + 0.5 * w_second

    if np.any(~np.isfinite(g_vals)):
        penalty = weight * 1e6
        if return_grad:
            return penalty, np.zeros(5, dtype=float)
        return penalty

    violations_mask = g_vals < 0.0
    violations = g_vals[violations_mask]
    if violations.size == 0:
        if return_grad:
            return 0.0, np.zeros(5, dtype=float)
        return 0.0

    dA = (
        (k_grid[:, None] * 0.5)
        * (dw_prime * inv_w[:, None] - w_prime[:, None] * inv_w_sq[:, None] * dw)
    )
    dterm1 = -dA
    dB = -0.25 * inv_w_sq[:, None] * dw
    dterm3 = 2.0 * w_prime[:, None] * B[:, None] * dw_prime + w_prime[:, None] ** 2 * dB
    dg = 2.0 * term1[:, None] * dterm1 - dterm3 + 0.5 * dw_second

    grad = np.zeros(5, dtype=float)
    scaled = 2.0 * weight * violations[:, None]
    grad += np.sum(scaled * dg[violations_mask], axis=0)

    penalty = weight * float(np.sum(violations**2))
    if return_grad:
        return penalty, grad
    return penalty


def _call_spread_penalty(
    params: SVIParameters,
    context: _CallSpreadContext | None,
    weight: float,
    *,
    return_grad: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Return call-spread penalty and optionally its gradient."""

    if weight <= 0.0 or context is None:
        if return_grad:
            return 0.0, np.zeros(5, dtype=float)
        return 0.0

    (
        _,
        w_combined,
        dw_combined,
        _,
        _,
        _,
        _,
    ) = _svi_total_variance_with_derivatives(context.combined_k, params)

    if np.any(~np.isfinite(w_combined)):
        penalty = weight * 1e6
        if return_grad:
            return penalty, np.zeros(5, dtype=float)
        return penalty

    variance = w_combined / context.maturity
    variance = np.maximum(variance, 0.0)
    sigma = np.sqrt(np.maximum(variance, 1e-12))

    denom = sigma * context.sqrt_t
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    d1 = (context.log_fk + 0.5 * sigma**2 * context.maturity) / denom
    d2 = d1 - sigma * context.sqrt_t
    cdf_d1 = special.ndtr(d1)
    cdf_d2 = special.ndtr(d2)
    prices = cdf_d1 - context.strikes * cdf_d2

    pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    vega = context.sqrt_t * pdf_d1

    dc_dw = np.zeros_like(sigma)
    valid_sigma = sigma > 1e-12
    dc_dw[valid_sigma] = (
        vega[valid_sigma]
        / (2.0 * sigma[valid_sigma] * context.maturity)
    )
    price_grads = dc_dw[:, None] * dw_combined

    prices_left = prices[context.left_slice]
    prices_central = prices[context.centre_slice]
    prices_right = prices[context.right_slice]

    left_spread = prices_left - prices_central
    right_spread = prices_central - prices_right

    penalty = 0.0
    grad = np.zeros(5, dtype=float)

    left_mask = left_spread < 0.0
    if np.any(left_mask):
        diff_val = weight * np.sum(left_spread[left_mask] ** 2)
        penalty += float(diff_val)
        scaled = 2.0 * weight * left_spread[left_mask][:, None]
        left_grad = (
            price_grads[context.left_slice][left_mask]
            - price_grads[context.centre_slice][left_mask]
        )
        grad += np.sum(scaled * left_grad, axis=0)

    right_mask = right_spread < 0.0
    if np.any(right_mask):
        diff_val = weight * np.sum(right_spread[right_mask] ** 2)
        penalty += float(diff_val)
        scaled = 2.0 * weight * right_spread[right_mask][:, None]
        right_grad = (
            price_grads[context.centre_slice][right_mask]
            - price_grads[context.right_slice][right_mask]
        )
        grad += np.sum(scaled * right_grad, axis=0)

    if return_grad:
        return penalty, grad
    return penalty


def _bid_ask_penalty(
    model_iv: np.ndarray,
    bid_iv: np.ndarray | None,
    ask_iv: np.ndarray | None,
    weight: float,
) -> float:
    """Return hinge penalties when model IV leaves the bid/ask envelope."""

    if weight <= 0.0:
        return 0.0
    penalty = 0.0

    if bid_iv is not None:
        bid_arr = np.asarray(bid_iv, dtype=float)
        if bid_arr.shape != model_iv.shape:
            raise ValueError("bid_iv must match the shape of model_iv")
        mask = np.isfinite(bid_arr)
        if mask.any():
            below = np.maximum(0.0, bid_arr[mask] - model_iv[mask])
            penalty += float(np.sum(below**2))

    if ask_iv is not None:
        ask_arr = np.asarray(ask_iv, dtype=float)
        if ask_arr.shape != model_iv.shape:
            raise ValueError("ask_iv must match the shape of model_iv")
        mask = np.isfinite(ask_arr)
        if mask.any():
            above = np.maximum(0.0, model_iv[mask] - ask_arr[mask])
            penalty += float(np.sum(above**2))

    return weight * penalty


def _adaptive_call_spread_step(
    k: np.ndarray | Iterable[float],
    maturity_years: float,
    options: SVICalibrationOptions,
) -> float:
    """Compute an adaptive finite-difference step for the call-spread penalty.

    Args:
        k: Log-moneyness coordinates for observed strikes.
        maturity_years: Time to expiry in years.
        options: SVI calibration options controlling the step behaviour.

    Returns:
        A non-negative step size tuned to the strike spacing and tenor.

    Raises:
        ValueError: If the configured floor and ceiling bounds are invalid.
    """

    override = options.callspread_step
    if override is not None:
        step = float(override)
        return max(step, 0.0)

    floor = float(options.callspread_step_floor or 0.01)
    ceiling = float(options.callspread_step_ceiling or 0.35)
    if floor <= 0.0 or ceiling <= 0.0:
        raise ValueError("callspread step bounds must be positive")
    if floor >= ceiling:
        raise ValueError("callspread_step_floor must be less than the ceiling")

    k_arr = np.asarray(k, dtype=float)
    if k_arr.size < 2:
        return floor

    ordered = np.sort(k_arr)
    diffs = np.diff(ordered)
    positive_diffs = diffs[diffs > 1e-8]
    if positive_diffs.size > 0:
        spacing = float(np.median(positive_diffs))
    else:
        spacing = float(np.std(ordered))
    if not np.isfinite(spacing) or spacing <= 0.0:
        span = float(np.max(ordered) - np.min(ordered)) if ordered.size > 1 else floor
        divisor = max(ordered.size - 1, 1)
        spacing = max(span / divisor, floor)

    baseline = 0.5 * spacing
    tenor = max(float(maturity_years), 1e-6)
    scale = np.sqrt(tenor / 0.25)
    scale = float(np.clip(scale, 0.5, 2.0))
    adaptive = baseline * scale
    adaptive = max(floor, adaptive)
    adaptive = min(ceiling, adaptive)
    return adaptive


def _compute_huber_delta(
    total_variance: np.ndarray | Iterable[float],
    options: SVICalibrationOptions,
    *,
    bid_iv: np.ndarray | None = None,
    ask_iv: np.ndarray | None = None,
    maturity_years: float | None = None,
) -> float:
    """Determine the Huber loss transition scale in total variance space.

    Args:
        total_variance: Observed total variance values for the slice.
        options: SVI calibration options providing scaling parameters.

    Returns:
        A positive scalar suitable for the Huberised residuals.
    """

    override = options.huber_delta
    if override is not None:
        delta = float(override)
        return max(delta, 0.0)

    beta = float(options.huber_beta or 0.01)
    floor = float(options.huber_delta_floor or 1e-4)
    tv_arr = np.asarray(total_variance, dtype=float)
    finite_tv = tv_arr[np.isfinite(tv_arr)]
    if finite_tv.size == 0:
        typical_scale = floor
    else:
        typical_scale = float(np.median(finite_tv))
        if not np.isfinite(typical_scale) or typical_scale <= 0.0:
            typical_scale = float(np.mean(np.abs(finite_tv)))

    if not np.isfinite(typical_scale) or typical_scale <= 0.0:
        typical_scale = 1.0

    delta = max(floor, beta * typical_scale)

    if (
        bid_iv is not None
        and ask_iv is not None
        and maturity_years is not None
        and maturity_years > 0.0
    ):
        bid_arr = np.asarray(bid_iv, dtype=float)
        ask_arr = np.asarray(ask_iv, dtype=float)
        if bid_arr.shape == ask_arr.shape == np.asarray(total_variance, dtype=float).shape:
            spread = np.abs((ask_arr**2) - (bid_arr**2)) * float(maturity_years)
            valid = np.isfinite(spread) & (spread > 0.0)
            if valid.any():
                measurement_scale = float(np.median(spread[valid]) * 0.5)
                if np.isfinite(measurement_scale) and measurement_scale > 0.0:
                    delta = max(delta, measurement_scale, floor)

    return delta


def _initial_guess(k: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    """Generate a heuristic starting point for SVI calibration.

    Args:
        k: Log-moneyness grid for the observations.
        total_variance: Observed total variance values aligned with ``k``.

    Returns:
        A NumPy array containing the initial parameter vector ``(a, b, rho, m,
        sigma)`` used by the optimiser.
    """
    k_arr = np.asarray(k, dtype=float)
    tv_arr = np.asarray(total_variance, dtype=float)
    order = np.argsort(k_arr)
    k_sorted = k_arr[order]
    tv_sorted = tv_arr[order]

    idx_min = int(np.argmin(tv_sorted))
    a0 = float(max(1e-6, tv_sorted[idx_min]))
    m0 = float(k_sorted[idx_min])

    # Estimate wing slopes around the minimum to seed b and rho
    if 0 < idx_min < k_sorted.size - 1:
        left_slope = (tv_sorted[idx_min] - tv_sorted[idx_min - 1]) / (
            k_sorted[idx_min] - k_sorted[idx_min - 1]
        )
        right_slope = (tv_sorted[idx_min + 1] - tv_sorted[idx_min]) / (
            k_sorted[idx_min + 1] - k_sorted[idx_min]
        )
        slope_scale = max(abs(left_slope), abs(right_slope), 0.05)
        b0 = float(np.clip(slope_scale, 0.05, 5.0))
        rho_estimate = (right_slope - left_slope) / (
            abs(right_slope) + abs(left_slope) + 1e-12
        )
        rho0 = float(np.clip(rho_estimate, -0.9, 0.9))
    else:
        b0 = 0.1
        rho0 = 0.0

    spread = float(np.std(k_sorted))
    sigma0 = float(np.clip(spread if spread > 1e-6 else 0.2, 0.05, 3.0))
    return np.array([a0, b0, rho0, m0, sigma0], dtype=float)


def _vega_based_weights(
    k: np.ndarray,
    total_variance: np.ndarray,
    maturity_years: float,
    mode: str,
    weight_cap: float,
    volumes: np.ndarray | None = None,
    bid_iv: np.ndarray | None = None,
    ask_iv: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, bool]:
    """Construct weighting vector for the calibration objective."""

    size = k.shape[0]
    if size == 0 or maturity_years <= 0:
        return np.ones(size, dtype=float), False, False

    mode_normalised = mode.lower()
    if mode_normalised in {"none", "off", "disabled"}:
        return np.ones(size, dtype=float), False, False

    total_var_arr = np.asarray(total_variance, dtype=float)
    total_var_arr = np.maximum(total_var_arr, 1e-12)
    iv = np.sqrt(total_var_arr / maturity_years)
    sqrt_t = np.sqrt(max(maturity_years, 1e-12))

    if not np.isfinite(sqrt_t):
        return np.ones(size, dtype=float), False, False

    strikes = np.exp(np.asarray(k, dtype=float))
    ln_fk = -np.log(np.maximum(strikes, 1e-12))
    denom = np.maximum(iv * sqrt_t, 1e-12)
    d1 = (ln_fk + 0.5 * (iv**2) * maturity_years) / denom
    phi = np.exp(-0.5 * np.square(d1)) / np.sqrt(2 * np.pi)
    vega = sqrt_t * phi

    weights = np.maximum(vega, 1e-8)
    volume_used = False
    measurement_used = False

    measurement_weights: np.ndarray | None = None
    if bid_iv is not None and ask_iv is not None:
        bid_arr = np.asarray(bid_iv, dtype=float)
        ask_arr = np.asarray(ask_iv, dtype=float)
        if bid_arr.shape != weights.shape or ask_arr.shape != weights.shape:
            raise ValueError("bid_iv and ask_iv must have the same shape as k")
        spread = ask_arr - bid_arr
        mask = np.isfinite(spread) & (spread > 0.0)
        if mask.any():
            spread_safe = np.maximum(spread, 1e-4)
            measurement_weights = np.ones_like(weights)
            measurement_weights[mask] = 1.0 / spread_safe[mask]
            measurement_used = True

    if volumes is not None:
        vol_arr = np.asarray(volumes, dtype=float)
        if vol_arr.shape != weights.shape:
            raise ValueError("volumes must have the same shape as k")
        valid_mask = np.isfinite(vol_arr) & (vol_arr > 0)
        if valid_mask.any():
            fill_value = float(np.nanmedian(vol_arr[valid_mask]))
            vol_weights = np.where(valid_mask, vol_arr, fill_value)
            weights *= vol_weights
            volume_used = True

    if mode_normalised in {"spread", "measurement"}:
        if measurement_weights is None:
            return np.ones(size, dtype=float), volume_used, False
        weights = measurement_weights.copy()
    elif mode_normalised in {"vega+spread", "spread+vega", "vega_spread"}:
        if measurement_weights is not None:
            weights = weights * measurement_weights
        measurement_used = measurement_used and measurement_weights is not None
    else:
        measurement_used = measurement_used and mode_normalised != "vega"

    mean_weight = float(np.mean(weights))
    if mean_weight > 0:
        weights = weights / mean_weight

    if np.isfinite(weight_cap) and weight_cap > 0:
        weights = np.clip(weights, 1e-6, weight_cap)

    return weights, volume_used, measurement_used


def _qe_split_seeds(
    k: np.ndarray,
    total_variance: np.ndarray,
    sqrt_weights: np.ndarray,
    heuristic_guess: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> list[np.ndarray]:
    """Generate quasi-explicit split seeds for SVI calibration.

    Args:
        k: Log-moneyness coordinates for the observed smile.
        total_variance: Observed total variance values aligned with ``k``.
        sqrt_weights: Square roots of the weighting vector used for regression.
        heuristic_guess: Initial heuristic parameter vector ``(a, b, rho, m, sigma)``.
        bounds: Parameter bounds supplied to the optimiser.

    Returns:
        A list of candidate starting points derived from the split heuristic.
    """

    k_arr = np.asarray(k, dtype=float)
    tv_arr = np.asarray(total_variance, dtype=float)
    w_arr = np.asarray(sqrt_weights, dtype=float)

    if k_arr.size < 3 or tv_arr.size != k_arr.size or w_arr.size != k_arr.size:
        return []

    m0 = float(heuristic_guess[3])
    sigma0 = float(heuristic_guess[4])

    ordered = np.sort(k_arr)
    diffs = np.diff(ordered)
    positive_diffs = diffs[diffs > 1e-8]
    if positive_diffs.size > 0:
        spacing = float(np.median(positive_diffs))
    else:
        spacing = float(np.std(ordered))
    if not np.isfinite(spacing) or spacing <= 0.0:
        span = float(np.max(ordered) - np.min(ordered)) if ordered.size > 1 else 0.1
        divisor = max(ordered.size - 1, 1)
        spacing = max(span / divisor, 0.05)

    m_candidates: list[float] = []
    for candidate in (m0, float(np.median(k_arr)), m0 - spacing, m0 + spacing):
        if not np.isfinite(candidate):
            continue
        clipped = float(np.clip(candidate, bounds[3][0], bounds[3][1]))
        if not any(np.isclose(clipped, existing, atol=1e-6) for existing in m_candidates):
            m_candidates.append(clipped)

    sigma_candidates: list[float] = []
    sigma_raw = [
        sigma0,
        sigma0 * 0.75,
        sigma0 * 1.25,
        max(bounds[4][0] * 1.05, spacing),
    ]
    for candidate in sigma_raw:
        if not np.isfinite(candidate) or candidate <= 0.0:
            continue
        clipped = float(np.clip(candidate, bounds[4][0], bounds[4][1]))
        if not any(np.isclose(clipped, existing, atol=1e-6) for existing in sigma_candidates):
            sigma_candidates.append(clipped)

    seeds: list[np.ndarray] = []

    for m_candidate in m_candidates:
        diff = k_arr - m_candidate
        design_diff = diff
        for sigma_candidate in sigma_candidates:
            phi = np.sqrt(np.square(diff) + sigma_candidate**2)
            design = np.column_stack(
                (
                    np.ones_like(k_arr),
                    design_diff,
                    phi,
                )
            )
            weighted_design = design * w_arr[:, None]
            weighted_response = tv_arr * w_arr
            try:
                coefficients, _, _, _ = np.linalg.lstsq(
                    weighted_design, weighted_response, rcond=None
                )
            except np.linalg.LinAlgError:
                continue

            a_candidate = float(coefficients[0])
            b_rho = float(coefficients[1])
            b_candidate = float(coefficients[2])
            if not np.isfinite(b_candidate) or b_candidate <= 0.0:
                continue

            rho_candidate = b_rho / b_candidate
            if not np.isfinite(rho_candidate):
                continue
            rho_candidate = float(np.clip(rho_candidate, -0.999, 0.999))

            candidate_vec = np.array(
                [a_candidate, b_candidate, rho_candidate, m_candidate, sigma_candidate],
                dtype=float,
            )
            clipped_vec = np.array(candidate_vec, dtype=float)
            for idx, bound in enumerate(bounds):
                clipped_vec[idx] = float(np.clip(clipped_vec[idx], bound[0], bound[1]))

            try:
                SVIParameters(*clipped_vec)
            except ValueError:
                continue

            if not any(np.allclose(clipped_vec, existing, atol=1e-6) for existing in seeds):
                seeds.append(clipped_vec)

    return seeds


def svi_options(**overrides: Any) -> SVICalibrationOptions:
    """Return SVI calibration options with the provided overrides applied.

    Args:
        **overrides: Keyword arguments overriding default option values.

    Returns:
        A dataclass containing the merged calibration options.

    Raises:
        TypeError: If an unknown option key is supplied.
    """

    return merge_svi_options(overrides)


def merge_svi_options(
    overrides: SVICalibrationOptions | Mapping[str, Any] | None,
) -> SVICalibrationOptions:
    """Merge default SVI options with user-provided overrides.

    Args:
        overrides: Mapping whose keys correspond to recognised SVI options or an
            existing :class:`SVICalibrationOptions` instance.

    Returns:
        A :class:`SVICalibrationOptions` instance containing the merged options.

    Raises:
        TypeError: If ``overrides`` contains keys that are not recognised.
    """

    return SVICalibrationOptions.from_mapping(overrides)


def _build_bounds(
    k: np.ndarray, maturity_years: float, config: SVICalibrationOptions
) -> Sequence[Tuple[float, float]]:
    """Construct parameter bounds for the SVI optimisation routine.

    Args:
        k: Log-moneyness grid for the smile being calibrated.
        config: Configuration dictionary controlling the optimiser bounds.

    Returns:
        A sequence of ``(lower, upper)`` bounds for each SVI parameter.
    """

    k_min = float(np.min(k))
    k_max = float(np.max(k))
    span = float(k_max - k_min)
    span = max(span, 1e-3)
    expansion = max(span, 1.0)

    rho_bound = float(config.rho_bound)
    sigma_min = float(config.sigma_min)
    tenor = max(float(maturity_years), 1e-4)

    m_lo = k_min - expansion
    m_hi = k_max + expansion

    # Wider wing slopes allowed for short-dated expiries
    wing_amp = max(1.0, 1.0 / np.sqrt(tenor))
    b_upper = min(50.0, 5.0 * wing_amp)

    sigma_upper = min(5.0, max(0.5, 2.0 * expansion))
    sigma_lower = max(sigma_min, 1e-4)

    return [
        (-1.0, 4.0),
        (0.0, b_upper),
        (-rho_bound, rho_bound),
        (m_lo, m_hi),
        (sigma_lower, sigma_upper),
    ]


def _penalty_terms(
    params_vec: np.ndarray, regularisation: float, *, return_grad: bool = False
) -> float | tuple[float, np.ndarray]:
    """Compute regularisation penalties for the optimisation objective.

    Args:
        params_vec: Candidate SVI parameter vector ``(a, b, rho, m, sigma)``.
        regularisation: L2 penalty magnitude applied to ``b``.
        return_grad: Whether to also return the gradient with respect to the
            SVI parameters.

    Returns:
        Penalty value and, when ``return_grad`` is set, a tuple of the value and
        the gradient vector.
    """

    penalty, grad = _penalty_terms_with_grad(params_vec, regularisation)
    if return_grad:
        return penalty, grad
    return penalty


def _penalty_terms_with_grad(
    params_vec: np.ndarray, regularisation: float
) -> tuple[float, np.ndarray]:
    """Return penalty value together with its gradient."""

    a, b, rho, _, sigma = params_vec
    if b < 0 or sigma <= 0 or abs(rho) >= 1:
        return 1e6, np.zeros(5, dtype=float)

    grad = np.zeros(5, dtype=float)
    penalty = 0.0

    one_minus_rho_sq = max(1e-12, 1.0 - rho * rho)
    sqrt_term = np.sqrt(one_minus_rho_sq)
    min_var = a + b * sigma * sqrt_term
    if min_var < 0.0:
        penalty += 1e5 * (-min_var)
        coeff = 1e5
        d_minvar = np.array(
            [
                1.0,
                sigma * sqrt_term,
                b * sigma * (-rho) / sqrt_term,
                0.0,
                b * sqrt_term,
            ],
            dtype=float,
        )
        grad -= coeff * d_minvar

    penalty += regularisation * (b**2)
    grad[1] += 2.0 * regularisation * b

    return penalty, grad


def calibrate_svi_parameters(
    k: np.ndarray,
    total_variance: np.ndarray,
    maturity_years: float,
    config: SVICalibrationOptions | Mapping[str, Any] | None,
    *,
    bid_iv: np.ndarray | None = None,
    ask_iv: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
) -> Tuple[SVIParameters, SVICalibrationDiagnostics]:
    """Calibrate raw SVI parameters to observed total variance data.

    Args:
        k: Log-moneyness grid corresponding to the observed smile.
        total_variance: Observed total variance values on ``k``.
        maturity_years: Time to expiry in year fractions.
        config: Optional SVI calibration configuration overrides.
        bid_iv: Optional array of observed bid implied volatilities aligned with
            ``k`` used by the bid/ask envelope penalty.
        ask_iv: Optional array of observed ask implied volatilities aligned with
            ``k`` used by the bid/ask envelope penalty.
        volumes: Optional array of trade volumes aligned with ``k`` used for
            vega × volume weighting of residuals.

    Returns:
        A tuple containing the calibrated :class:`SVIParameters` and an
        :class:`SVICalibrationDiagnostics` instance summarising the optimisation.

    Raises:
        ValueError: If the inputs fail validation checks.
        CalculationError: If the optimisation fails or violates constraints.
    """

    options = merge_svi_options(config)

    diagnostic_pad = float(options.diagnostic_grid_pad)
    diagnostic_points = int(options.diagnostic_grid_points)
    butterfly_weight = float(options.butterfly_weight)
    callspread_weight = float(options.callspread_weight)
    envelope_weight = float(options.envelope_weight or 0.0)
    weighting_mode = str(options.weighting_mode or "vega+spread")
    weight_cap = float(options.weight_cap or 25.0)

    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if k.shape != total_variance.shape:
        raise ValueError("k and total_variance must have the same shape")
    if k.ndim != 1 or k.size < 5:
        raise ValueError("Need at least 5 strikes for SVI calibration")

    volumes_arr: np.ndarray | None
    if volumes is None:
        volumes_arr = None
    else:
        volumes_arr = np.asarray(volumes, dtype=float)
        if volumes_arr.shape != k.shape:
            raise ValueError("volumes must have the same shape as k")

    bid_iv_arr: np.ndarray | None
    if bid_iv is None:
        bid_iv_arr = None
    else:
        bid_iv_arr = np.asarray(bid_iv, dtype=float)
        if bid_iv_arr.shape != k.shape:
            raise ValueError("bid_iv must have the same shape as k")

    ask_iv_arr: np.ndarray | None
    if ask_iv is None:
        ask_iv_arr = None
    else:
        ask_iv_arr = np.asarray(ask_iv, dtype=float)
        if ask_iv_arr.shape != k.shape:
            raise ValueError("ask_iv must have the same shape as k")

    callspread_step = _adaptive_call_spread_step(k, maturity_years, options)
    huber_delta = _compute_huber_delta(
        total_variance,
        options,
        bid_iv=bid_iv_arr,
        ask_iv=ask_iv_arr,
        maturity_years=maturity_years,
    )

    heuristic_guess = _initial_guess(k, total_variance)
    bounds = _build_bounds(k, maturity_years, options)

    weights_array, volume_used, measurement_used = _vega_based_weights(
        k,
        total_variance,
        maturity_years,
        weighting_mode,
        weight_cap,
        volumes_arr,
        bid_iv_arr,
        ask_iv_arr,
    )
    weights = np.asarray(weights_array, dtype=float)
    if weights.shape != k.shape:
        raise ValueError("weights must have the same shape as k")
    sqrt_weights = np.sqrt(weights)

    k_min, k_max = float(np.min(k)), float(np.max(k))
    diagnostic_grid = np.linspace(
        k_min - diagnostic_pad, k_max + diagnostic_pad, diagnostic_points
    )
    callspread_context = _CallSpreadContext.build(
        diagnostic_grid, callspread_step, maturity_years
    )

    cache_vec: np.ndarray | None = None
    cache_value: float | None = None
    cache_grad: np.ndarray | None = None

    def _objective_and_grad(vec: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal cache_vec, cache_value, cache_grad

        vec_arr = np.asarray(vec, dtype=float)
        if cache_vec is not None and np.array_equal(vec_arr, cache_vec):
            assert cache_value is not None and cache_grad is not None
            return cache_value, cache_grad

        try:
            params = SVIParameters(*vec_arr)
        except ValueError:
            value = 1e9
            grad = np.zeros(5, dtype=float)
            cache_vec = vec_arr.copy()
            cache_value = float(value)
            cache_grad = grad.copy()
            return cache_value, cache_grad

        (
            _,
            w,
            dw,
            w_prime,
            dw_prime,
            w_second,
            dw_second,
        ) = _svi_total_variance_with_derivatives(k, params)

        residual = w - total_variance
        abs_residual = np.abs(residual)
        mask = abs_residual <= huber_delta
        quadratic = 0.5 * residual**2
        linear = huber_delta * (abs_residual - 0.5 * huber_delta)
        huber_loss = np.where(mask, quadratic, linear)
        base = float(np.sum(weights * huber_loss))

        huber_grad = weights * np.where(mask, residual, huber_delta * np.sign(residual))
        grad = np.sum(huber_grad[:, None] * dw, axis=0)
        value = base

        penalty_val, penalty_grad = _penalty_terms(
            vec_arr, options.regularisation, return_grad=True
        )
        value += penalty_val
        grad += penalty_grad

        butterfly_val, butterfly_grad = _butterfly_penalty(
            params, diagnostic_grid, butterfly_weight, return_grad=True
        )
        value += butterfly_val
        grad += butterfly_grad

        callspread_val, callspread_grad = _call_spread_penalty(
            params,
            callspread_context,
            callspread_weight,
            return_grad=True,
        )
        value += callspread_val
        grad += callspread_grad

        envelope_val = 0.0
        envelope_grad = np.zeros(5, dtype=float)
        if envelope_weight > 0.0 and (bid_iv_arr is not None or ask_iv_arr is not None):
            maturity_safe = max(maturity_years, 1e-12)
            variance_obs = w / maturity_safe
            sqrt_var = np.maximum(variance_obs, 0.0)
            model_iv = np.sqrt(np.maximum(sqrt_var, 1e-12))
            coeff = np.zeros_like(model_iv)
            positive = model_iv > 1e-12
            coeff[positive] = 0.5 / (model_iv[positive] * maturity_safe)
            d_model_iv = coeff[:, None] * dw

            if bid_iv_arr is not None:
                bid_arr = bid_iv_arr
                mask_bid = np.isfinite(bid_arr)
                if mask_bid.any():
                    diff = bid_arr[mask_bid] - model_iv[mask_bid]
                    active = diff > 0.0
                    if np.any(active):
                        delta = diff[active]
                        envelope_val += float(np.sum(delta**2))
                        envelope_grad += np.sum(
                            (-2.0 * delta)[:, None] * d_model_iv[mask_bid][active],
                            axis=0,
                        )

            if ask_iv_arr is not None:
                ask_arr = ask_iv_arr
                mask_ask = np.isfinite(ask_arr)
                if mask_ask.any():
                    diff = model_iv[mask_ask] - ask_arr[mask_ask]
                    active = diff > 0.0
                    if np.any(active):
                        delta = diff[active]
                        envelope_val += float(np.sum(delta**2))
                        envelope_grad += np.sum(
                            (2.0 * delta)[:, None] * d_model_iv[mask_ask][active],
                            axis=0,
                        )

        if envelope_val != 0.0:
            value += envelope_weight * envelope_val
            grad += envelope_weight * envelope_grad

        cache_vec = vec_arr.copy()
        cache_value = float(value)
        cache_grad = grad.copy()
        return cache_value, cache_grad

    def objective(vec: np.ndarray) -> float:
        val, _ = _objective_and_grad(vec)
        return float(val)

    def objective_grad(vec: np.ndarray) -> np.ndarray:
        _, grad = _objective_and_grad(vec)
        return grad

    global_solver = str(options.global_solver or "none").lower()
    polish_solver = str(options.polish_solver or "lbfgsb").lower()
    n_starts = int(options.n_starts or 0)
    random_seed = options.random_seed
    global_max_iter = int(options.global_max_iter or 50)

    qe_seeds = _qe_split_seeds(k, total_variance, sqrt_weights, heuristic_guess, bounds)

    diagnostics = SVICalibrationDiagnostics(
        status="pending",
        objective=float("nan"),
        iterations=0,
        message="",
        global_solver=global_solver,
        polish_solver=polish_solver,
        n_starts=n_starts,
        weighting_mode=weighting_mode,
        huber_delta=float(huber_delta),
        callspread_step=float(callspread_step),
        weights_min=float(weights.min()),
        weights_max=float(weights.max()),
        envelope_weight=float(envelope_weight),
        weights_volume_used=bool(volume_used),
        weights_measurement_used=bool(measurement_used),
        qe_seed_count=len(qe_seeds),
    )
    diagnostics.random_seed = random_seed if random_seed is not None else None

    logger.info(
        "Starting SVI calibration: n_strikes=%d, maturity_years=%.6f, weighting=%s, seed=%s",
        k.size,
        maturity_years,
        weighting_mode,
        str(random_seed),
    )

    rng = np.random.default_rng(random_seed)

    start_candidates: list[tuple[np.ndarray, str]] = [
        (np.asarray(heuristic_guess, dtype=float), "heuristic")
    ]
    for seed in qe_seeds:
        start_candidates.append((np.asarray(seed, dtype=float), "qe"))

    global_result: optimize.OptimizeResult | None = None

    if global_solver in {"de", "differential_evolution"}:
        try:
            global_result = optimize.differential_evolution(
                objective,
                bounds,
                maxiter=global_max_iter,
                seed=random_seed,
                polish=False,
            )
        except Exception as exc:  # pragma: no cover - SciPy backend failures
            diagnostics.global_status = f"failure: {exc}"
            logger.warning("Global solver failed: %s", exc)
        else:
            diagnostics.global_status = "success"
            diagnostics.global_objective = float(global_result.fun)
            diagnostics.global_iterations = int(getattr(global_result, "nit", 0))
            logger.info(
                "Global solver success: objective=%.3e iterations=%d",
                float(global_result.fun),
                int(getattr(global_result, "nit", 0)),
            )
            start_candidates.insert(
                0, (np.asarray(global_result.x, dtype=float), "global")
            )

    # Multi-start: sample additional initial points within bounds deterministically when seeded
    if n_starts > 0:
        lowers = np.array([bound[0] for bound in bounds], dtype=float)
        uppers = np.array([bound[1] for bound in bounds], dtype=float)
        span = uppers - lowers

        # Generate JW-informed directions for seeding
        try:
            jw_heuristic = raw_to_jw(SVIParameters(*heuristic_guess))
        except Exception:
            jw_heuristic = None

        for idx in range(n_starts):
            if jw_heuristic is not None and idx < len(JW_SEED_DIRECTIONS):
                direction = JW_SEED_DIRECTIONS[idx]
                tweaked_jw = tuple(
                    (
                        jw_heuristic[d] * (1 + direction[d])
                        if direction[d] != 0
                        else jw_heuristic[d]
                    )
                    for d in range(5)
                )
                try:
                    candidate = jw_to_raw(tweaked_jw)
                    sample = np.array(
                        [
                            candidate.a,
                            candidate.b,
                            candidate.rho,
                            candidate.m,
                            candidate.sigma,
                        ],
                        dtype=float,
                    )
                    origin = "jw"
                except Exception:
                    sample = lowers + span * rng.random(size=lowers.shape)
                    origin = "random"
            else:
                sample = lowers + span * rng.random(size=lowers.shape)
                origin = "random"
            start_candidates.append((np.asarray(sample, dtype=float), origin))

    # Deduplicate starts while preserving order
    unique_starts: list[tuple[np.ndarray, str]] = []
    for vec, origin in start_candidates:
        if any(np.allclose(vec, existing_vec) for existing_vec, _ in unique_starts):
            continue
        unique_starts.append((vec, origin))

    def _run_local(start_vec: np.ndarray) -> optimize.OptimizeResult:
        clipped = np.array(
            [np.clip(val, bound[0], bound[1]) for val, bound in zip(start_vec, bounds)],
            dtype=float,
        )
        if polish_solver == "nelder" or polish_solver == "nelder-mead":
            return optimize.minimize(
                objective,
                clipped,
                method="Nelder-Mead",
                options={
                    "maxiter": int(options.max_iter),
                    "fatol": float(options.tol),
                },
            )
        primary = optimize.minimize(
            objective,
            clipped,
            method="L-BFGS-B",
            bounds=bounds,
            jac=objective_grad,
            options={
                "maxiter": int(options.max_iter),
                "ftol": float(options.tol),
            },
        )
        if primary.success:
            return primary
        secondary = optimize.minimize(
            objective,
            clipped,
            method="Nelder-Mead",
            options={
                "maxiter": int(options.max_iter),
                "fatol": float(options.tol),
            },
        )
        return secondary

    best_result: optimize.OptimizeResult | None = None
    best_fun = float("inf")
    local_results: list[optimize.OptimizeResult] = []

    for start_index, (start_vec, origin) in enumerate(unique_starts):
        result = _run_local(start_vec)
        local_results.append(result)
        record = SVITrialRecord(
            start_index=start_index,
            start=tuple(float(x) for x in start_vec),
            success=bool(result.success),
            objective=float(result.fun),
            start_origin=origin,
            params=tuple(float(x) for x in result.x) if result.success else None,
        )
        diagnostics.add_trial_record(record)
        logger.debug(
            "Local optimiser start %d (%s): success=%s objective=%.3e",
            start_index,
            origin,
            result.success,
            float(result.fun),
        )
        if result.success and result.fun < best_fun:
            best_fun = float(result.fun)
            best_result = result
            diagnostics.chosen_start_index = start_index
            diagnostics.chosen_start_origin = origin

    if best_result is None:
        messages = [str(res.message) for res in local_results]
        logger.error("All local optimiser starts failed: %s", "; ".join(messages))
        raise CalculationError(
            "SVI calibration failed: no successful local optimisation (messages="
            + "; ".join(messages)
            + ")"
        )

    result = best_result

    diagnostics.status = "success" if result.success else "failure"
    diagnostics.objective = float(result.fun)
    diagnostics.iterations = int(getattr(result, "nit", 0))
    diagnostics.message = str(result.message)

    if not result.success:
        raise CalculationError(f"SVI calibration failed: {result.message}")

    params_vec = result.x
    params = SVIParameters(*params_vec)

    k_grid = diagnostic_grid
    min_g = min_g_on_grid(params, k_grid)
    diagnostics.min_g = min_g
    if min_g < -1e-6:
        diagnostics.status = "warning"
        diagnostics.butterfly_warning = float(min_g)
        logger.warning("SVI calibration butterfly violation: min_g=%.3e", min_g)
        warnings.warn(
            f"SVI calibration violates butterfly condition: min_g={min_g:.3e}",
            UserWarning,
        )

    final_model = svi_total_variance(k, params)
    residual_final = final_model - total_variance
    diagnostics.rmse_unweighted = float(np.sqrt(np.mean(residual_final**2)))
    diagnostics.rmse_weighted = float(
        np.sqrt(np.mean((np.sqrt(weights) * residual_final) ** 2))
    )
    diagnostics.residual_mean = float(np.mean(residual_final))
    final_iv = from_total_variance(np.maximum(final_model, 0.0), maturity_years)
    if bid_iv_arr is not None or ask_iv_arr is not None:
        lower_valid = np.zeros(k.shape, dtype=bool)
        upper_valid = np.zeros(k.shape, dtype=bool)
        if bid_iv_arr is not None:
            lower_valid = np.isfinite(bid_iv_arr)
        if ask_iv_arr is not None:
            upper_valid = np.isfinite(ask_iv_arr)
        valid_mask = lower_valid | upper_valid
        violations = np.zeros(k.shape, dtype=bool)
        if bid_iv_arr is not None:
            violations |= (final_iv < bid_iv_arr) & lower_valid
        if ask_iv_arr is not None:
            violations |= (final_iv > ask_iv_arr) & upper_valid
        total_points = int(np.count_nonzero(valid_mask))
        if total_points > 0:
            diagnostics.envelope_violations_pct = (
                float(np.count_nonzero(violations)) / float(total_points) * 100.0
            )
        else:
            diagnostics.envelope_violations_pct = 0.0
    else:
        diagnostics.envelope_violations_pct = 0.0

    if diagnostics.status != "warning":
        diagnostics.status = "success"
    logger.info(
        "SVI calibration complete: status=%s, objective=%.3e, min_g=%.3e, seed=%s",
        diagnostics.status,
        diagnostics.objective,
        diagnostics.min_g if diagnostics.min_g is not None else float("nan"),
        str(diagnostics.random_seed),
    )
    return params, diagnostics


__all__ = [
    "DEFAULT_SVI_OPTIONS",
    "merge_svi_options",
    "svi_options",
    "RawSVI",
    "JWParams",
    "raw_to_jw",
    "jw_to_raw",
    "SVIParameters",
    "log_moneyness",
    "to_total_variance",
    "from_total_variance",
    "svi_total_variance",
    "svi_first_derivative",
    "svi_second_derivative",
    "g_function",
    "check_butterfly",
    "min_g_on_grid",
    "svi_minimum_location",
    "svi_minimum_variance",
    "calibrate_svi_parameters",
    "SVICalibrationOptions",
    "SVICalibrationDiagnostics",
    "SVITrialRecord",
]
@dataclass(frozen=True)
class _CallSpreadContext:
    """Pre-computed constants used by the call-spread penalty."""

    combined_k: np.ndarray
    left_slice: slice
    centre_slice: slice
    right_slice: slice
    strikes: np.ndarray
    log_fk: np.ndarray
    sqrt_t: float
    maturity: float

    @classmethod
    def build(
        cls, k_grid: np.ndarray, step: float, maturity_years: float
    ) -> _CallSpreadContext | None:
        if step <= 0.0 or maturity_years <= 0.0:
            return None

        size = k_grid.shape[0]
        k_minus = k_grid - step
        k_plus = k_grid + step
        combined_k = np.concatenate((k_minus, k_grid, k_plus))

        strikes = np.exp(combined_k)
        log_fk = -combined_k
        sqrt_t = np.sqrt(max(maturity_years, 1e-12))

        return cls(
            combined_k=combined_k,
            left_slice=slice(0, size),
            centre_slice=slice(size, 2 * size),
            right_slice=slice(2 * size, 3 * size),
            strikes=strikes,
            log_fk=log_fk,
            sqrt_t=sqrt_t,
            maturity=max(maturity_years, 1e-12),
        )
