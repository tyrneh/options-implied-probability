"""Analytical helpers for Stochastic Volatility Inspired (SVI) smiles.

This module contains parameter containers and closed-form utilities for the
single-expiry SVI model following Gatheral & Jacquier (2012).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import warnings

import numpy as np
from scipy import optimize

from oipd.core.errors import CalculationError
from oipd.pricing.black76 import black76_call_price

DEFAULT_SVI_OPTIONS: Dict[str, Any] = {
    "max_iter": 200,
    "tol": 1e-6,
    "regularisation": 1e-4,
    "rho_bound": 0.999,
    "sigma_min": 1e-4,
    "diagnostic_grid_pad": 0.5,
    "diagnostic_grid_points": 201,
    "butterfly_weight": 1e4,
    "callspread_weight": 1e3,
    "callspread_step": 0.05,
    "global_solver": "de",
    "global_max_iter": 20,
    "polish_solver": "lbfgsb",
    "n_starts": 5,
    "random_seed": 1,
    "weighting_mode": "vega",
    "weight_cap": 25.0,
    "huber_delta": 1e-3,
    "envelope_weight": 1e3,
    "volume_column": None,
}

RawSVI = Tuple[float, float, float, float, float]
JWParams = Tuple[float, float, float, float, float]

JW_SEED_DIRECTIONS: Tuple[Tuple[float, float, float, float, float], ...] = (
    (0.0, -0.2, -0.2, 0.3, 0.0),  # flatter wings
    (0.0, 0.3, 0.3, -0.2, 0.0),  # steeper wings
    (0.1, 0.0, 0.0, 0.0, 0.1),  # higher ATM level
    (-0.1, 0.0, 0.0, 0.0, -0.1),  # lower ATM level
    (0.0, 0.0, 0.2, -0.3, 0.0),  # skew heavier
)

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

    k_arr = np.asarray(k, dtype=float)
    diff = k_arr - params.m
    root = np.sqrt(np.square(diff) + params.sigma**2)
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

    k_arr = np.asarray(k, dtype=float)
    diff = k_arr - params.m
    denom = np.sqrt(np.square(diff) + params.sigma**2)
    return params.b * (params.rho + diff / denom)


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

    k_arr = np.asarray(k, dtype=float)
    diff = k_arr - params.m
    denom = np.sqrt(np.square(diff) + params.sigma**2)
    return params.b * params.sigma**2 / np.power(denom, 3)


def g_function(k: np.ndarray | Iterable[float], params: SVIParameters) -> np.ndarray:
    """Evaluate the Gatheral–Jacquier butterfly condition diagnostic ``g(k)``.

    The density is non-negative iff ``g(k) >= 0`` for all values of ``k``.

    Args:
        k: Log-moneyness coordinates for the diagnostic evaluation.
        params: Calibrated SVI parameters.

    Returns:
        Diagnostic values capturing the no-arbitrage butterfly constraint.
    """

    k_arr = np.asarray(k, dtype=float)
    w = svi_total_variance(k_arr, params)
    wp = svi_first_derivative(k_arr, params)
    wpp = svi_second_derivative(k_arr, params)

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

    total_variance = svi_total_variance(k, params)
    implied_variance = np.maximum(total_variance / maturity_years, 0.0)
    implied_vol = np.sqrt(np.maximum(implied_variance, 1e-12))
    strikes = np.exp(k)
    return black76_call_price(1.0, strikes, implied_vol, maturity_years, 0.0)


def _butterfly_penalty(
    params: SVIParameters, k_grid: np.ndarray, weight: float
) -> float:
    """Return a hinge penalty for butterfly arbitrage violations on ``k_grid``.

    Args:
        params: Candidate SVI parameters.
        k_grid: Diagnostic log-moneyness grid.
        weight: Penalty weight applied to squared violations.

    Returns:
        Penalty value; zero when no violation is detected.
    """

    if weight <= 0.0:
        return 0.0

    g_vals = g_function(k_grid, params)
    if np.any(~np.isfinite(g_vals)):
        return weight * 1e6

    violations = g_vals[g_vals < 0.0]
    if violations.size == 0:
        return 0.0
    return weight * float(np.sum(np.square(violations)))


def _call_spread_penalty(
    params: SVIParameters,
    k_grid: np.ndarray,
    maturity_years: float,
    step: float,
    weight: float,
) -> float:
    """Return hinge penalties enforcing positive call spreads on ``k_grid``.

    Args:
        params: Candidate SVI parameters.
        k_grid: Diagnostic log-moneyness grid.
        maturity_years: Time to expiry in years.
        step: Shift in log-moneyness applied to construct spreads.
        weight: Penalty weight applied to squared violations.

    Returns:
        Penalty value capturing violations of monotonicity in strike.
    """

    if weight <= 0.0 or step <= 0.0:
        return 0.0

    k_minus = k_grid - step
    k_plus = k_grid + step

    prices_central = _call_prices_from_params(k_grid, params, maturity_years)
    prices_left = _call_prices_from_params(k_minus, params, maturity_years)
    prices_right = _call_prices_from_params(k_plus, params, maturity_years)

    left_spread = prices_left - prices_central
    right_spread = prices_central - prices_right

    violations = np.concatenate(
        [left_spread[left_spread < 0.0], right_spread[right_spread < 0.0]]
    )
    if violations.size == 0:
        return 0.0
    return weight * float(np.sum(np.square(violations)))


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
) -> tuple[np.ndarray, bool]:
    """Construct weighting vector for the calibration objective."""

    size = k.shape[0]
    if size == 0 or maturity_years <= 0:
        return np.ones(size, dtype=float), False

    mode_normalised = mode.lower()
    if mode_normalised in {"none", "off", "disabled"}:
        return np.ones(size, dtype=float), False

    total_var_arr = np.asarray(total_variance, dtype=float)
    total_var_arr = np.maximum(total_var_arr, 1e-12)
    iv = np.sqrt(total_var_arr / maturity_years)
    sqrt_t = np.sqrt(max(maturity_years, 1e-12))

    if not np.isfinite(sqrt_t):
        return np.ones(size, dtype=float)

    strikes = np.exp(np.asarray(k, dtype=float))
    ln_fk = -np.log(np.maximum(strikes, 1e-12))
    denom = np.maximum(iv * sqrt_t, 1e-12)
    d1 = (ln_fk + 0.5 * (iv**2) * maturity_years) / denom
    phi = np.exp(-0.5 * np.square(d1)) / np.sqrt(2 * np.pi)
    vega = sqrt_t * phi

    weights = np.maximum(vega, 1e-8)
    volume_used = False

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

    mean_weight = float(np.mean(weights))
    if mean_weight > 0:
        weights = weights / mean_weight

    if np.isfinite(weight_cap) and weight_cap > 0:
        weights = np.clip(weights, 1e-6, weight_cap)

    return weights, volume_used


def svi_options(**overrides: Any) -> Dict[str, Any]:
    """Return SVI calibration options with the provided overrides applied.

    Args:
        **overrides: Keyword arguments overriding default option values.

    Returns:
        A dictionary containing the merged calibration options.

    Raises:
        TypeError: If an unknown option key is supplied.
    """

    return merge_svi_options(overrides)


def merge_svi_options(overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Merge default SVI options with user-provided overrides.

    Args:
        overrides: Mapping whose keys correspond to recognised SVI options.

    Returns:
        A dictionary containing the merged calibration options.

    Raises:
        TypeError: If ``overrides`` contains keys that are not recognised.
    """

    if overrides is None:
        return dict(DEFAULT_SVI_OPTIONS)

    unknown = set(overrides) - _SVI_OPTION_KEYS
    if unknown:
        raise TypeError(f"Unknown SVI option(s): {sorted(unknown)}")

    merged = dict(DEFAULT_SVI_OPTIONS)
    merged.update({key: overrides[key] for key in overrides})
    return merged


def _build_bounds(
    k: np.ndarray, maturity_years: float, config: Mapping[str, Any]
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

    rho_bound = float(config["rho_bound"])
    sigma_min = float(config["sigma_min"])
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


def _penalty_terms(params_vec: np.ndarray, config: Mapping[str, Any]) -> float:
    """Compute regularisation penalties for the optimisation objective.

    Args:
        params_vec: Candidate SVI parameter vector ``(a, b, rho, m, sigma)``.
        config: Configuration dictionary containing penalty multipliers.

    Returns:
        Scalar penalty value added to the least-squares objective.
    """
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
    config: Mapping[str, Any] | None,
    *,
    bid_iv: np.ndarray | None = None,
    ask_iv: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
) -> Tuple[SVIParameters, Dict[str, Any]]:
    """Calibrate raw SVI parameters to observed total variance data.

    Args:
        k: Log-moneyness grid corresponding to the observed smile.
        total_variance: Observed total variance values on ``k``.
        maturity_years: Time to expiry in year fractions.
        config: Optional configuration dictionary overriding defaults.
        bid_iv: Optional array of observed bid implied volatilities aligned with
            ``k`` used by the bid/ask envelope penalty.
        ask_iv: Optional array of observed ask implied volatilities aligned with
            ``k`` used by the bid/ask envelope penalty.
        volumes: Optional array of trade volumes aligned with ``k`` used for
            vega × volume weighting of residuals.

    Returns:
        A tuple containing the calibrated :class:`SVIParameters` and a
        diagnostics dictionary summarising the optimisation.

    Raises:
        ValueError: If the inputs fail validation checks.
        CalculationError: If the optimisation fails or violates constraints.
    """

    options = merge_svi_options(config)

    diagnostic_pad = float(options["diagnostic_grid_pad"])
    diagnostic_points = int(options["diagnostic_grid_points"])
    butterfly_weight = float(options["butterfly_weight"])
    callspread_weight = float(options["callspread_weight"])
    callspread_step = float(options["callspread_step"])
    envelope_weight = float(options.get("envelope_weight", 0.0))
    weighting_mode = str(options.get("weighting_mode", "vega") or "vega")
    weight_cap = float(options.get("weight_cap", 25.0) or 25.0)
    huber_delta = float(options.get("huber_delta", 1e-3) or 1e-3)

    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if k.shape != total_variance.shape:
        raise ValueError("k and total_variance must have the same shape")
    if k.ndim != 1 or k.size < 5:
        raise ValueError("Need at least 5 strikes for SVI calibration")

    heuristic_guess = _initial_guess(k, total_variance)
    bounds = _build_bounds(k, maturity_years, options)

    volumes_arr: np.ndarray | None
    if volumes is None:
        volumes_arr = None
    else:
        volumes_arr = np.asarray(volumes, dtype=float)
        if volumes_arr.shape != k.shape:
            raise ValueError("volumes must have the same shape as k")

    weights_array, volume_used = _vega_based_weights(
        k, total_variance, maturity_years, weighting_mode, weight_cap, volumes_arr
    )
    weights = np.asarray(weights_array, dtype=float)
    if weights.shape != k.shape:
        raise ValueError("weights must have the same shape as k")
    sqrt_weights = np.sqrt(weights)

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

    k_min, k_max = float(np.min(k)), float(np.max(k))
    diagnostic_grid = np.linspace(
        k_min - diagnostic_pad, k_max + diagnostic_pad, diagnostic_points
    )

    def objective(vec: np.ndarray) -> float:
        """Evaluate the penalised least-squares objective for calibration.

        Args:
            vec: Candidate SVI parameter vector.

        Returns:
            Scalar objective value combining residuals and regularisation.
        """

        try:
            params = SVIParameters(*vec)
        except ValueError:
            return 1e9
        model = svi_total_variance(k, params)
        residual = model - total_variance
        abs_residual = np.abs(residual)
        mask = abs_residual <= huber_delta
        quadratic = 0.5 * residual**2
        linear = huber_delta * (abs_residual - 0.5 * huber_delta)
        huber_loss = np.where(mask, quadratic, linear)
        base = float(np.sum(weights * huber_loss))
        model_iv = from_total_variance(np.maximum(model, 0.0), maturity_years)
        penalty = _penalty_terms(vec, options)
        penalty += _butterfly_penalty(params, diagnostic_grid, butterfly_weight)
        penalty += _call_spread_penalty(
            params, diagnostic_grid, maturity_years, callspread_step, callspread_weight
        )
        penalty += _bid_ask_penalty(model_iv, bid_iv_arr, ask_iv_arr, envelope_weight)
        return base + penalty

    global_solver = str(options.get("global_solver") or "none").lower()
    polish_solver = str(options.get("polish_solver") or "lbfgsb").lower()
    n_starts = int(options.get("n_starts", 0) or 0)
    random_seed = options.get("random_seed")
    global_max_iter = int(options.get("global_max_iter", 50) or 50)

    diagnostics: Dict[str, Any] = {
        "status": "pending",
        "objective": float("nan"),
        "iterations": 0,
        "message": "",
        "global_solver": global_solver,
        "polish_solver": polish_solver,
        "n_starts": n_starts,
        "weighting_mode": weighting_mode,
        "huber_delta": huber_delta,
        "weights_min": float(weights.min()),
        "weights_max": float(weights.max()),
        "envelope_weight": envelope_weight,
        "weights_volume_used": bool(volume_used),
    }

    rng = np.random.default_rng(random_seed)

    start_vectors: list[np.ndarray] = [np.asarray(heuristic_guess, dtype=float)]
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
            diagnostics["global_status"] = f"failure: {exc}"  # type: ignore[index]
        else:
            diagnostics["global_status"] = "success"  # type: ignore[index]
            diagnostics["global_objective"] = float(global_result.fun)  # type: ignore[index]
            diagnostics["global_iterations"] = int(getattr(global_result, "nit", 0))  # type: ignore[index]
            start_vectors.insert(0, np.asarray(global_result.x, dtype=float))

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
                except Exception:
                    sample = lowers + span * rng.random(size=lowers.shape)
            else:
                sample = lowers + span * rng.random(size=lowers.shape)
            start_vectors.append(sample)

    # Deduplicate starts while preserving order
    unique_starts: list[np.ndarray] = []
    for vec in start_vectors:
        if not any(np.allclose(vec, existing) for existing in unique_starts):
            unique_starts.append(vec)

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
                    "maxiter": int(options["max_iter"]),
                    "fatol": float(options["tol"]),
                },
            )
        primary = optimize.minimize(
            objective,
            clipped,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(options["max_iter"]),
                "ftol": float(options["tol"]),
            },
        )
        if primary.success:
            return primary
        secondary = optimize.minimize(
            objective,
            clipped,
            method="Nelder-Mead",
            options={
                "maxiter": int(options["max_iter"]),
                "fatol": float(options["tol"]),
            },
        )
        return secondary

    best_result: optimize.OptimizeResult | None = None
    best_fun = float("inf")
    local_results: list[optimize.OptimizeResult] = []
    trial_records: list[Dict[str, Any]] = []

    for start_index, start_vec in enumerate(unique_starts):
        result = _run_local(start_vec)
        local_results.append(result)
        record: Dict[str, Any] = {
            "start_index": start_index,
            "start": [float(x) for x in start_vec],
            "success": bool(result.success),
            "objective": float(result.fun),
        }
        if result.success:
            record["params"] = [float(x) for x in result.x]
        trial_records.append(record)
        if result.success and result.fun < best_fun:
            best_fun = float(result.fun)
            best_result = result
            diagnostics["chosen_start_index"] = start_index  # type: ignore[index]

    if best_result is None:
        messages = [str(res.message) for res in local_results]
        raise CalculationError(
            "SVI calibration failed: no successful local optimisation (messages="
            + "; ".join(messages)
            + ")"
        )

    result = best_result

    diagnostics["status"] = "success" if result.success else "failure"
    diagnostics["objective"] = float(result.fun)
    diagnostics["iterations"] = int(getattr(result, "nit", 0))
    diagnostics["message"] = str(result.message)
    diagnostics["trial_records"] = trial_records

    if not result.success:
        raise CalculationError(f"SVI calibration failed: {result.message}")

    params_vec = result.x
    params = SVIParameters(*params_vec)

    k_grid = diagnostic_grid
    min_g = min_g_on_grid(params, k_grid)
    diagnostics["min_g"] = min_g
    if min_g < -1e-6:
        diagnostics["status"] = "warning"
        diagnostics["butterfly_warning"] = float(min_g)
        warnings.warn(
            f"SVI calibration violates butterfly condition: min_g={min_g:.3e}",
            UserWarning,
        )

    final_model = svi_total_variance(k, params)
    residual_final = final_model - total_variance
    diagnostics["rmse_unweighted"] = float(np.sqrt(np.mean(residual_final**2)))
    diagnostics["rmse_weighted"] = float(
        np.sqrt(np.mean((np.sqrt(weights) * residual_final) ** 2))
    )
    diagnostics["residual_mean"] = float(np.mean(residual_final))
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
            diagnostics["envelope_violations_pct"] = (
                float(np.count_nonzero(violations)) / float(total_points) * 100.0
            )
        else:
            diagnostics["envelope_violations_pct"] = 0.0
    else:
        diagnostics["envelope_violations_pct"] = 0.0

    if diagnostics.get("status") != "warning":
        diagnostics["status"] = "success"
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
    "min_g_on_grid",
    "svi_minimum_location",
    "svi_minimum_variance",
    "calibrate_svi_parameters",
]
