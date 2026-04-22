"""Risk-neutral density utilities (Breeden–Litzenberger pipeline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from oipd.core.errors import InvalidInputError
from .finite_diff import finite_diff_first_derivative, finite_diff_second_derivative

CDF_EPSILON = 1e-5
CDF_UPPER_TAIL_CLIP_TOLERANCE = 1e-3
MONOTONICITY_EPSILON = 1e-6
NEAR_ZERO_STRIKE_ABS = 0.01
NEAR_ZERO_STRIKE_REL = 1e-6


@dataclass(frozen=True)
class DirectCdfResult:
    """Container for canonical direct-CDF output and diagnostics.

    Attributes:
        prices: Strike grid aligned with all CDF arrays.
        cdf_values: Minimally cleaned direct first-derivative CDF values.
        raw_cdf_values: Unrepaired direct first-derivative CDF values.
        diagnostics: Scalar numerical diagnostics and cleanup flags.
    """

    prices: np.ndarray
    cdf_values: np.ndarray
    raw_cdf_values: np.ndarray
    diagnostics: dict[str, float | int | bool | str]

    def __post_init__(self) -> None:
        """Copy arrays and diagnostics into the frozen result."""
        object.__setattr__(
            self, "prices", np.array(self.prices, dtype=float, copy=True)
        )
        object.__setattr__(
            self,
            "cdf_values",
            np.array(self.cdf_values, dtype=float, copy=True),
        )
        object.__setattr__(
            self,
            "raw_cdf_values",
            np.array(self.raw_cdf_values, dtype=float, copy=True),
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _resolve_time_to_expiry_years(
    *,
    time_to_expiry_years: Optional[float],
) -> float:
    """Validate one explicit year-fraction maturity value.

    Args:
        time_to_expiry_years: Canonical year-fraction maturity input.

    Returns:
        float: Finite maturity in year fractions.

    Raises:
        InvalidInputError: If no maturity is provided or the resolved value is
            not finite.
    """
    if time_to_expiry_years is None:
        raise InvalidInputError("time_to_expiry_years is required.")

    years = float(time_to_expiry_years)

    if not np.isfinite(years):
        raise InvalidInputError("time_to_expiry_years must be finite.")

    return years


def pdf_from_price_curve(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    *,
    risk_free_rate: float,
    time_to_expiry_years: Optional[float] = None,
    min_strike: float | None = None,
    max_strike: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Breeden-Litzenberger to obtain a PDF from call prices.

    Args:
        strikes: Strike grid for option prices.
        call_prices: Call prices aligned with ``strikes``.
        risk_free_rate: Continuously compounded risk-free rate.
        time_to_expiry_years: Explicit maturity in year fractions.
        min_strike: Optional left strike cutoff.
        max_strike: Optional right strike cutoff.

    Returns:
        Tuple containing filtered strikes and non-negative PDF values.

    Raises:
        InvalidInputError: If array shapes are invalid or no maturity input
            is provided.
    """

    strikes_arr = np.asarray(strikes, dtype=float)
    prices_arr = np.asarray(call_prices, dtype=float)
    if strikes_arr.shape != prices_arr.shape:
        raise InvalidInputError("Strikes and prices must have the same shape")

    second_derivative = finite_diff_second_derivative(prices_arr, strikes_arr)
    years = _resolve_time_to_expiry_years(
        time_to_expiry_years=time_to_expiry_years,
    )
    pdf = np.exp(risk_free_rate * years) * second_derivative
    pdf = np.maximum(pdf, 0.0)

    if min_strike is not None or max_strike is not None:
        left = 0
        right = len(strikes_arr) - 1
        if min_strike is not None:
            while left < len(strikes_arr) and strikes_arr[left] < min_strike:
                left += 1
        if max_strike is not None:
            while right >= 0 and strikes_arr[right] > max_strike:
                right -= 1
        strikes_arr = strikes_arr[left : right + 1]
        pdf = pdf[left : right + 1]

    return strikes_arr, pdf


def _resolve_near_zero_strike_threshold(
    strikes: np.ndarray,
    *,
    reference_price: float | None,
) -> float:
    """Resolve the strike threshold treated as effectively zero.

    Args:
        strikes: Strike grid used for the CDF estimate.
        reference_price: Optional spot or forward used to scale the relative
            near-zero tolerance.

    Returns:
        float: Absolute strike threshold below which the left endpoint may be
        snapped to zero for tiny numerical noise.
    """
    if reference_price is not None and np.isfinite(reference_price):
        reference = abs(float(reference_price))
    else:
        reference = float(np.max(np.abs(strikes)))

    return max(NEAR_ZERO_STRIKE_ABS, NEAR_ZERO_STRIKE_REL * reference)


def _fail_if_direct_cdf_violates_thresholds(
    cdf_values: np.ndarray,
    *,
    diagnostics: dict[str, float | int | bool],
) -> None:
    """Reject raw direct CDF values with meaningful numerical violations.

    Args:
        cdf_values: Raw direct first-derivative CDF values.
        diagnostics: Scalar diagnostics computed from ``cdf_values``.

    Raises:
        InvalidInputError: If the raw CDF is non-finite, materially below
            zero, materially non-monotone, or materially above one beyond the
            documented small upper-tail clipping tolerance.
    """
    if not np.all(np.isfinite(cdf_values)):
        raise InvalidInputError("Direct CDF contains non-finite values.")
    if float(diagnostics["min"]) < -CDF_EPSILON:
        raise InvalidInputError("Direct CDF is materially below zero.")
    if float(diagnostics["min_step"]) < -MONOTONICITY_EPSILON:
        raise InvalidInputError("Direct CDF is materially non-monotone.")
    if float(diagnostics["max"]) > 1.0 + CDF_UPPER_TAIL_CLIP_TOLERANCE:
        raise InvalidInputError("Direct CDF is materially above one.")


def _minimal_direct_cdf_cleanup(
    strikes: np.ndarray,
    raw_cdf_values: np.ndarray,
    *,
    reference_price: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Apply bounded numerical cleanup to a raw direct CDF.

    Args:
        strikes: Strike grid aligned with ``raw_cdf_values``.
        raw_cdf_values: Unrepaired direct first-derivative CDF values.
        reference_price: Optional spot or forward used to decide whether the
            left strike is effectively zero.

    Returns:
        tuple[np.ndarray, dict[str, float | int | bool | str]]: Cleaned CDF
        values plus diagnostics and cleanup flags. Cleanup is limited to tiny
        endpoint dust and explicit clipping of finite, monotone upper-tail
        overshoots no larger than ``CDF_UPPER_TAIL_CLIP_TOLERANCE``.

    Raises:
        InvalidInputError: If the raw CDF has meaningful bound, finiteness, or
            monotonicity violations.
    """
    values = np.asarray(raw_cdf_values, dtype=float)
    diagnostics = direct_cdf_diagnostics(
        values,
        tolerance=MONOTONICITY_EPSILON,
    )
    _fail_if_direct_cdf_violates_thresholds(values, diagnostics=diagnostics)

    cleaned = values.copy()
    near_zero_threshold = _resolve_near_zero_strike_threshold(
        np.asarray(strikes, dtype=float),
        reference_price=reference_price,
    )
    is_left_near_zero = bool(float(strikes[0]) <= near_zero_threshold)

    left_endpoint_snapped = False
    if is_left_near_zero and abs(float(cleaned[0])) <= CDF_EPSILON:
        cleaned[0] = 0.0
        left_endpoint_snapped = True

    lower_clip_mask = cleaned < 0.0
    upper_clip_mask = cleaned > 1.0
    lower_clip_count = int(np.count_nonzero(lower_clip_mask))
    upper_clip_count = int(np.count_nonzero(upper_clip_mask))
    upper_tail_max_excess = max(0.0, float(diagnostics["max"]) - 1.0)
    cleaned[lower_clip_mask] = 0.0
    cleaned[upper_clip_mask] = 1.0

    right_endpoint_snapped = False
    if abs(float(cleaned[-1]) - 1.0) <= CDF_EPSILON:
        cleaned[-1] = 1.0
        right_endpoint_snapped = True

    cleanup_diagnostics: dict[str, float | int | bool | str] = {
        "cdf_cleanup_policy": "minimal_epsilon_cleanup",
        "cdf_left_endpoint_snapped": left_endpoint_snapped,
        "cdf_right_endpoint_snapped": right_endpoint_snapped,
        "cdf_lower_clip_count": lower_clip_count,
        "cdf_upper_clip_count": upper_clip_count,
        "cdf_upper_tail_clip_policy": "clip_finite_monotone_small_overshoot",
        "cdf_upper_tail_clip_applied": bool(upper_clip_count > 0),
        "cdf_upper_tail_clip_tolerance": CDF_UPPER_TAIL_CLIP_TOLERANCE,
        "cdf_upper_tail_max_excess": upper_tail_max_excess,
        "cdf_upper_tail_clip_count": upper_clip_count,
        "cdf_near_zero_strike_threshold": near_zero_threshold,
        "raw_cdf_start": diagnostics["start"],
        "raw_cdf_end": diagnostics["end"],
        "raw_cdf_min": diagnostics["min"],
        "raw_cdf_max": diagnostics["max"],
        "raw_cdf_is_monotone": diagnostics["is_monotone"],
        "raw_cdf_negative_step_count": diagnostics["negative_step_count"],
        "raw_cdf_max_negative_step": diagnostics["max_negative_step"],
        "raw_cdf_below_zero_count": diagnostics["below_zero_count"],
        "raw_cdf_above_one_count": diagnostics["above_one_count"],
        "raw_cdf_points": diagnostics["points"],
    }
    return cleaned, cleanup_diagnostics


def raw_cdf_from_price_curve(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    *,
    risk_free_rate: float,
    time_to_expiry_years: Optional[float] = None,
    min_strike: float | None = None,
    max_strike: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the direct first-derivative CDF formula without repairs.

    Args:
        strikes: Strike grid for option prices.
        call_prices: Call prices aligned with ``strikes``.
        risk_free_rate: Continuously compounded risk-free rate.
        time_to_expiry_years: Explicit maturity in year fractions.
        min_strike: Optional left strike cutoff.
        max_strike: Optional right strike cutoff.

    Returns:
        Tuple containing filtered strikes and raw CDF values. The CDF values
        are not clipped, monotonicized, or normalized.

    Raises:
        InvalidInputError: If array shapes are invalid or no maturity input
            is provided.
    """
    strikes_arr = np.asarray(strikes, dtype=float)
    prices_arr = np.asarray(call_prices, dtype=float)
    if strikes_arr.shape != prices_arr.shape:
        raise InvalidInputError("Strikes and prices must have the same shape")

    years = _resolve_time_to_expiry_years(
        time_to_expiry_years=time_to_expiry_years,
    )
    first_derivative = finite_diff_first_derivative(prices_arr, strikes_arr)
    cdf_values = 1.0 + np.exp(risk_free_rate * years) * first_derivative

    if min_strike is not None or max_strike is not None:
        left = 0
        right = len(strikes_arr) - 1
        if min_strike is not None:
            while left < len(strikes_arr) and strikes_arr[left] < min_strike:
                left += 1
        if max_strike is not None:
            while right >= 0 and strikes_arr[right] > max_strike:
                right -= 1
        strikes_arr = strikes_arr[left : right + 1]
        cdf_values = cdf_values[left : right + 1]

    return strikes_arr, cdf_values


def cdf_from_price_curve(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    *,
    risk_free_rate: float,
    time_to_expiry_years: Optional[float] = None,
    min_strike: float | None = None,
    max_strike: float | None = None,
    reference_price: float | None = None,
) -> DirectCdfResult:
    """Apply Breeden-Litzenberger to obtain the canonical direct CDF.

    Args:
        strikes: Strike grid for option prices.
        call_prices: Call prices aligned with ``strikes``.
        risk_free_rate: Continuously compounded risk-free rate.
        time_to_expiry_years: Explicit maturity in year fractions.
        min_strike: Optional left strike cutoff.
        max_strike: Optional right strike cutoff.
        reference_price: Optional spot or forward used for near-zero endpoint
            cleanup.

    Returns:
        DirectCdfResult: Filtered strikes, raw CDF values, minimally cleaned
        CDF values, and diagnostics.

    Raises:
        InvalidInputError: If input arrays are invalid or the raw direct CDF
            has meaningful numerical violations.
    """
    cdf_prices, raw_cdf_values = raw_cdf_from_price_curve(
        strikes,
        call_prices,
        risk_free_rate=risk_free_rate,
        time_to_expiry_years=time_to_expiry_years,
        min_strike=min_strike,
        max_strike=max_strike,
    )
    cdf_values, diagnostics = _minimal_direct_cdf_cleanup(
        cdf_prices,
        raw_cdf_values,
        reference_price=reference_price,
    )
    return DirectCdfResult(
        prices=cdf_prices,
        cdf_values=cdf_values,
        raw_cdf_values=raw_cdf_values,
        diagnostics=diagnostics,
    )


def direct_cdf_diagnostics(
    cdf_values: np.ndarray,
    *,
    tolerance: float = MONOTONICITY_EPSILON,
) -> dict[str, float | int | bool]:
    """Summarize raw or repaired direct-CDF numerical health.

    Args:
        cdf_values: CDF values ordered by increasing strike.
        tolerance: Allowed monotonicity tolerance before a step is counted as
            a decrease.

    Returns:
        dict[str, float | int | bool]: Scalar diagnostics for endpoints,
        bounds, and monotonicity.
    """
    values = np.asarray(cdf_values, dtype=float)
    if values.size == 0:
        raise InvalidInputError("CDF values cannot be empty.")

    steps = np.diff(values)
    negative_steps = steps[steps < -float(tolerance)]
    min_step = float(np.min(steps)) if steps.size else 0.0
    max_negative_step = min_step if min_step < 0.0 else 0.0

    return {
        "start": float(values[0]),
        "end": float(values[-1]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "span": float(values[-1] - values[0]),
        "is_monotone": bool(np.all(steps >= -float(tolerance))),
        "negative_step_count": int(negative_steps.size),
        "max_negative_step": float(max_negative_step),
        "min_step": min_step,
        "below_zero_count": int(np.count_nonzero(values < -float(tolerance))),
        "above_one_count": int(np.count_nonzero(values > 1.0 + float(tolerance))),
        "points": int(values.size),
    }


def calculate_quartiles(
    cdf_point_arrays: Tuple[np.ndarray, np.ndarray],
) -> dict[float, float]:
    """Compute quartiles from a CDF curve.

    Args:
        cdf_point_arrays: Tuple containing the ordered price grid and CDF
            values aligned with that grid.

    Returns:
        dict[float, float]: Quartile probabilities mapped to price levels.
    """

    x_array, cdf_values = cdf_point_arrays
    cdf_interpolated = interp1d(x_array, cdf_values)
    x_start, x_end = x_array[0], x_array[-1]
    return {
        0.25: brentq(lambda x: cdf_interpolated(x) - 0.25, x_start, x_end),
        0.5: brentq(lambda x: cdf_interpolated(x) - 0.5, x_start, x_end),
        0.75: brentq(lambda x: cdf_interpolated(x) - 0.75, x_start, x_end),
    }


__all__ = [
    "DirectCdfResult",
    "cdf_from_price_curve",
    "pdf_from_price_curve",
    "raw_cdf_from_price_curve",
    "calculate_quartiles",
]
