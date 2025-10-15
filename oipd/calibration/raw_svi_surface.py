"""Penalty-stitched raw SVI surface calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from oipd.core.errors import CalculationError
from oipd.core.svi import SVIParameters, calibrate_svi_parameters, g_function, svi_total_variance
from oipd.core.svi_types import SVICalibrationDiagnostics, SVICalibrationOptions
from oipd.core.vol_model import VolModel
from oipd.calibration.ssvi_surface import SSVISliceObservations


@dataclass(frozen=True)
class RawSVISliceFit:
    """Fitted raw SVI parameters for a single maturity."""

    maturity: float
    params: SVIParameters
    diagnostics: SVICalibrationDiagnostics


@dataclass(frozen=True)
class RawSVISurfaceFit:
    """Joint fit diagnostics for the raw SVI surface."""

    slices: tuple[RawSVISliceFit, ...]
    min_calendar_margin: float
    min_butterfly: float
    objective: float
    alpha: float = 0.0
    raw_calendar_margins: list[float] | None = None
    raw_calendar_deltas: list[float] | None = None


def _calendar_metrics(
    slices: Sequence[RawSVISliceFit],
    *,
    grid: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    if len(slices) < 2:
        return float("inf"), [], []
    margins: list[float] = []
    deltas: list[float] = []
    for early, late in zip(slices[:-1], slices[1:]):
        w_early = svi_total_variance(grid, early.params)
        w_late = svi_total_variance(grid, late.params)
        margins.append(float(np.min(w_late - w_early)))
        deltas.append(float(late.maturity - early.maturity))
    min_margin = float(np.min(margins)) if margins else float("inf")
    return min_margin, margins, deltas


def _min_butterfly(
    slices: Sequence[RawSVISliceFit],
    *,
    grid: np.ndarray,
) -> float:
    minima = []
    for slice_fit in slices:
        values = g_function(grid, slice_fit.params)
        if np.isfinite(values).any():
            minima.append(np.min(values[np.isfinite(values)]))
    if not minima:
        return float("nan")
    return float(np.min(minima))


def calibrate_raw_svi_surface(
    observations: Sequence[SSVISliceObservations],
    vol_model: VolModel,
    *,
    options: SVICalibrationOptions | None = None,
) -> RawSVISurfaceFit:
    """Calibrate raw SVI slices independently and compute surface diagnostics."""

    if not observations:
        raise CalculationError("Raw SVI calibration requires at least one maturity")
    if vol_model.method not in {"raw_svi", None}:  # pragma: no cover - guarded upstream
        raise CalculationError("VolModel.method must be 'raw_svi' for raw SVI calibration")

    observations = tuple(sorted(observations, key=lambda obs: obs.maturity))
    configs = options or SVICalibrationOptions()

    fitted_slices: list[RawSVISliceFit] = []
    total_objective = 0.0

    for obs in observations:
        params, diagnostics = calibrate_svi_parameters(
            obs.log_moneyness,
            obs.total_variance,
            obs.maturity,
            configs,
        )
        if np.isnan(diagnostics.objective):
            diagnostics.objective = float(np.mean((
                svi_total_variance(obs.log_moneyness, params) - obs.total_variance
            ) ** 2))
        total_objective += float(diagnostics.objective)
        fitted_slices.append(
            RawSVISliceFit(
                maturity=obs.maturity,
                params=params,
                diagnostics=diagnostics,
            )
        )

    # Diagnostics grids
    all_k = np.concatenate([obs.log_moneyness for obs in observations])
    k_min = float(np.min(all_k)) - 0.5
    k_max = float(np.max(all_k)) + 0.5
    k_grid = np.linspace(k_min, k_max, 201)

    calendar_margin, pair_margins, pair_deltas = _calendar_metrics(fitted_slices, grid=k_grid)
    butterfly_min = _min_butterfly(fitted_slices, grid=k_grid)

    alpha_tilt = 0.0
    if vol_model.strict_no_arbitrage and pair_margins:
        requirements = [
            (-margin) / delta
            for margin, delta in zip(pair_margins, pair_deltas)
            if margin < -1e-6 and delta > 0.0
        ]
        if requirements:
            alpha_tilt = max(requirements)
            alpha_tilt = max(alpha_tilt, 0.0)
            adjusted_margins = [
                margin + alpha_tilt * max(delta, 0.0)
                for margin, delta in zip(pair_margins, pair_deltas)
            ]
            calendar_margin = float(np.min(adjusted_margins)) if adjusted_margins else calendar_margin

    if vol_model.strict_no_arbitrage and calendar_margin < -1e-3:
        raise CalculationError(
            f"Calendar arbitrage detected in raw SVI surface (min margin {calendar_margin:.3e})"
        )

    return RawSVISurfaceFit(
        slices=tuple(fitted_slices),
        min_calendar_margin=calendar_margin,
        min_butterfly=butterfly_min,
        objective=total_objective,
        alpha=alpha_tilt,
        raw_calendar_margins=pair_margins,
        raw_calendar_deltas=pair_deltas,
    )
