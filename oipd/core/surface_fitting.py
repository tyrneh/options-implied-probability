"""Surface fitting configuration, data structures, and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Literal, Optional

import numpy as np
from scipy import interpolate

from oipd.core.errors import CalculationError
from oipd.core.svi import (
    SVIParameters,
    calibrate_svi_parameters,
    from_total_variance,
    log_moneyness,
    svi_total_variance,
    to_total_variance,
)


VolCurve = Callable[[Iterable[float] | np.ndarray], np.ndarray]


__all__ = [
    "SurfaceConfig",
    "SVIConfig",
    "SmileSlice",
    "SVIFitDiagnostics",
    "available_surface_fits",
    "fit_surface",
    "register_surface_fit",
    "VolCurve",
]


_SURFACE_REGISTRY: Dict[str, Callable[..., VolCurve]] = {}


def register_surface_fit(name: str, factory: Callable[..., VolCurve]) -> None:
    """Register a surface-fitting factory under the given name."""

    if name in _SURFACE_REGISTRY:
        raise ValueError(f"Surface fit '{name}' is already registered")
    _SURFACE_REGISTRY[name] = factory


def available_surface_fits() -> list[str]:
    """Return the sorted list of registered surface fits."""

    return sorted(_SURFACE_REGISTRY.keys())


def _get_surface_factory(name: str) -> Callable[..., VolCurve]:
    try:
        return _SURFACE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown surface fit '{name}'. Available: {available_surface_fits()}"
        ) from exc


def fit_surface(
    config: SurfaceConfig,
    *,
    strikes: np.ndarray,
    iv: np.ndarray,
    **kwargs: Any,
) -> VolCurve:
    """Fit a volatility surface slice using the configured method."""

    strikes_arr = np.asarray(strikes, dtype=float)
    iv_arr = np.asarray(iv, dtype=float)
    if strikes_arr.shape != iv_arr.shape:
        raise ValueError("Strike and IV arrays must have identical shapes")

    factory = _get_surface_factory(config.name)
    merged_kwargs = config.kwargs().copy()
    merged_kwargs.update(kwargs)
    return factory(strikes_arr, iv_arr, **merged_kwargs)


@dataclass(frozen=True)
class SVIConfig:
    """Calibration safeguards and knobs for the SVI smile fitter."""

    max_iter: int = 200
    tol: float = 1e-8
    regularisation: float = 1e-4
    rho_bound: float = 0.999
    sigma_min: float = 1e-4
    use_natural: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "tol": self.tol,
            "regularisation": self.regularisation,
            "rho_bound": self.rho_bound,
            "sigma_min": self.sigma_min,
            "use_natural": self.use_natural,
        }


@dataclass(frozen=True)
class SurfaceConfig:
    """User-facing selection of the volatility surface fitting method."""

    name: Literal["bspline", "svi"] = "svi"
    params: Optional[Any] = None

    def kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for the fitting implementation."""

        if self.params is None:
            # Default SVI config when unspecified
            return SVIConfig().to_dict() if self.name == "svi" else {}

        if isinstance(self.params, dict):
            return dict(self.params)

        if isinstance(self.params, SVIConfig):
            return self.params.to_dict()

        raise TypeError(
            "SurfaceConfig.params must be None, dict, or SVIConfig instance"
        )


@dataclass(frozen=True)
class SmileSlice:
    """Container for a single-expiry smile prepared for calibration."""

    k: np.ndarray
    total_variance: np.ndarray
    maturity_years: float
    forward: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.maturity_years <= 0.0:
            raise ValueError("maturity_years must be positive")
        if self.forward <= 0.0:
            raise ValueError("forward must be positive")

        k_arr = np.asarray(self.k, dtype=float)
        tv_arr = np.asarray(self.total_variance, dtype=float)
        if k_arr.shape != tv_arr.shape:
            raise ValueError("k and total_variance must share the same shape")
        if k_arr.ndim != 1:
            raise ValueError("k and total_variance must be one-dimensional arrays")
        if k_arr.size < 5:
            # SVI typically needs at least 5 strikes to calibrate reliably
            raise ValueError("SmileSlice requires at least 5 strike points")
        if np.any(tv_arr < 0.0):
            raise ValueError("total_variance must be non-negative")

        object.__setattr__(self, "k", k_arr)
        object.__setattr__(self, "total_variance", tv_arr)


@dataclass(frozen=True)
class SVIFitDiagnostics:
    """Telemetry captured during SVI calibration."""

    status: Literal["success", "failure", "not_run"] = "not_run"
    objective: Optional[float] = None
    min_g: Optional[float] = None
    iterations: Optional[int] = None
    message: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def _extract_svi_config(kwargs: Dict[str, Any]) -> tuple[SVIConfig, Dict[str, Any]]:
    svi_fields = set(SVIConfig.__dataclass_fields__.keys())
    config_kwargs = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in svi_fields}
    return SVIConfig(**config_kwargs), kwargs


def _fit_svi(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    forward: Optional[float] = None,
    maturity_years: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> VolCurve:
    """Calibrate an SVI slice and return a volatility curve."""

    if forward is None or forward <= 0:
        raise ValueError("forward must be positive for SVI fitting")
    if maturity_years is None or maturity_years <= 0:
        raise ValueError("maturity_years must be positive for SVI fitting")

    config, remaining = _extract_svi_config(kwargs)
    if remaining:
        extras = {key: remaining[key] for key in remaining}
    else:
        extras = {}

    k = log_moneyness(strikes, forward)
    total_variance = to_total_variance(iv, maturity_years)

    params, diag = calibrate_svi_parameters(k, total_variance, maturity_years, config, weights=weights)

    fitted_total_variance = svi_total_variance(k, params)
    fitted_iv = from_total_variance(fitted_total_variance, maturity_years)

    def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        eval_array = np.asarray(eval_strikes, dtype=float)
        eval_k = log_moneyness(eval_array, forward)
        total_var_eval = svi_total_variance(eval_k, params)
        return from_total_variance(total_var_eval, maturity_years)

    params_dict = {
        "method": "svi",
        "a": params.a,
        "b": params.b,
        "rho": params.rho,
        "m": params.m,
        "sigma": params.sigma,
        "forward": forward,
        "maturity_years": maturity_years,
    }

    diag_extras = {k: v for k, v in diag.items() if k not in {"status", "objective", "min_g", "iterations", "message"}}
    diag_extras.update(extras)

    diagnostics = SVIFitDiagnostics(
        status=diag.get("status", "success"),
        objective=diag.get("objective"),
        min_g=diag.get("min_g"),
        iterations=diag.get("iterations"),
        message=diag.get("message"),
        extras=diag_extras,
    )

    vol_curve.grid = (strikes, fitted_iv)  # type: ignore[attr-defined]
    vol_curve.params = params_dict  # type: ignore[attr-defined]
    vol_curve.diagnostics = diagnostics  # type: ignore[attr-defined]
    return vol_curve


def _fit_bspline(
    strikes: np.ndarray,
    iv: np.ndarray,
    *,
    smoothing_factor: float = 10.0,
    degree: int = 3,
    dx: float = 0.1,
) -> VolCurve:
    """Legacy B-spline surface fit for implied volatility data."""

    if strikes.size < 4:
        raise CalculationError(
            f"Insufficient data for B-spline fitting: need at least 4 points, got {strikes.size}"
        )

    try:
        tck = interpolate.splrep(strikes, iv, s=smoothing_factor, k=degree)
    except Exception as exc:  # pragma: no cover - scipy error propagation
        raise CalculationError(f"Failed to fit B-spline to implied volatility data: {exc}")

    grid_points = max(2, int(np.ceil((strikes.max() - strikes.min()) / max(dx, 1e-6))))
    grid_x = np.linspace(strikes.min(), strikes.max(), grid_points)
    spline = interpolate.BSpline(*tck)
    grid_y = spline(grid_x)

    def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        eval_array = np.asarray(eval_strikes, dtype=float)
        return spline(eval_array)

    vol_curve.grid = (grid_x, grid_y)  # type: ignore[attr-defined]
    vol_curve.params = {
        "method": "bspline",
        "smoothing_factor": smoothing_factor,
        "degree": degree,
        "dx": dx,
    }  # type: ignore[attr-defined]
    vol_curve.diagnostics = {"points_used": strikes.size}  # type: ignore[attr-defined]
    return vol_curve


# Register legacy smoother by default
register_surface_fit("svi", _fit_svi)
register_surface_fit("bspline", _fit_bspline)
