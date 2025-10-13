"""Typed containers for SVI calibration configuration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace, asdict
from typing import Any, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class SVICalibrationOptions:
    """Configuration knobs controlling SVI calibration.

    Args:
        max_iter: Maximum iterations for the local optimiser.
        tol: Convergence tolerance for the optimiser objective.
        regularisation: L2 penalty applied to the SVI ``b`` parameter.
        rho_bound: Absolute value cap for the correlation parameter ``rho``.
        sigma_min: Strictly positive lower bound for ``sigma``.
        diagnostic_grid_pad: Log-moneyness padding added around observations for diagnostics.
        diagnostic_grid_points: Number of points in the diagnostic grid.
        butterfly_weight: Penalty factor for butterfly arbitrage violations.
        callspread_weight: Penalty factor for call-spread monotonicity violations.
        callspread_step: Optional manual finite-difference step for call-spread penalties.
        callspread_step_floor: Minimum adaptive step when ``callspread_step`` is unset.
        callspread_step_ceiling: Maximum adaptive step when ``callspread_step`` is unset.
        global_solver: Name of the stochastic/global optimiser to run before polishing.
        global_max_iter: Maximum iterations for the global solver.
        polish_solver: Name of the local polishing optimiser.
        n_starts: Number of additional multi-start seeds to try.
        random_seed: Seed forwarded to stochastic components of the calibration.
        weighting_mode: Weighting scheme applied to residuals (e.g. ``"vega"``).
        weight_cap: Upper cap applied to the residual weights.
        huber_delta: Optional manual transition scale for the Huber loss in total variance.
        huber_beta: Fraction of the typical slice variance used when ``huber_delta`` is inferred.
        huber_delta_floor: Minimum allowed value for the inferred Huber delta.
        envelope_weight: Penalty weight for breaching the bid/ask implied-volatility envelope.
        volume_column: Optional column name used when extracting volumes from input frames.
        spread_floor: Minimum allowed bid/ask spread when computing residual weights.
        global_tol: Convergence tolerance used by the global optimiser.
        global_workers: Number of worker processes supplied to the global optimiser.
    """

    max_iter: int = 200
    tol: float = 1e-6
    regularisation: float = 1e-4
    rho_bound: float = 0.999
    sigma_min: float = 1e-4
    diagnostic_grid_pad: float = 0.5
    diagnostic_grid_points: int = 201
    butterfly_weight: float = 1e4
    callspread_weight: float = 1e3
    callspread_step: float | None = None
    callspread_step_floor: float = 0.01
    callspread_step_ceiling: float = 0.35
    global_solver: str = "de"
    global_max_iter: int = 20
    polish_solver: str = "lbfgsb"
    n_starts: int = 5
    random_seed: int | None = 1
    weighting_mode: str = "vega"
    weight_cap: float = 25.0
    huber_delta: float | None = None
    huber_beta: float = 0.01
    huber_delta_floor: float = 1e-4
    envelope_weight: float = 1e3
    volume_column: str | None = None
    spread_floor: float = 1e-4
    global_tol: float = 1e-3
    global_workers: int = 1

    @classmethod
    def field_names(cls) -> set[str]:
        """Return the names of supported configuration fields."""

        return {f.name for f in fields(cls)}

    @classmethod
    def from_mapping(
        cls, overrides: SVICalibrationOptions | Mapping[str, Any] | None = None
    ) -> SVICalibrationOptions:
        """Build an options instance from optional overrides.

        Args:
            overrides: Either an existing :class:`SVICalibrationOptions` instance or a
                mapping of field overrides.

        Returns:
            A fully populated :class:`SVICalibrationOptions` instance.

        Raises:
            TypeError: If ``overrides`` contains unrecognised keys.
        """

        if overrides is None:
            return cls()
        if isinstance(overrides, cls):
            return overrides
        unknown = set(overrides) - cls.field_names()
        if unknown:
            raise TypeError(f"Unknown SVI option(s): {sorted(unknown)}")
        return replace(cls(), **{name: overrides[name] for name in overrides})

    def to_mapping(self) -> dict[str, Any]:
        """Return a mapping representation of the options."""

        return asdict(self)


@dataclass
class SVITrialRecord:
    """Diagnostics captured for an individual local optimiser start.

    Args:
        start_index: Integer index of the starting point.
        start: Tuple containing the initial SVI parameters passed to the optimiser.
        success: Whether the optimiser reported success.
        objective: Objective value returned by the optimiser from this start.
        start_origin: Source tag describing how the start vector was generated
            (e.g. ``"heuristic"``, ``"global"``, ``"jw"``, ``"qe"``, ``"random"``).
        params: Optional tuple containing the optimised parameter vector when
            ``success`` is ``True``.
    """

    start_index: int
    start: Tuple[float, ...]
    success: bool
    objective: float
    start_origin: str
    params: Tuple[float, ...] | None = None


@dataclass
class SVICalibrationDiagnostics:
    """Structured diagnostics summarising an SVI calibration run."""

    status: str = "pending"
    objective: float = float("nan")
    iterations: int = 0
    message: str = ""
    global_solver: str = "none"
    polish_solver: str = "lbfgsb"
    n_starts: int = 0
    weighting_mode: str = "vega"
    huber_delta: float = 0.0
    callspread_step: float = 0.0
    weights_min: float = 0.0
    weights_max: float = 0.0
    envelope_weight: float = 0.0
    weights_volume_used: bool = False
    weights_spread_used: bool = False
    qe_seed_count: int = 0
    random_seed: int | None = None
    global_status: str | None = None
    global_objective: float | None = None
    global_iterations: int | None = None
    chosen_start_index: int | None = None
    chosen_start_origin: str | None = None
    min_g: float | None = None
    butterfly_warning: float | None = None
    max_wing_slope: float | None = None
    wing_warning: float | None = None
    rmse_unweighted: float | None = None
    rmse_weighted: float | None = None
    residual_mean: float | None = None
    envelope_violations_pct: float | None = None
    trial_records: list[SVITrialRecord] = field(default_factory=list)

    def add_trial_record(self, record: SVITrialRecord) -> None:
        """Append a trial record to the diagnostics trace.

        Args:
            record: Trial diagnostics to store.
        """

        self.trial_records.append(record)

    def to_mapping(self) -> dict[str, Any]:
        """Return a dictionary representation of the diagnostics."""

        base = asdict(self)
        base["trial_records"] = [
            {
                "start_index": record.start_index,
                "start": list(record.start),
                "success": record.success,
                "objective": record.objective,
                "start_origin": record.start_origin,
                "params": list(record.params) if record.params is not None else None,
            }
            for record in self.trial_records
        ]
        return base

    def __getitem__(self, key: str) -> Any:
        """Allow read-only dict-style access for compatibility."""

        return getattr(self, key)

    def keys(self) -> Iterable[str]:
        """Return iterable access to field names."""

        return self.__dict__.keys()


__all__ = [
    "SVICalibrationOptions",
    "SVICalibrationDiagnostics",
    "SVITrialRecord",
]
