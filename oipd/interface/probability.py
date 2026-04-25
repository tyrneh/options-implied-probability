"""Stateful probability estimators derived from fitted volatility curves."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Mapping, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from oipd.core.maturity import format_timestamp_for_display
from oipd.interface.warning_diagnostics import (
    WarningDiagnostics,
    WarningEvent,
    _emit_warning_summaries,
)
from oipd.market_inputs import ResolvedMarket
from oipd.presentation.plot_rnd import plot_rnd
from oipd.presentation.probability_surface_plot import plot_probability_summary

from oipd.pipelines.probability import (
    build_density_results_frame,
    build_fan_quantile_summary_frame,
    build_surface_density_results_frame,
    quantile_from_cdf,
    resolve_surface_query_time,
)
from oipd.pipelines.probability.rnd_curve import (
    materialize_distribution_from_definition,
)
from oipd.pipelines.probability.models import (
    CurveProbabilityDefinition,
    DistributionSnapshot,
    MaterializationSpec,
    SurfaceProbabilityDefinition,
)

if TYPE_CHECKING:
    from oipd.market_inputs import MarketInputs
    from oipd.interface.volatility import VolCurve
    from oipd.interface.volatility import VolSurface

PROB_CURVE_MATERIALIZE_WARNING_SCOPE = "ProbCurve.materialize"
PROB_SURFACE_QUERY_WARNING_SCOPE = "ProbSurface.query"
PROB_SURFACE_DENSITY_WARNING_SCOPE = "ProbSurface.density_results"
PROB_SURFACE_FAN_WARNING_SCOPE = "ProbSurface.plot_fan"


def _warning_detail_value(value: Any) -> Any:
    """Convert probability metadata into small JSON-like diagnostic details.

    Arrays, DataFrames, and other model objects should be summarized before
    being passed here; warning diagnostics store audit facts, not live data.

    Args:
        value: Metadata value returned by a probability pipeline.

    Returns:
        Any: JSON-like scalar or nested container accepted by ``WarningEvent``.
    """
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _warning_detail_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _warning_detail_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_warning_detail_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_warning_detail_value(item) for item in sorted(value, key=str)]
    return value


def _warning_detail_mapping(details: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-like details suitable for a ``WarningEvent``.

    Args:
        details: Raw detail mapping built from probability metadata.

    Returns:
        dict[str, Any]: Sanitized warning-event details.
    """
    return {str(key): _warning_detail_value(value) for key, value in details.items()}


def _cdf_repair_warning_event(
    metadata: Mapping[str, Any],
    *,
    scope: str,
    expiry: Any | None = None,
    is_pillar: bool | None = None,
) -> WarningEvent | None:
    """Translate CDF repair metadata into one warning diagnostic event.

    Details must remain small JSON-like audit payloads. Arrays, DataFrames, and
    other rich objects should be summarized before being stored in metadata.

    Args:
        metadata: Probability materialization metadata.
        scope: Diagnostic replacement scope for the event.
        expiry: Optional expiry label to store when not already in metadata.
        is_pillar: Whether the slice is an original fitted pillar, when known.

    Returns:
        WarningEvent | None: CDF repair event when repair is warning-worthy.
    """
    if not bool(metadata.get("cdf_monotonicity_repair_applied")):
        return None

    monotonicity_severity = str(metadata.get("cdf_monotonicity_severity", "none"))
    if monotonicity_severity not in {"material", "severe"}:
        return None

    event_severity = "severe" if monotonicity_severity == "severe" else "warning"
    expiry_detail = metadata.get("expiry", expiry)
    details: dict[str, Any] = {
        "cdf_violation_policy": metadata.get("cdf_violation_policy"),
        "cdf_monotonicity_repair_tolerance": metadata.get(
            "cdf_monotonicity_repair_tolerance"
        ),
        "cdf_total_negative_variation_tolerance": metadata.get(
            "cdf_total_negative_variation_tolerance"
        ),
        "cdf_monotonicity_severity": monotonicity_severity,
        "raw_cdf_negative_step_count": metadata.get("raw_cdf_negative_step_count"),
        "raw_cdf_max_negative_step": metadata.get("raw_cdf_max_negative_step"),
        "raw_cdf_total_negative_variation": metadata.get(
            "raw_cdf_total_negative_variation"
        ),
        "raw_cdf_worst_step_strike": metadata.get("raw_cdf_worst_step_strike"),
    }
    if expiry_detail is not None:
        details["expiry"] = expiry_detail
    if is_pillar is not None:
        details["is_pillar"] = bool(is_pillar)
    for lineage_key in (
        "time_to_expiry_years",
        "interpolated",
        "interpolated_from_expiries",
    ):
        if lineage_key in metadata:
            details[lineage_key] = metadata.get(lineage_key)

    return WarningEvent(
        category="model_risk",
        event_type="cdf_repair",
        severity=event_severity,
        scope=scope,
        message="Direct CDF monotonicity was repaired during materialization.",
        details=_warning_detail_mapping(details),
    )


def _fan_skip_warning_events(
    skip_report: Mapping[str, Any] | None,
    *,
    scope: str,
) -> list[WarningEvent]:
    """Translate a fan-summary skip report into workflow warning events.

    Args:
        skip_report: Compact skip report from ``DataFrame.attrs``.
        scope: Diagnostic replacement scope for the event.

    Returns:
        list[WarningEvent]: Workflow events for skipped sampled expiries.
    """
    if not skip_report:
        return []

    skipped_count = int(skip_report.get("skipped_count", 0) or 0)
    if skipped_count <= 0:
        return []

    skipped_expiries = list(skip_report.get("skipped_expiries", []) or [])
    compact_expiries = skipped_expiries[:10]
    if len(skipped_expiries) > len(compact_expiries):
        compact_expiries.append(
            f"... {len(skipped_expiries) - len(compact_expiries)} more"
        )

    expiry_label = "expiry" if skipped_count == 1 else "expiries"
    return [
        WarningEvent(
            category="workflow",
            event_type="skipped_expiry",
            severity="warning",
            scope=scope,
            message=f"Skipped {skipped_count} sampled {expiry_label}.",
            details=_warning_detail_mapping(
                {
                    "skipped_count": skipped_count,
                    "reason_summary": skip_report.get("reason_summary"),
                    "skipped_expiries": compact_expiries,
                }
            ),
        )
    ]


class ProbCurve:
    """Single-expiry risk-neutral probability curve wrapper.

    Computes PDF/CDF from an option chain using the stateless probability pipeline
    (golden-master aligned) and exposes convenience probability queries.
    """

    def __init__(
        self,
        vol_curve: (
            Any | None
        ) = None,  # Type hint as Any to avoid circular import with VolCurve
        *,
        resolved_market: Optional[ResolvedMarket] = None,
        metadata: Optional[dict[str, Any]] = None,
        prices: Optional[np.ndarray] = None,
        pdf_values: Optional[np.ndarray] = None,
        cdf_values: Optional[np.ndarray] = None,
        cdf_violation_policy: Literal["warn", "raise"] = "warn",
    ) -> None:
        """Initialize a ProbCurve result container.

        Args:
            vol_curve: The fitted VolCurve instance.
            resolved_market: Optional resolved market for array-backed construction.
            metadata: Optional metadata for array-backed construction.
            prices: Optional precomputed price grid.
            pdf_values: Optional precomputed PDF values.
            cdf_values: Optional precomputed CDF values.
            cdf_violation_policy: Direct-CDF monotonicity policy for lazy
                materialization. ``"warn"`` repairs and warns; ``"raise"``
                fails on material violations.
        """
        if vol_curve is None and resolved_market is None:
            raise ValueError(
                "ProbCurve requires either a fitted vol_curve or resolved_market."
            )

        has_arrays = (
            prices is not None or pdf_values is not None or cdf_values is not None
        )
        if has_arrays and (prices is None or pdf_values is None or cdf_values is None):
            raise ValueError(
                "Array-backed ProbCurve construction requires prices, pdf_values, "
                "and cdf_values."
            )

        self._warning_diagnostics = WarningDiagnostics()
        self._native_spec: MaterializationSpec = MaterializationSpec(
            points=None,
            cdf_violation_policy=cdf_violation_policy,
        )
        self._definition: CurveProbabilityDefinition | None = None
        self._native_snapshot: DistributionSnapshot | None = None
        self._resolved_market: ResolvedMarket | None = None
        self._metadata: dict[str, Any] = {}
        self._emit_warning_summary = True

        if has_arrays:
            if vol_curve is not None:
                self._resolved_market = vol_curve.resolved_market
                snapshot_metadata = metadata or vol_curve._metadata or {}
            else:
                self._resolved_market = resolved_market
                snapshot_metadata = metadata or {}
            self._native_snapshot = DistributionSnapshot(
                prices=np.asarray(prices, dtype=float),
                pdf_values=np.asarray(pdf_values, dtype=float),
                cdf_values=np.asarray(cdf_values, dtype=float),
                metadata=snapshot_metadata,
            )
            self._metadata = self._native_snapshot.metadata
            return

        if vol_curve is None:
            self._resolved_market = resolved_market
            self._metadata = metadata or {}
            return

        self._definition = CurveProbabilityDefinition.from_vol_curve(
            vol_curve,
            native_spec=self._native_spec,
            metadata=metadata,
        )
        self._resolved_market = self._definition.resolved_market
        self._metadata = self._definition.vol_metadata

    @classmethod
    def _from_definition(
        cls,
        definition: CurveProbabilityDefinition,
        *,
        emit_warning_summary: bool = True,
    ) -> "ProbCurve":
        """Build a lazy ProbCurve from an internal probability definition.

        Args:
            definition: Frozen single-expiry probability recipe.
            emit_warning_summary: Whether materialization should emit a
                summarized Python warning for recorded diagnostics.

        Returns:
            ProbCurve: Lazy probability curve backed by ``definition``.
        """
        instance = cls.__new__(cls)
        instance._native_spec = definition.native_spec
        instance._definition = definition
        instance._native_snapshot = None
        instance._resolved_market = definition.resolved_market
        instance._metadata = definition.vol_metadata
        instance._warning_diagnostics = WarningDiagnostics()
        instance._emit_warning_summary = bool(emit_warning_summary)
        return instance

    @classmethod
    def _from_vol_curve(
        cls,
        vol_curve: Any,
        *,
        native_spec: MaterializationSpec | None = None,
        metadata: Optional[dict[str, Any]] = None,
        emit_warning_summary: bool = True,
    ) -> "ProbCurve":
        """Build a lazy ProbCurve from a fitted VolCurve using an internal spec.

        Args:
            vol_curve: Fitted public VolCurve object.
            native_spec: Internal native materialization policy. When omitted,
                the public default policy is used.
            metadata: Optional metadata override.
            emit_warning_summary: Whether materialization should emit a
                summarized Python warning for recorded diagnostics.

        Returns:
            ProbCurve: Lazy probability curve for the fitted volatility curve.
        """
        definition = CurveProbabilityDefinition.from_vol_curve(
            vol_curve,
            native_spec=native_spec or MaterializationSpec(points=None),
            metadata=metadata,
        )
        return cls._from_definition(
            definition,
            emit_warning_summary=emit_warning_summary,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        resolved_market: ResolvedMarket,
        metadata: dict[str, Any],
        prices: np.ndarray,
        pdf_values: np.ndarray,
        cdf_values: np.ndarray,
    ) -> "ProbCurve":
        """Build a ProbCurve from precomputed arrays.

        Args:
            resolved_market: Resolved market snapshot for the slice.
            metadata: Slice metadata dictionary.
            prices: Strike grid.
            pdf_values: PDF values aligned with ``prices``.
            cdf_values: CDF values aligned with ``prices``.

        Returns:
            ProbCurve: Array-backed probability curve.
        """
        return cls(
            None,
            resolved_market=resolved_market,
            metadata=metadata,
            prices=prices,
            pdf_values=pdf_values,
            cdf_values=cdf_values,
        )

    @classmethod
    def from_chain(
        cls,
        chain: pd.DataFrame,
        market: "MarketInputs",
        *,
        column_mapping: Optional[Mapping[str, str]] = None,
        max_staleness_days: int = 3,
        cdf_violation_policy: Literal["warn", "raise"] = "warn",
    ) -> "ProbCurve":
        """Build a ProbCurve directly from a single-expiry option chain.

        This convenience constructor fits an SVI volatility curve under the hood
        and then derives the risk-neutral distribution.

        Args:
            chain: Long-form option chain DataFrame containing a single expiry,
                with one row per contract. Expected standard columns are
                ``strike``, ``expiry``, ``option_type``, and ``last_price``.
                Optional quote metadata includes ``bid``, ``ask``, ``volume``,
                and ``last_trade_date``. ``bid`` and ``ask`` can improve quote
                handling when present, but ``last_price`` is still required by
                the current input contract.
            market: Market inputs required for calibration.
            column_mapping: Optional mapping from input columns to OIPD
                standard names.
            max_staleness_days: Maximum age of quotes in days to include.
            cdf_violation_policy: Direct-CDF monotonicity policy for lazy
                materialization. ``"warn"`` repairs and warns; ``"raise"``
                fails on material violations.

        Returns:
            ProbCurve: The fitted risk-neutral probability curve.

        Raises:
            ValueError: If the chain contains multiple expiries or invalid expiry values.
            CalculationError: If the underlying volatility calibration fails.
        """
        from oipd.interface.volatility import VolCurve

        vol_curve = VolCurve(method="svi", max_staleness_days=max_staleness_days)
        vol_curve.fit(chain, market, column_mapping=column_mapping)
        prob_curve = ProbCurve._from_vol_curve(
            vol_curve,
            native_spec=MaterializationSpec(
                points=None,
                cdf_violation_policy=cdf_violation_policy,
            ),
        )
        prob_curve._warning_diagnostics = WarningDiagnostics(
            vol_curve.warning_diagnostics.events
        )
        return prob_curve

    def _materialize_native_distribution(self) -> DistributionSnapshot:
        """Return the native distribution snapshot, materializing it if needed.

        Returns:
            DistributionSnapshot: Cached native probability arrays and metadata.

        Raises:
            ValueError: If no fitted volatility definition is available.
        """
        if self._native_snapshot is not None:
            return self._native_snapshot

        if self._definition is None:
            raise ValueError("Probability arrays are unavailable for this curve.")

        native_snapshot = materialize_distribution_from_definition(
            self._definition,
            self._native_spec,
        )
        metadata = native_snapshot.metadata
        warning_event = _cdf_repair_warning_event(
            metadata,
            scope=PROB_CURVE_MATERIALIZE_WARNING_SCOPE,
        )
        warning_events = [] if warning_event is None else [warning_event]
        if self._emit_warning_summary:
            _emit_warning_summaries(warning_events, owner="ProbCurve.materialize")

        self._native_snapshot = native_snapshot
        self._metadata = metadata
        self._warning_diagnostics._replace_scope_events(
            PROB_CURVE_MATERIALIZE_WARNING_SCOPE,
            warning_events,
        )
        return self._native_snapshot

    def _ensure_grid_generated(self) -> None:
        """Lazily generate and cache the native probability snapshot."""
        self._materialize_native_distribution()

    def _require_cached_distribution_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached distribution arrays after enforcing lazy initialization.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Cached price, PDF, and CDF arrays.

        Raises:
            ValueError: If the cached arrays remain unavailable after initialization.
        """
        snapshot = self._materialize_native_distribution()
        return snapshot.prices, snapshot.pdf_values, snapshot.cdf_values

    def pdf(self, price: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the Probability Density Function (PDF) at the given price level(s).

        Args:
            price: Price level(s) to evaluate.

        Returns:
            float | np.ndarray: PDF values.
        """
        prices, pdf_values, _ = self._require_cached_distribution_arrays()
        query_prices = np.asarray(price, dtype=float)
        interpolated = np.interp(
            query_prices,
            prices,
            pdf_values,
            left=0.0,
            right=0.0,
        )
        if np.isscalar(price):
            return float(interpolated)
        return np.asarray(interpolated, dtype=float)

    def __call__(self, price: float | np.ndarray) -> float | np.ndarray:
        """Alias for :meth:`pdf`."""
        return self.pdf(price)

    def prob_below(self, price: float) -> float:
        """Probability that the asset price is below ``price``.

        Args:
           price: Price level.

        Returns:
           float: Probability P(S < price).
        """

        prices, _, cdf_values = self._require_cached_distribution_arrays()
        cdf_monotone = np.maximum.accumulate(np.asarray(cdf_values, dtype=float))
        return float(
            np.interp(
                price,
                np.asarray(prices, dtype=float),
                cdf_monotone,
                left=0.0,
                right=1.0,
            )
        )

    def prob_above(self, price: float) -> float:
        """Probability that the asset price is at or above ``price``.

        Args:
            price: Price level.

        Returns:
            float: Probability P(S >= price).
        """

        return 1.0 - self.prob_below(price)

    def prob_between(self, low: float, high: float) -> float:
        """Probability that the asset price lies in ``[low, high]``.

        Args:
            low: Lower bound price.
            high: Upper bound price.

        Returns:
            float: Probability P(low <= S < high).
        """

        if low > high:
            raise ValueError("low must be <= high")
        return self.prob_below(high) - self.prob_below(low)

    def mean(self) -> float:
        """Return the expected value (mean) under the fitted PDF.

        Returns:
            float: Expected price E[S].
        """

        prices, pdf_values, _ = self._require_cached_distribution_arrays()
        return float(np.trapezoid(prices * pdf_values, prices))

    def variance(self) -> float:
        """Return the variance under the fitted PDF.

        Returns:
            float: Variance Var[S].
        """

        prices, pdf_values, _ = self._require_cached_distribution_arrays()
        mean = self.mean()
        return float(
            np.trapezoid(
                ((prices - mean) ** 2) * pdf_values,
                prices,
            )
        )

    def skew(self) -> float:
        """Return the skewness (3rd standardized moment) of the fitted PDF.

        Skew = E[(X - mu)^3] / sigma^3.

        Returns:
            float: Skewness. (Positive = lean to right/fat right tail, but for prices usually negative/fat left tail).
        """
        prices, pdf_values, _ = self._require_cached_distribution_arrays()
        mean = self.mean()
        var = self.variance()
        std = np.sqrt(var)

        moment3 = np.trapezoid(((prices - mean) ** 3) * pdf_values, prices)

        return float(moment3 / (std**3))

    def kurtosis(self) -> float:
        """Return the excess kurtosis (4th standardized moment - 3) of the fitted PDF.

        Excess Kurtosis = E[(X - mu)^4] / sigma^4 - 3.

        Returns:
            float: Excess Kurtosis (0 = Normal). Positive = Fat Tails.
        """
        prices, pdf_values, _ = self._require_cached_distribution_arrays()
        mean = self.mean()
        var = self.variance()
        std = np.sqrt(var)

        moment4 = np.trapezoid(((prices - mean) ** 4) * pdf_values, prices)

        # Excess Kurtosis
        return float(moment4 / (std**4) - 3.0)

    def quantile(self, q: float) -> float:
        """Return the price level at a given probability quantile (Inverse CDF).

        Args:
            q: Probability level (0 < q < 1).

        Returns:
            float: Price S such that P(Asset < S) = q.
        """
        if not (0 < q < 1):
            raise ValueError("Quantile q must be between 0 and 1")

        prices, _, cdf_values = self._require_cached_distribution_arrays()
        return quantile_from_cdf(prices, cdf_values, q)

    @property
    def prices(self) -> np.ndarray:
        """Default price grid used for standard visualization.

        Returns:
            np.ndarray: Array of price levels.
        """

        prices, _, _ = self._require_cached_distribution_arrays()
        return prices

    @property
    def pdf_values(self) -> np.ndarray:
        """PDF values over the stored price grid.

        Returns:
            np.ndarray: Probability densities in decimals.
        """

        _, pdf_values, _ = self._require_cached_distribution_arrays()
        return pdf_values

    @property
    def cdf_values(self) -> np.ndarray:
        """CDF values over the stored price grid.

        Returns:
            np.ndarray: Cumulative probabilities in decimals.
        """

        _, _, cdf_values = self._require_cached_distribution_arrays()
        return cdf_values

    @property
    def warning_diagnostics(self) -> WarningDiagnostics:
        """Return structured warning diagnostics for the current curve state.

        Returns:
            WarningDiagnostics: Current warning diagnostic events and summary.
        """
        return self._warning_diagnostics

    @property
    def resolved_market(self) -> ResolvedMarket:
        """Resolved market snapshot used for estimation.

        Returns:
            ResolvedMarket: Standardized market inputs.
        """

        if self._resolved_market is None:
            raise ValueError("Call fit before accessing the resolved market")
        return self._resolved_market

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata captured during estimation (vol curve, diagnostics, etc.).

        Returns:
            dict[str, Any]: Metadata dictionary.
        """

        return self._materialize_native_distribution().metadata

    def density_results(
        self,
        domain: tuple[float, float] | None = None,
        points: int = 200,
        *,
        full_domain: bool = False,
    ) -> pd.DataFrame:
        """Return a DataFrame view of the fitted probability density.

        Args:
            domain: Optional explicit export/view domain as ``(min_price, max_price)``.
                When provided, the native cached distribution is resampled onto
                this range. This does not change the upstream PDF domain chosen
                during implied-distribution construction.
            points: Number of resampled points for compact or explicit-domain
                exports. Ignored when ``full_domain`` returns native arrays
                exactly.
            full_domain: If ``True`` and ``domain`` is omitted, return the
                native full-domain arrays exactly. Explicit ``domain`` always
                takes precedence.

        Returns:
            DataFrame with columns ``price``, ``pdf``, and ``cdf``.
        """
        snapshot = self._materialize_native_distribution()
        return build_density_results_frame(
            snapshot.prices,
            snapshot.pdf_values,
            snapshot.cdf_values,
            domain=domain,
            points=points,
            default_domain=snapshot.metadata.get("default_view_domain"),
            full_domain=full_domain,
        )

    def plot(
        self,
        *,
        kind: Literal["pdf", "cdf", "both"] = "both",
        figsize: tuple[float, float] = (10.0, 5.0),
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        points: int = 800,
        full_domain: bool = False,
        **kwargs,
    ) -> Any:
        """Plot the risk-neutral probability distribution.

        Args:
            kind: Which distribution(s) to plot: ``"pdf"``, ``"cdf"``, or ``"both"``.
            figsize: Figure size as (width, height) in inches.
            title: Optional custom title for the plot.
            xlim: Optional x-axis limits as (min, max).
            ylim: Optional y-axis limits as (min, max).
            points: Number of display points used for plot resampling.
            full_domain: If ``True`` and ``xlim`` is omitted, plot across the
                full native probability domain. Otherwise the compact default
                view domain is used.
            **kwargs: Additional arguments forwarded to ``oipd.presentation.plot_rnd.plot_rnd``.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        resolved_market = self.resolved_market
        snapshot = self._materialize_native_distribution()
        metadata = snapshot.metadata

        underlying_price = resolved_market.underlying_price
        valuation_date = format_timestamp_for_display(
            resolved_market.valuation_timestamp
        )

        expiry_raw = metadata.get("expiry")
        expiry_date = (
            format_timestamp_for_display(expiry_raw) if expiry_raw is not None else None
        )

        native_domain = (float(snapshot.prices[0]), float(snapshot.prices[-1]))
        if xlim is not None:
            plot_domain = xlim
        elif full_domain:
            plot_domain = native_domain
        else:
            plot_domain = metadata.get("default_view_domain", native_domain)

        plot_frame = build_density_results_frame(
            snapshot.prices,
            snapshot.pdf_values,
            snapshot.cdf_values,
            domain=plot_domain,
            points=points,
        )
        grid_prices = plot_frame["price"].to_numpy(dtype=float)
        grid_pdf = plot_frame["pdf"].to_numpy(dtype=float)
        grid_cdf = plot_frame["cdf"].to_numpy(dtype=float)

        return plot_rnd(
            prices=grid_prices,
            pdf=grid_pdf,
            cdf=grid_cdf,
            kind=kind,
            figsize=figsize,
            title=title,
            current_price=underlying_price,
            valuation_date=valuation_date,
            expiry_date=expiry_date,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )


class ProbSurface:
    """Multi-expiry risk-neutral probability surface wrapper."""

    def __init__(
        self,
        *,
        vol_surface: "VolSurface",
        grid_points: int | None = None,
        cdf_violation_policy: Literal["warn", "raise"] = "warn",
    ) -> None:
        """Initialize a ProbSurface directly from a fitted VolSurface.

        Args:
            vol_surface: Fitted volatility surface used as the canonical source.
            grid_points: Native materialization grid size for each cached slice.
                ``None`` uses the smart native grid policy. Export
                ``density_results(points=...)`` controls downstream resampling,
                not this native grid.
            cdf_violation_policy: Direct-CDF monotonicity policy for lazy
                slice materialization. ``"warn"`` repairs and warns;
                ``"raise"`` fails on material violations.

        Raises:
            ValueError: If ``vol_surface`` has not been fitted.
        """
        if (
            vol_surface._model is None
            or vol_surface._interpolator is None
            or vol_surface._market is None
            or len(vol_surface.expiries) == 0
        ):
            raise ValueError(
                "ProbSurface requires a fitted VolSurface with interpolator and market."
            )
        if isinstance(grid_points, bool):
            raise ValueError("grid_points must be at least 5 for finite differences.")
        if grid_points is not None and not isinstance(grid_points, int):
            raise ValueError("grid_points must be at least 5 for finite differences.")
        if grid_points is not None and grid_points < 5:
            raise ValueError("grid_points must be at least 5 for finite differences.")

        native_points = None if grid_points is None else int(grid_points)
        native_spec = MaterializationSpec(
            points=native_points,
            cdf_violation_policy=cdf_violation_policy,
        )
        self._definition = SurfaceProbabilityDefinition.from_vol_surface(
            vol_surface,
            native_spec=native_spec,
        )
        self._vol_surface = self._definition.vol_surface
        self._market = self._vol_surface._market
        self._valuation_timestamp = self._market.valuation_timestamp
        self._grid_points = native_points
        self._curve_cache: dict[int, ProbCurve] = {}
        self._internal_curve_cache: dict[int, ProbCurve] = {}
        self._warning_diagnostics = WarningDiagnostics()

    def _resolve_query_time(
        self, t: float | str | date | pd.Timestamp
    ) -> tuple[pd.Timestamp, float]:
        """Normalize maturity input and enforce strict domain constraints.

        Args:
            t: Maturity input as year-fraction float or date-like object.

        Returns:
            tuple[pd.Timestamp, float]: Expiry timestamp and year fraction.

        Raises:
            ValueError: If maturity is invalid or unsupported.
        """
        return resolve_surface_query_time(self._vol_surface, t)

    def _distribution_arrays_for_t_years(
        self,
        t_years: float,
        *,
        expiry_timestamp: pd.Timestamp | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached distribution arrays for maturity ``t_years``.

        Args:
            t_years: Time to maturity in years.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Price grid, PDF, CDF.
        """
        if expiry_timestamp is None:
            expiry_timestamp, t_years = self._resolve_query_time(float(t_years))
        curve = self._internal_curve_for_time(
            pd.Timestamp(expiry_timestamp),
            float(t_years),
        )
        return curve.prices, curve.pdf_values, curve.cdf_values

    def _resolved_market_for_t_years(
        self,
        t_years: float,
        *,
        expiry_timestamp: pd.Timestamp | None = None,
    ) -> ResolvedMarket:
        """Return the cached resolved market snapshot for maturity ``t``.

        Args:
            t_years: Time to maturity in years.

        Returns:
            ResolvedMarket: Synthetic market snapshot aligned with maturity ``t``.
        """
        if expiry_timestamp is None:
            expiry_timestamp, t_years = self._resolve_query_time(float(t_years))
        curve = self._internal_curve_for_time(
            pd.Timestamp(expiry_timestamp),
            float(t_years),
        )
        return curve.resolved_market

    def _build_curve_for_time(
        self,
        expiry_timestamp: pd.Timestamp,
        *,
        emit_warning_summary: bool,
    ) -> ProbCurve:
        """Build a probability curve for one surface maturity.

        Args:
            expiry_timestamp: Maturity timestamp.
            emit_warning_summary: Whether direct curve materialization should
                emit concise summary warnings.

        Returns:
            ProbCurve: Probability curve built from the canonical single-slice path.
        """
        vol_curve = self._vol_surface.slice(pd.Timestamp(expiry_timestamp))
        return ProbCurve._from_vol_curve(
            vol_curve,
            native_spec=self._definition.native_spec,
            emit_warning_summary=emit_warning_summary,
        )

    def _curve_for_time(
        self, expiry_timestamp: pd.Timestamp, t_years: float
    ) -> ProbCurve:
        """Return a public cached ProbCurve slice for the requested maturity.

        Args:
            expiry_timestamp: Maturity timestamp.
            t_years: Time to maturity in years.

        Returns:
            ProbCurve: Warning-enabled probability curve for direct user access.
        """
        cache_key = int(pd.Timestamp(expiry_timestamp).value)
        if cache_key in self._curve_cache:
            return self._curve_cache[cache_key]

        curve = self._build_curve_for_time(
            pd.Timestamp(expiry_timestamp),
            emit_warning_summary=True,
        )
        self._curve_cache[cache_key] = curve
        return curve

    def _internal_curve_for_time(
        self, expiry_timestamp: pd.Timestamp, t_years: float
    ) -> ProbCurve:
        """Return a no-warning cached ProbCurve for surface-owned operations.

        Args:
            expiry_timestamp: Maturity timestamp.
            t_years: Time to maturity in years.

        Returns:
            ProbCurve: Probability curve whose diagnostics are aggregated by
            the owning ``ProbSurface`` operation.
        """
        cache_key = int(pd.Timestamp(expiry_timestamp).value)
        if cache_key in self._internal_curve_cache:
            return self._internal_curve_cache[cache_key]

        curve = self._build_curve_for_time(
            pd.Timestamp(expiry_timestamp),
            emit_warning_summary=False,
        )
        self._internal_curve_cache[cache_key] = curve
        return curve

    def _internal_slice(self, expiry: str | date | pd.Timestamp) -> ProbCurve:
        """Return a no-warning probability slice for surface internals.

        Args:
            expiry: Expiry identifier as a date-like object.

        Returns:
            ProbCurve: Internal no-warning probability slice.
        """
        resolved_expiry, t_years = self._resolve_query_time(expiry)
        return self._internal_curve_for_time(resolved_expiry, t_years)

    def _evict_transient_cache_entries(
        self,
        preserved_keys: set[int],
        *,
        cache: dict[int, ProbCurve] | None = None,
    ) -> None:
        """Remove cache entries created by bulk exports or fan plotting.

        Args:
            preserved_keys: Cache keys that existed before the bulk operation.
            cache: Cache to prune. Defaults to the internal no-warning cache.
        """
        target_cache = self._internal_curve_cache if cache is None else cache
        for cache_key in list(target_cache):
            if cache_key not in preserved_keys:
                target_cache.pop(cache_key, None)

    def _materialized_curve_items_for_expiries(
        self,
        expiries: Any,
    ) -> list[tuple[int, ProbCurve]]:
        """Return materialized internal cache items for exact expiries.

        Args:
            expiries: Iterable of expiry-like values produced by one operation.

        Returns:
            list[tuple[int, ProbCurve]]: Cache-key and curve pairs for touched
            materialized slices only.
        """
        curve_items: list[tuple[int, ProbCurve]] = []
        seen_keys: set[int] = set()
        for expiry in expiries:
            cache_key = int(pd.Timestamp(expiry).value)
            if cache_key in seen_keys:
                continue
            seen_keys.add(cache_key)
            curve = self._internal_curve_cache.get(cache_key)
            if (
                curve is not None
                and getattr(curve, "_native_snapshot", None) is not None
            ):
                curve_items.append((cache_key, curve))
        return curve_items

    def _cdf_repair_events_from_materialized_curves(
        self,
        *,
        scope: str,
        curve_items: list[tuple[int, ProbCurve]] | None = None,
    ) -> list[WarningEvent]:
        """Build CDF repair events from already materialized cached slices.

        Args:
            scope: Diagnostic replacement scope for generated events.
            curve_items: Optional cache-key and curve pairs to inspect. When
                omitted, no curves are inspected.

        Returns:
            list[WarningEvent]: Repair events for cached slices with material
            direct-CDF repairs.
        """
        events: list[WarningEvent] = []
        pillar_expiries = {pd.Timestamp(expiry) for expiry in self.expiries}
        materialized_curve_items = [] if curve_items is None else curve_items
        for cache_key, curve in materialized_curve_items:
            snapshot = getattr(curve, "_native_snapshot", None)
            if snapshot is None:
                continue

            metadata = snapshot.metadata
            expiry_detail = metadata.get("expiry")
            expiry_timestamp = (
                pd.Timestamp(expiry_detail)
                if expiry_detail is not None
                else pd.Timestamp(cache_key)
            )
            warning_event = _cdf_repair_warning_event(
                metadata,
                scope=scope,
                expiry=expiry_timestamp,
                is_pillar=expiry_timestamp in pillar_expiries,
            )
            if warning_event is not None:
                events.append(warning_event)
        return events

    def _record_surface_warning_events(
        self,
        *,
        scope: str,
        owner: str,
        extra_events: list[WarningEvent] | None = None,
        curve_items: list[tuple[int, ProbCurve]] | None = None,
    ) -> None:
        """Replace and summarize surface warning diagnostics for one operation.

        Args:
            scope: Diagnostic replacement scope for the public operation.
            owner: Interface operation name shown in summary warnings.
            extra_events: Additional same-scope events such as fan skip reports.
            curve_items: Optional cache-key and curve pairs to inspect for CDF
                repair events. When omitted, no curves are inspected.
        """
        warning_events = self._cdf_repair_events_from_materialized_curves(
            scope=scope,
            curve_items=curve_items,
        )
        if extra_events:
            warning_events.extend(extra_events)
        _emit_warning_summaries(warning_events, owner=owner)
        self._warning_diagnostics._replace_scope_events(scope, warning_events)

    @classmethod
    def from_chain(
        cls,
        chain: pd.DataFrame,
        market: "MarketInputs",
        *,
        column_mapping: Optional[Mapping[str, str]] = None,
        max_staleness_days: int = 3,
        failure_policy: Literal["raise", "skip_warn"] = "skip_warn",
        cdf_violation_policy: Literal["warn", "raise"] = "warn",
    ) -> "ProbSurface":
        """Build a ProbSurface directly from a multi-expiry option chain.

        This convenience constructor fits SVI slices under the hood and then
        derives the risk-neutral distribution surface.

        Args:
            chain: Long-form option chain DataFrame containing multiple expiries,
                with one row per contract. Expected standard columns are
                ``strike``, ``expiry``, ``option_type``, and ``last_price``.
                Optional quote metadata includes ``bid``, ``ask``, ``volume``,
                and ``last_trade_date``. ``bid`` and ``ask`` can improve quote
                handling when present, but ``last_price`` is still required by
                the current input contract.
            market: Market inputs required for calibration.
            column_mapping: Optional mapping from input columns to OIPD
                standard names.
            max_staleness_days: Maximum age of quotes in days to include.
            failure_policy: Slice-level failure handling policy. Use ``"raise"``
                for strict mode or ``"skip_warn"`` for best-effort surface
                calibration.
            cdf_violation_policy: Direct-CDF monotonicity policy for lazy
                materialization. ``"warn"`` repairs and warns; ``"raise"``
                fails on material violations.

        Returns:
            ProbSurface: The fitted risk-neutral probability surface.

        Raises:
            ValueError: If ``failure_policy`` is not a supported value.
            CalculationError: If the chain has fewer than two expiries or the
                underlying volatility calibration fails.
        """
        if failure_policy not in {"raise", "skip_warn"}:
            raise ValueError(
                "failure_policy must be either 'raise' or 'skip_warn', "
                f"got {failure_policy!r}."
            )

        from oipd.interface.volatility import VolSurface

        vol_surface = VolSurface(method="svi", max_staleness_days=max_staleness_days)
        vol_surface.fit(
            chain,
            market,
            column_mapping=column_mapping,
            failure_policy=failure_policy,
        )
        prob_surface = ProbSurface(
            vol_surface=vol_surface,
            cdf_violation_policy=cdf_violation_policy,
        )
        prob_surface._warning_diagnostics = WarningDiagnostics(
            vol_surface.warning_diagnostics.events
        )
        return prob_surface

    def slice(self, expiry: str | date | pd.Timestamp) -> ProbCurve:
        """Return a ProbCurve for the requested maturity.

        Args:
            expiry: Expiry identifier as a date-like object.

        Returns:
            ProbCurve: Probability slice at the requested maturity.

        Raises:
            ValueError: If the surface is empty or interpolation context is unavailable.
        """

        if not isinstance(expiry, (str, date, pd.Timestamp)):
            raise ValueError(
                "slice(expiry) requires a date-like expiry "
                "(str, datetime.date, or pandas.Timestamp)."
            )

        resolved_expiry, t_years = self._resolve_query_time(expiry)
        return self._curve_for_time(resolved_expiry, t_years)

    @property
    def expiries(self) -> tuple[pd.Timestamp, ...]:
        """Return fitted expiries available on the probability surface."""
        return self._definition.expiries

    @property
    def warning_diagnostics(self) -> WarningDiagnostics:
        """Return structured warning diagnostics for the current surface state.

        Returns:
            WarningDiagnostics: Current warning diagnostic events and summary.
        """
        return self._warning_diagnostics

    def pdf(
        self,
        price: float | np.ndarray,
        t: float | str | date | pd.Timestamp,
    ) -> np.ndarray:
        """Evaluate PDF at requested price level(s) and maturity.

        Args:
            price: Price level(s) where the density is evaluated.
            t: Maturity input as year-fraction float or date-like object.

        Returns:
            np.ndarray: Interpolated PDF values at ``price``.
        """
        expiry_timestamp, t_years = self._resolve_query_time(t)
        curve = self._internal_curve_for_time(expiry_timestamp, t_years)
        query_prices = np.asarray(price, dtype=float)
        interpolated = np.interp(
            query_prices,
            curve.prices,
            curve.pdf_values,
            left=0.0,
            right=0.0,
        )
        self._record_surface_warning_events(
            scope=PROB_SURFACE_QUERY_WARNING_SCOPE,
            owner="ProbSurface.pdf",
            curve_items=[(int(pd.Timestamp(expiry_timestamp).value), curve)],
        )
        return np.asarray(interpolated, dtype=float)

    def __call__(
        self,
        price: float | np.ndarray,
        t: float | str | date | pd.Timestamp,
    ) -> np.ndarray:
        """Alias for :meth:`pdf`."""
        return self.pdf(price, t)

    def cdf(
        self,
        price: float | np.ndarray,
        t: float | str | date | pd.Timestamp,
    ) -> np.ndarray:
        """Evaluate CDF at requested price level(s) and maturity.

        Args:
            price: Price level(s) where the cumulative probability is evaluated.
            t: Maturity input as year-fraction float or date-like object.

        Returns:
            np.ndarray: Interpolated CDF values at ``price``.
        """
        expiry_timestamp, t_years = self._resolve_query_time(t)
        curve = self._internal_curve_for_time(expiry_timestamp, t_years)
        cdf_monotone = np.maximum.accumulate(np.asarray(curve.cdf_values, dtype=float))
        query_prices = np.asarray(price, dtype=float)
        interpolated = np.interp(
            query_prices,
            curve.prices,
            cdf_monotone,
            left=0.0,
            right=1.0,
        )
        self._record_surface_warning_events(
            scope=PROB_SURFACE_QUERY_WARNING_SCOPE,
            owner="ProbSurface.cdf",
            curve_items=[(int(pd.Timestamp(expiry_timestamp).value), curve)],
        )
        return np.asarray(interpolated, dtype=float)

    def quantile(
        self,
        q: float,
        t: float | str | date | pd.Timestamp,
    ) -> float:
        """Return the inverse-CDF price level at quantile ``q`` for maturity ``t``.

        Args:
            q: Target quantile in the open interval ``(0, 1)``.
            t: Maturity input as year-fraction float or date-like object.

        Returns:
            float: Price level corresponding to quantile ``q``.

        Raises:
            ValueError: If ``q`` is outside the open interval ``(0, 1)``.
        """
        if not (0 < q < 1):
            raise ValueError("Quantile q must be between 0 and 1.")

        expiry_timestamp, t_years = self._resolve_query_time(t)
        curve = self._internal_curve_for_time(expiry_timestamp, t_years)
        quantile_value = quantile_from_cdf(curve.prices, curve.cdf_values, q)
        self._record_surface_warning_events(
            scope=PROB_SURFACE_QUERY_WARNING_SCOPE,
            owner="ProbSurface.quantile",
            curve_items=[(int(pd.Timestamp(expiry_timestamp).value), curve)],
        )
        return quantile_value

    def density_results(
        self,
        domain: tuple[float, float] | None = None,
        points: int = 200,
        start: str | date | pd.Timestamp | None = None,
        end: str | date | pd.Timestamp | None = None,
        step_days: int | None = 1,
        *,
        full_domain: bool = False,
    ) -> pd.DataFrame:
        """Return a long-format DataFrame view of surface probability slices.

        Args:
            domain: Optional explicit export domain as ``(min_price, max_price)``.
            points: Number of resampled points for compact or explicit-domain
                exports. Ignored when ``full_domain`` returns native arrays
                exactly.
            start: Optional lower expiry bound. If omitted, uses the first fitted
                pillar expiry.
            end: Optional upper expiry bound. If omitted, uses the last fitted
                pillar expiry.
            step_days: Calendar-day sampling interval. Defaults to ``1`` so the
                export includes a daily grid. Fitted pillar expiries are always
                included even when they fall off the stepped schedule. Use
                ``None`` to export fitted pillars only.
            full_domain: If ``True`` and ``domain`` is omitted, each slice
                exports its native full-domain arrays exactly. Explicit
                ``domain`` always takes precedence.

        Returns:
            DataFrame with columns ``expiry``, ``price``, ``pdf``, and ``cdf``.
        """
        preserved_keys = set(self._internal_curve_cache)
        try:
            results_frame = build_surface_density_results_frame(
                self,
                domain=domain,
                points=points,
                start=start,
                end=end,
                step_days=step_days,
                full_domain=full_domain,
            )
            touched_curve_items = self._materialized_curve_items_for_expiries(
                results_frame["expiry"].drop_duplicates().tolist()
                if "expiry" in results_frame
                else []
            )
            self._record_surface_warning_events(
                scope=PROB_SURFACE_DENSITY_WARNING_SCOPE,
                owner="ProbSurface.density_results",
                curve_items=touched_curve_items,
            )
            return results_frame
        finally:
            self._evict_transient_cache_entries(preserved_keys)

    def plot_fan(
        self,
        *,
        figsize: tuple[float, float] = (10.0, 6.0),
        title: Optional[str] = None,
    ) -> Any:
        """Plot a fixed fan chart of risk-neutral quantiles across expiries.

        Args:
            figsize: Figure size as (width, height) in inches.
            title: Optional custom title for the plot.

        Returns:
            matplotlib.figure.Figure: The plot figure.

        Raises:
            ValueError: If the surface is empty or if no valid sampled slices
                remain after invalid fan slices are skipped.
        """
        if len(self.expiries) == 0:
            raise ValueError("Call fit before plotting the probability surface")
        preserved_keys = set(self._internal_curve_cache)
        try:
            summary_frame = build_fan_quantile_summary_frame(self)
            fan_skip_events = _fan_skip_warning_events(
                summary_frame.attrs.get("fan_skip_report"),
                scope=PROB_SURFACE_FAN_WARNING_SCOPE,
            )
            touched_curve_items = self._materialized_curve_items_for_expiries(
                summary_frame["expiry"].drop_duplicates().tolist()
                if "expiry" in summary_frame
                else []
            )
            self._record_surface_warning_events(
                scope=PROB_SURFACE_FAN_WARNING_SCOPE,
                owner="ProbSurface.plot_fan",
                extra_events=fan_skip_events,
                curve_items=touched_curve_items,
            )
        finally:
            self._evict_transient_cache_entries(preserved_keys)

        return plot_probability_summary(
            summary_frame,
            figsize=figsize,
            title=title,
        )


__all__ = ["ProbCurve", "ProbSurface"]
