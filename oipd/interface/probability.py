"""Stateful probability estimators derived from fitted volatility curves."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Mapping, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from oipd.core.maturity import format_timestamp_for_display
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
    ) -> None:
        """Initialize a ProbCurve result container.

        Args:
            vol_curve: The fitted VolCurve instance.
            resolved_market: Optional resolved market for array-backed construction.
            metadata: Optional metadata for array-backed construction.
            prices: Optional precomputed price grid.
            pdf_values: Optional precomputed PDF values.
            cdf_values: Optional precomputed CDF values.
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

        self._native_spec: MaterializationSpec = MaterializationSpec(points=None)
        self._definition: CurveProbabilityDefinition | None = None
        self._native_snapshot: DistributionSnapshot | None = None
        self._resolved_market: ResolvedMarket | None = None
        self._metadata: dict[str, Any] = {}

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
    ) -> "ProbCurve":
        """Build a lazy ProbCurve from an internal probability definition.

        Args:
            definition: Frozen single-expiry probability recipe.

        Returns:
            ProbCurve: Lazy probability curve backed by ``definition``.
        """
        instance = cls.__new__(cls)
        instance._native_spec = definition.native_spec
        instance._definition = definition
        instance._native_snapshot = None
        instance._resolved_market = definition.resolved_market
        instance._metadata = definition.vol_metadata
        return instance

    @classmethod
    def _from_vol_curve(
        cls,
        vol_curve: Any,
        *,
        native_spec: MaterializationSpec | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ProbCurve":
        """Build a lazy ProbCurve from a fitted VolCurve using an internal spec.

        Args:
            vol_curve: Fitted public VolCurve object.
            native_spec: Internal native materialization policy. When omitted,
                the public default policy is used.
            metadata: Optional metadata override.

        Returns:
            ProbCurve: Lazy probability curve for the fitted volatility curve.
        """
        definition = CurveProbabilityDefinition.from_vol_curve(
            vol_curve,
            native_spec=native_spec or MaterializationSpec(points=None),
            metadata=metadata,
        )
        return cls._from_definition(definition)

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
    ) -> "ProbCurve":
        """Build a ProbCurve directly from a single-expiry option chain.

        This convenience constructor fits an SVI volatility curve under the hood
        and then derives the risk-neutral distribution.

        Args:
            chain: Option chain DataFrame containing a single expiry.
            market: Market inputs required for calibration.
            column_mapping: Optional mapping from input columns to OIPD
                standard names.
            max_staleness_days: Maximum age of quotes in days to include.

        Returns:
            ProbCurve: The fitted risk-neutral probability curve.

        Raises:
            ValueError: If the chain contains multiple expiries or invalid expiry values.
            CalculationError: If the underlying volatility calibration fails.
        """
        from oipd.interface.volatility import VolCurve

        vol_curve = VolCurve(method="svi", max_staleness_days=max_staleness_days)
        vol_curve.fit(chain, market, column_mapping=column_mapping)
        return ProbCurve._from_vol_curve(vol_curve)

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

        self._native_snapshot = materialize_distribution_from_definition(
            self._definition,
            self._native_spec,
        )
        self._metadata = self._native_snapshot.metadata
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
    ) -> None:
        """Initialize a ProbSurface directly from a fitted VolSurface.

        Args:
            vol_surface: Fitted volatility surface used as the canonical source.
            grid_points: Native materialization grid size for each cached slice.
                ``None`` uses the smart native grid policy. Export
                ``density_results(points=...)`` controls downstream resampling,
                not this native grid.

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
        native_spec = MaterializationSpec(points=native_points)
        self._definition = SurfaceProbabilityDefinition.from_vol_surface(
            vol_surface,
            native_spec=native_spec,
        )
        self._vol_surface = self._definition.vol_surface
        self._market = self._vol_surface._market
        self._valuation_timestamp = self._market.valuation_timestamp
        self._grid_points = native_points
        self._curve_cache: dict[int, ProbCurve] = {}

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
        curve = self._curve_for_time(
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
        curve = self._curve_for_time(
            pd.Timestamp(expiry_timestamp),
            float(t_years),
        )
        return curve.resolved_market

    def _curve_for_time(
        self, expiry_timestamp: pd.Timestamp, t_years: float
    ) -> ProbCurve:
        """Return a cached ProbCurve slice for the requested maturity.

        Args:
            expiry_timestamp: Maturity timestamp.
            t_years: Time to maturity in years.

        Returns:
            ProbCurve: Probability curve built from the canonical single-slice path.
        """
        cache_key = int(pd.Timestamp(expiry_timestamp).value)
        if cache_key in self._curve_cache:
            return self._curve_cache[cache_key]

        vol_curve = self._vol_surface.slice(pd.Timestamp(expiry_timestamp))
        curve = ProbCurve._from_vol_curve(
            vol_curve,
            native_spec=self._definition.native_spec,
        )
        self._curve_cache[cache_key] = curve
        return curve

    def _evict_transient_cache_entries(self, preserved_keys: set[int]) -> None:
        """Remove cache entries created by bulk exports or fan plotting.

        Args:
            preserved_keys: Cache keys that existed before the bulk operation.
        """
        for cache_key in list(self._curve_cache):
            if cache_key not in preserved_keys:
                self._curve_cache.pop(cache_key, None)

    @classmethod
    def from_chain(
        cls,
        chain: pd.DataFrame,
        market: "MarketInputs",
        *,
        column_mapping: Optional[Mapping[str, str]] = None,
        max_staleness_days: int = 3,
        failure_policy: Literal["raise", "skip_warn"] = "skip_warn",
    ) -> "ProbSurface":
        """Build a ProbSurface directly from a multi-expiry option chain.

        This convenience constructor fits SVI slices under the hood and then
        derives the risk-neutral distribution surface.

        Args:
            chain: Option chain DataFrame containing multiple expiries.
            market: Market inputs required for calibration.
            column_mapping: Optional mapping from input columns to OIPD
                standard names.
            max_staleness_days: Maximum age of quotes in days to include.
            failure_policy: Slice-level failure handling policy. Use ``"raise"``
                for strict mode or ``"skip_warn"`` for best-effort surface
                calibration.

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
        return ProbSurface(vol_surface=vol_surface)

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
        curve = self._curve_for_time(expiry_timestamp, t_years)
        query_prices = np.asarray(price, dtype=float)
        interpolated = np.interp(
            query_prices,
            curve.prices,
            curve.pdf_values,
            left=0.0,
            right=0.0,
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
        curve = self._curve_for_time(expiry_timestamp, t_years)
        cdf_monotone = np.maximum.accumulate(np.asarray(curve.cdf_values, dtype=float))
        query_prices = np.asarray(price, dtype=float)
        interpolated = np.interp(
            query_prices,
            curve.prices,
            cdf_monotone,
            left=0.0,
            right=1.0,
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
        curve = self._curve_for_time(expiry_timestamp, t_years)
        return quantile_from_cdf(curve.prices, curve.cdf_values, q)

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
        preserved_keys = set(self._curve_cache)
        try:
            return build_surface_density_results_frame(
                self,
                domain=domain,
                points=points,
                start=start,
                end=end,
                step_days=step_days,
                full_domain=full_domain,
            )
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
        preserved_keys = set(self._curve_cache)
        try:
            summary_frame = build_fan_quantile_summary_frame(self)
        finally:
            self._evict_transient_cache_entries(preserved_keys)

        return plot_probability_summary(
            summary_frame,
            figsize=figsize,
            title=title,
        )


__all__ = ["ProbCurve", "ProbSurface"]
