"""Stateful probability estimators derived from fitted volatility curves."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Mapping, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from oipd.market_inputs import ResolvedMarket
from oipd.presentation.plot_rnd import plot_rnd
from oipd.presentation.probability_surface_plot import plot_probability_summary

from oipd.core.utils import calculate_time_to_expiry
from oipd.pipelines.probability import (
    build_density_results_frame,
    build_daily_fan_density_frame,
    build_global_log_moneyness_grid,
    build_interpolated_resolved_market,
    build_probcurve_metadata,
    build_surface_density_results_frame,
    derive_distribution_from_curve,
    derive_surface_distribution_at_t,
    resolve_surface_query_time,
)


if TYPE_CHECKING:
    from oipd.market_inputs import MarketInputs
    from oipd.interface.volatility import VolSurface


class ProbCurve:
    """Single-date probability curve in physical or risk-neutral measure.

    Public constructors default to physical probabilities. Users can request a
    risk-neutral view explicitly or switch measures after creation.
    """

    def __init__(
        self,
        vol_curve: Any | None = None,
        *,
        resolved_market: Optional[ResolvedMarket] = None,
        metadata: Optional[dict[str, Any]] = None,
        prices: Optional[np.ndarray] = None,
        pdf_values: Optional[np.ndarray] = None,
        cdf_values: Optional[np.ndarray] = None,
        measure: Literal["physical", "risk_neutral"] = "physical",
        risk_aversion: Optional[float] = None,
        default_risk_aversion: float = 3.0,
        physical_cache: Optional[dict[float, tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        """Initialize a ProbCurve result container.

        Args:
            vol_curve: Optional fitted VolCurve instance for lazy RN generation.
            resolved_market: Optional resolved market for array-backed construction.
            metadata: Optional metadata dictionary.
            prices: Optional precomputed RN price grid.
            pdf_values: Optional precomputed RN PDF values.
            cdf_values: Optional precomputed RN CDF values.
            measure: Active measure for probability queries.
            risk_aversion: CRRA coefficient used for physical conversion.
            default_risk_aversion: Default CRRA coefficient used by
                ``to_physical`` when callers do not override it.
            physical_cache: Optional cache of precomputed physical distributions
                by risk-aversion key.

        Raises:
            ValueError: If neither a fitted curve nor resolved market is supplied.
        """
        if vol_curve is None and resolved_market is None:
            raise ValueError(
                "ProbCurve requires either a fitted vol_curve or resolved_market."
            )
        if measure not in {"physical", "risk_neutral"}:
            raise ValueError(
                "measure must be either 'physical' or 'risk_neutral', "
                f"got {measure!r}."
            )

        self._vol_curve = vol_curve
        if vol_curve is not None:
            self._resolved_market = vol_curve.resolved_market
            self._metadata: dict[str, Any] = dict(vol_curve._metadata or {})
        else:
            self._resolved_market = resolved_market
            self._metadata = dict(metadata or {})

        self._default_risk_aversion = float(default_risk_aversion)
        self._measure: Literal["physical", "risk_neutral"] = measure
        self._risk_aversion: Optional[float] = (
            float(risk_aversion)
            if risk_aversion is not None
            else (self._default_risk_aversion if measure == "physical" else None)
        )

        self._rn_prices: Optional[np.ndarray] = (
            np.asarray(prices, dtype=float) if prices is not None else None
        )
        self._rn_pdf: Optional[np.ndarray] = (
            np.asarray(pdf_values, dtype=float) if pdf_values is not None else None
        )
        self._rn_cdf: Optional[np.ndarray] = (
            np.asarray(cdf_values, dtype=float) if cdf_values is not None else None
        )

        self._physical_cache: dict[float, tuple[np.ndarray, np.ndarray]] = dict(
            physical_cache or {}
        )

        self._cached_prices: Optional[np.ndarray] = None
        self._cached_pdf: Optional[np.ndarray] = None
        self._cached_cdf: Optional[np.ndarray] = None

    @classmethod
    def from_arrays(
        cls,
        *,
        resolved_market: ResolvedMarket,
        metadata: dict[str, Any],
        prices: np.ndarray,
        pdf_values: np.ndarray,
        cdf_values: np.ndarray,
        measure: Literal["physical", "risk_neutral"] = "physical",
        risk_aversion: Optional[float] = None,
        default_risk_aversion: float = 3.0,
        physical_cache: Optional[dict[float, tuple[np.ndarray, np.ndarray]]] = None,
    ) -> "ProbCurve":
        """Build a ProbCurve from precomputed risk-neutral arrays.

        Args:
            resolved_market: Resolved market snapshot for the slice.
            metadata: Slice metadata dictionary.
            prices: RN strike grid.
            pdf_values: RN PDF values aligned with ``prices``.
            cdf_values: RN CDF values aligned with ``prices``.
            measure: Active measure for probability queries.
            risk_aversion: CRRA coefficient used when ``measure="physical"``.
            default_risk_aversion: Default CRRA coefficient for later
                ``to_physical`` calls.
            physical_cache: Optional cache of physical densities by
                risk-aversion key.

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
            measure=measure,
            risk_aversion=risk_aversion,
            default_risk_aversion=default_risk_aversion,
            physical_cache=physical_cache,
        )

    @classmethod
    def from_chain(
        cls,
        chain: pd.DataFrame,
        market: "MarketInputs",
        *,
        column_mapping: Optional[Mapping[str, str]] = None,
        max_staleness_days: int = 3,
        measure: Literal["physical", "risk_neutral"] = "physical",
        risk_aversion: float = 3.0,
    ) -> "ProbCurve":
        """Build a ProbCurve directly from a single-expiry option chain.

        This constructor fits an SVI volatility curve under the hood and then
        derives either physical or risk-neutral probabilities.

        Args:
            chain: Option chain DataFrame containing a single expiry.
            market: Market inputs required for calibration.
            column_mapping: Optional mapping from input columns to OIPD
                standard names.
            max_staleness_days: Maximum age of quotes in days to include.
            measure: Output measure, either physical or risk-neutral.
            risk_aversion: CRRA coefficient used when ``measure="physical"``.

        Returns:
            ProbCurve: The fitted probability curve in the requested measure.

        Raises:
            ValueError: If the chain contains multiple expiries or invalid expiry values.
            CalculationError: If underlying volatility calibration fails.
        """
        from oipd import VolCurve

        vol_curve = VolCurve(method="svi", max_staleness_days=max_staleness_days)
        vol_curve.fit(chain, market, column_mapping=column_mapping)
        return vol_curve.implied_distribution(
            measure=measure, risk_aversion=risk_aversion
        )

    @staticmethod
    def _risk_aversion_cache_key(risk_aversion: float) -> float:
        """Return a stable numeric key for risk-aversion caches.

        Args:
            risk_aversion: CRRA coefficient.

        Returns:
            float: Rounded risk-aversion key.
        """
        return round(float(risk_aversion), 12)

    def _ensure_rn_generated(self) -> None:
        """Ensure canonical RN arrays are available.

        Returns:
            None.

        Raises:
            ValueError: If RN arrays cannot be generated.
        """
        if (
            self._rn_prices is not None
            and self._rn_pdf is not None
            and self._rn_cdf is not None
        ):
            return

        if self._vol_curve is None:
            raise ValueError("Risk-neutral arrays are unavailable for this curve.")

        prices, pdf, cdf, metadata = derive_distribution_from_curve(
            self._vol_curve,
            self._resolved_market,
            pricing_engine=self._vol_curve.pricing_engine,
            vol_metadata=self._metadata,
        )
        self._rn_prices = np.asarray(prices, dtype=float)
        self._rn_pdf = np.asarray(pdf, dtype=float)
        self._rn_cdf = np.asarray(cdf, dtype=float)
        self._metadata.update(metadata)

        self._cached_prices = None
        self._cached_pdf = None
        self._cached_cdf = None

    def _ensure_physical_generated(self, risk_aversion: float) -> None:
        """Ensure physical arrays are available for the requested risk aversion.

        Args:
            risk_aversion: CRRA coefficient.

        Returns:
            None.
        """
        from oipd.core.probability_density_conversion import (
            physical_from_rn_crra,
        )

        risk_aversion_key = self._risk_aversion_cache_key(risk_aversion)
        if risk_aversion_key in self._physical_cache:
            return

        self._ensure_rn_generated()
        physical_pdf, physical_cdf = physical_from_rn_crra(
            self._rn_prices,
            self._rn_pdf,
            float(risk_aversion),
        )
        self._physical_cache[risk_aversion_key] = (
            np.asarray(physical_pdf, dtype=float),
            np.asarray(physical_cdf, dtype=float),
        )

    def _ensure_active_distribution(self) -> None:
        """Populate active arrays used by probability queries.

        Returns:
            None.
        """
        if (
            self._cached_prices is not None
            and self._cached_pdf is not None
            and self._cached_cdf is not None
        ):
            return

        self._ensure_rn_generated()
        if self._measure == "risk_neutral":
            self._cached_prices = np.asarray(self._rn_prices, dtype=float)
            self._cached_pdf = np.asarray(self._rn_pdf, dtype=float)
            self._cached_cdf = np.asarray(self._rn_cdf, dtype=float)
            return

        active_risk_aversion = float(
            self._risk_aversion
            if self._risk_aversion is not None
            else self._default_risk_aversion
        )
        self._ensure_physical_generated(active_risk_aversion)
        risk_aversion_key = self._risk_aversion_cache_key(active_risk_aversion)
        physical_pdf, physical_cdf = self._physical_cache[risk_aversion_key]
        self._cached_prices = np.asarray(self._rn_prices, dtype=float)
        self._cached_pdf = np.asarray(physical_pdf, dtype=float)
        self._cached_cdf = np.asarray(physical_cdf, dtype=float)

    def to_physical(self, *, risk_aversion: float = 3.0) -> "ProbCurve":
        """Return a new ProbCurve view in physical measure.

        Args:
            risk_aversion: CRRA coefficient used for the physical view.

        Returns:
            ProbCurve: New curve with active physical probabilities.
        """
        self._ensure_rn_generated()
        self._ensure_physical_generated(risk_aversion)
        return ProbCurve.from_arrays(
            resolved_market=self.resolved_market,
            metadata=dict(self._metadata),
            prices=np.asarray(self._rn_prices, dtype=float),
            pdf_values=np.asarray(self._rn_pdf, dtype=float),
            cdf_values=np.asarray(self._rn_cdf, dtype=float),
            measure="physical",
            risk_aversion=risk_aversion,
            default_risk_aversion=self._default_risk_aversion,
            physical_cache=dict(self._physical_cache),
        )

    def to_risk_neutral(self) -> "ProbCurve":
        """Return a new ProbCurve view in risk-neutral measure.

        Returns:
            ProbCurve: New curve with active risk-neutral probabilities.
        """
        self._ensure_rn_generated()
        return ProbCurve.from_arrays(
            resolved_market=self.resolved_market,
            metadata=dict(self._metadata),
            prices=np.asarray(self._rn_prices, dtype=float),
            pdf_values=np.asarray(self._rn_pdf, dtype=float),
            cdf_values=np.asarray(self._rn_cdf, dtype=float),
            measure="risk_neutral",
            risk_aversion=None,
            default_risk_aversion=self._default_risk_aversion,
            physical_cache=dict(self._physical_cache),
        )

    @property
    def measure(self) -> Literal["physical", "risk_neutral"]:
        """Return the active measure used by query methods.

        Returns:
            Literal["physical", "risk_neutral"]: Active measure.
        """
        return self._measure

    @property
    def risk_aversion(self) -> Optional[float]:
        """Return the active CRRA coefficient when using physical measure.

        Returns:
            Optional[float]: Risk-aversion value for physical view, otherwise ``None``.
        """
        if self._measure == "physical":
            return float(
                self._risk_aversion
                if self._risk_aversion is not None
                else self._default_risk_aversion
            )
        return None

    def pdf(self, price: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the Probability Density Function (PDF) at price level(s).

        Args:
            price: Price level(s) to evaluate.

        Returns:
            float | np.ndarray: PDF values in the active measure.
        """
        self._ensure_active_distribution()
        query_prices = np.asarray(price, dtype=float)
        interpolated = np.interp(
            query_prices,
            self._cached_prices,
            self._cached_pdf,
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
            float: Probability ``P(S < price)`` in the active measure.
        """
        self._ensure_active_distribution()
        cdf_monotone = np.maximum.accumulate(np.asarray(self._cached_cdf, dtype=float))
        return float(
            np.interp(
                price,
                np.asarray(self._cached_prices, dtype=float),
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
            float: Probability ``P(S >= price)`` in the active measure.
        """
        return 1.0 - self.prob_below(price)

    def prob_between(self, low: float, high: float) -> float:
        """Probability that the asset price lies in ``[low, high]``.

        Args:
            low: Lower bound price.
            high: Upper bound price.

        Returns:
            float: Probability ``P(low <= S < high)`` in the active measure.

        Raises:
            ValueError: If ``low`` exceeds ``high``.
        """
        if low > high:
            raise ValueError("low must be <= high")
        return self.prob_below(high) - self.prob_below(low)

    def mean(self) -> float:
        """Return the expected value under the active PDF.

        Returns:
            float: Expected price ``E[S]``.
        """
        self._ensure_active_distribution()
        return float(
            np.trapz(self._cached_prices * self._cached_pdf, self._cached_prices)
        )

    def variance(self) -> float:
        """Return the variance under the active PDF.

        Returns:
            float: Variance ``Var[S]``.
        """
        self._ensure_active_distribution()
        mean = self.mean()
        return float(
            np.trapz(
                ((self._cached_prices - mean) ** 2) * self._cached_pdf,
                self._cached_prices,
            )
        )

    def skew(self) -> float:
        """Return skewness of the active probability distribution.

        Returns:
            float: Third standardized moment.
        """
        self._ensure_active_distribution()
        mean = self.mean()
        var = self.variance()
        std = np.sqrt(var)
        moment3 = np.trapz(
            ((self._cached_prices - mean) ** 3) * self._cached_pdf, self._cached_prices
        )
        return float(moment3 / (std**3))

    def kurtosis(self) -> float:
        """Return excess kurtosis of the active probability distribution.

        Returns:
            float: Excess kurtosis ``E[(X-mu)^4]/sigma^4 - 3``.
        """
        self._ensure_active_distribution()
        mean = self.mean()
        var = self.variance()
        std = np.sqrt(var)
        moment4 = np.trapz(
            ((self._cached_prices - mean) ** 4) * self._cached_pdf, self._cached_prices
        )
        return float(moment4 / (std**4) - 3.0)

    def quantile(self, q: float) -> float:
        """Return the inverse-CDF price level for quantile ``q``.

        Args:
            q: Probability level in the open interval ``(0, 1)``.

        Returns:
            float: Price ``S`` such that ``P(S_t < S) = q``.

        Raises:
            ValueError: If ``q`` is outside ``(0, 1)``.
        """
        if not (0 < q < 1):
            raise ValueError("Quantile q must be between 0 and 1")
        self._ensure_active_distribution()
        return float(np.interp(q, self._cached_cdf, self._cached_prices))

    @property
    def prices(self) -> np.ndarray:
        """Return the price grid for the active measure view.

        Returns:
            np.ndarray: Price levels.
        """
        self._ensure_active_distribution()
        return self._cached_prices

    @property
    def pdf_values(self) -> np.ndarray:
        """Return active-measure PDF values on ``prices``.

        Returns:
            np.ndarray: Probability densities.
        """
        self._ensure_active_distribution()
        return self._cached_pdf

    @property
    def cdf_values(self) -> np.ndarray:
        """Return active-measure CDF values on ``prices``.

        Returns:
            np.ndarray: Cumulative probabilities.
        """
        self._ensure_active_distribution()
        return self._cached_cdf

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
    def metadata(self) -> dict[str, Any]:
        """Metadata captured during estimation and conversion.

        Returns:
            dict[str, Any]: Metadata dictionary with active measure annotations.
        """
        metadata = dict(self._metadata)
        metadata["measure"] = self._measure
        if self._measure == "physical":
            active_risk_aversion = float(
                self._risk_aversion
                if self._risk_aversion is not None
                else self._default_risk_aversion
            )
            metadata["risk_aversion"] = active_risk_aversion
            metadata["physical_transform"] = "crra_power_utility"
        else:
            metadata.pop("risk_aversion", None)
            metadata.pop("physical_transform", None)
        return metadata

    def density_results(
        self,
        domain: tuple[float, float] | None = None,
        points: int | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame view of the active probability density.

        Args:
            domain: Optional export domain as ``(min_price, max_price)``.
            points: Optional number of resampled points when ``domain`` is set.

        Returns:
            pd.DataFrame: DataFrame with columns ``price``, ``pdf``, and ``cdf``.
        """
        self._ensure_active_distribution()
        return build_density_results_frame(
            self._cached_prices,
            self._cached_pdf,
            self._cached_cdf,
            domain=domain,
            points=points,
        )

    def plot(
        self,
        *,
        kind: Literal["pdf", "cdf", "both"] = "both",
        figsize: tuple[float, float] = (10.0, 5.0),
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        points: int = 200,
        **kwargs,
    ) -> Any:
        """Plot the active probability distribution view.

        Args:
            kind: Which distribution(s) to plot: ``"pdf"``, ``"cdf"``, or ``"both"``.
            figsize: Figure size as (width, height) in inches.
            title: Optional custom title for the plot.
            xlim: Optional x-axis limits as (min, max).
            ylim: Optional y-axis limits as (min, max).
            points: Number of points in the grid (if generating dynamically).
            **kwargs: Additional arguments forwarded to ``plot_rnd``.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        market = self.resolved_market
        underlying_price = market.underlying_price
        valuation_date = market.valuation_date.strftime("%b %d, %Y")
        expiry_date_raw = self._metadata.get("expiry_date")
        expiry_date = expiry_date_raw.strftime("%b %d, %Y") if expiry_date_raw else None

        if xlim is not None:
            grid_prices = np.linspace(xlim[0], xlim[1], points)
            grid_pdf = self.pdf(grid_prices)
            grid_cdf = np.array([self.prob_below(p) for p in grid_prices])
        else:
            grid_prices = self.prices
            grid_pdf = self.pdf_values
            grid_cdf = self.cdf_values

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
    """Multi-date probability surface in physical or risk-neutral measure.

    Public constructors default to physical probabilities. Users can request a
    risk-neutral view explicitly or switch measures after creation.
    """

    def __init__(
        self,
        *,
        vol_surface: "VolSurface",
        grid_points: int = 241,
        measure: Literal["physical", "risk_neutral"] = "physical",
        risk_aversion: Optional[float] = None,
        default_risk_aversion: float = 3.0,
    ) -> None:
        """Initialize a ProbSurface directly from a fitted VolSurface.

        Args:
            vol_surface: Fitted volatility surface used as canonical source.
            grid_points: Number of log-moneyness points in the shared strike grid.
            measure: Active measure used by probability queries.
            risk_aversion: CRRA coefficient used when ``measure="physical"``.
            default_risk_aversion: Default CRRA coefficient used by
                ``to_physical`` when callers do not override it.

        Raises:
            ValueError: If ``vol_surface`` has not been fitted or measure is invalid.
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
        if grid_points < 5:
            raise ValueError("grid_points must be at least 5 for finite differences.")
        if measure not in {"physical", "risk_neutral"}:
            raise ValueError(
                "measure must be either 'physical' or 'risk_neutral', "
                f"got {measure!r}."
            )

        self._vol_surface = vol_surface
        self._market = vol_surface._market
        self._valuation_timestamp = pd.to_datetime(self._market.valuation_date)
        self._max_supported_t = calculate_time_to_expiry(
            max(vol_surface.expiries), self._market.valuation_date
        )
        self._k_grid = self._build_k_grid(points=grid_points)

        self._default_risk_aversion = float(default_risk_aversion)
        self._measure: Literal["physical", "risk_neutral"] = measure
        self._risk_aversion: Optional[float] = (
            float(risk_aversion)
            if risk_aversion is not None
            else (self._default_risk_aversion if measure == "physical" else None)
        )

        self._distribution_cache: dict[
            float, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._physical_distribution_cache: dict[
            tuple[float, float], tuple[np.ndarray, np.ndarray]
        ] = {}
        self._curve_cache: dict[tuple[str, Optional[float], float], ProbCurve] = {}
        self._resolved_market_cache: dict[float, ResolvedMarket] = {}

    @staticmethod
    def _time_cache_key(t_years: float) -> float:
        """Return a stable cache key for maturity.

        Args:
            t_years: Time to maturity in years.

        Returns:
            float: Rounded maturity key.
        """
        return round(float(t_years), 12)

    @staticmethod
    def _risk_aversion_cache_key(risk_aversion: float) -> float:
        """Return a stable cache key for risk aversion.

        Args:
            risk_aversion: CRRA coefficient.

        Returns:
            float: Rounded risk-aversion key.
        """
        return round(float(risk_aversion), 12)

    def _build_k_grid(self, *, points: int) -> np.ndarray:
        """Build a unified log-moneyness grid shared across maturities.

        Args:
            points: Number of points in the grid.

        Returns:
            np.ndarray: Uniform log-moneyness grid.
        """
        return build_global_log_moneyness_grid(self._vol_surface, points=points)

    def _resolve_query_time(
        self, t: float | str | date | pd.Timestamp
    ) -> tuple[pd.Timestamp, float]:
        """Normalize maturity input and enforce strict domain constraints.

        Args:
            t: Maturity input as year-fraction float or date-like object.

        Returns:
            tuple[pd.Timestamp, float]: Expiry timestamp and year fraction.
        """
        return resolve_surface_query_time(self._vol_surface, t)

    def _rn_distribution_arrays_for_t_years(
        self, t_years: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache RN distribution arrays for maturity ``t_years``.

        Args:
            t_years: Time to maturity in years.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Price grid, RN PDF, RN CDF.
        """
        cache_key = self._time_cache_key(t_years)
        if cache_key in self._distribution_cache:
            return self._distribution_cache[cache_key]

        result = derive_surface_distribution_at_t(
            self._vol_surface,
            t_years,
            log_moneyness_grid=self._k_grid,
        )
        self._distribution_cache[cache_key] = result
        return result

    def _physical_distribution_for_t_years(
        self, t_years: float, risk_aversion: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache physical distribution arrays for a maturity.

        Args:
            t_years: Time to maturity in years.
            risk_aversion: CRRA coefficient.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Price grid, physical PDF, physical CDF.
        """
        from oipd.core.probability_density_conversion import (
            physical_from_rn_crra,
        )

        t_key = self._time_cache_key(t_years)
        risk_aversion_key = self._risk_aversion_cache_key(risk_aversion)
        cache_key = (t_key, risk_aversion_key)
        rn_prices, rn_pdf, _ = self._rn_distribution_arrays_for_t_years(t_years)
        if cache_key not in self._physical_distribution_cache:
            physical_pdf, physical_cdf = physical_from_rn_crra(
                rn_prices,
                rn_pdf,
                float(risk_aversion),
            )
            self._physical_distribution_cache[cache_key] = (
                np.asarray(physical_pdf, dtype=float),
                np.asarray(physical_cdf, dtype=float),
            )

        physical_pdf, physical_cdf = self._physical_distribution_cache[cache_key]
        return (
            rn_prices,
            np.asarray(physical_pdf, dtype=float),
            np.asarray(physical_cdf, dtype=float),
        )

    def _distribution_arrays_for_t_years(
        self, t_years: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return active-measure distribution arrays for maturity ``t_years``.

        Args:
            t_years: Time to maturity in years.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Price grid, active PDF, active CDF.
        """
        if self._measure == "risk_neutral":
            return self._rn_distribution_arrays_for_t_years(t_years)

        active_risk_aversion = float(
            self._risk_aversion
            if self._risk_aversion is not None
            else self._default_risk_aversion
        )
        return self._physical_distribution_for_t_years(t_years, active_risk_aversion)

    def _resolved_market_for_t_years(self, t_years: float) -> ResolvedMarket:
        """Build and cache a synthetic resolved market snapshot for maturity ``t``.

        Args:
            t_years: Time to maturity in years.

        Returns:
            ResolvedMarket: Synthetic market snapshot aligned with maturity ``t``.
        """
        cache_key = self._time_cache_key(t_years)
        if cache_key in self._resolved_market_cache:
            return self._resolved_market_cache[cache_key]
        resolved_market = build_interpolated_resolved_market(self._vol_surface, t_years)
        self._resolved_market_cache[cache_key] = resolved_market
        return resolved_market

    def _curve_for_time(
        self, expiry_timestamp: pd.Timestamp, t_years: float
    ) -> ProbCurve:
        """Return a cached ProbCurve slice for the requested maturity.

        Args:
            expiry_timestamp: Maturity timestamp.
            t_years: Time to maturity in years.

        Returns:
            ProbCurve: Probability curve in the active measure.
        """
        t_key = self._time_cache_key(t_years)
        risk_aversion_key = (
            self._risk_aversion_cache_key(
                self._risk_aversion
                if self._risk_aversion is not None
                else self._default_risk_aversion
            )
            if self._measure == "physical"
            else None
        )
        cache_key = (self._measure, risk_aversion_key, t_key)
        if cache_key in self._curve_cache:
            return self._curve_cache[cache_key]

        rn_prices, rn_pdf, rn_cdf = self._rn_distribution_arrays_for_t_years(t_years)
        metadata = build_probcurve_metadata(
            self._vol_surface,
            expiry_timestamp,
            t_years,
        )
        resolved_market = self._resolved_market_for_t_years(t_years)

        if self._measure == "risk_neutral":
            curve = ProbCurve.from_arrays(
                resolved_market=resolved_market,
                metadata=metadata,
                prices=rn_prices,
                pdf_values=rn_pdf,
                cdf_values=rn_cdf,
                measure="risk_neutral",
                default_risk_aversion=self._default_risk_aversion,
            )
        else:
            active_risk_aversion = float(
                self._risk_aversion
                if self._risk_aversion is not None
                else self._default_risk_aversion
            )
            _, physical_pdf, physical_cdf = self._physical_distribution_for_t_years(
                t_years, active_risk_aversion
            )
            curve = ProbCurve.from_arrays(
                resolved_market=resolved_market,
                metadata=metadata,
                prices=rn_prices,
                pdf_values=rn_pdf,
                cdf_values=rn_cdf,
                measure="physical",
                risk_aversion=active_risk_aversion,
                default_risk_aversion=self._default_risk_aversion,
                physical_cache={
                    self._risk_aversion_cache_key(active_risk_aversion): (
                        physical_pdf,
                        physical_cdf,
                    )
                },
            )

        self._curve_cache[cache_key] = curve
        return curve

    @classmethod
    def from_chain(
        cls,
        chain: pd.DataFrame,
        market: "MarketInputs",
        *,
        column_mapping: Optional[Mapping[str, str]] = None,
        max_staleness_days: int = 3,
        failure_policy: Literal["raise", "skip_warn"] = "skip_warn",
        measure: Literal["physical", "risk_neutral"] = "physical",
        risk_aversion: float = 3.0,
    ) -> "ProbSurface":
        """Build a ProbSurface directly from a multi-expiry option chain.

        Args:
            chain: Option chain DataFrame containing multiple expiries.
            market: Market inputs required for calibration.
            column_mapping: Optional mapping from input columns to OIPD names.
            max_staleness_days: Maximum age of quotes in days to include.
            failure_policy: Slice-level failure handling policy.
            measure: Output measure, either physical or risk-neutral.
            risk_aversion: CRRA coefficient used when ``measure="physical"``.

        Returns:
            ProbSurface: Fitted probability surface in requested measure.

        Raises:
            ValueError: If ``failure_policy`` is unsupported.
        """
        if failure_policy not in {"raise", "skip_warn"}:
            raise ValueError(
                "failure_policy must be either 'raise' or 'skip_warn', "
                f"got {failure_policy!r}."
            )

        from oipd import VolSurface

        vol_surface = VolSurface(method="svi", max_staleness_days=max_staleness_days)
        vol_surface.fit(
            chain,
            market,
            column_mapping=column_mapping,
            failure_policy=failure_policy,
        )
        return vol_surface.implied_distribution(
            measure=measure, risk_aversion=risk_aversion
        )

    def _spawn(
        self,
        *,
        measure: Literal["physical", "risk_neutral"],
        risk_aversion: Optional[float],
    ) -> "ProbSurface":
        """Create a new ProbSurface view while preserving computed caches.

        Args:
            measure: Target active measure.
            risk_aversion: Target CRRA coefficient for physical measure.

        Returns:
            ProbSurface: New surface with copied canonical caches.
        """
        new_surface = ProbSurface(
            vol_surface=self._vol_surface,
            grid_points=len(self._k_grid),
            measure=measure,
            risk_aversion=risk_aversion,
            default_risk_aversion=self._default_risk_aversion,
        )
        new_surface._k_grid = np.asarray(self._k_grid, dtype=float)
        new_surface._distribution_cache = dict(self._distribution_cache)
        new_surface._physical_distribution_cache = dict(
            self._physical_distribution_cache
        )
        new_surface._resolved_market_cache = dict(self._resolved_market_cache)
        return new_surface

    def to_physical(self, *, risk_aversion: float = 3.0) -> "ProbSurface":
        """Return a new ProbSurface view in physical measure.

        Args:
            risk_aversion: CRRA coefficient used for the physical view.

        Returns:
            ProbSurface: New surface with active physical probabilities.
        """
        return self._spawn(measure="physical", risk_aversion=risk_aversion)

    def to_risk_neutral(self) -> "ProbSurface":
        """Return a new ProbSurface view in risk-neutral measure.

        Returns:
            ProbSurface: New surface with active risk-neutral probabilities.
        """
        return self._spawn(measure="risk_neutral", risk_aversion=None)

    @property
    def measure(self) -> Literal["physical", "risk_neutral"]:
        """Return active measure for probability queries.

        Returns:
            Literal["physical", "risk_neutral"]: Active measure.
        """
        return self._measure

    @property
    def risk_aversion(self) -> Optional[float]:
        """Return the active CRRA coefficient when in physical measure.

        Returns:
            Optional[float]: Active risk aversion for physical view, otherwise ``None``.
        """
        if self._measure == "physical":
            return float(
                self._risk_aversion
                if self._risk_aversion is not None
                else self._default_risk_aversion
            )
        return None

    def slice(self, expiry: str | date | pd.Timestamp) -> ProbCurve:
        """Return a ProbCurve slice at the requested maturity.

        Args:
            expiry: Expiry identifier as a date-like object.

        Returns:
            ProbCurve: Probability slice inheriting active measure and risk aversion.
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
        return self._vol_surface.expiries

    def pdf(
        self,
        price: float | np.ndarray,
        t: float | str | date | pd.Timestamp,
    ) -> np.ndarray:
        """Evaluate active-measure PDF at price level(s) and maturity.

        Args:
            price: Price level(s) where density is evaluated.
            t: Maturity as year-fraction float or date-like object.

        Returns:
            np.ndarray: Interpolated PDF values at ``price``.
        """
        _, t_years = self._resolve_query_time(t)
        prices_grid, pdf_values, _ = self._distribution_arrays_for_t_years(t_years)
        query_prices = np.asarray(price, dtype=float)
        interpolated = np.interp(
            query_prices, prices_grid, pdf_values, left=0.0, right=0.0
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
        """Evaluate active-measure CDF at price level(s) and maturity.

        Args:
            price: Price level(s) where cumulative probability is evaluated.
            t: Maturity as year-fraction float or date-like object.

        Returns:
            np.ndarray: Interpolated CDF values at ``price``.
        """
        _, t_years = self._resolve_query_time(t)
        prices_grid, _, cdf_values = self._distribution_arrays_for_t_years(t_years)
        cdf_monotone = np.maximum.accumulate(np.asarray(cdf_values, dtype=float))
        query_prices = np.asarray(price, dtype=float)
        interpolated = np.interp(
            query_prices, prices_grid, cdf_monotone, left=0.0, right=1.0
        )
        return np.asarray(interpolated, dtype=float)

    def quantile(
        self,
        q: float,
        t: float | str | date | pd.Timestamp,
    ) -> float:
        """Return inverse-CDF price level at quantile ``q`` for maturity ``t``.

        Args:
            q: Target quantile in the open interval ``(0, 1)``.
            t: Maturity as year-fraction float or date-like object.

        Returns:
            float: Price level corresponding to quantile ``q``.
        """
        if not (0 < q < 1):
            raise ValueError("Quantile q must be between 0 and 1.")
        _, t_years = self._resolve_query_time(t)
        prices_grid, _, cdf_values = self._distribution_arrays_for_t_years(t_years)
        cdf_monotone = np.maximum.accumulate(np.asarray(cdf_values, dtype=float))
        cdf_min = float(np.nanmin(cdf_monotone))
        cdf_max = float(np.nanmax(cdf_monotone))
        q_clamped = float(np.clip(q, cdf_min, cdf_max))
        return float(np.interp(q_clamped, cdf_monotone, prices_grid))

    def density_results(
        self,
        domain: tuple[float, float] | None = None,
        points: int | None = None,
        start: str | date | pd.Timestamp | None = None,
        end: str | date | pd.Timestamp | None = None,
        step_days: int | None = None,
    ) -> pd.DataFrame:
        """Return a long-format DataFrame view of surface probability slices.

        Args:
            domain: Optional export domain as ``(min_price, max_price)``.
            points: Optional number of resampled points when ``domain`` is set.
            start: Optional lower expiry bound.
            end: Optional upper expiry bound.
            step_days: Optional calendar-day sampling interval.

        Returns:
            pd.DataFrame: DataFrame with columns ``expiry``, ``price``, ``pdf``,
            and ``cdf``.
        """
        return build_surface_density_results_frame(
            self,
            domain=domain,
            points=points,
            start=start,
            end=end,
            step_days=step_days,
        )

    def plot_fan(
        self,
        *,
        lower_percentile: float = 25.0,
        upper_percentile: float = 75.0,
        figsize: tuple[float, float] = (10.0, 6.0),
        title: Optional[str] = None,
    ) -> Any:
        """Plot fan-chart quantiles across expiries for the active measure.

        Args:
            lower_percentile: Lower percentile bound for shaded band (0-100).
            upper_percentile: Upper percentile bound for shaded band (0-100).
            figsize: Figure size as (width, height) in inches.
            title: Optional custom title for the plot.

        Returns:
            matplotlib.figure.Figure: The plot figure.

        Raises:
            ValueError: If no probability slices are available for plotting.
        """
        if len(self.expiries) == 0:
            raise ValueError("Call fit before plotting the probability surface")

        density_data = build_daily_fan_density_frame(self)

        return plot_probability_summary(
            density_data,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            figsize=figsize,
            title=title,
        )


__all__ = ["ProbCurve", "ProbSurface"]
