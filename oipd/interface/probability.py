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
    build_global_log_moneyness_grid,
    build_interpolated_resolved_market,
    build_probcurve_metadata,
    build_surface_density_results_frame,
    derive_distribution_from_curve,
    derive_surface_distribution_at_t,
    quantile_from_cdf,
    resolve_surface_query_time,
)

if TYPE_CHECKING:
    from oipd.market_inputs import MarketInputs
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

        self._vol_curve = vol_curve
        if vol_curve is not None:
            self._resolved_market = vol_curve.resolved_market
            self._metadata = vol_curve._metadata or {}
        else:
            self._resolved_market = resolved_market
            self._metadata = metadata or {}

        # Cached grid values (lazy-loaded on property access)
        self._cached_prices: Optional[np.ndarray] = (
            np.asarray(prices, dtype=float) if prices is not None else None
        )
        self._cached_pdf: Optional[np.ndarray] = (
            np.asarray(pdf_values, dtype=float) if pdf_values is not None else None
        )
        self._cached_cdf: Optional[np.ndarray] = (
            np.asarray(cdf_values, dtype=float) if cdf_values is not None else None
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
        from oipd import VolCurve

        vol_curve = VolCurve(method="svi", max_staleness_days=max_staleness_days)
        vol_curve.fit(chain, market, column_mapping=column_mapping)
        return vol_curve.implied_distribution()

    def _ensure_grid_generated(self) -> None:
        """Lazily generate a default evaluation grid for array properties and plotting.

        Delegates to the stateless pipeline to generate a standard grid
        (based on ATM forward and time to expiry) if one hasn't been cached yet.
        This allows ``.pdf_values`` and ``.plot()`` to be used without providing explicit ranges.
        """
        if (
            self._cached_prices is not None
            and self._cached_pdf is not None
            and self._cached_cdf is not None
        ):
            return

        if self._vol_curve is None:
            raise ValueError("Probability arrays are unavailable for this curve.")

        prices, pdf, cdf, _ = derive_distribution_from_curve(
            self._vol_curve,
            self._resolved_market,
            pricing_engine=self._vol_curve.pricing_engine,
            vol_metadata=self._metadata,
        )

        self._cached_prices = prices
        self._cached_pdf = pdf
        self._cached_cdf = cdf

    def _require_cached_distribution_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached distribution arrays after enforcing lazy initialization.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Cached price, PDF, and CDF arrays.

        Raises:
            ValueError: If the cached arrays remain unavailable after initialization.
        """
        self._ensure_grid_generated()
        if (
            self._cached_prices is None
            or self._cached_pdf is None
            or self._cached_cdf is None
        ):
            raise ValueError("Probability arrays are unavailable for this curve.")
        return self._cached_prices, self._cached_pdf, self._cached_cdf

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

        return self._metadata

    def density_results(
        self,
        domain: tuple[float, float] | None = None,
        points: int = 200,
    ) -> pd.DataFrame:
        """Return a DataFrame view of the fitted probability density.

        Args:
            domain: Optional explicit export domain as ``(min_price, max_price)``.
            points: Number of resampled points when ``domain`` is set. Ignored
                when ``domain`` is omitted and the native fitted grid is used.

        Returns:
            DataFrame with columns ``price``, ``pdf``, and ``cdf``.
        """
        prices, pdf_values, cdf_values = self._require_cached_distribution_arrays()
        return build_density_results_frame(
            prices,
            pdf_values,
            cdf_values,
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
        """Plot the risk-neutral probability distribution.

        Args:
            kind: Which distribution(s) to plot: ``"pdf"``, ``"cdf"``, or ``"both"``.
            figsize: Figure size as (width, height) in inches.
            title: Optional custom title for the plot.
            xlim: Optional x-axis limits as (min, max).
            ylim: Optional y-axis limits as (min, max).
            points: Number of points in the grid (if generating dynamically).
            **kwargs: Additional arguments forwarded to ``oipd.presentation.plot_rnd.plot_rnd``.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        underlying_price = self._resolved_market.underlying_price
        valuation_date = format_timestamp_for_display(
            self._resolved_market.valuation_timestamp
        )

        expiry_raw = self._metadata.get("expiry")
        expiry_date = (
            format_timestamp_for_display(expiry_raw) if expiry_raw is not None else None
        )

        # Determine grid
        if xlim is not None:
            # Generate dynamic grid based on xlim
            grid_prices = np.linspace(xlim[0], xlim[1], points)
            grid_pdf = self.pdf(grid_prices)
            grid_cdf = np.array(
                [self.prob_below(p) for p in grid_prices]
            )  # list comp for scalar cdf
        else:
            # Use default cached grid
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
    """Multi-expiry risk-neutral probability surface wrapper."""

    def __init__(
        self,
        *,
        vol_surface: "VolSurface",
        grid_points: int = 241,
    ) -> None:
        """Initialize a ProbSurface directly from a fitted VolSurface.

        Args:
            vol_surface: Fitted volatility surface used as the canonical source.
            grid_points: Number of log-moneyness points in the unified strike grid.

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
        if grid_points < 5:
            raise ValueError("grid_points must be at least 5 for finite differences.")

        self._vol_surface = vol_surface
        self._market = vol_surface._market
        self._valuation_timestamp = self._market.valuation_timestamp
        self._k_grid = self._build_k_grid(points=grid_points)

        self._distribution_cache: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._curve_cache: dict[int, ProbCurve] = {}
        self._resolved_market_cache: dict[int, ResolvedMarket] = {}

    def _build_k_grid(self, *, points: int) -> np.ndarray:
        """Build a unified log-moneyness grid shared across all maturities.

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
        """Compute and cache a distribution slice for maturity ``t_years``.

        Args:
            t_years: Time to maturity in years.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Price grid, PDF, CDF.
        """
        if expiry_timestamp is None:
            expiry_timestamp, t_years = self._resolve_query_time(float(t_years))
        cache_key = int(pd.Timestamp(expiry_timestamp).value)
        if cache_key in self._distribution_cache:
            return self._distribution_cache[cache_key]

        result = derive_surface_distribution_at_t(
            self._vol_surface,
            t_years,
            log_moneyness_grid=self._k_grid,
        )
        self._distribution_cache[cache_key] = result
        return result

    def _resolved_market_for_t_years(
        self,
        t_years: float,
        *,
        expiry_timestamp: pd.Timestamp | None = None,
    ) -> ResolvedMarket:
        """Build and cache a synthetic resolved market snapshot for maturity ``t``.

        Args:
            t_years: Time to maturity in years.

        Returns:
            ResolvedMarket: Synthetic market snapshot aligned with maturity ``t``.
        """
        if expiry_timestamp is None:
            expiry_timestamp, t_years = self._resolve_query_time(float(t_years))
        cache_key = int(pd.Timestamp(expiry_timestamp).value)
        if cache_key in self._resolved_market_cache:
            return self._resolved_market_cache[cache_key]
        resolved_market = build_interpolated_resolved_market(
            self._vol_surface,
            t_years,
            expiry_timestamp=expiry_timestamp,
        )
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
            ProbCurve: Probability curve built from unified surface engine.
        """
        cache_key = int(pd.Timestamp(expiry_timestamp).value)
        if cache_key in self._curve_cache:
            return self._curve_cache[cache_key]

        prices, pdf_values, cdf_values = self._distribution_arrays_for_t_years(
            t_years,
            expiry_timestamp=expiry_timestamp,
        )
        metadata = build_probcurve_metadata(
            self._vol_surface,
            expiry_timestamp,
            t_years,
        )
        curve = ProbCurve.from_arrays(
            resolved_market=self._resolved_market_for_t_years(
                t_years,
                expiry_timestamp=expiry_timestamp,
            ),
            metadata=metadata,
            prices=prices,
            pdf_values=pdf_values,
            cdf_values=cdf_values,
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

        from oipd import VolSurface

        vol_surface = VolSurface(method="svi", max_staleness_days=max_staleness_days)
        vol_surface.fit(
            chain,
            market,
            column_mapping=column_mapping,
            failure_policy=failure_policy,
        )
        return vol_surface.implied_distribution()

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
        return self._vol_surface.expiries

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
        prices_grid, pdf_values, _ = self._distribution_arrays_for_t_years(
            t_years,
            expiry_timestamp=expiry_timestamp,
        )
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
        """Evaluate CDF at requested price level(s) and maturity.

        Args:
            price: Price level(s) where the cumulative probability is evaluated.
            t: Maturity input as year-fraction float or date-like object.

        Returns:
            np.ndarray: Interpolated CDF values at ``price``.
        """
        expiry_timestamp, t_years = self._resolve_query_time(t)
        prices_grid, _, cdf_values = self._distribution_arrays_for_t_years(
            t_years,
            expiry_timestamp=expiry_timestamp,
        )
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
        prices_grid, _, cdf_values = self._distribution_arrays_for_t_years(
            t_years,
            expiry_timestamp=expiry_timestamp,
        )
        return quantile_from_cdf(prices_grid, cdf_values, q)

    def density_results(
        self,
        domain: tuple[float, float] | None = None,
        points: int = 200,
        start: str | date | pd.Timestamp | None = None,
        end: str | date | pd.Timestamp | None = None,
        step_days: int | None = 1,
    ) -> pd.DataFrame:
        """Return a long-format DataFrame view of surface probability slices.

        Args:
            domain: Optional explicit export domain as ``(min_price, max_price)``.
            points: Number of resampled points when ``domain`` is set. Ignored
                when ``domain`` is omitted and the native slice grids are used.
            start: Optional lower expiry bound. If omitted, uses the first fitted
                pillar expiry.
            end: Optional upper expiry bound. If omitted, uses the last fitted
                pillar expiry.
            step_days: Calendar-day sampling interval. Defaults to ``1`` so the
                export includes a daily grid. Fitted pillar expiries are always
                included even when they fall off the stepped schedule. Use
                ``None`` to export fitted pillars only.

        Returns:
            DataFrame with columns ``expiry``, ``price``, ``pdf``, and ``cdf``.
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
        summary_frame = build_fan_quantile_summary_frame(self)

        return plot_probability_summary(
            summary_frame,
            figsize=figsize,
            title=title,
        )


__all__ = ["ProbCurve", "ProbSurface"]
