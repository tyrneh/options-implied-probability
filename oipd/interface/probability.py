"""Stateful probability estimators derived from fitted volatility curves."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError


from oipd.market_inputs import (
    ResolvedMarket,
)
from oipd.presentation.plot_rnd import plot_rnd
from oipd.presentation.probability_surface_plot import plot_probability_summary


from oipd.core.utils import calculate_days_to_expiry, calculate_time_to_expiry
from oipd.pipelines.probability import derive_distribution_from_curve


if TYPE_CHECKING:
    from oipd.market_inputs import MarketInputs


class ProbCurve:
    """Single-expiry risk-neutral probability curve wrapper.

    Computes PDF/CDF from an option chain using the stateless probability pipeline
    (golden-master aligned) and exposes convenience probability queries.
    """

    def __init__(
        self,
        vol_curve: Any,  # Type hint as Any to avoid circular import with VolCurve
    ) -> None:
        """Initialize a ProbCurve result container.

        Args:
            vol_curve: The fitted VolCurve instance.
        """
        self._vol_curve = vol_curve
        self._resolved_market = vol_curve.resolved_market
        self._metadata = vol_curve._metadata or {}
        
        # Cached grid values (lazy-loaded on property access)
        self._cached_prices: Optional[np.ndarray] = None
        self._cached_pdf: Optional[np.ndarray] = None
        self._cached_cdf: Optional[np.ndarray] = None

    @classmethod
    def from_chain(
        cls,
        chain: pd.DataFrame,
        market: "MarketInputs",
        *,
        column_mapping: Optional[Mapping[str, str]] = None,
    ) -> "ProbCurve":
        """Build a ProbCurve directly from a single-expiry option chain.

        This convenience constructor fits an SVI volatility curve under the hood
        and then derives the risk-neutral distribution.

        Args:
            chain: Option chain DataFrame containing a single expiry.
            market: Market inputs required for calibration.
            column_mapping: Optional mapping from input columns to OIPD
                standard names.

        Returns:
            ProbCurve: The fitted risk-neutral probability curve.

        Raises:
            ValueError: If the chain contains multiple expiries or invalid expiry values.
            CalculationError: If the underlying volatility calibration fails.
        """
        from oipd import VolCurve

        vol_curve = VolCurve(method="svi")
        vol_curve.fit(chain, market, column_mapping=column_mapping)
        return vol_curve.implied_distribution()

    def _ensure_grid_generated(self) -> None:
        """Lazily generate a default evaluation grid for array properties and plotting.

        Delegates to the stateless pipeline to generate a standard grid
        (based on ATM forward and time to expiry) if one hasn't been cached yet.
        This allows ``.pdf_values`` and ``.plot()`` to be used without providing explicit ranges.
        """
        if self._cached_prices is not None:
             return

        prices, pdf, cdf, _ = derive_distribution_from_curve(
            self._vol_curve, 
            self._resolved_market,
            pricing_engine=self._vol_curve.pricing_engine,
            vol_metadata=self._metadata
        )
        
        self._cached_prices = prices
        self._cached_pdf = pdf
        self._cached_cdf = cdf

    def pdf(self, price: float | np.ndarray) -> np.ndarray:
        """Evaluate the Probability Density Function (PDF) at the given price level(s).

        Args:
            price: Price level(s) to evaluate.

        Returns:
            np.ndarray: PDF values.
        """

        if self._vol_curve is None:
            raise ValueError("ProbCurve must be initialized with a fitted VolCurve")
        
        prices = np.asarray(price, dtype=float)
        
        # Breeden-Litzenberger: PDF = e^{rT} * d^2C / dK^2
        # We approximate d^2C/dK^2 using finite differences on the VolCurve.price() 
        # centered around the requested prices.
        
        # Use a small epsilon for finite difference
        # Ideally relative to strike, but absolute is safer for 0 strike edge cases
        h = 1e-4 * prices 
        h = np.maximum(h, 1e-4) # Floor epsilon
        
        # Vectorized 2nd derivative calculation
        # price(K+h) - 2*price(K) + price(K-h) / h^2
        
        # We need T and r for the discount factor
        r = self._resolved_market.risk_free_rate
        
        # Get T from metadata (standardized in VolCurve)
        if "time_to_expiry_years" in self._metadata:
             T = float(self._metadata["time_to_expiry_years"])
        elif "expiry_date" in self._metadata:
             # Recalculate if not cached
             T = calculate_time_to_expiry(self._metadata["expiry_date"], self._resolved_market.valuation_date)
        else:
             raise ValueError("Expiry date not found in metadata. Cannot calculate T.")
             
        # Clip T to avoid division by zero or extreme scaling
        T = max(T, 1e-5)
        
        factor = np.exp(r * T)
        
        # Calculate C(K+h), C(K), C(K-h)
        # Note: VolCurve.price() is vectorized
        c_up = self._vol_curve.price(prices + h, call_or_put="call")
        c_mid = self._vol_curve.price(prices, call_or_put="call")
        c_dn = self._vol_curve.price(prices - h, call_or_put="call")
        
        d2c_dk2 = (c_up - 2 * c_mid + c_dn) / (h ** 2)
        
        pdf_vals = factor * d2c_dk2
        
        return pdf_vals

    def __call__(self, price: float | np.ndarray) -> np.ndarray:
        """Alias for :meth:`pdf`."""
        return self.pdf(price)

    def prob_below(self, price: float) -> float:
        """Probability that the asset price is below ``price``.

        Args:
           price: Price level.

        Returns:
           float: Probability P(S < price).
        """

        # CDF = 1 + e^{rT} * dC/dK (for Calls)
        # dC/dK = (C(K+h) - C(K-h)) / 2h
        
        if self._vol_curve is None:
            raise ValueError("ProbCurve must be initialized with a fitted VolCurve")

        # Get T and r
        r = self._resolved_market.risk_free_rate
        if "time_to_expiry_years" in self._metadata:
             T = float(self._metadata["time_to_expiry_years"])
        else:
             T = calculate_time_to_expiry(self._metadata.get("expiry_date"), self._resolved_market.valuation_date)
        
        T = max(T, 1e-5)
        factor = np.exp(r * T)
        
        h = 1e-4 * price
        h = max(h, 1e-4)
        
        c_up = self._vol_curve.price(price + h, call_or_put="call")
        c_dn = self._vol_curve.price(price - h, call_or_put="call")
        
        dc_dk = (c_up - c_dn) / (2 * h)
        
        # CDF = 1 + e^{rT} * dC/dK
        cdf_val = 1.0 + factor * dc_dk
        
        return float(cdf_val)



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

        if self._vol_curve is None:
             raise ValueError("Call fit before accessing moments")
             
        # Moments require integration over a grid
        self._ensure_grid_generated()
        return float(np.trapz(self._cached_prices * self._cached_pdf, self._cached_prices))

    def variance(self) -> float:
        """Return the variance under the fitted PDF.

        Returns:
            float: Variance Var[S].
        """

        if self._vol_curve is None:
            raise ValueError("Call fit before accessing moments")
            
        self._ensure_grid_generated()
        mean = self.mean()
        return float(np.trapz(((self._cached_prices - mean) ** 2) * self._cached_pdf, self._cached_prices))

    def skew(self) -> float:
        """Return the skewness (3rd standardized moment) of the fitted PDF.

        Skew = E[(X - mu)^3] / sigma^3.

        Returns:
            float: Skewness. (Positive = lean to right/fat right tail, but for prices usually negative/fat left tail).
        """
        if self._vol_curve is None:
            raise ValueError("Call fit before accessing moments")

        self._ensure_grid_generated()
        mean = self.mean()
        var = self.variance()
        std = np.sqrt(var)

        moment3 = np.trapz(
            ((self._cached_prices - mean) ** 3) * self._cached_pdf, self._cached_prices
        )
        
        return float(moment3 / (std ** 3))

    def kurtosis(self) -> float:
        """Return the excess kurtosis (4th standardized moment - 3) of the fitted PDF.

        Excess Kurtosis = E[(X - mu)^4] / sigma^4 - 3.

        Returns:
            float: Excess Kurtosis (0 = Normal). Positive = Fat Tails.
        """
        if self._vol_curve is None:
            raise ValueError("Call fit before accessing moments")

        self._ensure_grid_generated()
        mean = self.mean()
        var = self.variance()
        std = np.sqrt(var)

        moment4 = np.trapz(
            ((self._cached_prices - mean) ** 4) * self._cached_pdf, self._cached_prices
        )
        
        # Excess Kurtosis
        return float(moment4 / (std ** 4) - 3.0)

    def quantile(self, q: float) -> float:
        """Return the price level at a given probability quantile (Inverse CDF).

        Args:
            q: Probability level (0 < q < 1).

        Returns:
            float: Price S such that P(Asset < S) = q.
        """
        if not (0 < q < 1):
             raise ValueError("Quantile q must be between 0 and 1")
             
        self._ensure_grid_generated()
        
        # Use interpolation on the cached CDF
        # cdf_values are y, prices are x. We want x for a given y.
        return float(np.interp(q, self._cached_cdf, self._cached_prices))



    @property
    def prices(self) -> np.ndarray:
        """Default price grid used for standard visualization.

        Returns:
            np.ndarray: Array of price levels.
        """

        self._ensure_grid_generated()
        return self._cached_prices

    @property
    def pdf_values(self) -> np.ndarray:
        """PDF values over the stored price grid.

        Returns:
            np.ndarray: Probability densities in decimals.
        """

        self._ensure_grid_generated()
        return self._cached_pdf

    @property
    def cdf_values(self) -> np.ndarray:
        """CDF values over the stored price grid.

        Returns:
            np.ndarray: Cumulative probabilities in decimals.
        """

        self._ensure_grid_generated()
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
    def metadata(self) -> dict[str, Any] | None:
        """Metadata captured during estimation (vol curve, diagnostics, etc.).

        Returns:
            dict[str, Any]: Metadata dictionary.
        """

        return self._metadata

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
        valuation_date = self._resolved_market.valuation_date.strftime("%b %d, %Y")
        
        # Get expiry_date from metadata (stored by vol_curve_pipeline)
        expiry_date_raw = self._metadata.get("expiry_date")
        expiry_date = expiry_date_raw.strftime("%b %d, %Y") if expiry_date_raw else None

        # Determine grid
        if xlim is not None:
             # Generate dynamic grid based on xlim
             grid_prices = np.linspace(xlim[0], xlim[1], points)
             grid_pdf = self.pdf(grid_prices)
             grid_cdf = np.array([self.prob_below(p) for p in grid_prices]) # list comp for scalar cdf
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
        distributions: Mapping[pd.Timestamp, ProbCurve],
    ) -> None:
        """Initialize a ProbSurface result container.

        Args:
            distributions: Dictionary mapping expiry timestamps to ProbCurve objects.
        """
        self._distributions = dict(distributions)
        # We can infer resolved markets from the distributions themselves
        self._resolved_markets = {
            ts: dist.resolved_market for ts, dist in self._distributions.items()
        }

    @classmethod
    def from_chain(
        cls,
        chain: pd.DataFrame,
        market: "MarketInputs",
        *,
        column_mapping: Optional[Mapping[str, str]] = None,
    ) -> "ProbSurface":
        """Build a ProbSurface directly from a multi-expiry option chain.

        This convenience constructor fits SVI slices under the hood and then
        derives the risk-neutral distribution surface.

        Args:
            chain: Option chain DataFrame containing multiple expiries.
            market: Market inputs required for calibration.
            column_mapping: Optional mapping from input columns to OIPD
                standard names.

        Returns:
            ProbSurface: The fitted risk-neutral probability surface.

        Raises:
            CalculationError: If the chain has fewer than two expiries or the
                underlying volatility calibration fails.
        """
        from oipd import VolSurface

        vol_surface = VolSurface(method="svi")
        vol_surface.fit(chain, market, column_mapping=column_mapping)
        return vol_surface.implied_distribution()

    def slice(self, expiry: Any) -> ProbCurve:
        """Return a ProbCurve (probability distribution function) for a specific expiry."""

        if not self._distributions:
            raise ValueError("Call fit before slicing the distribution surface")

        expiry_timestamp = pd.to_datetime(expiry).tz_localize(None)
        if expiry_timestamp not in self._distributions:
            raise ValueError(
                f"Expiry {expiry} not found in fitted probability surface"
            )
        return self._distributions[expiry_timestamp]

    @property
    def expiries(self) -> tuple[pd.Timestamp, ...]:
        """Return fitted expiries available on the probability surface."""

        return tuple(self._distributions.keys())

    def plot_fan(
        self,
        *,
        lower_percentile: float = 25.0,
        upper_percentile: float = 75.0,
        figsize: tuple[float, float] = (10.0, 6.0),
        title: Optional[str] = None,
    ) -> Any:
        """Plot a fan chart of risk-neutral quantiles across expiries.

        Args:
            lower_percentile: Lower percentile bound for the shaded band (0-100).
            upper_percentile: Upper percentile bound for the shaded band (0-100).
            figsize: Figure size as (width, height) in inches.
            title: Optional custom title for the plot.

        Returns:
            matplotlib.figure.Figure: The plot figure.

        Raises:
            ValueError: If the surface is empty or unavailable.
        """
        if not self._distributions:
            raise ValueError("Call fit before plotting the probability surface")

        frames: list[pd.DataFrame] = []
        for expiry, curve in self._distributions.items():
            strikes = np.asarray(curve.prices, dtype=float)
            cdf_values = np.asarray(curve.cdf_values, dtype=float)
            expiry_ts = pd.to_datetime(expiry)

            frame = pd.DataFrame(
                {
                    "expiry_date": np.full(strikes.shape, expiry_ts),
                    "strike": strikes,
                    "cdf": cdf_values,
                }
            )
            frames.append(frame)

        if not frames:
            raise ValueError("No probability slices available for plotting")

        density_data = pd.concat(frames, ignore_index=True)

        return plot_probability_summary(
            density_data,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            figsize=figsize,
            title=title,
        )


__all__ = ["ProbCurve", "ProbSurface"]
