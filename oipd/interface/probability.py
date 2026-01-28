"""Stateful probability estimators derived from fitted volatility curves."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError


from oipd.market_inputs import (
    ResolvedMarket,
)


class ProbCurve:
    """Single-expiry risk-neutral probability curve wrapper.

    Computes PDF/CDF from an option chain using the stateless probability pipeline
    (golden-master aligned) and exposes convenience probability queries.
    """

    def __init__(
        self,
        prices: np.ndarray,
        pdf: np.ndarray,
        cdf: np.ndarray,
        market: ResolvedMarket,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a ProbCurve result container.

        Args:
            prices: Price grid.
            pdf: Probability density values.
            cdf: Cumulative distribution values.
            market: Resolved market snapshot.
            metadata: Optional metadata (provenance, diagnostics).
        """
        self._prices = np.asarray(prices, dtype=float)
        self._pdf = np.asarray(pdf, dtype=float)
        self._cdf = np.asarray(cdf, dtype=float)
        self._resolved_market = market
        self._metadata = metadata or {}

    def __call__(self, price: float | np.ndarray) -> np.ndarray:
        """Evaluate the PDF at the given price level(s)."""

        if self._prices is None or self._pdf is None:
            raise ValueError("Call fit before evaluating the distribution")
        return np.interp(np.asarray(price, dtype=float), self._prices, self._pdf)

    def prob_below(self, price: float) -> float:
        """Probability that the asset price is below ``price``."""

        if self._prices is None or self._cdf is None:
            raise ValueError("Call fit before querying probabilities")
        if price <= self._prices.min():
            return 0.0
        if price >= self._prices.max():
            return 1.0
        return float(np.interp(price, self._prices, self._cdf))

    def prob_above(self, price: float) -> float:
        """Probability that the asset price is at or above ``price``."""

        return 1.0 - self.prob_below(price)

    def prob_at_or_above(self, price: float) -> float:
        """Alias for prob_above."""
        return self.prob_above(price)

    def prob_between(self, low: float, high: float) -> float:
        """Probability that the asset price lies in ``[low, high]``."""

        if low > high:
            raise ValueError("low must be <= high")
        return max(self.prob_below(high) - self.prob_below(low), 0.0)

    def expected_value(self) -> float:
        """Return the expected value under the fitted PDF."""

        if self._prices is None or self._pdf is None:
            raise ValueError("Call fit before accessing moments")
        return float(np.trapz(self._prices * self._pdf, self._prices))

    def variance(self) -> float:
        """Return the variance under the fitted PDF."""

        if self._prices is None or self._pdf is None:
            raise ValueError("Call fit before accessing moments")
        mean = self.expected_value()
        return float(np.trapz(((self._prices - mean) ** 2) * self._pdf, self._prices))

    @property
    def prices(self) -> np.ndarray:
        """Price grid used for PDF/CDF."""

        if self._prices is None:
            raise ValueError("Call fit before accessing prices")
        return self._prices

    @property
    def pdf(self) -> np.ndarray:
        """PDF values over the stored price grid."""

        if self._pdf is None:
            raise ValueError("Call fit before accessing pdf")
        return self._pdf

    @property
    def cdf(self) -> np.ndarray:
        """CDF values over the stored price grid."""

        if self._cdf is None:
            raise ValueError("Call fit before accessing cdf")
        return self._cdf

    @property
    def resolved_market(self) -> ResolvedMarket:
        """Resolved market snapshot used for estimation."""

        if self._resolved_market is None:
            raise ValueError("Call fit before accessing the resolved market")
        return self._resolved_market

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata captured during estimation (vol curve, diagnostics, etc.)."""

        return self._metadata

    def plot(
        self,
        *,
        kind: Literal["pdf", "cdf", "both"] = "both",
        figsize: tuple[float, float] = (10.0, 5.0),
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> Any:
        """Plot the risk-neutral probability distribution.

        Args:
            kind: Which distribution(s) to plot: ``"pdf"``, ``"cdf"``, or ``"both"``.
            figsize: Figure size as (width, height) in inches.
            title: Optional custom title for the plot.
            xlim: Optional x-axis limits as (min, max).
            ylim: Optional y-axis limits as (min, max).
            **kwargs: Additional arguments forwarded to ``oipd.presentation.plot_rnd.plot_rnd``.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        from oipd.presentation.plot_rnd import plot_rnd

        underlying_price = self._resolved_market.underlying_price
        valuation_date = self._resolved_market.valuation_date.strftime("%b %d, %Y")
        
        # Get expiry_date from metadata (stored by vol_curve_pipeline)
        expiry_date_raw = self._metadata.get("expiry_date")
        expiry_date = expiry_date_raw.strftime("%b %d, %Y") if expiry_date_raw else None

        return plot_rnd(
            prices=self.prices,
            pdf=self.pdf,
            cdf=self.cdf,
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


__all__ = ["ProbCurve", "ProbSurface"]
