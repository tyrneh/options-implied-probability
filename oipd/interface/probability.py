"""Stateful probability estimators derived from fitted volatility curves."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.pipelines.market_inputs import (
    FillMode,
    MarketInputs,
    ResolvedMarket,
    VendorSnapshot,
    resolve_market,
)
from oipd.pipelines.prob_estimation import derive_distribution_internal


class Distribution:
    """Single-expiry risk-neutral distribution wrapper.

    Computes PDF/CDF from an option chain using the stateless probability pipeline
    (golden-master aligned) and exposes convenience probability queries.
    """

    def __init__(
        self,
        *,
        method: str = "svi",
        method_options: Optional[Mapping[str, Any]] = None,
        solver: str = "brent",
        pricing_engine: str = "black76",
        price_method: str = "mid",
        max_staleness_days: int = 3,
    ) -> None:
        """Configure the distribution estimator.

        Args:
            method: Volatility fitting method (``"svi"`` or ``"bspline"``).
            method_options: Method-specific overrides (e.g., ``{"random_seed": 42}``).
            solver: Implied-vol solver (``"brent"`` or ``"newton"``).
            pricing_engine: Pricing engine (``"black76"`` or ``"bs"``).
            price_method: Price selection strategy (``"mid"`` or ``"last"``).
            max_staleness_days: Maximum allowed quote age before filtering.
        """

        self.method = method
        self.method_options = method_options
        self.solver = solver
        self.pricing_engine = pricing_engine
        self.price_method = price_method
        self.max_staleness_days = max_staleness_days

        self._prices: np.ndarray | None = None
        self._pdf: np.ndarray | None = None
        self._cdf: np.ndarray | None = None
        self._metadata: dict[str, Any] | None = None
        self._resolved_market: ResolvedMarket | None = None

    def fit(
        self,
        chain: pd.DataFrame,
        market: MarketInputs,
        *,
        vendor: Optional[VendorSnapshot] = None,
        fill_mode: FillMode = "strict",
        column_mapping: Optional[Mapping[str, str]] = None,
        method_options: Optional[Mapping[str, Any]] = None,
    ) -> "Distribution":
        """Compute the risk-neutral PDF/CDF and store on self.

        Args:
            chain: Option chain DataFrame; columns may be remapped via
                ``column_mapping``.
            market: User-supplied market inputs.
            vendor: Optional vendor snapshot to fill missing market fields.
            fill_mode: How to combine user/vendor inputs (``"strict"`` by default).
            column_mapping: Optional mapping from user column names to OIPD
                standard names.
            method_options: Per-fit overrides for the volatility calibration.

        Returns:
            The fitted ``Distribution`` instance.

        Raises:
            CalculationError: If the pipeline fails to produce a distribution.
        """

        chain_input = chain.copy()
        if column_mapping:
            chain_input = chain_input.rename(columns=column_mapping)

        resolved_market = resolve_market(market, vendor, mode=fill_mode)
        prices, pdf, cdf, metadata = derive_distribution_internal(
            chain_input,
            resolved_market,
            solver=self.solver,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
            method=self.method,
            method_options=method_options or self.method_options,
        )

        if prices is None or pdf is None or cdf is None:
            raise CalculationError("Probability estimation returned no results")

        self._prices = prices
        self._pdf = pdf
        self._cdf = cdf
        self._metadata = metadata
        self._resolved_market = resolved_market
        return self

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


class DistributionSurface:
    """Multi-expiry risk-neutral distribution surface wrapper."""

    def __init__(
        self,
        *,
        method: str = "svi",
        method_options: Optional[Mapping[str, Any]] = None,
        solver: str = "brent",
        pricing_engine: str = "black76",
        price_method: str = "mid",
        max_staleness_days: int = 3,
        expiration_column: str = "expiration",
    ) -> None:
        """Configure the distribution surface estimator.

        Args:
            method: Volatility fitting method (``"svi"`` or ``"bspline"``).
            method_options: Method-specific overrides.
            solver: Implied-vol solver (``"brent"`` or ``"newton"``).
            pricing_engine: Pricing engine (``"black76"`` or ``"bs"``).
            price_method: Price selection strategy (``"mid"`` or ``"last"``).
            max_staleness_days: Maximum allowed quote age before filtering.
            expiration_column: Column name holding option expiry values.
        """

        self.method = method
        self.method_options = method_options
        self.solver = solver
        self.pricing_engine = pricing_engine
        self.price_method = price_method
        self.max_staleness_days = max_staleness_days
        self.expiration_column = expiration_column

        self._distributions: dict[pd.Timestamp, Distribution] = {}
        self._resolved_markets: dict[pd.Timestamp, ResolvedMarket] = {}

    def fit(
        self,
        chain: pd.DataFrame,
        market: MarketInputs,
        *,
        vendor: Optional[VendorSnapshot] = None,
        fill_mode: FillMode = "strict",
        column_mapping: Optional[Mapping[str, str]] = None,
        method_options: Optional[Mapping[str, Any]] = None,
    ) -> "DistributionSurface":
        """Compute distributions for all expiries in the chain.

        Args:
            chain: Option chain containing multiple expiries.
            market: Base market inputs (valuation date, rates, underlying price).
            vendor: Optional vendor snapshot to fill missing fields.
            fill_mode: How to combine user/vendor inputs (``"strict"`` by default).
            column_mapping: Optional mapping from user column names to OIPD standard names.
            method_options: Per-fit overrides for the volatility calibration.

        Returns:
            DistributionSurface: The fitted surface instance.

        Raises:
            CalculationError: If expiries cannot be parsed or distributions fail.
        """

        chain_input = chain.copy()
        if column_mapping:
            chain_input = chain_input.rename(columns=column_mapping)

        if self.expiration_column not in chain_input.columns:
            raise CalculationError(
                f"Expiration column '{self.expiration_column}' not found in input data"
            )

        expiries = pd.to_datetime(chain_input[self.expiration_column], errors="coerce")
        if expiries.isna().any():
            raise CalculationError("Invalid expiration values encountered during parsing")
        expiries = expiries.dt.tz_localize(None)
        chain_input[self.expiration_column] = expiries

        self._distributions.clear()
        self._resolved_markets.clear()

        unique_expiries = sorted(expiries.unique())
        for expiry_ts in unique_expiries:
            expiry_date = expiry_ts.date()
            slice_df = chain_input[chain_input[self.expiration_column] == expiry_ts]

            slice_market = MarketInputs(
                risk_free_rate=market.risk_free_rate,
                valuation_date=market.valuation_date,
                risk_free_rate_mode=market.risk_free_rate_mode,
                underlying_price=market.underlying_price,
                dividend_yield=market.dividend_yield,
                dividend_schedule=market.dividend_schedule,
                expiry_date=expiry_date,
            )

            resolved_market = resolve_market(slice_market, vendor, mode=fill_mode)
            dist = Distribution(
                method=self.method,
                method_options=method_options or self.method_options,
                solver=self.solver,
                pricing_engine=self.pricing_engine,
                price_method=self.price_method,
                max_staleness_days=self.max_staleness_days,
            )
            dist.fit(
                slice_df,
                slice_market,
                vendor=vendor,
                fill_mode=fill_mode,
                column_mapping=None,
                method_options=method_options,
            )

            self._distributions[expiry_ts] = dist
            self._resolved_markets[expiry_ts] = resolved_market

        return self

    def slice(self, expiry: Any) -> Distribution:
        """Return a Distribution for a specific expiry."""

        if not self._distributions:
            raise ValueError("Call fit before slicing the distribution surface")

        expiry_ts = pd.to_datetime(expiry).tz_localize(None)
        if expiry_ts not in self._distributions:
            raise ValueError(f"Expiry {expiry} not found in fitted distribution surface")
        return self._distributions[expiry_ts]

    @property
    def expiries(self) -> tuple[pd.Timestamp, ...]:
        """Return fitted expiries available on the distribution surface."""

        return tuple(self._distributions.keys())


__all__ = ["Distribution", "DistributionSurface"]
