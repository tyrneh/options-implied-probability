"""Stateful volatility estimators wrapping stateless pipelines."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.interface.probability import Distribution, DistributionSurface
from oipd.pipelines.market_inputs import (
    FillMode,
    MarketInputs,
    ResolvedMarket,
    VendorSnapshot,
    resolve_market,
)
from oipd.pipelines.vol_estimation import fit_vol_curve_internal


class VolCurve:
    """Single-expiry implied-volatility estimator with sklearn-style API.

    Configure once, call ``fit`` to calibrate, then evaluate the fitted smile via
    ``__call__``. Heavy lifting is delegated to the stateless pipeline to keep
    behavior identical to the legacy implementation and the golden master.
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
        """Initialize a VolCurve with calibration configuration.

        Args:
            method: Smile fitting method (e.g., ``"svi"``).
            method_options: Optional method-specific options (e.g., random seed).
            solver: Implied-vol solver to use (``"brent"`` or ``"newton"``).
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

        self._vol_curve = None
        self._metadata: dict[str, Any] | None = None
        self._resolved_market: ResolvedMarket | None = None
        self._chain: pd.DataFrame | None = None

    def fit(
        self,
        chain: pd.DataFrame,
        market: MarketInputs,
        *,
        vendor: Optional[VendorSnapshot] = None,
        fill_mode: FillMode = "strict",
        column_mapping: Optional[Mapping[str, str]] = None,
        method_options: Optional[Mapping[str, Any]] = None,
    ) -> "VolCurve":
        """Fit the volatility smile and store fitted attributes on self.

        Args:
            chain: Option chain DataFrame. Column names can be remapped via
                ``column_mapping``.
            market: User-supplied market inputs (dates, rates, spot/forward).
            vendor: Optional vendor snapshot to fill missing market fields.
            fill_mode: How to combine user/vendor inputs (``"strict"`` by default).
            column_mapping: Optional mapping from user column names to OIPD
                standard names (e.g., ``{"type": "option_type"}``).
            method_options: Per-fit overrides for the calibration method. Falls
                back to the configuration supplied at initialization when not
                provided.

        Returns:
            VolCurve: The fitted estimator (for chaining).

        Raises:
            CalculationError: If calibration fails or produces no vol curve.
        """

        chain_input = chain.copy()
        if column_mapping:
            chain_input = chain_input.rename(columns=column_mapping)

        resolved_market = resolve_market(market, vendor, mode=fill_mode)
        vol_curve, metadata = fit_vol_curve_internal(
            chain_input,
            resolved_market,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
            solver=self.solver,
            method=self.method,
            method_options=method_options or self.method_options,
        )

        if vol_curve is None:
            raise CalculationError("Volatility calibration returned no curve")

        self._vol_curve = vol_curve
        self._metadata = metadata
        self._resolved_market = resolved_market
        self._chain = chain_input
        return self

    def __call__(self, strikes: np.ndarray | list[float]) -> np.ndarray:
        """Evaluate the fitted smile at the requested strikes.

        Args:
            strikes: Strike values (array-like).

        Returns:
            numpy.ndarray: Implied volatilities.

        Raises:
            ValueError: If ``fit`` has not been called.
        """

        if self._vol_curve is None:
            raise ValueError("Call fit before evaluating the curve")
        return self._vol_curve(np.asarray(strikes, dtype=float))

    @property
    def at_money_vol(self) -> float:
        """Return the at-the-money implied volatility.

        Returns:
            float: ATM implied volatility.

        Raises:
            ValueError: If ``fit`` has not been called.
        """
        if self._vol_curve is None:
            raise ValueError("Call fit before accessing ATM volatility")
        return getattr(self._vol_curve, "at_money_vol", np.nan)

    @property
    def params(self) -> dict[str, Any]:
        """Return fitted parameter dictionary from the underlying curve.
        
        Returns:
            dict[str, Any]: Fitted parameters (e.g., SVI ``a, b, rho, m, sigma``).

        Raises:
            ValueError: If ``fit`` has not been called.
        """

        if self._vol_curve is None:
            raise ValueError("Call fit before accessing parameters")
        return getattr(self._vol_curve, "params", {})

    @property
    def forward(self) -> float | None:
        """Return the parity-implied forward used in calibration if available."""

        if self._metadata is None:
            raise ValueError("Call fit before accessing forward")
        return self._metadata.get("forward_price")

    @property
    def diagnostics(self) -> dict[str, Any] | None:
        """Return calibration diagnostics captured during fitting."""

        if self._metadata is None:
            raise ValueError("Call fit before accessing diagnostics")
        return self._metadata.get("fit_diagnostics")

    @property
    def resolved_market(self) -> ResolvedMarket:
        """Return the resolved market snapshot used for calibration."""

        if self._resolved_market is None:
            raise ValueError("Call fit before accessing the resolved market")
        return self._resolved_market

    def implied_distribution(self) -> Distribution:
        """Return the risk-neutral distribution implied by the fitted smile.

        Returns:
            Distribution: Fitted distribution object.

        Raises:
            ValueError: If ``fit`` has not been called.
        """

        if self._chain is None or self._resolved_market is None or self._metadata is None:
            raise ValueError("Call fit before deriving the distribution")

        dist = Distribution(
            method=self.method,
            method_options=self.method_options,
            solver=self.solver,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
        )
        dist.fit(
            self._chain,
            MarketInputs(
                valuation_date=self._resolved_market.valuation_date,
                expiry_date=self._resolved_market.expiry_date,
                risk_free_rate=self._resolved_market.source_meta["risk_free_rate_input"],
                risk_free_rate_mode=self._resolved_market.source_meta["risk_free_rate_mode"],
                underlying_price=self._resolved_market.underlying_price,
                dividend_yield=self._resolved_market.dividend_yield,
                dividend_schedule=self._resolved_market.dividend_schedule,
            ),
            vendor=None,
            fill_mode="strict",
        )
        return dist


class VolSurface:
    """Multi-expiry volatility surface estimator.

    Calibrates individual expiries using the same pipeline as ``VolCurve`` and
    exposes ``slice(expiry)`` snapshots as ``VolCurve`` objects.
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
        expiration_column: str = "expiration",
    ) -> None:
        """Initialize a VolSurface with calibration configuration.

        Args:
            method: Smile fitting method (e.g., ``"svi"``).
            method_options: Optional method-specific options (e.g., random seed).
            solver: Implied-vol solver to use (``"brent"`` or ``"newton"``).
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

        self._slices: dict[pd.Timestamp, dict[str, Any]] = {}
        self._resolved_markets: dict[pd.Timestamp, ResolvedMarket] = {}
        self._slice_chains: dict[pd.Timestamp, pd.DataFrame] = {}

    def fit(
        self,
        chain: pd.DataFrame,
        market: MarketInputs,
        *,
        vendor: Optional[VendorSnapshot] = None,
        fill_mode: FillMode = "strict",
        column_mapping: Optional[Mapping[str, str]] = None,
        method_options: Optional[Mapping[str, Any]] = None,
    ) -> "VolSurface":
        """Fit all expiries in the chain and store slice curves.

        Args:
            chain: Option chain DataFrame containing multiple expiries.
            market: Base market inputs (valuation date, rates, underlying price).
            vendor: Optional vendor snapshot to fill missing market fields.
            fill_mode: How to combine user/vendor inputs (``"strict"`` by default).
            column_mapping: Optional mapping from user column names to OIPD standard names.
            method_options: Per-fit overrides for the calibration method.

        Returns:
            VolSurface: The fitted surface instance.

        Raises:
            CalculationError: If calibration fails for any expiry or expiration column is missing.
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

        unique_expiries = sorted(expiries.unique())
        self._slices.clear()
        self._resolved_markets.clear()
        self._slice_chains.clear()

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

            resolved = resolve_market(slice_market, vendor, mode=fill_mode)
            vol_curve, metadata = fit_vol_curve_internal(
                slice_df,
                resolved,
                pricing_engine=self.pricing_engine,
                price_method=self.price_method,
                max_staleness_days=self.max_staleness_days,
                solver=self.solver,
                method=self.method,
                method_options=method_options or self.method_options,
            )

            self._slices[expiry_ts] = {
                "curve": vol_curve,
                "metadata": metadata,
            }
            self._resolved_markets[expiry_ts] = resolved
            self._slice_chains[expiry_ts] = slice_df

        return self

    def slice(self, expiry: Any) -> VolCurve:
        """Return a VolCurve snapshot for the requested expiry.

        Args:
            expiry: Expiry identifier (string, date, or pandas-compatible timestamp).

        Returns:
            VolCurve: Snapshot carrying the fitted slice.

        Raises:
            ValueError: If fit has not been called or the expiry is unavailable.
        """

        if not self._slices:
            raise ValueError("Call fit before slicing the surface")

        expiry_ts = pd.to_datetime(expiry).tz_localize(None)
        if expiry_ts not in self._slices:
            raise ValueError(f"Expiry {expiry} not found in fitted surface")

        entry = self._slices[expiry_ts]
        resolved_market = self._resolved_markets[expiry_ts]
        chain_df = self._slice_chains[expiry_ts]

        vc = VolCurve(
            method=self.method,
            method_options=self.method_options,
            solver=self.solver,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
        )
        vc._vol_curve = entry["curve"]  # type: ignore[attr-defined]
        vc._metadata = entry["metadata"]  # type: ignore[attr-defined]
        vc._resolved_market = resolved_market  # type: ignore[attr-defined]
        vc._chain = chain_df  # type: ignore[attr-defined]
        return vc

    @property
    def expiries(self) -> tuple[pd.Timestamp, ...]:
        """Return the fitted expiries available on this surface."""

        return tuple(self._slices.keys())

    def implied_distribution(self) -> DistributionSurface:
        """Return the risk-neutral distribution surface for all fitted expiries.

        Returns:
            DistributionSurface: Surface with per-expiry distributions.

        Raises:
            ValueError: If ``fit`` has not been called.
        """

        if not self._slice_chains:
            raise ValueError("Call fit before deriving the distribution surface")

        dist_surface = DistributionSurface(
            method=self.method,
            method_options=self.method_options,
            solver=self.solver,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
            expiration_column=self.expiration_column,
        )

        combined = []
        for expiry_ts, df_slice in self._slice_chains.items():
            df = df_slice.copy()
            df[self.expiration_column] = expiry_ts
            combined.append(df)
        all_chain = pd.concat(combined, ignore_index=True)

        sample_resolved = next(iter(self._resolved_markets.values()))
        base_market = MarketInputs(
            valuation_date=sample_resolved.valuation_date,
            expiry_date=None,
            risk_free_rate=sample_resolved.source_meta["risk_free_rate_input"],
            risk_free_rate_mode=sample_resolved.source_meta["risk_free_rate_mode"],
            underlying_price=sample_resolved.underlying_price,
            dividend_yield=sample_resolved.dividend_yield,
            dividend_schedule=sample_resolved.dividend_schedule,
        )

        dist_surface.fit(
            all_chain,
            base_market,
            vendor=None,
            fill_mode="strict",
            column_mapping=None,
        )
        return dist_surface


__all__ = ["VolCurve", "VolSurface"]
