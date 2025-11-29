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
from oipd.pipelines.prob_estimation import derive_distribution_from_curve
from oipd.pipelines.vol_surface import fit_surface
from oipd.pipelines.vol_surface.models import FittedSurface


class VolCurve:
    """Single-expiry implied-volatility estimator with sklearn-style API.

    Configure once, call ``fit`` to calibrate, then evaluate the fitted smile via
    ``__call__``. Heavy lifting is delegated to the stateless pipeline.
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

        # resolved_market: A complete snapshot of market conditions (rates, spot, dividends)
        # derived by merging user inputs with vendor data (if provided).
        resolved_market = resolve_market(market, vendor, mode=fill_mode)

        # vol_curve: The fitted volatility model (callable) that maps strikes to implied vols.
        # metadata: A dictionary containing fit diagnostics (RMSE), the implied forward price,
        # and other artifacts from the calibration process.
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

        # Delegate to the stateless pipeline
        prices, pdf, cdf, metadata = derive_distribution_from_curve(
            self._vol_curve,
            self._resolved_market,
            pricing_engine=self.pricing_engine,
            vol_metadata=self._metadata,
        )

        # Return Result Container
        return Distribution(
            prices=prices,
            pdf=pdf,
            cdf=cdf,
            market=self._resolved_market,
            metadata=metadata,
        )


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
    ) -> None:
        """Initialize a VolSurface with calibration configuration.

        Args:
            method: Smile fitting method (e.g., ``"svi"``).
            method_options: Optional method-specific options (e.g., random seed).
            solver: Implied-vol solver to use (``"brent"`` or ``"newton"``).
            pricing_engine: Pricing engine (``"black76"`` or ``"bs"``).
            price_method: Price selection strategy (``"mid"`` or ``"last"``).
            max_staleness_days: Maximum allowed quote age before filtering.
            expiry_column: Column name holding option expiry values.
        """

        self.method = method
        self.method_options = method_options
        self.solver = solver
        self.pricing_engine = pricing_engine
        self.price_method = price_method
        self.max_staleness_days = max_staleness_days

        self._model: FittedSurface | None = None

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
            CalculationError: If calibration fails for any expiry or expiry column is missing.
        """

        chain_input = chain.copy()
        if column_mapping:
            chain_input = chain_input.rename(columns=column_mapping)

        if "expiry" not in chain_input.columns:
            raise CalculationError(
                "Expiry column 'expiry' not found in input data. "
                "Use column_mapping to map your expiry column to 'expiry'."
            )

        self._model = fit_surface(
            chain=chain_input,
            market=market,
            vendor=vendor,
            fill_mode=fill_mode,
            column_mapping=column_mapping,
            method_options=method_options or self.method_options,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
            solver=self.solver,
            method=self.method,
        )

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

        if self._model is None:
            raise ValueError("Call fit before slicing the surface")

        expiry_timestamp = pd.to_datetime(expiry).tz_localize(None)
        
        # Delegate to the model to get the slice data
        try:
            slice_data = self._model.get_slice(expiry_timestamp)
        except ValueError as e:
            # Re-raise with the same message format as before if needed, or just let it bubble
            raise ValueError(f"Expiry {expiry} not found in fitted surface") from e

        # slice_data is a dict with keys: "curve", "metadata", "resolved_market", "chain"

        vol_curve = VolCurve(
            method=self.method,
            method_options=self.method_options,
            solver=self.solver,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
        )
        vol_curve._vol_curve = slice_data["curve"]  # type: ignore[attr-defined]
        vol_curve._metadata = slice_data["metadata"]  # type: ignore[attr-defined]
        vol_curve._resolved_market = slice_data["resolved_market"]  # type: ignore[attr-defined]
        vol_curve._chain = slice_data["chain"]  # type: ignore[attr-defined]
        return vol_curve

    @property
    def expiries(self) -> tuple[pd.Timestamp, ...]:
        """Return the fitted expiries available on this surface."""

        if self._model is None:
            return ()
        return self._model.expiries

    def implied_distribution(self) -> DistributionSurface:
        """Return the risk-neutral distribution surface for all fitted expiries.

        Returns:
            DistributionSurface: Surface with per-expiry distributions.

        Raises:
            ValueError: If ``fit`` has not been called.
        """

        if self._model is None:
            raise ValueError("Call fit before deriving the distribution surface")

        distributions = {}
        
        # Iterate over fitted slices and derive distribution for each
        for expiry_timestamp in self.expiries:
            # We can use the slice() method to get a VolCurve, then ask it for distribution
            # This is slightly inefficient as it creates a VolCurve object just to discard it,
            # but it ensures consistent logic.
            # Alternatively, we can manually construct the VolCurve from stored state.
            
            # Let's use the slice() method for correctness and simplicity
            vol_curve = self.slice(expiry_timestamp)
            distributions[expiry_timestamp] = vol_curve.implied_distribution()

        return DistributionSurface(distributions)


__all__ = ["VolCurve", "VolSurface"]
