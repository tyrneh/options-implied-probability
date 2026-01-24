"""Stateful volatility estimators wrapping stateless pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import warnings

from oipd.core.errors import CalculationError
from oipd.interface.probability import ProbCurve, ProbSurface
from oipd.market_inputs import (
    FillMode,
    MarketInputs,
    ResolvedMarket,
    VendorSnapshot,
    resolve_market,
)
from oipd.pipelines.vol_curve import fit_vol_curve_internal, compute_fitted_smile
from oipd.pipelines.probability import derive_distribution_from_curve
from oipd.pipelines.vol_surface import fit_surface
from oipd.pipelines.vol_surface.models import FittedSurface
from oipd.pipelines.vol_surface.interpolator import build_surface_interpolator

from oipd.presentation.iv_plotting import plot_iv_smile, ForwardPriceAnnotation


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
            suppress_price_warning=True,
        )

        if metadata.get("mid_price_filled"):
            warnings.warn(
                "Filled missing mid prices with last_price due to unavailable bid/ask",
                UserWarning,
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

    def iv_smile(
        self,
        domain: Optional[tuple[float, float]] = None,
        points: int = 200,
        include_observed: bool = True,
    ) -> pd.DataFrame:
        """Return the implied-volatility smile with fitted and market-observed values.

        Args:
            domain: Optional (min, max) strike range. If None, inferred from observed data.
            points: Number of points in the grid.
            include_observed: Whether to include market-observed IVs. If False, only
                ``strike`` and ``fitted_iv`` columns are returned.

        Returns:
            DataFrame containing:

            - ``strike``: Strike levels.
            - ``fitted_iv``: Fitted implied volatility from the calibrated model.
            - ``market_iv``: *(if include_observed=True)* Mid-price implied volatility
              computed by inverting the mid option price using the same pricing model
              (Black-76 or Black-Scholes), risk-free rate, dividend assumptions, and
              time-to-expiry as specified during ``.fit()``.
            - ``market_bid_iv``: *(if include_observed=True)* Bid-price implied volatility
              computed using the same methodology.
            - ``market_ask_iv``: *(if include_observed=True)* Ask-price implied volatility
              computed using the same methodology.
            - ``market_last_iv``: *(if include_observed=True)* Last-price implied volatility
              computed using the same methodology (only included if bid/ask are unavailable).

        Note:
            The market IVs are **not** raw quotes from exchanges—they are computed by
            the library by solving the inverse problem: "What σ makes the model price
            match the observed option price?" This ensures consistency with the fitted
            curve, which uses the same pricing assumptions.
        """
        return compute_fitted_smile(
            vol_curve=self,
            metadata=self._metadata,
            domain=domain,
            points=points,
            include_observed=include_observed,
        )

    def plot(
        self,
        *,
        x_axis: Literal["strike", "log_moneyness"] = "strike",
        y_axis: Literal["iv", "total_variance"] = "iv",
        include_observed: bool = True,
        figsize: tuple[float, float] = (10.0, 5.0),
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> Any:
        """Plot the fitted implied volatility smile.

        Args:
            include_observed: Whether to include observed market data points.
            x_axis: X-axis mode ("strike", "log_moneyness").
            y_axis: Metric to plot on y-axis ("iv" or "total_variance").
            figsize: Figure size as (width, height) in inches.
            title: Optional plot title.
            xlim: Optional x-axis limits as (min, max).
            ylim: Optional y-axis limits as (min, max).
            **kwargs: Additional arguments forwarded to ``oipd.presentation.iv_plotting.plot_iv_smile``.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """

        smile_df = self.iv_smile(include_observed=include_observed)

        # Map our column names to what plot_iv_smile expects
        plot_df = smile_df.rename(
            columns={
                "market_bid_iv": "bid_iv",
                "market_ask_iv": "ask_iv",
                "market_last_iv": "last_iv",
            }
        )

        # Extract forward price for log-moneyness plotting
        forward_price_annotation = None
        forward_price = self._metadata.get("forward_price")
        if forward_price is not None:
            forward_price_annotation = ForwardPriceAnnotation(
                value=float(forward_price), label=f"Forward: {float(forward_price):.2f}"
            )

        if x_axis == "log_moneyness":
            if forward_price is None or forward_price <= 0:
                raise ValueError("Positive forward price required for log-moneyness axis, but not found in fit metadata.")

        # Extract expiry date for default title generation
        expiry_date = None
        if self._resolved_market is not None:
            expiry_date = self._resolved_market.expiry_date

        # Calculate time to expiry for total variance conversion if needed
        t_to_expiry = None
        if self._resolved_market is not None:
            t_to_expiry = self._resolved_market.days_to_expiry / 365.0

        return plot_iv_smile(
            plot_df,
            forward_price=forward_price_annotation,
            show_forward=False,  # Can add an argument to VolCurve.plot if we want to expose this
            include_observed=include_observed,
            x_axis=x_axis,
            y_axis=y_axis,
            figsize=figsize,
            title=title,
            expiry_date=expiry_date,
            xlim=xlim,
            ylim=ylim,
            t_to_expiry=t_to_expiry,
            **kwargs,
        )

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

    def implied_distribution(self) -> ProbCurve:
        """Return the risk-neutral distribution implied by the fitted smile.

        Returns:
            ProbCurve: Fitted distribution object.

        Raises:
            ValueError: If ``fit`` has not been called.
        """

        if (
            self._chain is None
            or self._resolved_market is None
            or self._metadata is None
        ):
            raise ValueError("Call fit before deriving the distribution")

        # Delegate to the stateless pipeline
        prices, pdf, cdf, metadata = derive_distribution_from_curve(
            self._vol_curve,
            self._resolved_market,
            pricing_engine=self.pricing_engine,
            vol_metadata=self._metadata,
        )

        # Return Result Container
        return ProbCurve(
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
        self._interpolator: Any = None  # TotalVarianceInterpolator if enabled

    def fit(
        self,
        chain: pd.DataFrame,
        market: MarketInputs,
        *,
        vendor: Optional[VendorSnapshot] = None,
        fill_mode: FillMode = "strict",
        column_mapping: Optional[Mapping[str, str]] = None,
        method_options: Optional[Mapping[str, Any]] = None,
        horizon: Optional[Union[str, date, pd.Timestamp]] = None,
        interpolation: Literal["none", "linear"] = "none",
        check_arbitrage: bool = False,
    ) -> "VolSurface":
        """Fit all expiries in the chain and store slice curves.

        Args:
            chain: Option chain DataFrame containing multiple expiries.
            market: Base market inputs (valuation date, rates, underlying price).
            vendor: Optional vendor snapshot to fill missing market fields.
            fill_mode: How to combine user/vendor inputs (``"strict"`` by default).
            column_mapping: Optional mapping from user column names to OIPD standard names.
            method_options: Per-fit overrides for the calibration method.
            horizon: Optional fit horizon (e.g., "30d", "1y" or explicit date). 
                     Expiries after this horizon will be ignored.
            interpolation: If ``"linear"``, build a total variance interpolator for
                           continuous time queries. Default ``"none"`` for discrete slices only.
            check_arbitrage: If True and interpolation is enabled, clamps variance to
                             prevent negative forward variance (calendar arbitrage).

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

        # Apply horizon filtering
        if horizon is not None:
            # Resolve cutoff date
            if isinstance(horizon, (date, pd.Timestamp, str)):
                # formatted string date "YYYY-MM-DD"
                try:
                    cutoff = pd.to_datetime(horizon).tz_localize(None)
                except (ValueError, TypeError):
                    # Handle relative horizon strings
                    # Custom handling for "m" (months) and "y" (years) first
                    # because pd.to_timedelta("3m") means 3 minutes, not months.
                    val_date = pd.to_datetime(market.valuation_date).tz_localize(None)
                    
                    if isinstance(horizon, str) and horizon[-1].lower() in ("m", "y"):
                         amount = int(horizon[:-1])
                         unit = horizon[-1].lower()
                         if unit == "m":
                             cutoff = val_date + pd.DateOffset(months=amount)
                         elif unit == "y":
                             cutoff = val_date + pd.DateOffset(years=amount)
                         else:
                             # Should not reach here due to if condition
                             raise ValueError(f"Could not parse horizon string: {horizon}")
                    else:
                        # Fallback to pandas timedelta (supports "d", "w", etc.)
                        try:
                            delta = pd.to_timedelta(horizon)
                            cutoff = val_date + delta
                        except Exception:
                             raise ValueError(f"Could not parse horizon: {horizon}")
            else:
                raise TypeError("horizon must be a string, date, or Timestamp")

            # Filter chain
            # Ensure expiry col is datetime
            chain_input["expiry"] = pd.to_datetime(chain_input["expiry"]).dt.tz_localize(None)
            chain_input = chain_input[chain_input["expiry"] <= cutoff]

            if chain_input.empty:
                raise CalculationError(f"No expiries found within horizon {horizon} (cutoff {cutoff})")

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

        # Build interpolator if requested
        if interpolation == "linear":
            # Extract slices and forwards from the fitted model
            slices = {}
            forwards = {}
            for expiry_ts in self._model.expiries:
                slice_data = self._model.get_slice(expiry_ts)
                # Time in years
                days = slice_data["resolved_market"].days_to_expiry
                t = days / 365.0
                slices[t] = slice_data["curve"]
                forward = slice_data["metadata"].get("forward_price")
                if forward is None:
                    raise ValueError(
                        f"Forward price missing for expiry slice t={t:.4f}. Cannot build interpolator."
                    )
                forwards[t] = forward

            self._interpolator = build_surface_interpolator(
                slices, forwards, check_arbitrage=check_arbitrage
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


    def total_variance(self, K: float, t: float) -> float:
        """Return total variance at strike K and time t (years).

        Requires interpolation="linear" during fit.

        Args:
            K: Strike price.
            t: Time to maturity in years.

        Returns:
            Total variance w(K, t).

        Raises:
            ValueError: If interpolator is not available.
        """
        if self._interpolator is None:
            raise ValueError(
                "Interpolator not available. Fit with interpolation='linear'."
            )
        return self._interpolator(K, t)

    def implied_vol(self, K: float, t: float) -> float:
        """Return implied volatility at strike K and time t (years).

        Requires interpolation="linear" during fit.

        Args:
            K: Strike price.
            t: Time to maturity in years.

        Returns:
            Implied volatility sigma(K, t).

        Raises:
            ValueError: If interpolator is not available.
        """
        if self._interpolator is None:
            raise ValueError(
                "Interpolator not available. Fit with interpolation='linear'."
            )
        return self._interpolator.implied_vol(K, t)

    def __call__(self, K: float, t: float) -> float:
        """Return implied volatility at strike K and time t (years).

        Alias for implied_vol. Requires interpolation="linear" during fit.
        """
        return self.implied_vol(K, t)

    def implied_distribution(self) -> ProbSurface:
        """Return the risk-neutral distribution surface for all fitted expiries.

        Returns:
            ProbSurface: Surface with per-expiry distributions.

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

        return ProbSurface(distributions)

    def plot(
        self,
        *,
        x_axis: Literal["strike", "log_moneyness"] = "log_moneyness",
        y_axis: Literal["iv", "total_variance"] = "total_variance",
        figsize: tuple[float, float] = (10.0, 5.0),
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        label_format: Literal["date", "days"] = "date",
        **kwargs,
    ) -> Any:
        """Plot overlayed IV smiles for all fitted expiries.

        Args:
            x_axis: X-axis mode ("strike", "log_moneyness").
            y_axis: Metric to plot on y-axis ("iv" or "total_variance"). Defaults to "total_variance".
            figsize: Figure size as (width, height) in inches.
            title: Optional plot title.
            xlim: Optional x-axis limits as (min, max).
            ylim: Optional y-axis limits as (min, max).
            label_format: How to label each curve - ``"date"`` (e.g., "Jan 17, 2025")
                or ``"days"`` (e.g., "30d").
            **kwargs: Additional arguments forwarded to matplotlib plot calls.

        Returns:
            matplotlib.figure.Figure: The plot figure.

        Raises:
            ValueError: If ``fit`` has not been called.
        """
        if self._model is None:
            raise ValueError("Call fit before plotting the surface")

        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from exc

        # Apply publication style
        from oipd.presentation.publication import (
            _apply_publication_style,
            _style_publication_axes,
        )

        _apply_publication_style(plt)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot each expiry's IV smile
        for expiry_timestamp in self.expiries:
            vol_curve = self.slice(expiry_timestamp)
            smile_df = vol_curve.iv_smile(include_observed=False)

            # Compute x-axis values
            strikes = smile_df["strike"].to_numpy(dtype=float)
            if x_axis == "log_moneyness":
                forward = vol_curve._metadata.get("forward_price")
                if forward is None or forward <= 0:
                    raise ValueError(
                        "Forward price required for log-moneyness axis but not available"
                    )
                x_values = np.log(strikes / forward)
            else:
                x_values = strikes

            # Generate label
            if label_format == "days":
                resolved_market = vol_curve._resolved_market
                if resolved_market is not None:
                    days = resolved_market.days_to_expiry
                    label = f"{days}d"
                else:
                    label = expiry_timestamp.strftime("%b %d, %Y")
            else:
                label = expiry_timestamp.strftime("%b %d, %Y")

            iv_values = smile_df["fitted_iv"].to_numpy(dtype=float)
            y_values = iv_values

            if y_axis == "total_variance":
                resolved_market = vol_curve._resolved_market
                if resolved_market is None:
                    # Fallback if resolved_market missing, though unlikely after fit
                    warnings.warn(f"Missing market data for expiry {expiry_timestamp}, cannot compute total variance.", UserWarning)
                    y_values = np.full_like(iv_values, np.nan)
                else:
                    t_to_expiry = resolved_market.days_to_expiry / 365.0
                    y_values = np.square(iv_values) * t_to_expiry

            ax.plot(
                x_values,
                y_values,
                label=label,
                linewidth=1.5,
                **kwargs,
            )

        # Axis labels
        if x_axis == "log_moneyness":
            ax.set_xlabel("Log Moneyness (ln(K/F))", fontsize=11)
        else:
            ax.set_xlabel("Strike", fontsize=11)

        if y_axis == "total_variance":
             ax.set_ylabel("Total Variance", fontsize=11)
        else:
            ax.set_ylabel("Implied Volatility", fontsize=11)
            # Format y-axis as percentage only for IV
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

        # Apply limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)

        # Legend
        legend = ax.legend(loc="best", frameon=False)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#333333")

        # Apply publication styling to axes
        _style_publication_axes(ax)

        # Title
        resolved_title = title or "Implied Volatility Surface"
        if y_axis == "total_variance":
            resolved_title = resolved_title.replace("Implied Volatility", "Total Variance")
            
        plt.subplots_adjust(top=0.88)
        fig.suptitle(
            resolved_title,
            fontsize=16,
            fontweight="bold",
            y=0.95,
            color="#333333",
        )

        return fig


__all__ = ["VolCurve", "VolSurface"]
