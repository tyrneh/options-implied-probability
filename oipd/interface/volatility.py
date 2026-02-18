"""Stateful volatility estimators wrapping stateless pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

from oipd.core.errors import CalculationError
from oipd.core.utils import (
    calculate_days_to_expiry,
    resolve_horizon,
    calculate_time_to_expiry,
    convert_days_to_years,
    resolve_risk_free_rate,
)
from oipd.core.vol_surface_fitting import fit_slice

from oipd.pipelines.vol_curve import fit_vol_curve_internal
from oipd.pipelines.vol_surface import fit_surface as fit_vol_surface_internal
from oipd.presentation.publication import (
    _apply_publication_style,
    _style_publication_axes,
)
from oipd.pricing import get_pricer
from oipd.pricing.black76 import (
    black76_call_price,
    black76_delta,
    black76_gamma,
    black76_vega,
    black76_theta,
    black76_rho,
    ArrayLike,
)
from oipd.pricing.black_scholes import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
    black_scholes_rho,
)

from oipd.market_inputs import (
    MarketInputs,
    ResolvedMarket,
    Provenance,
    resolve_market,
)
from oipd.pipelines.vol_curve import fit_vol_curve_internal, compute_fitted_smile
from oipd.pipelines.probability import (
    derive_distribution_from_curve,
    resolve_surface_query_time,
)
from oipd.pipelines.vol_surface import fit_surface
from oipd.pipelines.vol_surface.models import FittedSurface
from oipd.pipelines.vol_surface.interpolator import (
    build_interpolator_from_fitted_surface,
)
from oipd.presentation.iv_plotting import ForwardPriceAnnotation, plot_iv_smile


class VolCurve:
    """Single-expiry implied-volatility smile fitter with sklearn-style API.

    Configure once, call ``fit`` to calibrate, then evaluate the fitted smile via
    ``__call__``.
    """

    def __init__(
        self,
        *,
        method: str = "svi",
        pricing_engine: str = "black76",
        price_method: str = "mid",
        max_staleness_days: int = 3,
    ) -> None:
        """Initialize a VolCurve single-expiry volatility fitter.

        Parameters
        ----------
        method : str, default "svi"
            Calibration algorithm to use (e.g., "svi").
        pricing_engine : str, default "black76"
            Pricing model for IV inversion: "black76" (futures) or "bs" (spot).
        price_method : str, default "mid"
            Quote type to fit against: "mid", "last", "bid", or "ask".
        max_staleness_days : int, default 3
            Maximum age of quotes in days (relative to valuation date) to include.
        """
        self.method = method
        self.method_options = None  # Hidden advanced configuration
        self.solver = "brent"  # Default solver, not exposed in __init__
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
        column_mapping: Optional[Mapping[str, str]] = None,
    ) -> "VolCurve":
        """Fit the volatility smile and store fitted attributes on self.

        Args:
            chain: Option chain DataFrame. Column names can be remapped via
                ``column_mapping``.
            market: User-supplied market inputs (dates, rates, spot/forward).
            column_mapping: Optional mapping from user column names to OIPD
                standard names (e.g., ``{"type": "option_type"}``).

        Returns:
            VolCurve: The fitted vol smile instance.

        Raises:
            ValueError: If the expiry column is missing, invalid, or contains multiple expiries.
            CalculationError: If calibration fails or produces no vol curve.
        """

        chain_input = chain.copy()
        if column_mapping:
            chain_input = chain_input.rename(columns=column_mapping)

        # Strict validation: Expiry column is mandatory because MarketInputs no longer carries it.
        # We need the expiry to calculate time-to-maturity (T) for the pricing engine.
        if "expiry" not in chain_input.columns:
            raise ValueError(
                "Input DataFrame must contain an 'expiry' column. "
                "Use column_mapping={'your_col': 'expiry'} if needed."
            )

        expiry_series = pd.to_datetime(chain_input["expiry"], errors="coerce")
        if expiry_series.isna().any():
            raise ValueError("Invalid expiry values encountered during parsing.")
        expiry_series = expiry_series.dt.tz_localize(None)
        chain_input["expiry"] = expiry_series

        unique_expiries = expiry_series.unique()
        if len(unique_expiries) != 1:
            raise ValueError(
                "VolCurve.fit requires a single expiry in the input chain. "
                f"Found {len(unique_expiries)} unique expiries. "
                "Use VolSurface.fit for multiple expiries."
            )

        expiry_date = pd.to_datetime(unique_expiries[0]).date()
        # NOTE: If we ever move to intraday time-to-expiry (minutes/seconds),
        # this guard should be revisited to allow same-day expiries.
        if expiry_date <= market.valuation_date:
            raise CalculationError(
                "Expiry must be strictly after valuation_date. "
                "Choose a later expiry to fit a volatility curve."
            )

        # resolved_market: A complete snapshot of market conditions (rates, spot, dividends)
        # derived from explicit user inputs.
        resolved_market = resolve_market(market)

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
            method_options=self.method_options,
            suppress_price_warning=True,
            suppress_staleness_warning=True,
        )

        # Handle warnings manually to ensure consistency
        if metadata.get("mid_price_filled"):
            count = metadata["mid_price_filled"]
            # It might be a boolean in older versions or legacy paths, handle graceful fallback
            count_str = f"{count}" if isinstance(count, int) and count > 1 else ""
            warnings.warn(
                f"Filled {count_str} missing mid prices with last_price due to unavailable bid/ask",
                UserWarning,
            )

        if metadata.get("staleness_report"):
            stats = metadata["staleness_report"]
            removed_count = stats.get("removed_count", 0)
            strike_desc = stats.get("strike_desc", "N/A")
            max_staleness = stats.get("max_staleness_days", self.max_staleness_days)
            min_age = stats.get("min_age", "N/A")
            max_age = stats.get("max_age", "N/A")

            warnings.warn(
                f"Filtered {removed_count} option rows (covering {strike_desc} strikes) "
                f"older than {max_staleness} days "
                f"(most recent: {min_age} days old, oldest: {max_age} days old)",
                UserWarning,
            )

        if vol_curve is None:
            raise CalculationError("Volatility calibration returned no curve")

        self._vol_curve = vol_curve
        self._metadata = metadata
        self._resolved_market = resolved_market
        self._chain = chain_input
        return self

    def implied_vol(self, strikes: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Return implied volatilities for the given strikes.

        Args:
            strikes: Strike price(s) to evaluate. Can be a single float or array-like.

        Returns:
            np.ndarray: Array of implied volatilities corresponding to the input strikes.

        Raises:
             ValueError: If ``fit`` has not been called.
        """
        if self._vol_curve is None:
            raise ValueError("Call fit before evaluating the curve")

        # Convert inputs to array
        # The underlying vol_curve (pipeline result) is a callable that expects floats/arrays
        return self._vol_curve(np.asarray(strikes, dtype=float))

    def __call__(self, strikes: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Alias for :meth:`implied_vol`."""
        return self.implied_vol(strikes)

    def total_variance(
        self, strikes: float | Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Return total variance (w = sigma^2 * T) for the given strikes.

        Args:
            strikes: Strike price(s) to evaluate.

        Returns:
            np.ndarray: Array of total variances.
        """
        sigma = self.implied_vol(strikes)

        # Calculate T
        if self._resolved_market is None or self._metadata is None:
            raise ValueError("Model not fitted. Cannot calculate T.")

        expiry = self._metadata.get("expiry_date")
        if expiry is None:
            # Should not happen for standard fitted curves
            raise ValueError("Expiry date not found in metadata.")

        T = calculate_time_to_expiry(expiry, self._resolved_market.valuation_date)
        # Ensure non-negative T logic is handled in calculate_days_to_expiry (returns 0 if expired)
        return (sigma**2) * T

    @property
    def atm_vol(self) -> float:
        """Return the At-The-Money (ATM) implied volatility.
        ATM is defined as Strike = Forward Price.

        Returns:
            float: Implied volatility at the forward strike (decimal, e.g. 0.20 = 20%).
        """
        if self._vol_curve is None:
            raise ValueError("Call fit before accessing ATM volatility")

        if self._metadata is None:
            raise ValueError("Call fit before accessing ATM volatility")

        atm_vol = self._metadata.get("at_money_vol")
        if atm_vol is None or not np.isfinite(atm_vol):
            raise ValueError(
                "ATM volatility missing or invalid in fit metadata. "
                "Check that calibration succeeded and produced at_money_vol."
            )
        return float(atm_vol)

    @property
    def expiries(self) -> tuple[Any]:
        """Return the single expiry of this curve as a tuple.

        Returns:
             tuple[Any]: A 1-element tuple containing the expiry (Timestamp or Date), or empty if invalid.
        """
        if self._metadata is None:
            return ()
        return (self._metadata.get("expiry_date"),)

    def iv_results(
        self,
        domain: Optional[tuple[float, float]] = None,
        points: int = 200,
        include_observed: bool = True,
    ) -> pd.DataFrame:
        """Return a DataFrame for inspecting the fitted smile against market data.

        Useful for plotting or debugging the calibration quality.

        Args:
            domain: (min_strike, max_strike) range. If None, inferred from data.
            points: Number of points to sample for the fitted curve.
            include_observed: Whether to include discrete market data points (Bid/Ask/Mid IV).

        Returns:
            pd.DataFrame: DataFrame with columns ['strike', 'fitted_iv', ...]
        """
        if self._vol_curve is None or self._metadata is None:
            raise ValueError("Call fit before inspecting the smile.")

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

        smile_df = self.iv_results(include_observed=include_observed)

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
                raise ValueError(
                    "Positive forward price required for log-moneyness axis, but not found in fit metadata."
                )

        # Extract expiry date for default title generation
        # expiry_date is stored in metadata from the vol_curve_pipeline
        expiry_date = self._metadata.get("expiry_date")

        # Calculate time to expiry for total variance conversion if needed
        t_to_expiry = None
        if self._resolved_market is not None and expiry_date is not None:
            t_to_expiry = calculate_time_to_expiry(
                expiry_date, self._resolved_market.valuation_date
            )

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
    def forward_price(self) -> float | None:
        """Return the parity-implied forward used in calibration if available.

        Returns:
            float: The forward price F, or None if not available/fitted.
        """

        if self._metadata is None:
            raise ValueError("Call fit before accessing forward")
        return self._metadata.get("forward_price")

    @property
    def diagnostics(self) -> dict[str, Any] | None:
        """Return calibration diagnostics captured during fitting.

        Returns:
             dict[str, Any]: Diagnostics dictionary (e.g. {"rmse": 0.001}).
        """

        if self._metadata is None:
            raise ValueError("Call fit before accessing diagnostics")
        diagnostics = self._metadata.get("diagnostics")
        if diagnostics is None:
            raise ValueError(
                "Diagnostics missing from fit metadata. "
                "Check that calibration succeeded and stored diagnostics."
            )
        return diagnostics

    @property
    def resolved_market(self) -> ResolvedMarket:
        """Return the resolved market snapshot used for calibration.

        Returns:
             ResolvedMarket: The standardized market inputs (dates, rates, spot).
        """

        if self._resolved_market is None:
            raise ValueError("Call fit before accessing the resolved market")
        return self._resolved_market

    def implied_distribution(self) -> ProbCurve:
        """Return the risk-neutral probability distribution implied by the fitted vol smile.

        Returns:
            ProbCurve: Fitted distribution object.

        Raises:
            ValueError: If ``fit`` has not been called.
        """

        # For interpolated slices, _chain is None but _metadata["interpolated"] is True
        is_interpolated = self._metadata is not None and self._metadata.get(
            "interpolated"
        )

        if not is_interpolated and (
            self._chain is None
            or self._resolved_market is None
            or self._metadata is None
        ):
            raise ValueError("Call fit before deriving the distribution")

        if is_interpolated and self._resolved_market is None:
            raise ValueError("Interpolated slice missing resolved market parameters")

        # Delegate to the stateless pipeline
        prices, pdf, cdf, metadata = derive_distribution_from_curve(
            self._vol_curve,
            self._resolved_market,
            pricing_engine=self.pricing_engine,
            vol_metadata=self._metadata,
        )

        # Return Result Container
        from oipd.interface.probability import ProbCurve

        return ProbCurve(self)

    def price(
        self,
        strikes: ArrayLike,
        call_or_put: Literal["call", "put"] = "call",
    ) -> np.ndarray:
        """Calculate theoretical option prices using the fitted volatility.

        Uses the pricing engine specified at initialization (default "black76").

        Args:
            strikes: Strike price(s) to value.
            call_or_put: Option type, "call" or "put".

        Returns:
            np.ndarray: Theoretical option prices.

        Raises:
            ValueError: If ``fit`` has not been called.
        """
        if self._resolved_market is None:
            raise ValueError("Call fit before pricing options")

        # 1. Gather Inputs
        K = np.asarray(strikes, dtype=float)
        sigma = self(K)

        # Determine time to expiry
        # If this is a synthetic curve, 'time_to_expiry_years' is in metadata
        if self._metadata and "time_to_expiry_years" in self._metadata:
            t = float(self._metadata["time_to_expiry_years"])
        else:
            # Otherwise derive from dates
            expiry_date = self._metadata.get("expiry_date") if self._metadata else None
            if expiry_date is None:
                raise ValueError("Expiry date not found in metadata")

            t = calculate_time_to_expiry(
                expiry_date, self._resolved_market.valuation_date
            )

        # Safety clamp for t
        t = max(t, 1e-6)

        rate_mode = self._resolved_market.source_meta["risk_free_rate_mode"]
        r = resolve_risk_free_rate(self._resolved_market.risk_free_rate, rate_mode, t)

        # 2. Select Engine & Dispatch
        # We allow "black76" (futures/forward) or "bs" (spot)
        engine_name = self.pricing_engine
        pricer = get_pricer(engine_name)

        if engine_name == "black76":
            # Black-76 requires Forward (F)
            F = self.forward_price
            if F is None:
                raise ValueError("Forward price not available for Black-76 pricing")

            # Helper handles the math: CallPrice = e^{-rT} [F N(d1) - K N(d2)]
            call_price = pricer(F, K, sigma, t, r)  # type: ignore

            if call_or_put == "put":
                # Put-Call Parity (Futures/Forward): P = C - e^{-rT}(F - K)
                # discount factor D = e^{-rT}
                # P = C - D*F + D*K
                df = np.exp(-r * t)
                put_price = call_price - df * (F - K)
                return put_price
            else:
                return call_price

        elif engine_name == "bs":
            # Black-Scholes requires Spot (S) and Div Yield (q)
            S = self._resolved_market.underlying_price
            q = self._resolved_market.dividend_yield
            if q is None:
                raise ValueError(
                    "Dividend yield (q) is required for Black-Scholes pricing but was not provided."
                )

            call_price = pricer(S, K, sigma, t, r, q)  # type: ignore

            if call_or_put == "put":
                # Put-Call Parity (Spot): P = C - S e^{-qT} + K e^{-rT}
                put_price = call_price - S * np.exp(-q * t) + K * np.exp(-r * t)
                return put_price
            else:
                return call_price

        else:
            raise ValueError(f"Unsupported pricing engine for .price(): {engine_name}")

    # -------------------------------------------------------------------------
    # Greeks
    # -------------------------------------------------------------------------

    def _get_greeks_inputs(
        self, strikes: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, float, float, float | None]:
        """Internal helper to gather common inputs for Greeks calculations."""
        if self._resolved_market is None:
            raise ValueError("Call fit before computing Greeks")

        K = np.asarray(strikes, dtype=float)
        sigma = self(K)

        # Time to expiry
        if self._metadata and "time_to_expiry_years" in self._metadata:
            t = float(self._metadata["time_to_expiry_years"])
        else:
            expiry_date = self._metadata.get("expiry_date") if self._metadata else None
            if expiry_date is None:
                raise ValueError("Expiry date not found in metadata")
            t = calculate_time_to_expiry(
                expiry_date, self._resolved_market.valuation_date
            )
        t = max(t, 1e-6)

        q = self._resolved_market.dividend_yield
        rate_mode = self._resolved_market.source_meta["risk_free_rate_mode"]
        r = resolve_risk_free_rate(self._resolved_market.risk_free_rate, rate_mode, t)
        return K, sigma, t, r, q

    def _dispatch_greek(
        self,
        func_black76: Callable,
        func_bs: Callable,
        strikes: ArrayLike,
        **kwargs,
    ) -> np.ndarray:
        """Dispatch controller: routes the calculation to the appropriate pricing kernel (Black-76 or Black-Scholes)."""
        K, sigma, t, r, q = self._get_greeks_inputs(strikes)

        if self.pricing_engine == "black76":
            F = self.forward_price
            if F is None:
                raise ValueError(f"Forward price required for Black-76 Greeks")
            return func_black76(F, K, sigma, t, r, **kwargs)
        else:  # bs
            S = self._resolved_market.underlying_price
            if q is None:
                raise ValueError(f"Dividend yield required for BS Greeks")
            return func_bs(S, K, sigma, t, r, q, **kwargs)

    def delta(
        self,
        strikes: ArrayLike,
        call_or_put: Literal["call", "put"] = "call",
    ) -> np.ndarray:
        """Calculate Delta (∂V/∂S or ∂V/∂F) for the fitted smile.

        Args:
            strikes: Strike price(s).
            call_or_put: Option type.

        Returns:
            np.ndarray: Delta values.
        """
        return self._dispatch_greek(
            black76_delta,
            black_scholes_delta,
            strikes,
            call_or_put=call_or_put,
        )

    def gamma(self, strikes: ArrayLike) -> np.ndarray:
        """Calculate Gamma (∂²V/∂S² or ∂²V/∂F²) for the fitted smile.

        Args:
            strikes: Strike price(s).

        Returns:
            np.ndarray: Gamma values (same for Call and Put).
        """
        return self._dispatch_greek(black76_gamma, black_scholes_gamma, strikes)

    def vega(self, strikes: ArrayLike) -> np.ndarray:
        """Calculate Vega (∂V/∂σ) for the fitted smile.

        Args:
            strikes: Strike price(s).

        Returns:
            np.ndarray: Vega values (same for Call and Put).
        """
        return self._dispatch_greek(black76_vega, black_scholes_vega, strikes)

    def theta(
        self,
        strikes: ArrayLike,
        call_or_put: Literal["call", "put"] = "call",
    ) -> np.ndarray:
        """Calculate Theta (∂V/∂t) for the fitted smile (per year).

        Args:
            strikes: Strike price(s).
            call_or_put: Option type.

        Returns:
            np.ndarray: Theta values (negative = time decay).
        """
        return self._dispatch_greek(
            black76_theta,
            black_scholes_theta,
            strikes,
            call_or_put=call_or_put,
        )

    def rho(
        self,
        strikes: ArrayLike,
        call_or_put: Literal["call", "put"] = "call",
    ) -> np.ndarray:
        """Calculate Rho (∂V/∂r) for the fitted smile.

        Args:
            strikes: Strike price(s).
            call_or_put: Option type.

        Returns:
            np.ndarray: Rho values.
        """
        return self._dispatch_greek(
            black76_rho,
            black_scholes_rho,
            strikes,
            call_or_put=call_or_put,
        )

    def greeks(
        self,
        strikes: ArrayLike,
        call_or_put: Literal["call", "put"] = "call",
    ) -> pd.DataFrame:
        """Calculate all Greeks for the fitted smile.

        Args:
            strikes: Strike price(s).
            call_or_put: Option type.

        Returns:
            pd.DataFrame: Table with columns [strike, delta, gamma, vega, theta, rho].
        """
        K = np.asarray(strikes, dtype=float)
        return pd.DataFrame(
            {
                "strike": K,
                "delta": self.delta(K, call_or_put),
                "gamma": self.gamma(K),
                "vega": self.vega(K),
                "theta": self.theta(K, call_or_put),
                "rho": self.rho(K, call_or_put),
            }
        )


class VolSurface:
    """Multi-expiry volatility surface fitter.

    Calibrates individual expiries using the same pipeline as ``VolCurve`` and
    exposes ``slice(expiry)`` snapshots as ``VolCurve`` objects.
    """

    def __init__(
        self,
        *,
        method: str = "svi",
        pricing_engine: str = "black76",
        price_method: str = "mid",
        max_staleness_days: int = 3,
    ) -> None:
        """Initialize a VolSurface multi-expiry volatility surface fitter.

        Parameters
        ----------
        method : str, default "svi"
            Calibration algorithm to use (e.g., "svi" or "cubic_spline").
        pricing_engine : str, default "black76"
            Pricing model for IV inversion: "black76" (futures) or "bs" (spot).
        price_method : str, default "mid"
            Quote type to fit against: "mid", "last", "bid", or "ask".
        max_staleness_days : int, default 3
            Maximum age of quotes in days (relative to valuation date) to include.
        """
        self.method = method
        self.method_options = None  # Hidden advanced configuration
        self.solver = "brent"  # Default solver, not exposed in __init__
        self.pricing_engine = pricing_engine
        self.price_method = price_method
        self.max_staleness_days = max_staleness_days

        self._model: FittedSurface | None = None
        self._interpolator: Any = None  # TotalVarianceInterpolator if enabled
        self._market: Optional[MarketInputs] = None  # Stored for interpolated slices

    def fit(
        self,
        chain: pd.DataFrame,
        market: MarketInputs,
        *,
        column_mapping: Optional[Mapping[str, str]] = None,
        horizon: Optional[Union[str, date, pd.Timestamp]] = None,
        failure_policy: Literal["raise", "skip_warn"] = "skip_warn",
    ) -> "VolSurface":
        """Fit all expiries in the chain and store slice curves.

        Args:
            chain: Option chain DataFrame containing multiple expiries.
            market: Explicit market parameters (price, rates, dividends).
            column_mapping: Optional mapping from user column names to OIPD standard names.
            horizon: Optional fit horizon (e.g., "30d", "1y" or explicit date).
                     Expiries after this horizon will be ignored.
            failure_policy: Slice-level failure handling policy. Use ``"raise"``
                to fail on the first problematic expiry or ``"skip_warn"`` to
                skip failures and continue fitting the surface.

        Returns:
            VolSurface: The fitted surface instance.

        Raises:
            ValueError: If ``failure_policy`` is not a supported value.
            CalculationError: If calibration fails, expiry column is missing or invalid,
                or fewer than two expiries are provided.
        """
        if failure_policy not in {"raise", "skip_warn"}:
            raise ValueError(
                "failure_policy must be either 'raise' or 'skip_warn', "
                f"got {failure_policy!r}."
            )

        chain_input = chain.copy()
        if column_mapping:
            chain_input = chain_input.rename(columns=column_mapping)

        if "expiry" not in chain_input.columns:
            raise CalculationError(
                "Expiry column 'expiry' not found in input data. "
                "Use column_mapping to map your expiry column to 'expiry'."
            )

        expiry_series = pd.to_datetime(chain_input["expiry"], errors="coerce")
        if expiry_series.isna().any():
            raise CalculationError("Invalid expiry values encountered during parsing.")
        expiry_series = expiry_series.dt.tz_localize(None)
        chain_input["expiry"] = expiry_series

        # Apply horizon filtering
        if horizon is not None:
            # Resolve cutoff date
            cutoff = resolve_horizon(horizon, market.valuation_date)

            # Filter chain
            chain_input = chain_input[chain_input["expiry"] <= cutoff]

            if chain_input.empty:
                raise CalculationError(
                    f"No expiries found within horizon {horizon} (cutoff {cutoff})"
                )

        drop_same_day_expiries = True
        # NOTE: If we ever move to intraday time-to-expiry (minutes/seconds),
        # this should be disabled so we can fit same-day slices.
        if drop_same_day_expiries:
            valuation_date = market.valuation_date
            chain_input = chain_input[chain_input["expiry"].dt.date > valuation_date]
            if chain_input.empty:
                raise CalculationError(
                    "All expiries are on or before valuation_date. "
                    "Surface fitting requires positive time to expiry."
                )

        unique_expiries = chain_input["expiry"].unique()
        if len(unique_expiries) < 2:
            raise CalculationError(
                "VolSurface.fit requires at least two unique expiries. "
                f"Found {len(unique_expiries)} expiry. "
                "Use VolCurve.fit for a single-expiry chain."
            )

        self._model = fit_surface(
            chain=chain_input,
            market=market,
            column_mapping=column_mapping,
            method_options=self.method_options,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
            solver=self.solver,
            method=self.method,
            failure_policy=failure_policy,
        )

        # Always build linear total variance interpolator
        self._interpolator = build_interpolator_from_fitted_surface(
            self._model, check_arbitrage=True
        )

        # Store market for interpolated slice creation
        self._market = market

        return self

    def slice(self, expiry: str | date | pd.Timestamp) -> VolCurve:
        """Return a VolCurve snapshot for the requested maturity.

        If the maturity maps to a fitted pillar, returns the original parametric
        curve. Otherwise, returns a synthetic curve derived from the Total
        Variance interpolator.

        Args:
            expiry: Expiry identifier as a date-like object.

        Returns:
            VolCurve: A VolCurve object representing the smile at that expiry.

        Raises:
            ValueError: If fit has not been called or maturity is unsupported.
        """

        if self._model is None:
            raise ValueError("Call fit before slicing the surface")

        if not isinstance(expiry, (str, date, pd.Timestamp)):
            raise ValueError(
                "slice(expiry) requires a date-like expiry "
                "(str, datetime.date, or pandas.Timestamp)."
            )

        expiry_timestamp, _ = self._resolve_query_time(expiry)

        # Check if this is an exact match to a fitted expiry
        if expiry_timestamp in self.expiries:
            # Return the real fitted VolCurve
            slice_data = self._model.get_slice(expiry_timestamp)

            vol_curve = VolCurve(
                method=self.method,
                pricing_engine=self.pricing_engine,
                price_method=self.price_method,
                max_staleness_days=self.max_staleness_days,
            )
            vol_curve.method_options = self.method_options
            vol_curve.solver = self.solver
            vol_curve._vol_curve = slice_data["curve"]  # type: ignore[attr-defined]
            vol_curve._metadata = slice_data["metadata"]  # type: ignore[attr-defined]
            vol_curve._resolved_market = slice_data["resolved_market"]  # type: ignore[attr-defined]
            vol_curve._chain = slice_data["chain"]  # type: ignore[attr-defined]
            return vol_curve

        # Interpolated slice: create a synthetic VolCurve
        return self._create_interpolated_slice(expiry_timestamp)

    def _create_interpolated_slice(self, expiry_timestamp: pd.Timestamp) -> VolCurve:
        """Create a synthetic VolCurve for an arbitrary expiry via interpolation.

        Args:
            expiry_timestamp: The target expiry date.

        Returns:
            VolCurve: A synthetic curve that evaluates IVs using the interpolator.
        """
        if self._interpolator is None:
            raise ValueError("Interpolator not available")
        if self._market is None:
            raise ValueError("Market data not stored")

        # Enforce strict maturity domain shared across all surface methods.
        resolved_expiry_timestamp, t = self._resolve_query_time(expiry_timestamp)
        days = calculate_days_to_expiry(
            resolved_expiry_timestamp, self._market.valuation_date
        )

        # Get interpolated forward price
        interpolator = self._interpolator
        forward_price = interpolator._forward_interp(t)

        # Construct synthetic ResolvedMarket for this expiry
        synthetic_market = ResolvedMarket(
            risk_free_rate=self._market.risk_free_rate,
            underlying_price=forward_price,  # Use forward as "underlying" for Black-76
            valuation_date=self._market.valuation_date,
            dividend_yield=self._market.dividend_yield,
            dividend_schedule=None,
            provenance=Provenance(price="user", dividends="none"),
            source_meta={
                "interpolated": True,
                "expiry": resolved_expiry_timestamp,
                "risk_free_rate_mode": self._market.risk_free_rate_mode,
            },
        )

        # Derive default strike domain from nearest fitted slices
        # Use the union of min/max strikes across all fitted slices
        all_strikes = []
        for exp_ts in self.expiries:
            try:
                slice_data = self._model.get_slice(exp_ts)
                if (
                    slice_data.get("chain") is not None
                    and "strike" in slice_data["chain"].columns
                ):
                    all_strikes.extend(slice_data["chain"]["strike"].tolist())
            except (ValueError, KeyError):
                pass

        if all_strikes:
            default_domain = (min(all_strikes), max(all_strikes))
        else:
            default_domain = None

        # Create a callable that wraps the interpolator
        def interpolated_vol_curve(strikes: np.ndarray) -> np.ndarray:
            """Synthetic vol curve that evaluates IV via surface interpolation."""
            strike_array = np.asarray(strikes, dtype=float)
            interpolated_ivs = interpolator.implied_vol(strike_array, t)
            return np.asarray(interpolated_ivs, dtype=float)

        # Build a shell VolCurve
        vol_curve = VolCurve(
            method=self.method,
            pricing_engine=self.pricing_engine,
            price_method=self.price_method,
            max_staleness_days=self.max_staleness_days,
        )
        vol_curve.method_options = self.method_options
        vol_curve.solver = self.solver
        vol_curve._vol_curve = interpolated_vol_curve  # type: ignore[attr-defined]
        vol_curve._metadata = {  # type: ignore[attr-defined]
            "interpolated": True,
            "expiry": resolved_expiry_timestamp,
            "expiry_date": resolved_expiry_timestamp.date(),  # Required for probability derivation
            "time_to_expiry_years": t,
            "days_to_expiry": days,
            "method": "total_variance_interpolation",
            "default_domain": default_domain,
        }
        vol_curve._resolved_market = synthetic_market  # type: ignore[attr-defined]
        vol_curve._chain = None  # type: ignore[attr-defined]
        return vol_curve

    @property
    def expiries(self) -> tuple[pd.Timestamp, ...]:
        """Return the fitted expiries available on this surface.

        Returns:
            tuple[pd.Timestamp, ...]: Sorted tuple of expiry timestamps.
        """

        if self._model is None:
            return ()
        return self._model.expiries

    def _resolve_query_time(
        self,
        t: float | str | date | pd.Timestamp,
    ) -> tuple[pd.Timestamp, float]:
        """Normalize maturity input and enforce strict domain constraints.

        Args:
            t: Maturity input as year-fraction float or date-like object.

        Returns:
            tuple[pd.Timestamp, float]: Expiry timestamp and year fraction.

        Raises:
            ValueError: If the surface has not been fitted or maturity is invalid.
        """
        if self._interpolator is None or self._market is None or self._model is None:
            raise ValueError("Surface not fitted. Call fit() first.")
        return resolve_surface_query_time(self, t)

    def total_variance(self, K: float, t: float | str | date | pd.Timestamp) -> float:
        """Return total variance at strike K and time t (years).

        Args:
            K: Strike price.
            t: Time to maturity in years (float) or Expiry Date.

        Returns:
            float: Total variance, defined as w(K, t) = sigma(K, t)^2 * t.

        """
        _, t_years = self._resolve_query_time(t)
        return self._interpolator(K, t_years)

    def implied_vol(self, K: float, t: float | str | date | pd.Timestamp) -> float:
        """Return implied volatility at strike K and time t (years).

        Args:
            K: Strike price.
            t: Time to maturity in years (float) or Expiry Date.

        Returns:
            float: Implied volatility in decimal form (e.g., 0.20 for 20%).

        Raises:
            ValueError: If the surface has not been fitted.
        """
        _, t_years = self._resolve_query_time(t)
        return self._interpolator.implied_vol(K, t_years)

    def __call__(self, K: float, t: float | str | date | pd.Timestamp) -> float:
        """Return implied volatility at strike K and time t (years).

        Alias for :meth:`implied_vol`.
        """
        return self.implied_vol(K, t)

    def forward_price(self, t: float | str | date | pd.Timestamp) -> float:
        """Return the interpolated forward price at time t.

        Args:
            t: Time to maturity in years (float) or Expiry Date.

        Returns:
            float: Forward price F(t).
        """
        _, t_years = self._resolve_query_time(t)
        return self._interpolator._forward_interp(t_years)

    def price(
        self,
        strikes: ArrayLike,
        t: float | str | date | pd.Timestamp,
        call_or_put: Literal["call", "put"] = "call",
    ) -> np.ndarray:
        """Calculate theoretical option prices at arbitrary time t.

        Args:
            strikes: Strike price(s).
            t: Time to expiry (years as float) or Expiry Date (string/date).
            call_or_put: "call" or "put".

        Returns:
            np.ndarray: Option prices.
        """
        _, t_years = self._resolve_query_time(t)

        # 2. Direct Math Kernel Queries
        # Get Forward F and Vol sigma directly from interpolator (fast)
        F = self._interpolator._forward_interp(t_years)
        sigma = self._interpolator.implied_vol(np.asarray(strikes), t_years)

        K = np.asarray(strikes, dtype=float)
        r = resolve_risk_free_rate(
            self._market.risk_free_rate, self._market.risk_free_rate_mode, t_years
        )

        # 3. Dispatch to internal engine
        # Using self.pricing_engine to stay consistent with fit()
        engine_name = self.pricing_engine

        if engine_name == "black76":
            # Direct Black-76 call
            call_price = black76_call_price(F, K, sigma, t_years, r)

            if call_or_put == "put":
                # Parity: P = C - D(F - K)
                df = np.exp(-r * t_years)
                return call_price - df * (F - K)
            return call_price

        elif engine_name == "bs":
            # If using BS, we need Spot S and Yield q

            S = self._market.underlying_price
            q = self._market.dividend_yield
            if q is None:
                raise ValueError(
                    "Dividend yield (q) is required for Black-Scholes pricing but was not provided."
                )

            call_price = black_scholes_call_price(S, K, sigma, t_years, r, q)

            if call_or_put == "put":
                return call_price - S * np.exp(-q * t_years) + K * np.exp(-r * t_years)
            return call_price

        else:
            raise ValueError(f"Unsupported pricing engine for .price(): {engine_name}")

    def implied_distribution(self) -> ProbSurface:
        """Return the risk-neutral distribution surface for all fitted expiries.

        Returns:
            ProbSurface: Surface with per-expiry distributions.

        Raises:
            ValueError: If ``fit`` has not been called.
        """

        if self._model is None:
            raise ValueError("Call fit before deriving the distribution surface")

        from oipd.interface.probability import ProbSurface

        return ProbSurface(vol_surface=self)

    def atm_vol(self, t: float | str | date | pd.Timestamp) -> float:
        """Return At-The-Money (ATM) implied volatility at time t.

        ATM is defined as Strike = Forward Price at time t.

        Args:
            t: Time to maturity in years (float) or Expiry Date.

        Returns:
            float: Interpolated ATM implied volatility.
        """
        _, t_years = self._resolve_query_time(t)

        F = self.forward_price(t_years)
        return self.implied_vol(F, t_years)

    def iv_results(self) -> pd.DataFrame:
        """Return a concatenated DataFrame of calibration results for all fitted expiries.

        Returns:
            pd.DataFrame: Long-format DataFrame with 'expiry' column added.
        """
        if self._model is None:
            raise ValueError("Surface not fitted.")

        dfs = []
        for expiry in self.expiries:
            # slice() ensures we get the real fitted curve
            curve = self.slice(expiry)
            try:
                # curve.iv_results() returns the DataFrame for that slice
                df = curve.iv_results()
                df["expiry"] = expiry
                dfs.append(df)
            except ValueError:
                continue

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    @property
    def params(self) -> dict[pd.Timestamp, Any]:
        """Return a dictionary of fitted parameters for each expiry pillar.

        Returns:
            dict: {expiry_timestamp: params_object}
        """
        if self._model is None:
            return {}

        params_dict = {}
        for expiry in self.expiries:
            curve = self.slice(expiry)
            params_dict[expiry] = curve.params
        return params_dict

    # -------------------------------------------------------------------------
    # Greeks
    # -------------------------------------------------------------------------

    def _dispatch_surface_greek(
        self,
        func_black76: Callable,
        func_bs: Callable,
        strikes: ArrayLike,
        t: float | str | date | pd.Timestamp,
        **kwargs,
    ) -> np.ndarray:
        """Centralized dispatch for Surface Greek calculations at arbitrary time t."""
        if self._interpolator is None or self._market is None:
            raise ValueError("Call fit before computing Greeks")

        # 1. Resolve Time t with strict domain checks
        _, t_years = self._resolve_query_time(t)

        # 2. Resolve Inputs from Interpolator
        K = np.asarray(strikes, dtype=float)
        sigma = self._interpolator.implied_vol(K, t_years)
        r = resolve_risk_free_rate(
            self._market.risk_free_rate, self._market.risk_free_rate_mode, t_years
        )
        q = self._market.dividend_yield

        # 3. Dispatch to Match Pricing Engine
        if self.pricing_engine == "black76":
            F = self._interpolator._forward_interp(t_years)
            return func_black76(F, K, sigma, t_years, r, **kwargs)
        else:  # bs
            S = self._market.underlying_price
            if q is None:
                raise ValueError("Dividend yield required for BS Greeks")
            return func_bs(S, K, sigma, t_years, r, q, **kwargs)

    def delta(
        self,
        strikes: ArrayLike,
        t: float | str | date | pd.Timestamp,
        call_or_put: Literal["call", "put"] = "call",
    ) -> np.ndarray:
        """Calculate Delta (∂V/∂S or ∂V/∂F) for the surface at time t.

        Args:
            strikes: Strike price(s).
            t: Time to expiry.
            call_or_put: "call" or "put".

        Returns:
            np.ndarray: Delta values.
        """
        return self._dispatch_surface_greek(
            black76_delta,
            black_scholes_delta,
            strikes,
            t,
            call_or_put=call_or_put,
        )

    def gamma(
        self,
        strikes: ArrayLike,
        t: float | str | date | pd.Timestamp,
    ) -> np.ndarray:
        """Calculate Gamma (∂²V/∂S² or ∂²V/∂F²) for the surface at time t.

        Args:
            strikes: Strike price(s).
            t: Time to expiry.

        Returns:
            np.ndarray: Gamma values.
        """
        return self._dispatch_surface_greek(
            black76_gamma,
            black_scholes_gamma,
            strikes,
            t,
        )

    def vega(
        self,
        strikes: ArrayLike,
        t: float | str | date | pd.Timestamp,
    ) -> np.ndarray:
        """Calculate Vega (∂V/∂σ) for the surface at time t.

        Args:
            strikes: Strike price(s).
            t: Time to expiry.

        Returns:
            np.ndarray: Vega values.
        """
        return self._dispatch_surface_greek(
            black76_vega,
            black_scholes_vega,
            strikes,
            t,
        )

    def theta(
        self,
        strikes: ArrayLike,
        t: float | str | date | pd.Timestamp,
        call_or_put: Literal["call", "put"] = "call",
    ) -> np.ndarray:
        """Calculate Theta (∂V/∂t) for the surface at time t (per year).

        Args:
            strikes: Strike price(s).
            t: Time to expiry.
            call_or_put: "call" or "put".

        Returns:
            np.ndarray: Theta values.
        """
        return self._dispatch_surface_greek(
            black76_theta,
            black_scholes_theta,
            strikes,
            t,
            call_or_put=call_or_put,
        )

    def rho(
        self,
        strikes: ArrayLike,
        t: float | str | date | pd.Timestamp,
        call_or_put: Literal["call", "put"] = "call",
    ) -> np.ndarray:
        """Calculate Rho (∂V/∂r) for the surface at time t.

        Args:
            strikes: Strike price(s).
            t: Time to expiry.
            call_or_put: "call" or "put".

        Returns:
            np.ndarray: Rho values.
        """
        return self._dispatch_surface_greek(
            black76_rho,
            black_scholes_rho,
            strikes,
            t,
            call_or_put=call_or_put,
        )

    def greeks(
        self,
        strikes: ArrayLike,
        t: float | str | date | pd.Timestamp,
        call_or_put: Literal["call", "put"] = "call",
    ) -> pd.DataFrame:
        """Calculate all Greeks for the surface at time t.

        Args:
            strikes: Strike price(s).
            t: Time to expiry.
            call_or_put: "call" or "put".

        Returns:
            pd.DataFrame: Table with columns [strike, delta, gamma, vega, theta, rho].
        """
        K = np.asarray(strikes, dtype=float)
        return pd.DataFrame(
            {
                "strike": K,
                "delta": self.delta(K, t, call_or_put),
                "gamma": self.gamma(K, t),
                "vega": self.vega(K, t),
                "theta": self.theta(K, t, call_or_put),
                "rho": self.rho(K, t, call_or_put),
            }
        )

    def plot(
        self,
        *,
        x_axis: Literal["strike", "log_moneyness"] = "log_moneyness",
        y_axis: Literal["iv", "total_variance"] = "total_variance",
        include_observed: bool = False,
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
            include_observed: Whether to include observed market data points (default False).
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

        _apply_publication_style(plt)

        fig, ax = plt.subplots(figsize=figsize)

        # Get color cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        # Plot each expiry's IV smile by delegating to VolCurve.plot
        for i, expiry_timestamp in enumerate(self.expiries):
            vol_curve = self.slice(expiry_timestamp)

            # Generate label
            if label_format == "days":
                resolved_market = vol_curve._resolved_market
                if resolved_market is not None:
                    days = calculate_days_to_expiry(
                        expiry_timestamp, resolved_market.valuation_date
                    )
                    label = f"{days}d"
                else:
                    label = expiry_timestamp.strftime("%b %d, %Y")
            else:
                label = expiry_timestamp.strftime("%b %d, %Y")

            # Determine color: use kwargs if provided, else cycle
            current_line_kwargs = {"label": label, **kwargs}
            if "color" not in current_line_kwargs and "c" not in current_line_kwargs:
                current_line_kwargs["color"] = colors[i % len(colors)]

            # Delegate to curve plotting
            # We suppress individual chart decorations (titles, labels, legends)
            # and handle them globally for the surface.
            vol_curve.plot(
                ax=ax,
                x_axis=x_axis,
                y_axis=y_axis,
                include_observed=include_observed,
                title="",  # Suppress per-curve title
                show_axis_labels=False,  # Suppress per-curve axis labels
                show_legend=False,  # Suppress per-curve legend
                line_kwargs=current_line_kwargs,
                xlim=xlim,
                ylim=ylim,
            )

        # Global Surface Decorations

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

        # Legend
        legend = ax.legend(loc="best", frameon=False)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#333333")

        # Apply publication styling to axes
        _style_publication_axes(ax)

        # Title
        resolved_title = title or "Implied Volatility Surface"
        if y_axis == "total_variance" and title is None:
            resolved_title = "Total Variance Surface"

        plt.subplots_adjust(top=0.88)
        fig.suptitle(
            resolved_title,
            fontsize=16,
            fontweight="bold",
            y=0.95,
            color="#333333",
        )

        return fig

    def plot_3d(
        self,
        *,
        strike_range: Optional[tuple[float, float]] = None,
        expiry_range: Optional[tuple[Any, Any]] = None,
        n_strikes: int = 80,
        n_expiries: int = 50,
        view_angle: tuple[float, float] = (30, -45),
        figsize: tuple[float, float] = (14, 10),
        title: Optional[str] = None,
        dark_mode: bool = True,
        show_projections: bool = False,
        cmap: str = "coolwarm",
        zlim: Optional[tuple[float, float]] = None,
        projection_type: Literal["ortho", "persp"] = "ortho",
    ) -> Any:
        """Visualize the fitted volatility surface in 3D.

        Generates a 3D surface plot of Implied Volatility vs Strike and Expiry.

        Args:
            strike_range: Optional (min, max) tuple for strike axis. Auto-determined if None.
            expiry_range: Optional (min, max) tuple for expiry axis (in days). Auto-determined if None.
            n_strikes: Number of grid points for strike axis.
            n_expiries: Number of grid points for expiry axis.
            view_angle: (elevation, azimuth) for 3D camera.
            figsize: Figure size (width, height).
            title: Custom plot title.
            dark_mode: If True, use dark theme.
            show_projections: Deprecated. Projections are no longer rendered.
            cmap: Matplotlib colormap.
            zlim: Explicit limits for Z-axis (IV).
            projection_type: 'ortho' for isometric-like view, 'persp' for perspective.

        Returns:
            matplotlib.figure.Figure: The plot figure.

        Raises:
            ValueError: If ``fit`` has not been called.
        """
        if self._model is None or self._interpolator is None:
            raise ValueError("Call fit before plotting the 3D surface")

        from oipd.presentation.surface_3d import plot_surface_3d

        # -------------------------------------------------------------------------
        # 1. Determine Time (Y-Axis) Domain
        # -------------------------------------------------------------------------
        all_expiries = sorted(self.expiries)
        valuation_date = self._market.valuation_date

        if expiry_range is not None:
            t_min = calculate_days_to_expiry(
                pd.to_datetime(expiry_range[0]), valuation_date
            )
            t_max = calculate_days_to_expiry(
                pd.to_datetime(expiry_range[1]), valuation_date
            )
        else:
            t_min = calculate_days_to_expiry(all_expiries[0], valuation_date)
            t_max = calculate_days_to_expiry(all_expiries[-1], valuation_date)

        t_min = max(t_min, 1)  # At least 1 day
        t_grid_days = np.linspace(t_min, t_max, n_expiries)
        t_grid_years = convert_days_to_years(t_grid_days)

        # -------------------------------------------------------------------------
        # 2. Determine Strike (X-Axis) Domain
        # -------------------------------------------------------------------------
        if strike_range is not None:
            k_min, k_max = strike_range
        else:
            # Auto-determine from the fitted data
            # Use log-moneyness bounds from observations, then convert to strikes
            all_lm = []
            for t in self.expiries:
                # Use public API to get slice for each expiry
                vc = self.slice(t)
                chain = vc._chain
                # Parity-implied forward for this slice
                F = vc.forward_price

                if (
                    chain is not None
                    and "strike" in chain.columns
                    and F is not None
                    and F > 0
                ):
                    strikes = chain["strike"].to_numpy(dtype=float)
                    # Calculate log-moneyness for these observations
                    lm = np.log(strikes / F)
                    all_lm.extend(lm[np.isfinite(lm)])

            if not all_lm:
                # Fallback if no valid observations found
                lm_min, lm_max = -0.5, 0.5
            else:
                k_arr = np.array(all_lm)
                # Trim extreme outliers (top/bottom 2%) to focus on liquid strikes
                lm_min, lm_max = np.nanquantile(k_arr, [0.02, 0.98])

            # Convert log-moneyness to strikes using median forward
            median_t = np.median(t_grid_years)
            median_forward = self._interpolator._forward_interp(median_t)
            k_min = median_forward * np.exp(lm_min)
            k_max = median_forward * np.exp(lm_max)

        k_grid = np.linspace(k_min, k_max, n_strikes)

        # -------------------------------------------------------------------------
        # 3. Build Meshgrid & Evaluate Implied Vol
        # -------------------------------------------------------------------------
        X, Y = np.meshgrid(k_grid, t_grid_days)  # X = strikes, Y = days
        Z = np.zeros_like(X)

        for i, t_years in enumerate(t_grid_years):
            forward = self._interpolator._forward_interp(t_years)
            # The interpolator is callable with (K, t)
            # It expects Strikes K (not log-moneyness)
            total_var = self._interpolator(k_grid, t_years)
            iv = np.sqrt(np.maximum(total_var / max(t_years, 1e-8), 1e-12))
            # Convert from decimal (0.30) to percentage (30) for display
            Z[i, :] = iv * 100

        if show_projections:
            warnings.warn(
                "Wall projections are no longer rendered for 3D vol surfaces.",
                UserWarning,
            )

        # -------------------------------------------------------------------------
        # 4. Delegate to Generic Plotter
        # -------------------------------------------------------------------------
        resolved_title = title or "Implied Volatility Surface"
        return plot_surface_3d(
            X,
            Y,
            Z,
            xlabel="Strike Price",
            ylabel="Expiration (days)",
            zlabel="Implied Vol (%)",
            title=resolved_title,
            cmap=cmap,
            view_angle=view_angle,
            figsize=figsize,
            show_projections=False,
            dark_mode=dark_mode,
            zlim=zlim,
            colorbar_label="Implied Volatility",
            projection_type=projection_type,
        )

    def plot_term_structure(
        self,
        at_money: Literal["forward", "spot"] = "forward",
        title: Optional[str] = None,
        figsize: tuple[float, float] = (10, 6),
        marker: str = "",
        line_color: Optional[str] = None,
        num_points: int = 100,
    ) -> Figure:
        """Plot the ATM implied volatility term structure.

        Visualizes how implied volatility changes across different expirations
        for At-The-Money (ATM) options using surface interpolation.

        Args:
            at_money: Definition of ATM. "forward" (K=F) is standard for vol surfaces.
                      "spot" (K=S) uses the current spot price.
            title: Optional plot title.
            figsize: Figure size (width, height) in inches.
            marker: Marker style for data points. Use "" for no markers.
            line_color: Optional color for the line.
            num_points: Number of interpolation points across the expiry range.

        Returns:
            matplotlib.figure.Figure: The generated plot.

        Raises:
            ValueError: If the surface has not been fitted.
        """
        if self._model is None or self._interpolator is None:
            raise ValueError("Call fit before plotting term structure")

        from oipd.presentation.term_structure import plot_term_structure
        from oipd.pipelines.vol_surface.term_structure import build_atm_term_structure

        valuation_date = self._market.valuation_date

        term_structure = build_atm_term_structure(
            expiries=self.expiries,
            valuation_date=valuation_date,
            implied_vol=self._interpolator.implied_vol,
            forward_price=self.forward_price,
            spot_price=self._market.underlying_price,
            at_money=at_money,
            num_points=num_points,
        )

        return plot_term_structure(
            days_to_expiry=term_structure["days_to_expiry"].to_numpy(),
            atm_ivs=term_structure["atm_iv"].to_numpy() * 100.0,
            title=title or f"ATM ({at_money.capitalize()}) Term Structure",
            figsize=figsize,
            marker=marker,
            line_color=line_color,
        )


__all__ = ["VolCurve", "VolSurface"]
