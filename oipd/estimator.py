from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, Optional, Dict, Literal, Any, Mapping, Sequence
from datetime import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np
import warnings

from oipd.core.errors import InvalidInputError, CalculationError
from oipd.core.data_processing import (
    apply_put_call_parity,
    compute_iv,
    filter_stale_options,
    select_price_column,
)
from oipd.core.vol_surface_fitting import AVAILABLE_SURFACE_FITS, fit_surface
from oipd.core.vol_surface_fitting.shared.svi import SVICalibrationOptions, svi_options
from oipd.core.vol_surface_fitting.shared.vol_model import VolModel, SLICE_METHODS
from oipd.core.probability_density_conversion import (
    price_curve_from_iv,
    pdf_from_price_curve,
    calculate_cdf_from_pdf,
)
from oipd.data_access.readers import CSVReader, DataFrameReader
from oipd.data_access.vendors import get_reader
from oipd.pricing.utils import (
    prepare_dividends,
    implied_dividend_yield_from_forward,
)
from oipd.market_inputs import (
    MarketInputs,
    VendorSnapshot,
    ResolvedMarket,
    resolve_market,
    FillMode,
)
from oipd.presentation.iv_plotting import ReferenceAnnotation, plot_iv_smile


# ---------------------------------------------------------------------------
# Dataclasses holding user configurable parameters
# ---------------------------------------------------------------------------


# MarketParams class has been removed - use MarketInputs from market_inputs.py instead


@dataclass
class ModelParams:
    """Model / algorithm specific knobs that users may tune."""

    solver: Literal["brent", "newton"] = "brent"
    american_to_european: bool = False  # placeholder for future functionality
    pricing_engine: Literal["black76", "bs"] = "black76"
    price_method: Optional[Literal["last", "mid"]] = None
    max_staleness_days: Optional[int] = (
        3  # in calendar days; set to 3 by default to accomodate weekends
    )
    surface_method: Literal["svi", "bspline"] = "svi"
    surface_options: Mapping[str, Any] | SVICalibrationOptions | None = None
    surface_random_seed: int | None = None
    price_method_explicit: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.price_method is None:
            self.price_method = "mid"
            self.price_method_explicit = False
        else:
            self.price_method_explicit = True

        if self.surface_method not in AVAILABLE_SURFACE_FITS:
            raise ValueError(
                f"surface_method must be one of {AVAILABLE_SURFACE_FITS}, got {self.surface_method}"
            )

        if self.surface_method == "svi":
            if self.surface_options is None:
                self.surface_options = svi_options()
            elif isinstance(self.surface_options, SVICalibrationOptions):
                pass
            elif isinstance(self.surface_options, Mapping):
                self.surface_options = svi_options(**dict(self.surface_options))
            else:
                raise TypeError(
                    "surface_options must be an SVICalibrationOptions instance or mapping for SVI"
                )

            if self.surface_random_seed is not None:
                self.surface_options = replace(
                    self.surface_options, random_seed=self.surface_random_seed
                )
        else:
            if self.surface_options is None:
                self.surface_options = {}
            elif isinstance(self.surface_options, Mapping):
                self.surface_options = dict(self.surface_options)
            else:
                raise TypeError("surface_options must be a mapping or None")

            if self.surface_random_seed is not None:
                warnings.warn(
                    "surface_random_seed is only used for SVI calibration and will be ignored.",
                    UserWarning,
                )



@dataclass(frozen=True)
class RNDResult:
    """Container for the resulting PDF / CDF arrays with convenience helpers."""

    prices: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    market: ResolvedMarket
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Return results as a tidy DataFrame."""
        return pd.DataFrame({"Price": self.prices, "PDF": self.pdf, "CDF": self.cdf})

    def to_csv(self, path: str, **kwargs) -> None:
        """Persist results to csv on disk."""
        self.to_frame().to_csv(path, index=False, **kwargs)

    def summary(self) -> str:
        """Return a one-line summary of resolved parameters and their sources."""
        # Build message in desired order, optionally including implied yield
        underlying = self.market.underlying_price
        price_src = self.market.provenance.price
        div_src = self.market.provenance.dividends
        days = self.market.days_to_expiry
        r = self.market.risk_free_rate

        # Dividends wording with explicit yield when available
        if div_src == "vendor_yield" and self.market.dividend_yield is not None:
            div_text = f"vendor yield of {self.market.dividend_yield:.4%}"
        elif div_src == "user_yield" and self.market.dividend_yield is not None:
            div_text = f"user yield of {self.market.dividend_yield:.4%}"
        elif div_src == "vendor_schedule":
            div_text = "vendor schedule"
        elif div_src == "user_schedule":
            div_text = "user schedule"
        else:
            div_text = "none"

        msg = f"Underlying price {underlying:.4f} (source: {price_src})"

        F = self.meta.get("forward_price")
        if F is not None:
            try:
                msg += f", implied forward price {float(F):.4f}"
            except Exception:
                pass

        msg += f"; dividends: {div_text}"

        if F is not None:
            try:
                q = self.implied_dividend_yield()
                msg += f", forward-implied annualised dividend yield of {q:.4%}"
            except Exception:
                pass

        msg += f"; days_to_expiry={days}; r={r};"
        return msg

    def prob_at_or_above(self, price: float) -> float:
        """
        Calculate the probability that the future price will be at or above a specified price.

        This is computed as 1 - CDF(price), where CDF is the cumulative distribution function.

        Parameters
        ----------
        price : float
            The price threshold to evaluate

        Returns
        -------
        float
            Probability (between 0 and 1) that the future price will be at or above the specified price
        """
        # Handle edge cases
        if price <= self.prices.min():
            return 1.0  # If price is below minimum, probability is 100%
        if price >= self.prices.max():
            return 0.0  # If price is above maximum, probability is 0%

        # Interpolate CDF at the specified price
        cdf_at_price = np.interp(price, self.prices, self.cdf)
        return 1.0 - cdf_at_price

    def prob_below(self, price: float) -> float:
        """
        Calculate the probability that the future price will be below a specified price.

        This is computed as CDF(price), where CDF is the cumulative distribution function.

        Parameters
        ----------
        price : float
            The price threshold to evaluate

        Returns
        -------
        float
            Probability (between 0 and 1) that the future price will be below the specified price
        """
        # Handle edge cases
        if price <= self.prices.min():
            return 0.0  # If price is at or below minimum, probability is 0%
        if price >= self.prices.max():
            return 1.0  # If price is at or above maximum, probability is 100%

        # Interpolate CDF at the specified price
        cdf_at_price = np.interp(price, self.prices, self.cdf)
        return cdf_at_price

    def implied_dividend_yield(self) -> float:
        """
        Compute the annualized implied continuous dividend yield q implied by
        put-call parity when a forward was inferred.

        Uses q = r - (1/T) * ln(F / S) with T in years, where:
        - r is the risk-free rate from the resolved market
        - F is the parity-inferred forward price captured in meta['forward_price']
        - S is the resolved underlying price

        Returns
        -------
        float
            Implied continuous dividend yield. Raises ValueError if forward is
            not available in metadata.
        """
        F = self.meta.get("forward_price")
        if F is None:
            raise ValueError(
                "No parity-inferred forward available to imply dividend yield."
            )
        S = float(self.market.underlying_price)
        if S <= 0:
            raise ValueError("Invalid underlying price for implied yield calculation.")
        T = float(self.market.days_to_expiry) / 365.0
        if T <= 0:
            raise ValueError("Non-positive time to expiry.")
        r = float(self.market.risk_free_rate)
        return implied_dividend_yield_from_forward(S, float(F), r, T)

    def plot(
        self,
        kind: Literal["pdf", "cdf", "both"] = "both",
        figsize: tuple[float, float] = (10, 5),
        title: Optional[str] = None,
        show_current_price: bool = True,
        style: Literal["publication", "default"] = "publication",
        source: Optional[str] = None,
        **kwargs,
    ):
        """Plot risk-neutral density outputs.

        Args:
            kind: Plot type to render. Set to ``"pdf"``, ``"cdf"``, or ``"both"``
                to visualize distribution outputs.
            figsize: Matplotlib figure size in inches.
            title: Optional custom title. When omitted, an informative default
                is constructed.
            show_current_price: Whether to highlight the current underlying
                price on the plot.
            style: Visual theme to apply. ``"publication"`` uses the package's
                publication-ready styling, while ``"default"`` relies on
                Matplotlib defaults.
            source: Optional attribution text displayed when ``style`` is
                ``"publication"``.
            **kwargs: Extra options forwarded to the underlying plotting
                routine.

        Returns:
            matplotlib.figure.Figure: Figure containing the requested plot.

        Raises:
            ImportError: If Matplotlib is unavailable.
            ValueError: If ``kind`` is unsupported.
        """

        from oipd.presentation import plot_rnd

        underlying_price = self.market.underlying_price
        valuation_date_obj = self.market.valuation_date
        expiry_date_obj = self.market.expiry_date
        valuation_date = valuation_date_obj.strftime("%b %d, %Y")
        expiry_date = expiry_date_obj.strftime("%b %d, %Y")

        return plot_rnd(
            prices=self.prices,
            pdf=self.pdf,
            cdf=self.cdf,
            kind=kind,
            figsize=figsize,
            title=title,
            show_current_price=show_current_price,
            current_price=underlying_price,
            valuation_date=valuation_date,
            expiry_date=expiry_date,
            style=style,
            source=source,
            **kwargs,
        )

    def iv_smile(
        self,
        strikes: Sequence[float] | np.ndarray | None = None,
        *,
        num_points: int = 200,
    ) -> pd.DataFrame:
        """Return the fitted implied-volatility smile evaluated on a strike grid.

        Args:
            strikes: Optional sequence of strikes at which to evaluate the fitted
                implied-volatility curve. When omitted, the method uses the stored
                calibration grid when available, otherwise a linspace spanning the
            observed strikes with ``num_points`` samples.
            num_points: Number of evaluation points when ``strikes`` is not
                provided and a grid must be generated. Must be at least two when
                the observed strike range is non-zero.

        Returns:
            DataFrame containing the fitted smile and available observed
            implied volatilities with columns:

            - ``strike`` – strike levels used for evaluation.
            - ``fitted_iv`` – implied volatilities from the calibrated smile.
            - ``bid_iv`` – implied volatilities backed out from observed bid
              quotes when available.
            - ``ask_iv`` – implied volatilities backed out from observed ask
              quotes when available.

        Raises:
            ValueError: If the fitted volatility curve is unavailable or a strike
                grid cannot be constructed.
        """
        vol_curve = self.meta.get("vol_curve")
        if vol_curve is None:
            raise ValueError("No fitted implied-volatility curve is available.")

        observed_iv = self.meta.get("observed_iv")

        if strikes is None:
            curve_grid = getattr(vol_curve, "grid", None)
            if curve_grid is not None:
                strike_grid = np.asarray(curve_grid[0], dtype=float)
                fitted_values = np.asarray(curve_grid[1], dtype=float)
            else:
                if observed_iv is None or observed_iv.empty:
                    raise ValueError(
                        "Unable to infer an evaluation grid for the implied-volatility smile."
                    )
                strike_min = float(observed_iv["strike"].min())
                strike_max = float(observed_iv["strike"].max())
                if np.isclose(strike_min, strike_max):
                    strike_grid = np.array([strike_min])
                else:
                    if num_points < 2:
                        raise ValueError(
                            "num_points must be at least 2 when generating a strike grid."
                        )
                    strike_grid = np.linspace(strike_min, strike_max, num_points)
                fitted_values = vol_curve(strike_grid)
        else:
            strike_grid = np.asarray(list(strikes), dtype=float)
            if strike_grid.size == 0:
                raise ValueError("strikes must contain at least one value.")
            fitted_values = vol_curve(strike_grid)

        smile_df = pd.DataFrame(
            {
                "strike": strike_grid,
                "fitted_iv": fitted_values,
            }
        )

        observed_bid = self.meta.get("observed_iv_bid")
        observed_last = self.meta.get("observed_iv_last")
        if isinstance(observed_bid, pd.DataFrame) and not observed_bid.empty:
            observed_bid = observed_bid.copy()
            observed_bid["strike"] = observed_bid["strike"].astype(float)
            bid_subset = (
                observed_bid.loc[:, ["strike", "iv"]]
                .dropna()
                .drop_duplicates(subset="strike")
                .rename(columns={"iv": "bid_iv"})
            )
            smile_df = smile_df.merge(bid_subset, on="strike", how="left")

        observed_ask = self.meta.get("observed_iv_ask")
        if isinstance(observed_ask, pd.DataFrame) and not observed_ask.empty:
            observed_ask = observed_ask.copy()
            observed_ask["strike"] = observed_ask["strike"].astype(float)
            ask_subset = (
                observed_ask.loc[:, ["strike", "iv"]]
                .dropna()
                .drop_duplicates(subset="strike")
                .rename(columns={"iv": "ask_iv"})
            )
            smile_df = smile_df.merge(ask_subset, on="strike", how="left")

        if (
            ("bid_iv" not in smile_df.columns or smile_df["bid_iv"].isna().all())
            and ("ask_iv" not in smile_df.columns or smile_df["ask_iv"].isna().all())
            and isinstance(observed_last, pd.DataFrame)
            and not observed_last.empty
        ):
            last_subset = (
                observed_last.copy()
                .astype({"strike": float})
                .loc[:, ["strike", "iv"]]
                .dropna()
                .rename(columns={"iv": "last_iv"})
            )
            smile_df = smile_df.merge(last_subset, on="strike", how="left")
            if "last_iv" not in smile_df.columns:
                smile_df["last_iv"] = np.nan
        else:
            smile_df["last_iv"] = np.nan

        if "bid_iv" not in smile_df.columns:
            smile_df["bid_iv"] = np.nan
        if "ask_iv" not in smile_df.columns:
            smile_df["ask_iv"] = np.nan
        if "last_iv" not in smile_df.columns:
            smile_df["last_iv"] = np.nan

        # Ensure column ordering
        return smile_df.loc[:, ["strike", "fitted_iv", "bid_iv", "ask_iv", "last_iv"]]

    def plot_iv(
        self,
        *,
        figsize: tuple[float, float] = (10, 5),
        title: Optional[str] = None,
        observations: Literal["range", "markers"] = "range",
        x_axis: Literal["log_moneyness", "strike"] = "log_moneyness",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
    ):
        """Plot the implied-volatility smile with observed quotes.

        Args:
            figsize: tuple[float, float], default (10, 5)
                Matplotlib figure size in inches.
            title: str | None, optional
                Custom chart title. When omitted a default based on expiry metadata is
                used.
            observations: Literal["range", "markers"], default "range"
                ``"range"`` displays bid/ask ranges; ``"markers"`` scatters the
                observed quotes.
            x_axis: Literal["log_moneyness", "strike"], default "log_moneyness"
                Axis definition for the smile plot.
            xlim: tuple[float, float] | None, optional
                Bounds for the x-axis when supplied.
            ylim: tuple[float, float] | None, optional
                Bounds for the y-axis when supplied.

        Returns:
            matplotlib.figure.Figure: Figure containing the plotted smile.

        Raises:
            ValueError: If the requested axis or observed style is unsupported.
        """
        smile_df = self.iv_smile()

        forward_price = self.meta.get("forward_price")
        spot_price = (
            float(self.market.underlying_price)
            if self.market.underlying_price is not None
            else None
        )
        reference_price = (
            float(forward_price) if forward_price is not None else spot_price
        )

        reference_annotation = None
        if reference_price is not None:
            valuation_date = self.market.valuation_date.strftime("%b %d, %Y")
            label_prefix = "Parity forward on" if forward_price is not None else "Price on"
            reference_annotation = ReferenceAnnotation(
                value=float(reference_price),
                label=f"{label_prefix} {valuation_date}\n${reference_price:,.2f}",
            )

        if title is not None:
            resolved_title = title
        else:
            expiry_obj = self.market.expiry_date
            resolved_title = (
                f"Implied Volatility Smile (Expiry {expiry_obj.strftime('%b %d, %Y')})"
                if expiry_obj is not None
                else None
            )

        figure = plot_iv_smile(
            smile_df,
            include_observed=True,
            observed_bid=self.meta.get("observed_iv_bid"),
            observed_ask=self.meta.get("observed_iv_ask"),
            observed_last=self.meta.get("observed_iv_last"),
            figsize=figsize,
            title=resolved_title,
            style="publication",
            source=None,
            show_reference=False,
            reference=reference_annotation,
            axis_mode=x_axis,
            observed_style=observations,
            xlim=xlim,
            ylim=ylim,
        )

        return figure


# ---------------------------------------------------------------------------
# Data-loading abstraction
# ---------------------------------------------------------------------------


class DataSource(Protocol):
    """Minimal interface every data source must implement."""

    def load(self) -> pd.DataFrame:  # pragma: no cover – Protocol, no runtime
        ...


class CSVSource:
    """Load options data from an on-disk CSV file."""

    def __init__(self, path: str, column_mapping: Optional[Dict[str, str]] = None):
        self._path = path
        self._column_mapping = column_mapping or {}
        self._reader = CSVReader()

    def load(self) -> pd.DataFrame:
        return self._reader.read(self._path, column_mapping=self._column_mapping)


class DataFrameSource:
    """Wrap an in-memory DataFrame so that it satisfies the *DataSource* Protocol."""

    def __init__(
        self, df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None
    ):
        self._df = df
        self._column_mapping = column_mapping or {}
        self._reader = DataFrameReader()

    def load(self) -> pd.DataFrame:
        return self._reader.read(self._df, column_mapping=self._column_mapping)


class TickerSource:
    """Load options data from a vendor for a given ticker and expiry."""

    def __init__(
        self,
        ticker: str,
        expiry: str,
        vendor: str = "yfinance",
        column_mapping: Optional[Dict[str, str]] = None,
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 15,
    ):
        self._ticker = ticker
        self._expiry = expiry
        self._column_mapping = column_mapping or {}

        reader_cls = get_reader(vendor)
        # Most readers accept cache flags; if not, Python will ignore unexpected kwargs.
        try:
            self._reader = reader_cls(
                cache_enabled=cache_enabled,
                cache_ttl_minutes=cache_ttl_minutes,
            )
        except TypeError:
            self._reader = reader_cls()

        self._underlying_price: Optional[float] = None
        self._dividend_yield: Optional[float] = None
        self._dividend_schedule: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load options data and extract current price"""
        ticker_expiry = f"{self._ticker}:{self._expiry}"
        df = self._reader.read(ticker_expiry, column_mapping=self._column_mapping)

        # Extract current/underlying price from DataFrame metadata
        self._underlying_price = df.attrs.get("underlying_price")
        self._dividend_yield = df.attrs.get("dividend_yield")
        self._dividend_schedule = df.attrs.get("dividend_schedule")

        return df

    @property
    def underlying_price(self) -> Optional[float]:
        """Get the current underlying price fetched from vendor"""
        return self._underlying_price

    @property
    def dividend_yield(self) -> Optional[float]:
        """Get the dividend yield fetched from vendor"""
        return self._dividend_yield

    @property
    def dividend_schedule(self) -> Optional[pd.DataFrame]:
        """Get the dividend schedule fetched from vendor"""
        return self._dividend_schedule


# ---------------------------------------------------------------------------
# Core estimation routine (non-public)
# ---------------------------------------------------------------------------


def _estimate(
    options_data: pd.DataFrame,
    resolved_market: ResolvedMarket,
    model: ModelParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run the core RND estimation given fully validated input data."""

    # Note when the market snapshot was taken so all filters line up with it
    valuation_date = resolved_market.valuation_date

    # Ensure we actually have the data needed to price with mid quotes if requested
    if model.price_method == "mid" and model.price_method_explicit:
        missing_optional_columns = options_data.attrs.get(
            "_oipd_missing_optional_columns", set()
        )
        has_mid_column = "mid" in options_data.columns
        has_bid_ask_columns = {"bid", "ask"}.issubset(options_data.columns)
        has_option_type_column = (
            "option_type" in options_data.columns
            and not options_data["option_type"].isna().all()
        )
        bid_ask_missing_in_source = {"bid", "ask"}.issubset(missing_optional_columns)
        if not (
            has_mid_column
            or has_option_type_column
            or (has_bid_ask_columns and not bid_ask_missing_in_source)
        ):
            raise CalculationError(
                "Requested price_method='mid' but input data lacks bid/ask, mid, or option_type columns."
            )

    # Work out the effective spot and dividend inputs required by the chosen engine
    if model.pricing_engine == "bs":
        effective_spot_price, effective_dividend_yield = prepare_dividends(
            underlying=resolved_market.underlying_price,
            dividend_schedule=resolved_market.dividend_schedule,
            dividend_yield=resolved_market.dividend_yield,
            r=resolved_market.risk_free_rate,
            valuation_date=valuation_date,
        )
    else:
        effective_spot_price = resolved_market.underlying_price
        effective_dividend_yield = None

    # Get rid of ITM calls and replace with synthetic calls from OTM puts
    parity_adjusted_options, parity_implied_forward_price = apply_put_call_parity(
        options_data, effective_spot_price, resolved_market
    )

    # get implied forward price
    if model.pricing_engine == "black76":
        if parity_implied_forward_price is None:
            warnings.warn(
                "Black-76 requires a parity-implied forward but put quotes are missing. "
                "Rerun with ModelParams(pricing_engine='bs') and provide dividend_yield or dividend_schedule.",
                UserWarning,
            )
            raise ValueError(
                "Put options missing: switch to Black-Scholes with explicit dividend inputs"
            )
        underlying_price = parity_implied_forward_price
    else:
        underlying_price = effective_spot_price

    # Remove quotes that are too old relative to the valuation date
    staleness_filtered_options = filter_stale_options(
        parity_adjusted_options,
        valuation_date,
        model.max_staleness_days,
        emit_warning=True,
    )

    # Pick which observed price column we will treat as the option premium
    options_with_selected_price = select_price_column(
        staleness_filtered_options, model.price_method
    )
    if options_with_selected_price.empty:
        raise CalculationError("No valid options data after price selection")

    # Back out implied volatilities from those option prices
    options_with_calculated_iv = compute_iv(
        options_with_selected_price,
        underlying_price,
        resolved_market,
        model.solver,
        model.pricing_engine,
        effective_dividend_yield,
    )

    # Smooth the implied vol smile so we can evaluate it anywhere we need
    # Capture the strike and IV columns as plain arrays for the fitting helper
    strike_array = options_with_calculated_iv["strike"].to_numpy()
    implied_volatility_array = options_with_calculated_iv["iv"].to_numpy()
    volume_array: np.ndarray | None = None
    if "volume" in options_with_calculated_iv.columns:
        volume_array = options_with_calculated_iv["volume"].to_numpy(dtype=float)
        if not np.isfinite(volume_array).any() or np.all(volume_array <= 0):
            volume_array = None

    def _compute_observed_iv(
        source_df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame | None:
        """Compute implied volatility for an alternate observed price column."""

        if price_column not in source_df.columns:
            return None

        priced = source_df.loc[
            source_df[price_column].notna() & (source_df[price_column] > 0)
        ].copy()
        if priced.empty:
            return None

        priced["price"] = priced[price_column]
        try:
            iv_df = compute_iv(
                priced,
                underlying_price,
                resolved_market,
                model.solver,
                model.pricing_engine,
                effective_dividend_yield,
            )
        except Exception:
            return None

        columns = ["strike", "iv"]
        if "option_type" in iv_df.columns:
            columns.append("option_type")
        return iv_df.loc[:, columns]

    observed_bid_iv = _compute_observed_iv(options_with_selected_price, "bid")
    observed_ask_iv = _compute_observed_iv(options_with_selected_price, "ask")
    observed_last_iv = _compute_observed_iv(options_with_selected_price, "last_price")

    surface_fit_kwargs: Dict[str, Any] = {}
    if model.surface_method == "svi":
        # SVI needs the forward level and time to expiry to calibrate properly
        surface_fit_kwargs.update(
            {
                "forward": underlying_price,
                "maturity_years": resolved_market.days_to_expiry / 365.0,
            }
        )

        def _align_iv_series(iv_df: pd.DataFrame | None) -> np.ndarray | None:
            if iv_df is None or iv_df.empty:
                return None
            joined = (
                pd.DataFrame({"strike": strike_array})
                .merge(iv_df, on="strike", how="left")
                .sort_index()
            )
            iv_values = joined["iv"].to_numpy(dtype=float)
            if np.all(np.isnan(iv_values)):
                return None
            return iv_values

        aligned_bid_iv = _align_iv_series(observed_bid_iv)
        aligned_ask_iv = _align_iv_series(observed_ask_iv)

        if aligned_bid_iv is not None or aligned_ask_iv is not None:
            surface_fit_kwargs["bid_iv"] = aligned_bid_iv
            surface_fit_kwargs["ask_iv"] = aligned_ask_iv
        if volume_array is not None:
            surface_fit_kwargs["volumes"] = volume_array

    try:
        fitted_volatility_curve = fit_surface(
            model.surface_method,
            strikes=strike_array,
            iv=implied_volatility_array,
            options=model.surface_options,
            **surface_fit_kwargs,
        )
        surface_method_used = model.surface_method
    except Exception as exc:
        if model.surface_method == "svi":
            warnings.warn(
                f"SVI calibration failed ({exc}); falling back to B-spline smoothing.",
                UserWarning,
            )
            fitted_volatility_curve = fit_surface(
                "bspline",
                strikes=strike_array,
                iv=implied_volatility_array,
                options={},
            )
            surface_method_used = "bspline"
        else:
            raise CalculationError(
                f"Failed to smooth implied volatility data: {exc}"
            ) from exc

    observed_iv_data = options_with_calculated_iv.copy()

    # Price calls on a dense strike grid using the smoothed volatility curve
    pricing_strike_grid, pricing_call_prices = price_curve_from_iv(
        fitted_volatility_curve,
        underlying_price,
        days_to_expiry=resolved_market.days_to_expiry,
        risk_free_rate=resolved_market.risk_free_rate,
        pricing_engine=model.pricing_engine,
        dividend_yield=effective_dividend_yield,
    )

    # Remember the strike bounds observed in the original data for later trimming
    observed_min_strike = float(options_with_selected_price["strike"].min())
    observed_max_strike = float(options_with_selected_price["strike"].max())

    # Apply Breeden-Litzenberger to turn call prices into the risk-neutral PDF
    pdf_strike_values, pdf_values = pdf_from_price_curve(
        pricing_strike_grid,
        pricing_call_prices,
        risk_free_rate=resolved_market.risk_free_rate,
        days_to_expiry=resolved_market.days_to_expiry,
        min_strike=observed_min_strike,
        max_strike=observed_max_strike,
    )

    try:
        # Numerically integrate the PDF to obtain the matching CDF
        _, cdf_values = calculate_cdf_from_pdf(pdf_strike_values, pdf_values)
    except Exception as exc:
        raise CalculationError(f"Failed to compute CDF: {exc}") from exc

    # Assemble metadata that callers might want for diagnostics
    result_metadata: Dict[str, Any] = {
        "model_params": model,
        "vol_curve": fitted_volatility_curve,
        "observed_iv": observed_iv_data,
    }
    if observed_bid_iv is not None:
        result_metadata["observed_iv_bid"] = observed_bid_iv
    if observed_ask_iv is not None:
        result_metadata["observed_iv_ask"] = observed_ask_iv
    if observed_last_iv is not None:
        result_metadata["observed_iv_last"] = observed_last_iv
    if parity_implied_forward_price is not None:
        try:
            result_metadata["forward_price"] = float(parity_implied_forward_price)
        except Exception:
            pass
    result_metadata["surface_fit"] = surface_method_used
    return pdf_strike_values, pdf_values, cdf_values, result_metadata


def _resolve_slice_vol_model(
    base_model: ModelParams,
    vol: Optional[VolModel],
) -> tuple[ModelParams, VolModel, str]:
    """Resolve the effective slice model given a requested ``VolModel``.

    Args:
        base_model: Baseline :class:`ModelParams` prior to any overrides.
        vol: Optional :class:`VolModel` describing user intent.

    Returns:
        A tuple ``(effective_model, resolved_vol, requested_method)`` where
        ``effective_model`` is safe to pass downstream, ``resolved_vol`` has a
        concrete ``method`` value, and ``requested_method`` records the user's
        canonical intent (e.g. ``"svi-jw"`` vs ``"svi"``).

    Raises:
        ValueError: If an incompatible ``VolModel.method`` is supplied for a
        single-expiry smile.
    """

    resolved_vol = vol or VolModel()
    requested_method = resolved_vol.method

    if requested_method is None:
        canonical = base_model.surface_method.lower()
        if canonical not in SLICE_METHODS:
            canonical = "svi"
        return base_model, replace(resolved_vol, method=canonical), canonical

    if requested_method not in SLICE_METHODS:
        raise ValueError(
            "VolModel.method must be one of {'svi', 'svi-jw', 'bspline'} for single expiries"
        )

    effective_surface_method = "svi" if requested_method in {"svi", "svi-jw"} else "bspline"

    if base_model.surface_method != effective_surface_method:
        # Reset surface_options so ModelParams.__post_init__ rebuilds the correct type.
        new_options: Any = None if effective_surface_method == "svi" else {}
        base_model = replace(
            base_model,
            surface_method=effective_surface_method,
            surface_options=new_options,
        )

    return base_model, replace(resolved_vol, method=requested_method), requested_method


# ---------------------------------------------------------------------------
# Public façade – what casual users will interact with
# ---------------------------------------------------------------------------

