from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, Dict, Literal, cast
from datetime import date, datetime

import pandas as pd
import numpy as np

from oipd.core import (
    calculate_pdf,
    calculate_cdf,
    fit_kde,
    InvalidInputError,
    CalculationError,
)
from oipd.io import CSVReader, DataFrameReader
from oipd.vendor import get_reader
from oipd.pricing.utils import prepare_dividends


# ---------------------------------------------------------------------------
# Dataclasses holding user configurable parameters
# ---------------------------------------------------------------------------


@dataclass
class MarketParams:
    """Market–specific parameters for the RND estimation."""

    current_price: Optional[float]
    risk_free_rate: float
    # Either provide `days_forward` directly *or* specify `current_date` & `expiry_date`.
    days_forward: Optional[int] = None
    # Alternative way to specify the horizon ---------------------------------
    current_date: Optional[date] = None
    expiry_date: Optional[date] = None
    # --- Optional dividend inputs for future use
    dividend_yield: Optional[float] = None
    dividend_schedule: Optional[pd.DataFrame] = None  # columns: ex_date, amount

    # ------------------------------------------------------------------
    # Validation & convenience
    # ------------------------------------------------------------------
    def __post_init__(self):
        # If days_forward not given, derive it from dates.
        if self.days_forward is None:
            if self.expiry_date is None:
                raise ValueError(
                    "Either `days_forward` or `expiry_date` must be provided."
                )

            # default current_date to today if missing
            current = self.current_date or date.today()
            if isinstance(current, datetime):  # allow datetime
                current = current.date()

            exp = self.expiry_date
            if isinstance(exp, datetime):
                exp = exp.date()

            delta = (exp - current).days
            if delta <= 0:
                raise ValueError(
                    "`expiry_date` must be in the future relative to `current_date`."
                )
            object.__setattr__(self, "days_forward", delta)
        else:
            # days_forward provided directly. Optionally check consistency if dates also supplied.
            if self.expiry_date is not None and self.current_date is not None:
                current = (
                    self.current_date
                    if not isinstance(self.current_date, datetime)
                    else self.current_date.date()
                )
                exp = (
                    self.expiry_date
                    if not isinstance(self.expiry_date, datetime)
                    else self.expiry_date.date()
                )
                delta = (exp - current).days
                if delta != self.days_forward:
                    raise ValueError(
                        "`days_forward` does not match the difference between `current_date` and `expiry_date`."
                    )


@dataclass
class ModelParams:
    """Model / algorithm specific knobs that users may tune."""

    solver: Literal["brent", "newton"] = "brent"
    fit_kde: bool = False
    american_to_european: bool = False  # placeholder for future functionality
    pricing_engine: Literal["bs"] = "bs"


@dataclass(frozen=True)
class RNDResult:
    """Container for the resulting PDF / CDF arrays with convenience helpers."""

    prices: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Return results as a tidy DataFrame."""
        return pd.DataFrame({"Price": self.prices, "PDF": self.pdf, "CDF": self.cdf})

    def to_csv(self, path: str, **kwargs) -> None:
        """Persist results to csv on disk."""
        self.to_frame().to_csv(path, index=False, **kwargs)


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
                cache_enabled=cache_enabled, cache_ttl_minutes=cache_ttl_minutes
            )
        except TypeError:
            self._reader = reader_cls()

        self._current_price: Optional[float] = None
        self._dividend_yield: Optional[float] = None
        self._dividend_schedule: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load options data and extract current price"""
        ticker_expiry = f"{self._ticker}:{self._expiry}"
        df = self._reader.read(ticker_expiry, column_mapping=self._column_mapping)

        # Extract current price from DataFrame metadata
        self._current_price = df.attrs.get("current_price")
        self._dividend_yield = df.attrs.get("dividend_yield")
        self._dividend_schedule = df.attrs.get("dividend_schedule")

        return df

    @property
    def current_price(self) -> Optional[float]:
        """Get the current price fetched from yfinance"""
        return self._current_price

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
    options_data: pd.DataFrame, market: MarketParams, model: ModelParams
) -> RNDResult:
    """Run the core RND estimation given fully validated input data."""

    # Validate that current_price is provided
    if market.current_price is None:
        raise ValueError(
            "current_price must be provided in MarketParams for RND estimation"
        )

    # Determine effective spot and q considering dividend inputs
    val_date = market.current_date or date.today()
    spot_eff, q_eff = prepare_dividends(
        spot=market.current_price,
        dividend_schedule=market.dividend_schedule,
        dividend_yield=market.dividend_yield,
        r=market.risk_free_rate,
        valuation_date=val_date,
    )

    # 1. Calculate PDF
    try:
        pdf_point_arrays = calculate_pdf(
            options_data,
            spot_eff,
            cast(int, market.days_forward),
            market.risk_free_rate,
            model.solver,
            dividend_yield=q_eff,
            pricing_engine=model.pricing_engine,
        )
    except (InvalidInputError, CalculationError):
        raise  # preserve stack & message
    except Exception as exc:
        raise CalculationError(f"Unexpected error during PDF calculation: {exc}")

    # 2. Optionally smooth via KDE
    if model.fit_kde:
        pdf_point_arrays = fit_kde(pdf_point_arrays)

    # 3. Convert PDF → CDF
    price_array, pdf_array = cast(tuple[np.ndarray, np.ndarray], pdf_point_arrays)
    try:
        _, cdf_array = calculate_cdf(
            cast(tuple[np.ndarray, np.ndarray], pdf_point_arrays)
        )
    except Exception as exc:
        raise CalculationError(f"Failed to compute CDF: {exc}")

    return RNDResult(prices=price_array, pdf=pdf_array, cdf=cdf_array)


# ---------------------------------------------------------------------------
# Public façade – what casual users will interact with
# ---------------------------------------------------------------------------


class RND:
    """High-level, user-friendly estimator of the option-implied risk-neutral density (RND)."""

    def __init__(self, model: Optional[ModelParams] = None):
        self.model = model or ModelParams()
        self._result: Optional[RNDResult] = None
        self._market_params: Optional[MarketParams] = None

    # ------------------------------------------------------------------
    # Fit helpers
    # ------------------------------------------------------------------

    def fit(self, source: DataSource, market: MarketParams) -> "RND":
        """Estimate the RND from the given *DataSource* and market parameters."""
        options_data = source.load()
        self._result = _estimate(options_data, market, self.model)
        self._market_params = market  # Store the market parameters
        return self

    # Convenience constructors -------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str,
        market: MarketParams,
        *,
        model: Optional[ModelParams] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> "RND":
        instance = cls(model)
        source = CSVSource(path, column_mapping=column_mapping)
        return instance.fit(source, market)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        market: MarketParams,
        *,
        model: Optional[ModelParams] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> "RND":
        instance = cls(model)
        source = DataFrameSource(df, column_mapping=column_mapping)
        return instance.fit(source, market)

    @classmethod
    def list_expiry_dates(cls, ticker: str, vendor: str = "yfinance") -> list[str]:
        """
        List available expiry dates for a given ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., "AAPL", "SPY")
        vendor : str, default "yfinance"
            Data vendor to use (currently only "yfinance" is supported)

        Returns
        -------
        list[str]
            List of available expiry dates in YYYY-MM-DD format

        Examples
        --------
        >>> expiry_dates = RND.list_expiry_dates("AAPL")
        >>> print(expiry_dates)
        ['2025-01-17', '2025-01-24', '2025-02-21', ...]
        """
        if vendor not in ("yfinance"):
            raise NotImplementedError(f"Vendor '{vendor}' is not supported yet.")

        reader_cls = get_reader(vendor)
        return reader_cls.list_expiry_dates(ticker)

    @classmethod
    def from_ticker(
        cls,
        ticker: str,
        market: MarketParams,
        *,
        model: Optional[ModelParams] = None,
        vendor: str = "yfinance",
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 15,
        **kwargs,
    ) -> "RND":
        """
        Fetch option chain from a data vendor and estimate RND.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., "AAPL", "SPY")
        market : MarketParams
            Market parameters including expiry date and risk-free rate.
            If current_price is not provided, it will be fetched automatically from yfinance.
        model : ModelParams, optional
            Model configuration parameters
        vendor : str, default "yfinance"
            Data vendor to use (currently only "yfinance" is supported)
        cache_enabled : bool, default True
            Whether to enable caching of yfinance data
        cache_ttl_minutes : int, default 15
            Cache time-to-live in minutes
        **kwargs
            Additional keyword arguments

        Returns
        -------
        RND
            Fitted RND estimator

        Examples
        --------
        >>> # First, discover available expiry dates
        >>> expiry_dates = RND.list_expiry_dates("AAPL")
        >>> print(expiry_dates[0])  # e.g., '2025-01-17'
        >>>
        >>> # Then use the expiry date to fetch options data (current price fetched automatically)
        >>> market = MarketParams(expiry_date=date(2025, 1, 17), risk_free_rate=0.045)
        >>> est = RND.from_ticker("AAPL", market)
        >>>
        >>> # Or override the current price if needed
        >>> market = MarketParams(current_price=150.0, expiry_date=date(2025, 1, 17), risk_free_rate=0.045)
        >>> est = RND.from_ticker("AAPL", market, current_price_override=155.0)
        """
        # Validate vendor
        reader_cls = get_reader(vendor)

        if market is None:
            raise ValueError("market parameters must be provided")

        if market.expiry_date is None:
            raise ValueError(
                "expiry_date must be provided in MarketParams for ticker-based data fetching"
            )

        expiry = market.expiry_date.strftime("%Y-%m-%d")

        # yfinance returns columns with different names, so we need to map them
        column_mapping = {
            "strike": "strike",
            "lastPrice": "last_price",
            "bid": "bid",
            "ask": "ask",
        }

        # Create ticker source and fetch data
        source = TickerSource(
            ticker=ticker,
            expiry=expiry,
            vendor=vendor,
            column_mapping=column_mapping,
            cache_enabled=cache_enabled,
            cache_ttl_minutes=cache_ttl_minutes,
        )

        # Load data to get current price
        _df = source.load()

        # ------------------------------------------------------------------
        # Update *the same* MarketParams instance so the caller sees changes
        # ------------------------------------------------------------------

        # 1️⃣  Handle current price (support explicit override)
        current_price_override: Optional[float] = kwargs.pop(
            "current_price_override", None
        )

        if current_price_override is not None:
            # Caller explicitly provided an override – always trust it
            market.current_price = current_price_override
        else:
            fetched_price = source.current_price
            if fetched_price is None:
                raise ValueError(
                    f"Could not fetch current price for {ticker}. "
                    "Please provide current_price in MarketParams."
                )
            # Always refresh with the latest fetched price so stale values don't persist
            market.current_price = fetched_price  # mutate in-place

        # 2️⃣  Refresh dividend information using precedence rules:
        #     user schedule > user yield > auto schedule > auto yield

        # Determine whether the existing dividend fields were previously auto-filled.
        # We record this via a private attribute `_auto_dividends` on the object.
        auto_prev: bool = getattr(market, "_auto_dividends", False)

        user_has_schedule = market.dividend_schedule is not None and not auto_prev
        user_has_yield = (
            market.dividend_schedule is None
            and market.dividend_yield is not None
            and not auto_prev
        )

        auto_schedule = source.dividend_schedule
        auto_yield = source.dividend_yield

        if user_has_schedule:
            # Keep user-provided schedule; do not overwrite.
            pass  # nothing to change
        elif user_has_yield:
            # Keep user-provided dividend yield.
            pass
        else:
            # Either there was no dividend info or it was previously auto-filled.
            # Refresh with the new vendor data following auto precedence.
            if auto_schedule is not None:
                market.dividend_schedule = auto_schedule
                market.dividend_yield = None  # clear yield if schedule now present
            elif auto_yield is not None:
                market.dividend_schedule = None
                market.dividend_yield = auto_yield
            else:
                # No data available from vendor; clear previous auto fields to
                # avoid carrying stale information forward.
                market.dividend_schedule = None
                market.dividend_yield = None

        # Mark whether the current dividend data originated from the vendor so
        # that future calls can decide if it is safe to overwrite.
        setattr(
            market,
            "_auto_dividends",
            market.dividend_schedule is None
            and market.dividend_yield is None
            or (auto_schedule is not None or auto_yield is not None),
        )

        # Create instance and fit using the **mutated** market params (original object)
        instance = cls(model)
        return instance.fit(source, market)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def result(self) -> RNDResult:
        if self._result is None:
            raise ValueError("You must call `fit` first before accessing results.")
        return self._result

    @property
    def pdf_(self):
        """Alias for result.pdf for scikit-learn similarity."""
        return self.result.pdf

    @property
    def cdf_(self):
        return self.result.cdf

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def market_params(self) -> Optional[MarketParams]:
        """Get the market parameters used for estimation."""
        return self._market_params

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return the RND as a tidy DataFrame (convenience)."""
        return self.result.to_frame()

    def to_csv(self, path: str, **kwargs) -> None:
        self.result.to_csv(path, **kwargs)

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

        Examples
        --------
        >>> est = RND.from_csv('data.csv', market)
        >>> prob = est.prob_at_or_above(39.39)  # Returns probability like 0.605 (60.5%)
        >>> print(f"Probability price >= $39.39: {prob:.1%}")  # Prints: "Probability price >= $39.39: 60.5%"
        """
        prices = self.result.prices
        cdf = self.result.cdf

        # Handle edge cases
        if price <= prices.min():
            return 1.0  # If price is below minimum, probability is 100%
        if price >= prices.max():
            return 0.0  # If price is above maximum, probability is 0%

            # Interpolate CDF at the specified price
        cdf_at_price = np.interp(price, prices, cdf)

        # Return 1 - CDF (probability of being at or above)
        return float(1.0 - cdf_at_price)

    def plot(
        self,
        kind: Literal["pdf", "cdf", "both"] = "both",
        figsize: tuple[float, float] = (10, 5),
        title: Optional[str] = None,
        show_current_price: bool = True,
        market_params: Optional[MarketParams] = None,
        style: Literal["publication", "default"] = "publication",
        source: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot the PDF and/or CDF with a clean, customizable interface.

        Parameters
        ----------
        kind : {'pdf', 'cdf', 'both'}, default 'both'
            Which distribution(s) to plot. When 'both', overlays PDF and CDF on same plot with dual y-axes
        figsize : tuple of float, default (10, 5)
            Figure size in inches (width, height)
        title : str, optional
            Main title for the plot. If None, auto-generates based on kind
        show_current_price : bool, default True
            Whether to show a vertical line at current price
        market_params : MarketParams, optional
            Market parameters to use for plotting. If None, uses the parameters from estimation.
        style : {'publication', 'default'}, default 'publication'
            Visual style for the plots
        source : str, optional
            Source attribution text (e.g., "Source: Bloomberg, Author analysis")
        **kwargs
            Additional keyword arguments passed to matplotlib plot()

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object

        Examples
        --------
        >>> est = RND.from_csv('data.csv', market)
        >>> est.plot()  # Uses market parameters from estimation automatically
        >>> est.plot(kind='pdf')  # Shows only PDF
        >>> est.plot(source='Source: Bloomberg, Author analysis')
        """
        from oipd.graphics import plot_rnd

        # Use stored market parameters if none provided
        if market_params is None:
            market_params = self._market_params

        # Extract current price and date from market_params if provided
        current_price = None
        current_date = None
        expiry_date = None
        if market_params and hasattr(market_params, "current_price"):
            current_price = market_params.current_price
        if market_params and hasattr(market_params, "current_date"):
            # Format the date nicely
            if market_params.current_date:
                date_obj = market_params.current_date
                current_date = date_obj.strftime("%b %d, %Y")  # e.g., "Mar 3, 2025"
        if market_params and hasattr(market_params, "expiry_date"):
            # Format the expiry date nicely
            if market_params.expiry_date:
                expiry_obj = market_params.expiry_date
                expiry_date = expiry_obj.strftime("%b %d, %Y")  # e.g., "Dec 19, 2025"

        return plot_rnd(
            prices=self.result.prices,
            pdf=self.pdf_,
            cdf=self.cdf_,
            kind=kind,
            figsize=figsize,
            title=title,
            show_current_price=show_current_price,
            current_price=current_price,
            current_date=current_date,
            expiry_date=expiry_date,
            style=style,
            source=source,
            **kwargs,
        )
