from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, Dict, Literal, cast, Any
from datetime import datetime

import pandas as pd
import numpy as np
import warnings

from oipd.core import (
    calculate_pdf,
    calculate_cdf,
    fit_kde,
    InvalidInputError,
    CalculationError,
)
from oipd.core.parity import preprocess_with_parity
from oipd.io import CSVReader, DataFrameReader
from oipd.vendor import get_reader
from oipd.pricing.utils import prepare_dividends
from oipd.market_inputs import (
    MarketInputs,
    VendorSnapshot,
    ResolvedMarket,
    resolve_market,
    FillMode,
)


# ---------------------------------------------------------------------------
# Dataclasses holding user configurable parameters
# ---------------------------------------------------------------------------


# MarketParams class has been removed - use MarketInputs from market_inputs.py instead


@dataclass
class ModelParams:
    """Model / algorithm specific knobs that users may tune."""

    solver: Literal["brent", "newton"] = "brent"
    fit_kde: bool = False
    american_to_european: bool = False  # placeholder for future functionality
    pricing_engine: Literal["black76", "bs"] = "black76"
    price_method: Literal["last", "mid"] = "last"
    max_staleness_days: Optional[int] = 3


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
        return self.market.summary()

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

    def plot(
        self,
        kind: Literal["pdf", "cdf", "both"] = "both",
        figsize: tuple[float, float] = (10, 5),
        title: Optional[str] = None,
        show_spot_price: bool = True,
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
        show_spot_price : bool, default True
            Whether to show a vertical line at spot price
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
        """
        from oipd.graphics import plot_rnd

        # Extract spot price and formatted dates from resolved market
        spot_price = self.market.spot_price

        # Format dates nicely from resolved market parameters
        valuation_date = None
        expiry_date = None

        # Use actual dates from the resolved market
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
            show_spot_price=show_spot_price,
            spot_price=spot_price,
            valuation_date=valuation_date,
            expiry_date=expiry_date,
            style=style,
            source=source,
            **kwargs,
        )


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

        self._spot_price: Optional[float] = None
        self._dividend_yield: Optional[float] = None
        self._dividend_schedule: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load options data and extract current price"""
        ticker_expiry = f"{self._ticker}:{self._expiry}"
        df = self._reader.read(ticker_expiry, column_mapping=self._column_mapping)

        # Extract spot price from DataFrame metadata
        self._spot_price = df.attrs.get("spot_price")
        self._dividend_yield = df.attrs.get("dividend_yield")
        self._dividend_schedule = df.attrs.get("dividend_schedule")

        return df

    @property
    def spot_price(self) -> Optional[float]:
        """Get the spot price fetched from yfinance"""
        return self._spot_price

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
    options_data: pd.DataFrame, resolved_market: ResolvedMarket, model: ModelParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run the core RND estimation given fully validated input data."""

    val_date = resolved_market.valuation_date

    if model.pricing_engine == "bs":
        spot_eff, q_eff = prepare_dividends(
            spot=resolved_market.spot_price,
            dividend_schedule=resolved_market.dividend_schedule,
            dividend_yield=resolved_market.dividend_yield,
            r=resolved_market.risk_free_rate,
            valuation_date=val_date,
        )
        forward_price = None
    else:
        spot_eff = resolved_market.spot_price
        q_eff = None
        forward_price = None

    # Apply put-call parity preprocessing if beneficial
    discount_factor = np.exp(-resolved_market.risk_free_rate * resolved_market.days_to_expiry / 365.0)
    options_data = preprocess_with_parity(options_data, spot_eff, discount_factor)

    if model.pricing_engine == "black76":
        if "F_used" not in options_data.columns:
            warnings.warn(
                "Black-76 requires a parity-implied forward but put quotes are missing. "
                "Rerun with ModelParams(pricing_engine='bs') and provide dividend_yield or dividend_schedule.",
                UserWarning,
            )
            raise ValueError(
                "Put options missing: switch to Black-Scholes with explicit dividend inputs"
            )
        forward_price = float(options_data["F_used"].iloc[0])

    # Filter stale data if configured and last_trade_date column is present
    if (
        model.max_staleness_days is not None
        and "last_trade_date" in options_data.columns
    ):

        # Check if the column has valid date data
        last_trade_datetimes = pd.to_datetime(options_data["last_trade_date"])

        # Skip filtering if column has NaT values (common in CSV files without date data)
        if last_trade_datetimes.isna().any():
            # Column exists but has missing values - skip filtering
            pass
        else:
            # Calculate days since trade relative to valuation_date
            options_data = options_data.copy()  # Don't modify original
            last_trade_dates = last_trade_datetimes.dt.date

            # Calculate days difference manually
            days_old = []
            for trade_date in last_trade_dates:
                days_diff = (resolved_market.valuation_date - trade_date).days
                days_old.append(days_diff)
            days_old = pd.Series(days_old)

            # Filter out stale data
            fresh_mask = days_old <= model.max_staleness_days
            stale_count = (~fresh_mask).sum()

            if stale_count > 0:
                warnings.warn(
                    f"Filtered {stale_count} strikes older than {model.max_staleness_days} days "
                    f"(most recent: {days_old.min()} days old, oldest: {days_old.max()} days old)",
                    UserWarning,
                )
                options_data = options_data[fresh_mask].reset_index(drop=True)

    # 1. Calculate PDF
    try:
        pdf_point_arrays = calculate_pdf(
            options_data,
            spot_eff,
            resolved_market.days_to_expiry,
            resolved_market.risk_free_rate,
            model.solver,
            model.pricing_engine,
            q_eff,
            model.price_method,
            forward_price=forward_price,
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

    meta = {"model_params": model}
    return price_array, pdf_array, cdf_array, meta


# ---------------------------------------------------------------------------
# Public façade – what casual users will interact with
# ---------------------------------------------------------------------------


class RND:
    """High-level, user-friendly estimator of the option-implied risk-neutral density (RND)."""

    def __init__(self, model: Optional[ModelParams] = None):
        self.model = model or ModelParams()
        self._result: Optional[RNDResult] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_vendor_snapshot(
        ticker: str,
        expiry_str: str,
        vendor: str = "yfinance",
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 15,
    ) -> tuple[pd.DataFrame, VendorSnapshot]:
        """Fetch options chain and vendor snapshot for a ticker."""
        # Create ticker source to fetch data
        # Note: yfinance doesn't have option_type - we create it when combining calls/puts
        column_mapping = {
            "lastPrice": "last_price",
            "lastTradeDate": "last_trade_date",
            # strike, bid, ask are already correctly named in yfinance
        }

        source = TickerSource(
            ticker=ticker,
            expiry=expiry_str,
            vendor=vendor,
            column_mapping=column_mapping,
            cache_enabled=cache_enabled,
            cache_ttl_minutes=cache_ttl_minutes,
        )

        # Load the options chain
        chain = source.load()

        # Create vendor snapshot
        snapshot = VendorSnapshot(
            asof=datetime.now(),
            vendor=vendor,
            spot_price=source.spot_price,
            dividend_yield=source.dividend_yield,
            dividend_schedule=source.dividend_schedule,
        )

        return chain, snapshot

    def fit(self, source: DataSource, resolved_market: ResolvedMarket) -> "RND":
        """Estimate the RND from the given *DataSource* and resolved market parameters."""
        options_data = source.load()
        prices, pdf, cdf, meta = _estimate(options_data, resolved_market, self.model)
        self._result = RNDResult(
            prices=prices, pdf=pdf, cdf=cdf, market=resolved_market, meta=meta
        )
        return self

    # Convenience constructors -------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str,
        market: MarketInputs,
        *,
        model: Optional[ModelParams] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> RNDResult:
        """Load options data from CSV and estimate RND.

        In CSV mode, spot_price and dividend information must be provided
        by the user in the market parameters.
        """
        # Read chain from CSV
        source = CSVSource(path, column_mapping=column_mapping)
        chain = source.load()

        # Resolve market parameters (strict mode - no vendor)
        resolved = resolve_market(market, vendor=None, mode="strict")

        # Run estimation
        prices, pdf, cdf, meta = _estimate(chain, resolved, model or ModelParams())

        return RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=resolved, meta=meta)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        market: MarketInputs,
        *,
        model: Optional[ModelParams] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> RNDResult:
        """Load options data from DataFrame and estimate RND.

        Similar to from_csv, spot_price and dividend information must be
        provided by the user in the market parameters.
        """
        # Process DataFrame
        source = DataFrameSource(df, column_mapping=column_mapping)
        chain = source.load()

        # Resolve market parameters (strict mode - no vendor)
        resolved = resolve_market(market, vendor=None, mode="strict")

        # Run estimation
        prices, pdf, cdf, meta = _estimate(chain, resolved, model or ModelParams())

        return RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=resolved, meta=meta)

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
        market: MarketInputs,
        *,
        model: Optional[ModelParams] = None,
        vendor: str = "yfinance",
        fill: FillMode = "missing",
        echo: bool = True,
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 15,
    ) -> RNDResult:
        """
        Fetch option chain from a data vendor and estimate RND.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., "AAPL", "SPY")
        market : MarketInputs
            Market parameters including expiry date and risk-free rate.
            If spot_price is not provided, it will be fetched automatically.
        model : ModelParams, optional
            Model configuration parameters
        vendor : str, default "yfinance"
            Data vendor to use (currently only "yfinance" is supported)
        fill : FillMode, default "missing"
            How to fill missing market data:
            - "missing": Use user values when available, fill missing from vendor
            - "vendor_only": Use only vendor data, ignore user inputs
            - "strict": Require all fields from user (no vendor filling)
        echo : bool, default True
            Whether to print a summary of resolved parameters
        cache_enabled : bool, default True
            Whether to enable caching of vendor data
        cache_ttl_minutes : int, default 15
            Cache time-to-live in minutes

        Returns
        -------
        RNDResult
            Result containing PDF/CDF arrays and resolved market parameters

        Examples
        --------
        >>> # Discover available expiry dates
        >>> expiry_dates = RND.list_expiry_dates("AAPL")
        >>>
        >>> # Auto-fetch current price and dividends
        >>> market = MarketInputs(
        ...     valuation_date=date.today(),
        ...     expiry_date=date(2025, 1, 17),
        ...     risk_free_rate=0.045
        ... )
        >>> result = RND.from_ticker("AAPL", market)
        >>> print(result.summary())  # Shows what was auto-fetched
        >>>
        >>> # Override with your own price
        >>> market = MarketInputs(
        ...     valuation_date=date.today(),
        ...     spot_price=150.0,
        ...     expiry_date=date(2025, 1, 17),
        ...     risk_free_rate=0.045
        ... )
        >>> result = RND.from_ticker("AAPL", market)
        """
        # Validate inputs
        if market.expiry_date is None:
            raise ValueError(
                "expiry_date must be provided in MarketInputs for ticker-based data fetching"
            )

        expiry = market.expiry_date.strftime("%Y-%m-%d")

        # Fetch chain and vendor snapshot
        chain, snapshot = cls._fetch_vendor_snapshot(
            ticker, expiry, vendor, cache_enabled, cache_ttl_minutes
        )

        # Resolve market parameters by merging user inputs with vendor snapshot
        resolved = resolve_market(market, snapshot, mode=fill)

        # Run estimation
        prices, pdf, cdf, meta = _estimate(chain, resolved, model or ModelParams())

        # Add ticker and vendor info to metadata
        meta.update(
            {
                "ticker": ticker,
                "vendor": snapshot.vendor,
                "asof": snapshot.asof.isoformat(),
            }
        )

        result = RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=resolved, meta=meta)

        # Print summary if requested
        if echo:
            print(result.summary())

        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def result(self) -> RNDResult:
        if self._result is None:
            raise ValueError("You must call `fit` first before accessing results.")
        return self._result

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return the RND as a tidy DataFrame (convenience)."""
        return self.result.to_frame()

    def to_csv(self, path: str, **kwargs) -> None:
        self.result.to_csv(path, **kwargs)

    def prob_at_or_above(self, price: float) -> float:
        """Delegate to result.prob_at_or_above() for backward compatibility."""
        return self.result.prob_at_or_above(price)

    def prob_below(self, price: float) -> float:
        """Delegate to result.prob_below() for backward compatibility."""
        return self.result.prob_below(price)

    def plot(self, **kwargs):
        """Delegate to result.plot() for backward compatibility."""
        return self.result.plot(**kwargs)
