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


# ---------------------------------------------------------------------------
# Dataclasses holding user configurable parameters
# ---------------------------------------------------------------------------


@dataclass
class MarketParams:
    """Market–specific parameters for the RND estimation."""

    current_price: float
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


# ---------------------------------------------------------------------------
# Core estimation routine (non-public)
# ---------------------------------------------------------------------------


def _estimate(
    options_data: pd.DataFrame, market: MarketParams, model: ModelParams
) -> RNDResult:
    """Run the core RND estimation given fully validated input data."""

    # 1. Calculate PDF
    try:
        pdf_point_arrays = calculate_pdf(
            options_data,
            market.current_price,
            cast(
                int, market.days_forward
            ),  # days_forward is guaranteed int after validation
            market.risk_free_rate,
            model.solver,
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

    # ------------------------------------------------------------------
    # Fit helpers
    # ------------------------------------------------------------------

    def fit(self, source: DataSource, market: MarketParams) -> "RND":
        """Estimate the RND from the given *DataSource* and market parameters."""
        options_data = source.load()
        self._result = _estimate(options_data, market, self.model)
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

    # Stub for future ticker-based constructor
    @classmethod
    def from_ticker(
        cls,
        ticker: str,
        expiry: Optional[str] = None,
        vendor: str = "yfinance",
        **kwargs,
    ) -> "RND":
        """Fetch option chain from a data vendor (not yet implemented)."""
        raise NotImplementedError("Ticker-based data fetching is not implemented yet.")

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
            If provided and show_current_price is True, uses the current_price from here
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
        >>> est.plot()  # Shows overlayed PDF and CDF with dual y-axes
        >>> est.plot(kind='pdf')  # Shows only PDF
        >>> est.plot(source='Source: Bloomberg, Author analysis')
        """
        from oipd.graphics import plot_rnd

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
