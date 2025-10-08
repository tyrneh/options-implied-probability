from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, Dict, Literal, Any, Mapping, Sequence, Callable
from datetime import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np
import warnings

from oipd.core.errors import InvalidInputError, CalculationError
from oipd.core.prep import (
    apply_put_call_parity,
    filter_stale_options,
    select_price_column,
    compute_iv,
)
from oipd.core.surface_fitting import AVAILABLE_SURFACE_FITS, fit_surface
from oipd.core.density import (
    price_curve_from_iv,
    pdf_from_price_curve,
    calculate_cdf_from_pdf,
)
from oipd.io import CSVReader, DataFrameReader
from oipd.vendor import get_reader
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
    surface_options: Mapping[str, Any] | None = None
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

        if self.surface_options is None:
            self.surface_options = {}
        elif isinstance(self.surface_options, Mapping):
            self.surface_options = dict(self.surface_options)
        else:
            raise TypeError("surface_options must be a mapping or None")


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
        kind: Literal["pdf", "cdf", "both", "iv_smile"] = "both",
        figsize: tuple[float, float] = (10, 5),
        title: Optional[str] = None,
        show_current_price: bool = True,
        style: Literal["publication", "default"] = "publication",
        source: Optional[str] = None,
        **kwargs,
    ):
        """Plot risk-neutral outputs or the fitted implied-volatility smile.

        Args:
            kind: Plot type to render. Set to ``"pdf"``, ``"cdf"``, or ``"both"``
                for distribution plots, or ``"iv_smile"`` to visualize the
                implied-volatility smile.
            figsize: Matplotlib figure size in inches.
            title: Optional custom title. When omitted, an informative default
                is constructed.
            show_current_price: Whether to highlight the current underlying
                price on the plot (applies to distribution and smile plots).
            style: Visual theme to apply. ``"publication"`` uses the package's
                publication-ready styling, while ``"default"`` relies on
                Matplotlib defaults.
            source: Optional attribution text displayed when ``style`` is
                ``"publication"``.
            **kwargs: Extra options forwarded to the underlying plotting
                routine. For ``kind="iv_smile"`` the supported keywords are
                ``strikes``, ``num_points``, ``include_observed``,
                ``line_kwargs``, ``scatter_kwargs``, ``xlim``, and ``ylim``.
                ``scatter_kwargs`` customize the bid/ask range bars (colour,
                linewidth, cap width via ``cap_ratio``). Any additional keywords
                are applied to the fitted line.

        Returns:
            matplotlib.figure.Figure: Figure containing the requested plot.

        Raises:
            ImportError: If Matplotlib is unavailable.
            ValueError: If the implied-volatility smile cannot be produced.
        """
        if kind == "iv_smile":
            strikes_arg = kwargs.pop("strikes", None)
            num_points_arg = kwargs.pop("num_points", 200)
            include_observed_points = kwargs.pop("include_observed", True)
            line_kwargs_input = kwargs.pop("line_kwargs", None)
            scatter_kwargs = kwargs.pop("scatter_kwargs", None)
            xlim = kwargs.pop("xlim", None)
            ylim = kwargs.pop("ylim", None)

            line_kwargs = dict(line_kwargs_input or {})
            line_kwargs.update(kwargs)

            return self._plot_iv_smile(
                strikes=strikes_arg,
                num_points=num_points_arg,
                include_observed=include_observed_points,
                figsize=figsize,
                title=title,
                style=style,
                source=source,
                show_current_price=show_current_price,
                line_kwargs=line_kwargs,
                scatter_kwargs=scatter_kwargs,
                xlim=xlim,
                ylim=ylim,
            )

        from oipd.graphics import plot_rnd

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

    def _plot_iv_smile(
        self,
        *,
        strikes: Sequence[float] | np.ndarray | None,
        num_points: int,
        include_observed: bool,
        figsize: tuple[float, float],
        title: Optional[str],
        style: Literal["publication", "default"],
        source: Optional[str],
        show_current_price: bool,
        line_kwargs: Dict[str, Any],
        scatter_kwargs: Optional[Dict[str, Any]],
        xlim: Optional[tuple[float, float]],
        ylim: Optional[tuple[float, float]],
    ):
        """Render the implied-volatility smile with optional observed points.

        Args:
            strikes: Optional evaluation strikes for the fitted smile.
            num_points: Number of points for an auto-generated grid when
                ``strikes`` is omitted.
            include_observed: Whether to display the observed implied
                volatilities alongside the fitted curve.
            figsize: Matplotlib figure size in inches.
            title: Custom plot title. If ``None``, a default based on the expiry
                date is used.
            style: Visual theme, mirroring :meth:`plot`.
            source: Optional attribution text for publication style.
            show_current_price: Whether to draw a vertical reference line at the
                current underlying price.
            line_kwargs: Styling overrides for the fitted curve.
            scatter_kwargs: Styling overrides for the observed points.
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            matplotlib.figure.Figure: Figure containing the plotted smile.

        Raises:
            ImportError: If Matplotlib is unavailable.
            ValueError: If the smile cannot be evaluated.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from exc

        style_axes: Optional[Callable[[Any], None]] = None
        if style == "publication":
            from oipd.graphics.publication import (
                _apply_publication_style,
                _style_publication_axes,
            )

            _apply_publication_style(plt)
            style_axes = _style_publication_axes

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        smile_df = self.iv_smile(strikes=strikes, num_points=num_points)

        line_config = dict(line_kwargs or {})
        if "color" not in line_config:
            line_config["color"] = "#1976D2" if style == "publication" else "tab:blue"
        if "linewidth" not in line_config:
            line_config["linewidth"] = 1.5
        if "label" not in line_config:
            line_config["label"] = "Fitted IV"

        ax.plot(smile_df["strike"], smile_df["fitted_iv"], **line_config)

        if include_observed:
            range_kwargs = dict(scatter_kwargs or {})
            range_color = range_kwargs.pop(
                "color", "#C62828" if style == "publication" else "tab:red"
            )
            range_alpha = range_kwargs.pop("alpha", 0.9)
            linewidth = range_kwargs.pop("linewidth", None)
            if linewidth is None:
                linewidth = range_kwargs.pop("linewidths", 1.5)
            cap_ratio = range_kwargs.pop("cap_ratio", 0.2)
            observed_bid = self.meta.get("observed_iv_bid")
            observed_ask = self.meta.get("observed_iv_ask")
            observed_last = self.meta.get("observed_iv_last")
            if (
                isinstance(observed_bid, pd.DataFrame)
                and isinstance(observed_ask, pd.DataFrame)
                and not observed_bid.empty
                and not observed_ask.empty
            ):
                observed_bid = observed_bid.copy()
                observed_ask = observed_ask.copy()
                observed_bid["strike"] = observed_bid["strike"].astype(float)
                observed_ask["strike"] = observed_ask["strike"].astype(float)
                observed_ranges = (
                    observed_bid.rename(columns={"iv": "bid_iv"})
                    .merge(
                        observed_ask.rename(columns={"iv": "ask_iv"}),
                        on="strike",
                        how="inner",
                    )
                    .dropna()
                )
            else:
                observed_ranges = smile_df.dropna(subset=["bid_iv", "ask_iv"])

            if not observed_ranges.empty:
                strikes = observed_ranges["strike"].to_numpy(dtype=float)
                bid_values = observed_ranges["bid_iv"].to_numpy(dtype=float)
                ask_values = observed_ranges["ask_iv"].to_numpy(dtype=float)

                vlines = ax.vlines(
                    strikes,
                    bid_values,
                    ask_values,
                    colors=range_color,
                    linewidth=linewidth,
                    alpha=range_alpha,
                )
                vlines.set_label("Bid/Ask IV range")

                unique_strikes = np.unique(strikes)
                if unique_strikes.size > 1:
                    min_diff = np.min(np.diff(unique_strikes))
                    cap_half_width = max(min_diff * cap_ratio, 1e-6)
                else:
                    cap_half_width = max(abs(strikes[0]) * 0.05, 0.1)

                left = strikes - cap_half_width
                right = strikes + cap_half_width
                ax.hlines(
                    bid_values,
                    left,
                    right,
                    colors=range_color,
                    linewidth=linewidth,
                    alpha=range_alpha,
                )
                ax.hlines(
                    ask_values,
                    left,
                    right,
                    colors=range_color,
                    linewidth=linewidth,
                    alpha=range_alpha,
                )

            if (
                (not isinstance(observed_bid, pd.DataFrame) or observed_bid.empty)
                and (not isinstance(observed_ask, pd.DataFrame) or observed_ask.empty)
                and isinstance(observed_last, pd.DataFrame)
                and not observed_last.empty
            ):
                last_df = observed_last.copy()
                last_df["strike"] = last_df["strike"].astype(float)
                valid_last = last_df.dropna(subset=["iv"])
                if not valid_last.empty:
                    marker_kwargs = dict(scatter_kwargs or {})
                    marker_kwargs.setdefault(
                        "color", "#C62828" if style == "publication" else "tab:red"
                    )
                    marker_kwargs.setdefault("alpha", 0.9)
                    marker_kwargs.setdefault("marker", "o")
                    marker_kwargs.setdefault("s", 20)
                    marker_kwargs.setdefault("label", "Observed IV")
                    ax.scatter(
                        valid_last["strike"],
                        valid_last["iv"],
                        **marker_kwargs,
                    )

        ax.set_xlabel("Strike", fontsize=11)
        ax.set_ylabel("Implied Volatility", fontsize=11)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

        if style_axes is not None:
            style_axes(ax)

        ax.relim()
        ax.autoscale_view()

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        forward_price = self.meta.get("forward_price")
        spot_price = (
            float(self.market.underlying_price)
            if self.market.underlying_price is not None
            else None
        )
        reference_price = (
            float(forward_price) if forward_price is not None else spot_price
        )

        if show_current_price and reference_price is not None:
            ax.axvline(
                x=reference_price,
                color="#555555",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
            y_min, y_max = ax.get_ylim()
            valuation_date = self.market.valuation_date.strftime("%b %d, %Y")
            label_text = (
                f"Parity forward on {valuation_date}\n${reference_price:,.2f}"
                if forward_price is not None
                else f"Price on {valuation_date}\n${reference_price:,.2f}"
            )
            ax.annotate(
                label_text,
                xy=(reference_price, y_min),
                xytext=(5, 15),
                textcoords="offset points",
                fontsize=10,
                color="#555555",
                ha="left",
                va="bottom",
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(frameon=False, loc="best")
            if style == "publication":
                for text in legend.get_texts():
                    text.set_color("#333333")

        resolved_title = title
        if resolved_title is None:
            expiry_obj = self.market.expiry_date
            if expiry_obj is not None:
                resolved_title = f"Implied Volatility Smile (Expiry {expiry_obj.strftime('%b %d, %Y')})"
            else:
                resolved_title = "Implied Volatility Smile"

        if style == "publication":
            plt.subplots_adjust(top=0.85, bottom=0.08 if source else 0.06)
            fig.suptitle(
                resolved_title,
                fontsize=16,
                fontweight="bold",
                y=0.92,
                color="#333333",
            )
            if source:
                fig.text(
                    0.99,
                    0.02,
                    source,
                    ha="right",
                    fontsize=9,
                    color="#666666",
                    style="italic",
                )
        else:
            fig.suptitle(resolved_title, fontsize=14, fontweight="bold")
            fig.tight_layout()

        return fig


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

    surface_fit_kwargs: Dict[str, Any] = {}
    if model.surface_method == "svi":
        # SVI needs the forward level and time to expiry to calibrate properly
        surface_fit_kwargs.update(
            {
                "forward": underlying_price,
                "maturity_years": resolved_market.days_to_expiry / 365.0,
            }
        )

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

        return iv_df.loc[:, ["strike", "iv"]]

    observed_bid_iv = _compute_observed_iv(options_with_selected_price, "bid")
    observed_ask_iv = _compute_observed_iv(options_with_selected_price, "ask")
    observed_last_iv = _compute_observed_iv(options_with_selected_price, "last_price")

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


# ---------------------------------------------------------------------------
# Public façade – what casual users will interact with
# ---------------------------------------------------------------------------


class RND:
    """High-level, user-friendly estimator of the option-implied risk-neutral density (RND)."""

    def __init__(self, model: Optional[ModelParams] = None, *, verbose: bool = True):
        self.model = model or ModelParams()
        self._result: Optional[RNDResult] = None
        self._verbose: bool = verbose

    # ------------------------------------------------------------------
    # Warning control
    # ------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def _suppress_oipd_warnings(suppress: bool):
        """Context manager to optionally silence UserWarnings from this package.

        When `suppress` is True, filters out UserWarning emitted from modules under
        the `oipd` package so demos/notebooks are not cluttered. Errors are never
        suppressed.
        """
        if not suppress:
            # Do nothing – propagate warnings normally
            yield
            return

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"oipd(\.|$)",
            )
            yield

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
            underlying_price=source.underlying_price,
            dividend_yield=source.dividend_yield,
            dividend_schedule=source.dividend_schedule,
        )

        return chain, snapshot

    def fit(self, source: DataSource, resolved_market: ResolvedMarket) -> "RND":
        """Estimate the RND from the given *DataSource* and resolved market parameters."""
        with self._suppress_oipd_warnings(suppress=not self._verbose):
            options_data = source.load()
            prices, pdf, cdf, meta = _estimate(
                options_data, resolved_market, self.model
            )
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
        verbose: bool = True,
    ) -> RNDResult:
        """Load options data from CSV and estimate RND.

        In CSV mode, underlying_price and dividend information must be provided
        by the user in the market parameters.
        """
        # Read chain from CSV
        source = CSVSource(path, column_mapping=column_mapping)
        with cls._suppress_oipd_warnings(suppress=not verbose):
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
        verbose: bool = True,
    ) -> RNDResult:
        """Load options data from DataFrame and estimate RND.

        Similar to from_csv, underlying_price and dividend information must be
        provided by the user in the market parameters.
        """
        # Process DataFrame
        source = DataFrameSource(df, column_mapping=column_mapping)
        with cls._suppress_oipd_warnings(suppress=not verbose):
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
        echo: Optional[bool] = None,
        verbose: bool = True,
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
            If underlying_price is not provided, it will be fetched automatically.
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
        ...     underlying_price=150.0,
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
        with cls._suppress_oipd_warnings(suppress=not verbose):
            chain, snapshot = cls._fetch_vendor_snapshot(
                ticker, expiry, vendor, cache_enabled, cache_ttl_minutes
            )

            # Resolve market parameters by merging user inputs with vendor snapshot
            resolved = resolve_market(market, snapshot, mode=fill)

            # Choose effective model: default to last price for yfinance
            effective_model = (
                model
                if model is not None
                else (
                    ModelParams(price_method="last")
                    if vendor == "yfinance"
                    else ModelParams()
                )
            )

            # Run estimation
            prices, pdf, cdf, meta = _estimate(chain, resolved, effective_model)

        # Add ticker and vendor info to metadata
        meta.update(
            {
                "ticker": ticker,
                "vendor": snapshot.vendor,
                "asof": snapshot.asof.isoformat(),
            }
        )

        result = RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=resolved, meta=meta)

        # Determine whether to echo summary
        echo_flag = verbose if echo is None else echo

        # Print summary if requested
        if echo_flag:
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

    def iv_smile(
        self,
        strikes: Sequence[float] | np.ndarray | None = None,
        *,
        num_points: int = 200,
    ) -> pd.DataFrame:
        """Retrieve the fitted implied-volatility smile on a strike grid.

        Args:
            strikes: Optional sequence of strike values where the smile should
                be evaluated. If omitted, the stored calibration grid or a
                generated linspace is used.
            num_points: Number of points used to generate the evaluation grid
                when ``strikes`` is not provided. Must be at least two for a
                non-degenerate strike range.

        Returns:
            DataFrame containing strike levels and fitted implied volatilities,
            along with observed bid/ask implied volatilities when available.
        """
        return self.result.iv_smile(strikes, num_points=num_points)
