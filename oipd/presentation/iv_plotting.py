"""Plotting helpers for implied volatility smiles and surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Mapping,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class HasLogMoneyness(Protocol):
    """Protocol describing observation slices required for surface plotting."""

    maturity: float
    log_moneyness: np.ndarray


@dataclass(frozen=True)
class ForwardPriceAnnotation:
    """Configuration describing how to annotate the smile forward/reference price."""

    value: float
    label: str


def _resolve_plot_style(
    style: Literal["publication", "default"],
) -> tuple[Any, Any, Optional[Callable[[Any], None]]]:
    """Import Matplotlib and prepare the requested visual style.

    Args:
        style: Name of the visual style to apply.

    Returns:
        Tuple containing the Matplotlib pyplot module, the ticker submodule,
        and an optional callable that formats axes according to the chosen
        publication palette.

    Raises:
        ImportError: If Matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.ticker as ticker  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    style_axes: Optional[Callable[[Any], None]] = None
    if style == "publication":
        from oipd.presentation.publication import (
            _apply_publication_style,
            _style_publication_axes,
        )

        _apply_publication_style(plt)
        style_axes = _style_publication_axes

    return plt, ticker, style_axes


def plot_iv_smile(
    smile: pd.DataFrame,
    *,
    include_observed: bool = True,
    observed_bid: pd.DataFrame | None = None,
    observed_ask: pd.DataFrame | None = None,
    observed_last: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (10.0, 5.0),
    title: Optional[str] = None,
    expiry_date: Optional[date] = None,
    style: Literal["publication", "default"] = "publication",
    source: Optional[str] = None,
    show_forward: bool = False,
    forward_price: ForwardPriceAnnotation | None = None,
    x_axis: Literal["log_moneyness", "strike"] = "log_moneyness",
    y_axis: Literal["iv", "total_variance"] = "iv",
    t_to_expiry: Optional[float] = None,
    line_kwargs: Optional[Dict[str, Any]] = None,
    scatter_kwargs: Optional[Dict[str, Any]] = None,
    observed_style: Literal["range", "markers"] = "range",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    ax: "Axes | None" = None,
    show_axis_labels: bool = True,
    show_legend: bool = True,
    title_fontsize: Optional[float] = None,
    tick_labelsize: Optional[float] = None,
) -> "Figure":
    """Render a single implied-volatility smile.

    Args:
        smile: DataFrame containing ``strike`` and ``fitted_iv`` columns with
            optional observed quote columns (``bid_iv``, ``ask_iv``, ``last_iv``).
        include_observed: Whether to display observed implied-volatility inputs.
        observed_bid: Optional bid quote DataFrame for richer annotations.
        observed_ask: Optional ask quote DataFrame for richer annotations.
        observed_last: Optional last trade implied-volatility DataFrame.
        figsize: Matplotlib figure dimensions in inches.
        title: Title text for the chart. When omitted a default is inferred.
        expiry_date: Optional expiry date used to generate a default title when
            ``title`` is not provided.
        style: Requested visual style, matching the estimator API.
        source: Optional attribution text used for publication styling.
        show_forward: Whether to draw a vertical reference line at the
            supplied forward price.
        forward_price: Optional configuration describing the forward price.
        x_axis: Coordinate system for the x-axis ("log_moneyness", "strike").
        y_axis: Metric to plot on y-axis ("iv" or "total_variance").
        t_to_expiry: Time to expiry in years (required if y_axis="total_variance").
        line_kwargs: Keyword overrides forwarded to ``ax.plot``.
        scatter_kwargs: Keyword overrides forwarded to observed quote markers.
        observed_style: Strategy for rendering observed quotes.
        xlim: Optional x-axis limits.
        ylim: Optional y-axis limits.
        ax: Optional Matplotlib Axes to draw on. When provided, the helper
            reuses the axes and returns the corresponding figure.
        show_axis_labels: Whether to render axis titles.
        show_legend: Whether to draw a legend for the plotted data.
        title_fontsize: Optional override for the subplot title font size when
            ``ax`` is supplied.
        tick_labelsize: Optional override for x/y tick label font size.

    Returns:
        Matplotlib Figure populated with the smile visualisation.

    Raises:
        ValueError: If the axis mode requires but lacks a positive forward price.
    """
    plt, ticker, style_axes = _resolve_plot_style(style)


    line_config = dict(line_kwargs or {})
    if "color" not in line_config:
        line_config["color"] = "#1976D2" if style == "publication" else "tab:blue"
    if "linewidth" not in line_config:
        line_config["linewidth"] = 1.5
    if "label" not in line_config:
        line_config["label"] = "Fitted IV"

    axis_choice = x_axis.lower()
    if axis_choice not in {"log_moneyness", "strike"}:
        raise ValueError("x_axis must be 'log_moneyness' or 'strike'")
    if axis_choice == "log_moneyness":
        if forward_price is None or forward_price.value <= 0:
            raise ValueError("Positive forward price required for log-moneyness axis")

    if y_axis == "total_variance":
        if t_to_expiry is None or t_to_expiry <= 0:
            raise ValueError(
                "Positive t_to_expiry is required when plotting total variance"
            )

    def _transform_y(iv_values: np.ndarray | pd.Series) -> np.ndarray:
        if y_axis == "total_variance":
            # w = sigma^2 * T
            return np.square(iv_values) * t_to_expiry
        return np.asarray(iv_values)

    def _to_axis(strike_values: np.ndarray) -> np.ndarray:
        if axis_choice == "log_moneyness":
            return np.log(strike_values / forward_price.value)  # type: ignore[union-attr]
        return strike_values

    created_fig = False
    target_ax = ax
    if target_ax is None:
        fig, target_ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = target_ax.figure

    strike_values = smile["strike"].to_numpy(dtype=float)
    target_ax.plot(_to_axis(strike_values), _transform_y(smile["fitted_iv"]), **line_config)

    if include_observed:
        observed_kwargs = dict(scatter_kwargs or {})
        observed_style_normalised = observed_style.lower()
        if observed_style_normalised not in {"range", "markers"}:
            raise ValueError("observed_style must be 'range' or 'markers'")

        def _prepare_observed_frame(
            frame: pd.DataFrame | None, *, label: str
        ) -> pd.DataFrame:
            if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
                return pd.DataFrame(columns=["strike", "iv", "option_type", "label"])
            df = frame.copy()
            df["strike"] = df["strike"].astype(float)
            df = df.dropna(subset=["iv"])
            if "option_type" in df.columns:
                df["option_type"] = df["option_type"].astype(str).str.upper().str[0]
            else:
                df["option_type"] = "?"
            df["label"] = label
            return df.loc[:, ["strike", "iv", "option_type", "label"]]

        if observed_style_normalised == "range":
            range_color = observed_kwargs.pop(
                "color", "#C62828" if style == "publication" else "tab:red"
            )
            range_alpha = observed_kwargs.pop("alpha", 0.9)
            linewidth = observed_kwargs.pop("linewidth", None)
            if linewidth is None:
                linewidth = observed_kwargs.pop("linewidths", 1.5)
            cap_ratio = observed_kwargs.pop("cap_ratio", 0.2)

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
                required_cols = ["bid_iv", "ask_iv"]
                if set(required_cols).issubset(smile.columns):
                    observed_ranges = smile.dropna(subset=required_cols)
                else:
                    observed_ranges = pd.DataFrame()

            if not observed_ranges.empty:
                strikes_series = observed_ranges["strike"].to_numpy(dtype=float)
                bid_values = observed_ranges["bid_iv"].to_numpy(dtype=float)
                ask_values = observed_ranges["ask_iv"].to_numpy(dtype=float)
                x_coords = _to_axis(strikes_series)

                vlines = target_ax.vlines(
                    x_coords,
                    _transform_y(bid_values),
                    _transform_y(ask_values),
                    colors=range_color,
                    linewidth=linewidth,
                    alpha=range_alpha,
                )
                vlines.set_label("Bid/Ask IV range")

                unique_positions = np.unique(x_coords)
                if unique_positions.size > 1:
                    sorted_positions = np.sort(unique_positions)
                    min_diff = np.min(np.diff(sorted_positions))
                    cap_half_width = max(min_diff * cap_ratio, 1e-6)
                else:
                    cap_half_width = max(abs(x_coords[0]) * 0.05, 0.1)

                left = x_coords - cap_half_width
                right = x_coords + cap_half_width
                target_ax.hlines(
                    _transform_y(bid_values),
                    left,
                    right,
                    colors=range_color,
                    linewidth=linewidth,
                    alpha=range_alpha,
                )
                target_ax.hlines(
                    _transform_y(ask_values),
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
                    target_ax.scatter(
                        _to_axis(valid_last["strike"].to_numpy(dtype=float)),
                        _transform_y(valid_last["iv"]),
                        **marker_kwargs,
                    )
        else:
            call_color = observed_kwargs.pop(
                "call_color", "#EF6C00" if style == "publication" else "tab:orange"
            )
            put_color = observed_kwargs.pop(
                "put_color", "#2E7D32" if style == "publication" else "tab:green"
            )
            other_color = observed_kwargs.pop(
                "other_color", "#5D4037" if style == "publication" else "tab:brown"
            )
            marker_symbol = observed_kwargs.pop("marker", "x")
            bid_alpha = observed_kwargs.pop("bid_alpha", 0.9)
            ask_alpha = observed_kwargs.pop("ask_alpha", 0.6)
            marker_size = observed_kwargs.pop("s", 36)
            observed_kwargs.pop("alpha", None)
            observed_kwargs.pop("linewidth", None)
            observed_kwargs.pop("linewidths", None)
            observed_kwargs.pop("color", None)
            observed_kwargs.pop("c", None)

            def _colour_for(option_type: str) -> str:
                if option_type == "C":
                    return call_color
                if option_type == "P":
                    return put_color
                return other_color

            def _label_for(option_type: str, quote: str) -> str:
                base = (
                    "Call"
                    if option_type == "C"
                    else "Put" if option_type == "P" else "Quote"
                )
                if quote == "bid":
                    return f"{base} bids"
                if quote == "ask":
                    return f"{base} asks"
                return f"{base} IV"

            def _scatter_points(frame: pd.DataFrame, *, alpha: float):
                if frame.empty:
                    return
                plotted_labels: set[tuple[str, str]] = set()
                for option_type, grouped in frame.groupby("option_type"):
                    for quote_kind, quote_df in grouped.groupby("label"):
                        if quote_df.empty:
                            continue
                        label_key = (option_type, quote_kind)
                        label = (
                            _label_for(option_type, quote_kind)
                            if label_key not in plotted_labels
                            else None
                        )
                        plotted_labels.add(label_key)
                        target_ax.scatter(
                            _to_axis(quote_df["strike"].to_numpy(dtype=float)),
                            _transform_y(quote_df["iv"]),
                            color=_colour_for(option_type),
                            marker=marker_symbol,
                            s=marker_size,
                            alpha=alpha,
                            label=label,
                            **observed_kwargs,
                        )

            bid_frame = _prepare_observed_frame(observed_bid, label="bid")
            if bid_frame.empty and "bid_iv" in smile.columns:
                fallback_bid = (
                    smile.loc[:, ["strike", "bid_iv"]]
                    .rename(columns={"bid_iv": "iv"})
                    .dropna()
                )
                fallback_bid["option_type"] = "?"
                fallback_bid["label"] = "bid"
                bid_frame = fallback_bid

            ask_frame = _prepare_observed_frame(observed_ask, label="ask")
            if ask_frame.empty and "ask_iv" in smile.columns:
                fallback_ask = (
                    smile.loc[:, ["strike", "ask_iv"]]
                    .rename(columns={"ask_iv": "iv"})
                    .dropna()
                )
                fallback_ask["option_type"] = "?"
                fallback_ask["label"] = "ask"
                ask_frame = fallback_ask

            _scatter_points(bid_frame, alpha=bid_alpha)
            _scatter_points(ask_frame, alpha=ask_alpha)

            if (
                bid_frame.empty
                and ask_frame.empty
                and isinstance(observed_last, pd.DataFrame)
                and not observed_last.empty
            ):
                last_df = observed_last.copy()
                last_df["strike"] = last_df["strike"].astype(float)
                valid_last = last_df.dropna(subset=["iv"])
                if not valid_last.empty:
                    marker_kwargs = dict(observed_kwargs)
                    marker_kwargs.setdefault(
                        "color", "#C62828" if style == "publication" else "tab:red"
                    )
                    marker_kwargs.setdefault("alpha", 0.9)
                    marker_kwargs.setdefault("marker", "o")
                    marker_kwargs.setdefault("s", 20)
                    marker_kwargs.setdefault("label", "Observed IV")
                    target_ax.scatter(
                        _to_axis(valid_last["strike"].to_numpy(dtype=float)),
                        _transform_y(valid_last["iv"]),
                        **marker_kwargs,
                    )

    if show_axis_labels:
        if axis_choice == "log_moneyness":
            target_ax.set_xlabel("Log Moneyness (ln(K/F))", fontsize=11)
        else:
            target_ax.set_xlabel("Strike", fontsize=11)
    
        if y_axis == "total_variance":
            target_ax.set_ylabel("Total Variance", fontsize=11)
        else:
            target_ax.set_ylabel("Implied Volatility", fontsize=11)
    else:
        target_ax.set_xlabel("")
        target_ax.set_ylabel("")
    
    if y_axis != "total_variance":
        target_ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    if style_axes is not None:
        style_axes(target_ax)

    if tick_labelsize is not None:
        target_ax.tick_params(axis="both", labelsize=tick_labelsize)

    target_ax.relim()
    target_ax.autoscale_view()

    if xlim is not None:
        target_ax.set_xlim(xlim)

    if ylim is not None:
        target_ax.set_ylim(ylim)
    else:
        iv_arrays: list[np.ndarray] = [_transform_y(smile["fitted_iv"])]
        for column in ("bid_iv", "ask_iv", "last_iv"):
            if column in smile.columns:
                iv_arrays.append(_transform_y(smile[column]))
        iv_values = np.concatenate(
            [arr[np.isfinite(arr)] for arr in iv_arrays if arr.size]
        )
        if iv_values.size == 0:
            max_iv = 1.0
        else:
            max_iv = float(np.max(iv_values))
        if not np.isfinite(max_iv) or max_iv <= 0:
            max_iv = 1.0
        target_ax.set_ylim(0.0, max_iv * 1.05)

    if show_forward and forward_price is not None:
        ref_axis_value = 0.0 if axis_choice == "log_moneyness" else forward_price.value
        target_ax.axvline(
            x=ref_axis_value,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        y_min, _ = target_ax.get_ylim()
        target_ax.annotate(
            forward_price.label,
            xy=(ref_axis_value, y_min),
            xytext=(5, 15),
            textcoords="offset points",
            fontsize=10,
            color="#555555",
            ha="left",
            va="bottom",
        )

    if show_legend:
        handles, labels = target_ax.get_legend_handles_labels()
        if handles:
            legend = target_ax.legend(frameon=False, loc="best")
            for text in legend.get_texts():
                text.set_color("#333333")

    if title is not None:
        resolved_title = title
    elif expiry_date is not None:
        resolved_title = (
            f"Implied Volatility Smile (Expiry {expiry_date.strftime('%b %d, %Y')})"
        )
    else:
        resolved_title = "Implied Volatility Smile"
    
    if y_axis == "total_variance":
        resolved_title = resolved_title.replace("Implied Volatility", "Total Variance")
        
    if not created_fig:
        target_ax.set_title(
            resolved_title,
            fontsize=title_fontsize if title_fontsize is not None else None,
        )

    if created_fig:
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


def plot_iv_surface(
    observations: Sequence[HasLogMoneyness],
    *,
    total_variance: Callable[[np.ndarray, float], np.ndarray],
    infer_forward: Callable[[float], float],
    maturities: Optional[Sequence[float]] = None,
    num_points: int = 200,
    x_axis: Literal["log_moneyness", "strike"] = "log_moneyness",
    figsize: tuple[float, float] = (10.0, 5.0),
    title: Optional[str] = None,
    style: Literal["publication", "default"] = "publication",
    source: Optional[str] = None,
    layout: Literal["overlay", "grid"] = "overlay",
    markets_by_maturity: Optional[Mapping[float, Any]] = None,
    include_observed: bool = False,
    observed_style: Literal["range", "markers"] = "range",
    scatter_kwargs: Optional[Dict[str, Any]] = None,
    observed_iv_by_maturity: Optional[Mapping[float, Dict[str, pd.DataFrame]]] = None,
) -> "Figure":
    """Render implied-volatility slices across maturities.

    Args:
        observations: Calibrated slice observations containing log-moneyness
            and maturity information.
        total_variance: Callable returning total variance for ``k`` and
            ``t`` inputs.
        infer_forward: Callable returning the forward price for a maturity.
        maturities: Optional subset of maturities to display.
        num_points: Number of strike or log-moneyness samples per slice.
        x_axis: Coordinate system for the x-axis ("log_moneyness", "strike").
        figsize: Matplotlib figure dimensions in inches.
        title: Custom chart title.
        style: Requested visual style, matching other plotting APIs.
        source: Optional attribution text used for publication styling.
        layout: Arrangement of the rendered slices. ``"overlay"`` draws all
            maturities on a single axes, while ``"grid"`` produces a square
            subplot grid with one smile per maturity.
        markets_by_maturity: Optional mapping providing resolved market
            metadata per maturity. Used to derive annotations in grid mode.
        include_observed: Whether to overlay observed implied volatilities.
        observed_style: Rendering strategy for observed IVs when displayed.
        scatter_kwargs: Keyword overrides forwarded to observed IV markers.
        observed_iv_by_maturity: Optional mapping containing observed IV
            DataFrames keyed by maturity.

    Returns:
        Matplotlib Figure populated with the surface slices.

    Raises:
        CalculationError: If observations are empty or num_points is too small.
        ValueError: If layout is invalid.
    """
    from oipd.core.errors import CalculationError

    if not observations:
        raise CalculationError("Surface calibration has no observations to plot")

    if num_points < 5:
        raise ValueError("num_points must be at least 5 for plotting")

    axis_choice = x_axis.lower()
    if axis_choice not in {"log_moneyness", "strike"}:
        raise ValueError("x_axis must be 'log_moneyness' or 'strike'")

    plt, ticker, style_axes = _resolve_plot_style(style)

    available_maturities = [float(obs.maturity) for obs in observations]

    if maturities is None:
        plot_maturities = available_maturities
    else:
        plot_maturities = []
        for maturity in maturities:
            tolerance = 5e-3
            matched = None
            for available in available_maturities:
                if abs(available - maturity) < tolerance:
                    matched = available
                    break
            if matched is None:
                matched = float(maturity)
            plot_maturities.append(matched)

    all_k = np.concatenate(
        [np.asarray(obs.log_moneyness, dtype=float) for obs in observations]
    )
    k_min = float(np.min(all_k))
    k_max = float(np.max(all_k))
    if not np.isfinite(k_min) or not np.isfinite(k_max):
        raise CalculationError("Invalid log-moneyness range for plotting")
    padding = 0.05 * (k_max - k_min if k_max > k_min else 1.0)
    k_grid = np.linspace(k_min - padding, k_max + padding, num_points)

    layout_mode = layout.lower()
    if layout_mode not in {"overlay", "grid"}:
        raise ValueError("layout must be 'overlay' or 'grid'")

    if layout_mode == "overlay" and include_observed:
        warnings.warn(
            "Observed implied volatilities are only displayed in grid layout.",
            UserWarning,
        )
        include_observed = False

    if layout_mode == "overlay":
        fig, ax = plt.subplots(figsize=figsize)

        def _axis_values(k: np.ndarray, forward: float) -> np.ndarray:
            if axis_choice == "log_moneyness":
                return k
            return forward * np.exp(k)

        for maturity in plot_maturities:
            forward = infer_forward(maturity)
            total_var = total_variance(k_grid, maturity)
            iv_curve = np.sqrt(np.maximum(total_var / max(maturity, 1e-8), 1e-12))
            label_days = int(round(maturity * 365))
            label = f"{label_days}d"
            ax.plot(_axis_values(k_grid, forward), iv_curve, label=label)

        ax.set_xlabel("Log Moneyness" if axis_choice == "log_moneyness" else "Strike")
        ax.set_ylabel("Implied Volatility")
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

        resolved_title = title or "Implied Volatility Surface"
        ax.set_title(resolved_title)
        legend = ax.legend(loc="best")
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#333333")

        if style_axes is not None:
            style_axes(ax)
        if source and style == "publication":
            fig.text(0.99, 0.01, source, ha="right", va="bottom", fontsize=8, alpha=0.6)

        return fig

    # Grid layout
    num_plots = len(plot_maturities)
    if num_plots == 0:
        raise CalculationError("No maturities available for grid layout")

    cols = min(num_plots, math.ceil(math.sqrt(num_plots)))
    rows = math.ceil(num_plots / cols)
    if isinstance(figsize, tuple):
        cell_size = min(figsize)
    else:  # pragma: no cover - defensive
        cell_size = float(figsize)
    width_per_axis = height_per_axis = cell_size
    fig, axes_grid = plt.subplots(
        rows,
        cols,
        figsize=(width_per_axis * cols, height_per_axis * rows),
    )
    axes_iter = np.atleast_1d(axes_grid).ravel()

    for idx, maturity in enumerate(plot_maturities):
        axis = axes_iter[idx]
        forward = float(infer_forward(maturity))
        total_var = total_variance(k_grid, maturity)
        iv_curve = np.sqrt(np.maximum(total_var / max(maturity, 1e-8), 1e-12))
        strikes = forward * np.exp(k_grid)
        smile_df = pd.DataFrame(
            {
                "strike": strikes,
                "fitted_iv": iv_curve,
            }
        )

        market = None
        observed_payload: Dict[str, pd.DataFrame] | None = None
        tolerance = 5e-3
        if markets_by_maturity:
            for key, resolved_market in markets_by_maturity.items():
                if abs(float(key) - maturity) < tolerance:
                    market = resolved_market
                    break
        if observed_iv_by_maturity:
            for key, payload in observed_iv_by_maturity.items():
                if abs(float(key) - maturity) < tolerance:
                    observed_payload = payload
                    break
        if market is not None and getattr(market, "valuation_date", None) is not None:
            valuation_text = market.valuation_date.strftime("%b %d, %Y")
        else:
            valuation_text = None
        if market is not None and getattr(market, "expiry_date", None) is not None:
            expiry_obj = market.expiry_date
            expiry_text = (
                expiry_obj.strftime("%b %d, %Y") if expiry_obj is not None else None
            )
        else:
            expiry_text = None

        reference_label = (
            f"Parity forward on {valuation_text}\n${forward:,.2f}"
            if valuation_text is not None
            else f"Forward ${forward:,.2f}"
        )
        forward_price = ForwardPriceAnnotation(value=forward, label=reference_label)

        days_label = f"{int(round(maturity * 365))}d"
        if expiry_text is not None:
            slice_title = f"{expiry_text} ({days_label})"
        else:
            slice_title = days_label

        observed_bid = (
            observed_payload.get("bid")
            if include_observed and observed_payload
            else None
        )
        observed_ask = (
            observed_payload.get("ask")
            if include_observed and observed_payload
            else None
        )
        observed_last = (
            observed_payload.get("last")
            if include_observed and observed_payload
            else None
        )

        if include_observed:
            for column in ("bid_iv", "ask_iv", "last_iv"):
                if column not in smile_df.columns:
                    smile_df[column] = np.nan

            def _merge_observed(column: str, frame: pd.DataFrame | None) -> None:
                if frame is None or frame.empty:
                    return
                joined = smile_df.loc[:, ["strike"]].merge(
                    frame.loc[:, ["strike", "iv"]], on="strike", how="left"
                )
                smile_df[column] = joined["iv"].to_numpy(dtype=float)

            _merge_observed("bid_iv", observed_bid)
            _merge_observed("ask_iv", observed_ask)
            _merge_observed("last_iv", observed_last)

        plot_iv_smile(
            smile_df,
            include_observed=include_observed,
            observed_bid=observed_bid,
            observed_ask=observed_ask,
            observed_last=observed_last,
            figsize=figsize,
            title=slice_title,
            style=style,
            source=None,
            show_forward=True,
            forward_price=forward_price,
            x_axis=x_axis,
            y_axis="total_variance",  # Surface usually plots total variance, but let's respect the outer call or default? The outer loop calculates total_var logic. Actually the outer function logic computes iv_curve from total_var.
            # Wait, plot_iv_surface usually plots IV slices.
            # The outer function calculates `iv_curve`.
            # Let's check y_axis logic.
            # In the loop: iv_curve = np.sqrt(total_var / ...). So we are plotting IV.
            # But we might want to plot total variance?
            # Creating a DataFrame with "fitted_iv".
            # plot_iv_smile will transform it back to total_variance if y_axis="total_variance".
            t_to_expiry=maturity,
            line_kwargs=None,
            scatter_kwargs=scatter_kwargs,
            observed_style=observed_style,
            xlim=None,
            ylim=None,
            ax=axis,
            show_axis_labels=False,
            show_legend=False,
            title_fontsize=14 if style == "publication" else 12,
            tick_labelsize=12 if style == "publication" else 11,
        )

    for idx in range(num_plots, axes_iter.size):
        axes_iter[idx].set_visible(False)

    grid_title = title or "Implied Volatility Surface"
    top_margin = 0.93 if style == "publication" else 0.93
    fig.suptitle(
        grid_title,
        fontsize=16 if style == "publication" else 14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, top_margin - 0.02))
    fig.subplots_adjust(top=top_margin)
    if source and style == "publication":
        fig.text(
            0.99,
            0.01,
            source,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#666666",
            style="italic",
        )

    return fig
