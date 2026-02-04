"""Plotting helpers for risk-neutral density visualisations."""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Any

import numpy as np

from oipd.presentation.publication import (
    _apply_publication_style,
    _style_publication_axes,
)


def plot_rnd(
    prices: np.ndarray,
    pdf: np.ndarray,
    cdf: np.ndarray,
    kind: Literal["pdf", "cdf", "both"] = "both",
    figsize: Tuple[float, float] = (10.0, 5.0),
    title: Optional[str] = None,
    show_current_price: bool = True,
    current_price: Optional[float] = None,
    valuation_date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    style: Literal["publication", "default"] = "publication",
    source: Optional[str] = None,
    **kwargs: Any,
):
    """Visualise risk-neutral density PDF and CDF outputs.

    Args:
        prices: np.ndarray
            Price grid used to evaluate the distribution.
        pdf: np.ndarray
            Probability density function values aligned with ``prices``.
        cdf: np.ndarray
            Cumulative distribution function values aligned with ``prices``.
        kind: Literal["pdf", "cdf", "both"], default "both"
            Selection of which distribution(s) to display.
        figsize: tuple[float, float], default (10.0, 5.0)
            Matplotlib figure size in inches.
        title: str | None, optional
            Custom title. When omitted a descriptive default is generated.
        show_current_price: bool, default True
            Whether to annotate the current underlying price.
        current_price: float | None, optional
            Underlying price used for the annotation when available.
        valuation_date: str | None, optional
            Valuation date string appended to the annotation.
        expiry_date: str | None, optional
            Expiry date string incorporated into auto-generated titles.
        style: Literal["publication", "default"], default "publication"
            Visual palette to apply.
        source: str | None, optional
            Attribution text displayed in publication mode.
        **kwargs: Any
            Additional keyword arguments forwarded to Matplotlib plot calls.

    Returns:
        matplotlib.figure.Figure: Figure containing the requested plot.

    Raises:
        ImportError: If Matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    if style == "publication":
        _apply_publication_style(plt)

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)

    if style == "publication":
        pdf_color = "#1976D2"
        cdf_color = "#C62828"
        spot_price_color = "black"
    else:
        pdf_color = kwargs.get("color", "tab:blue")
        cdf_color = kwargs.get("color", "tab:orange")
        spot_price_color = "black"

    plot_kwargs = {key: value for key, value in kwargs.items() if key != "color"}

    if kind in {"pdf", "both"}:
        if kind == "both":
            ax1.plot(
                prices,
                pdf,
                color=pdf_color,
                linewidth=1.5,
                label="PDF",
                **plot_kwargs,
            )
        else:
            ax1.plot(
                prices,
                pdf,
                color=pdf_color,
                linewidth=1.5,
                **plot_kwargs,
            )
        ax1.set_xlabel("Price at expiry", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11, color="#333333")
        ax1.tick_params(axis="y", labelcolor=pdf_color)
        ax1.set_ylim(bottom=0)

        if style == "publication":
            _style_publication_axes(ax1)

    if kind in {"cdf", "both"}:
        if kind == "both":
            ax2 = ax1.twinx()
            ax2.plot(
                prices,
                cdf,
                color=cdf_color,
                linewidth=1.5,
                linestyle="-",
                alpha=0.9,
                label="CDF",
                **plot_kwargs,
            )
            ax2.set_ylabel("Cumulative Probability", fontsize=11, color="#333333")
            ax2.tick_params(axis="y", labelcolor=cdf_color)
            ax2.set_ylim(0, 1)

            if style == "publication":
                ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
                ax2.spines["right"].set_color(cdf_color)
                ax2.spines["right"].set_linewidth(0.5)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if style == "publication":
                legend = ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    frameon=False,
                    fontsize=10,
                    loc="upper left",
                )
                for text in legend.get_texts():
                    text.set_color("#333333")
            else:
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            ax1.plot(
                prices,
                cdf,
                color=cdf_color,
                linewidth=1.5,
                alpha=0.9,
                **plot_kwargs,
            )
            ax1.set_xlabel("Price at expiry", fontsize=11)
            ax1.set_ylabel("Cumulative Probability", fontsize=11, color="#333333")
            ax1.set_ylim(0, 1)

            if style == "publication":
                _style_publication_axes(ax1)
                ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    if title is None:
        if kind == "pdf":
            plot_title = (
                f"Implied PDF of price on {expiry_date}" if expiry_date else "PDF"
            )
        elif kind == "cdf":
            plot_title = (
                f"Implied CDF of price on {expiry_date}" if expiry_date else "CDF"
            )
        else:
            plot_title = (
                f"Implied PDF and CDF of price on {expiry_date}"
                if expiry_date
                else "Risk-Neutral Distribution"
            )
    else:
        plot_title = title

    if show_current_price and current_price is not None:
        date_text = valuation_date or "current date"
        price_text = f"Price on {date_text}\nis ${current_price:,.2f}"
        if kind == "both":
            ax1.axvline(
                x=current_price,
                color=spot_price_color,
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
                label=f"Price: ${current_price:,.2f}",
            )
            y_min, y_max = ax1.get_ylim()
            y_text_pos = y_min + (y_max - y_min) * 0.15
            ax1.text(
                current_price + (prices.max() - prices.min()) * 0.02,
                y_text_pos,
                price_text,
                color=spot_price_color,
                fontsize=12,
                fontstyle="italic",
                va="bottom",
                ha="left",
            )
        else:
            ax1.axvline(
                x=current_price,
                color=spot_price_color,
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )
            if kind == "pdf":
                y_min, y_max = ax1.get_ylim()
                y_text_pos = y_min + (y_max - y_min) * 0.15
            else:
                y_text_pos = 0.15
            ax1.text(
                current_price + (prices.max() - prices.min()) * 0.02,
                y_text_pos,
                price_text,
                color=spot_price_color,
                fontsize=12,
                fontstyle="italic",
                va="bottom",
                ha="left",
            )

    if xlim is not None:
        ax1.set_xlim(xlim)
    else:
        try:
            ax1.relim()
            ax1.autoscale_view()
            left, right = ax1.get_xlim()
            if left < 0:
                ax1.set_xlim(left=0, right=right)
        except Exception:
            pass

    if ylim is not None:
        ax1.set_ylim(ylim)

    try:
        ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    except Exception:
        pass

    if style == "publication":
        plt.subplots_adjust(top=0.85, bottom=0.1 if source else 0.05)
        fig.suptitle(
            plot_title,
            fontsize=16,
            fontweight="bold",
            y=0.925,
            color="#333333",
        )
        if source:
            fig.text(
                0.99,
                0.01,
                source,
                ha="right",
                fontsize=9,
                color="#666666",
                style="italic",
            )
    else:
        plt.tight_layout()
        fig.suptitle(plot_title, fontsize=14, fontweight="bold")

    return fig
