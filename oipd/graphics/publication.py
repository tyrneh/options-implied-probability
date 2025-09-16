"""
Publication-ready plotting functionality for risk-neutral distributions.

This module provides clean, professional visualization tools with modern styling
suitable for academic papers, reports, and presentations.
"""

from typing import Optional, Literal, Tuple
import numpy as np


def plot_rnd(
    prices: np.ndarray,
    pdf: np.ndarray,
    cdf: np.ndarray,
    kind: Literal["pdf", "cdf", "both"] = "both",
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
    show_current_price: bool = True,
    current_price: Optional[float] = None,
    valuation_date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    style: Literal["publication", "default"] = "publication",
    source: Optional[str] = None,
    **kwargs,
):
    """
    Plot the PDF and/or CDF with a clean, customizable interface.

    Parameters
    ----------
    prices : np.ndarray
        Array of price values
    pdf : np.ndarray
        Array of probability density values
    cdf : np.ndarray
        Array of cumulative probability values
    kind : {'pdf', 'cdf', 'both'}, default 'both'
        Which distribution(s) to plot. When 'both', overlays PDF and CDF on same plot with dual y-axes
    figsize : tuple of float, default (10, 5)
        Figure size in inches (width, height)
    title : str, optional
        Main title for the plot. If None, auto-generates based on kind
    show_current_price : bool, default True
        Whether to show a vertical line at the current price
    current_price : float, optional
        Current price value for reference line
    valuation_date : str, optional
        Current date for price annotation (e.g., "Mar 3, 2025")
    expiry_date : str, optional
        Expiry date for the distribution (e.g., "Mar 3, 2025")
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
    >>> plot_rnd(prices, pdf, cdf)  # Shows overlayed PDF and CDF line plots with dual y-axes
    >>> plot_rnd(prices, pdf, cdf, kind='pdf')  # Shows only PDF line plot
    >>> plot_rnd(prices, pdf, cdf, expiry_date='Dec 19, 2025')  # Titles use "price on Dec 19, 2025"
    >>> plot_rnd(prices, pdf, cdf, source='Source: Bloomberg, Author analysis')
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        raise ImportError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    # Apply publication style if requested
    if style == "publication":
        _apply_publication_style(plt)

    # Create single plot
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # Extract axis-related parameters that shouldn't go to plot()
    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)

    # Define colors for publication style
    if style == "publication":
        pdf_color = "#1976D2"  # blue for PDF
        cdf_color = "#C62828"  # Red for CDF
        spot_price_color = "black"  # black for spot price
    else:
        pdf_color = kwargs.get("color", "tab:blue")
        cdf_color = kwargs.get("color", "tab:orange")
        spot_price_color = "black"  # black for spot price

    # Remove color from kwargs if we're setting it
    plot_kwargs = {k: v for k, v in kwargs.items() if k != "color"}

    if kind in ["pdf", "both"]:
        # Plot PDF as line plot on left y-axis
        if kind == "both":
            ax1.plot(
                prices,
                pdf,
                color=pdf_color,
                linewidth=1.5,
                label="PDF",  # Simplified label
                **plot_kwargs,
            )
        else:
            # Individual PDF plot: no label for legend
            ax1.plot(
                prices,
                pdf,
                color=pdf_color,
                linewidth=1.5,
                **plot_kwargs,
            )
        ax1.set_xlabel("Price at expiry", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11, color="#333333")  # Black y-axis title
        ax1.tick_params(axis="y", labelcolor=pdf_color)
        # Ensure PDF y-axis starts at 0
        ax1.set_ylim(bottom=0)

        if style == "publication":
            _style_publication_axes(ax1)

    if kind in ["cdf", "both"]:
        if kind == "both":
            # Create second y-axis for CDF
            ax2 = ax1.twinx()
            ax2.plot(
                prices,
                cdf,
                color=cdf_color,
                linewidth=1.5,
                linestyle="-",
                alpha=0.9,
                label="CDF",  # Simplified label
                **plot_kwargs,
            )
            ax2.set_ylabel(
                "Cumulative Probability", fontsize=11, color="#333333"
            )  # Black y-axis title
            ax2.tick_params(axis="y", labelcolor=cdf_color)
            # Ensure CDF y-axis starts at 0 and aligns with PDF axis
            ax2.set_ylim(0, 1)

            if style == "publication":
                # Format y-axis as percentages for CDF
                ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
                # Apply minimal styling to right axis
                ax2.spines["right"].set_color(cdf_color)
                ax2.spines["right"].set_linewidth(0.5)

            # Create combined legend for overlay plot
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
                # Explicitly set legend text color to black
                for text in legend.get_texts():
                    text.set_color("#333333")
            else:
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            # CDF only - no legend for individual plot
            ax1.plot(
                prices, cdf, color=cdf_color, linewidth=1.5, alpha=0.9, **plot_kwargs
            )
            ax1.set_xlabel("Price at expiry", fontsize=11)
            ax1.set_ylabel(
                "Cumulative Probability", fontsize=11, color="#333333"
            )  # Black y-axis title
            ax1.set_ylim(0, 1)

            if style == "publication":
                _style_publication_axes(ax1)
                # Format y-axis as percentages for CDF
                ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    # Set title
    if title is None:
        if kind == "pdf":
            if expiry_date:
                plot_title = f"Implied PDF of price on {expiry_date}"
            else:
                plot_title = "PDF"
        elif kind == "cdf":
            if expiry_date:
                plot_title = f"Implied CDF of price on {expiry_date}"
            else:
                plot_title = "CDF"
        else:
            # For overlay plots, use expiry date if available
            if expiry_date:
                plot_title = f"Implied PDF and CDF of price on {expiry_date}"
            else:
                plot_title = "Risk-Neutral Distribution"
    else:
        plot_title = title

    # Optional: show current price
    if show_current_price and current_price is not None:
        # Format date string for annotation
        if valuation_date:
            date_text = valuation_date
        else:
            date_text = "current date"

        # Display current price label without the word "spot"
        price_text = f"Price on {date_text}\nis ${current_price:,.2f}"

        # For overlay plots, add to legend; for individual plots, no legend
        if kind == "both":
            ax1.axvline(
                x=current_price,
                color=spot_price_color,
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,  # Thinner line
                # Legend label aligns with on-chart text
                label=f"Price: ${current_price:,.2f}",
            )
            # Add text annotation beside the line
            # Position text right above the x-axis
            y_min, y_max = ax1.get_ylim()
            y_text_pos = y_min + (y_max - y_min) * 0.15  # 15% from bottom

            ax1.text(
                current_price
                + (prices.max() - prices.min()) * 0.02,  # Slight offset to the right
                y_text_pos,
                price_text,
                color=spot_price_color,
                fontsize=12,  # Bigger font
                fontstyle="italic",  # Italicized
                va="bottom",  # Align to bottom so text sits above the position
                ha="left",
            )
        else:
            # Individual plots: spot price line without legend
            ax1.axvline(
                x=current_price,
                color=spot_price_color,
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,  # Thinner line
            )
            # Add text annotation for individual plots too
            if kind == "pdf":
                y_min, y_max = ax1.get_ylim()
                y_text_pos = y_min + (y_max - y_min) * 0.15  # 15% from bottom
            else:  # CDF
                y_text_pos = 0.15  # 15% on CDF scale

            ax1.text(
                current_price + (prices.max() - prices.min()) * 0.02,
                y_text_pos,
                price_text,
                color=spot_price_color,
                fontsize=12,  # Bigger font
                fontstyle="italic",  # Italicized
                va="bottom",  # Align to bottom so text sits above the position
                ha="left",
            )

    # Apply axis limits if provided; otherwise keep auto limits but clamp left >= 0
    if xlim is not None:
        ax1.set_xlim(xlim)
    else:
        try:
            # Trigger Matplotlib autoscale to compute limits
            ax1.relim()
            ax1.autoscale_view()
            left, right = ax1.get_xlim()
            if left < 0:
                ax1.set_xlim(left=0, right=right)
        except Exception:
            pass
    if ylim is not None:
        ax1.set_ylim(ylim)

    # Format x-axis with thousands separator (e.g., 10,000), always applied
    try:
        ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    except Exception:
        # Fallback silently if StrMethodFormatter is unavailable
        pass

    # Handle title and layout
    if style == "publication":
        # Set up the layout: 15% title, 85% graph (no subtitle anymore)
        plt.subplots_adjust(top=0.85, bottom=0.1 if source else 0.05)

        # Add main title at the top
        fig.suptitle(
            plot_title,
            fontsize=16,
            fontweight="bold",
            y=0.925,  # Center of top 15%
            color="#333333",
        )

        # Add source attribution at bottom
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
        # Default style
        plt.tight_layout()
        fig.suptitle(plot_title, fontsize=14, fontweight="bold")

    return fig


def _apply_publication_style(plt):
    """Apply publication-ready style to matplotlib."""
    # Set the figure background color
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F8F8",  # Light gray background
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.5,
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "axes.labelcolor": "#333333",
            "axes.titlecolor": "#333333",
        }
    )


def _style_publication_axes(ax):
    """Apply publication-specific axis styling."""
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Style the remaining spines
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # Grid styling
    ax.grid(True, alpha=0.3, color="#CCCCCC", linewidth=0.5)
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=10, colors="#666666", length=0)

    # Add subtle padding
    ax.margins(x=0.02, y=0.02)
