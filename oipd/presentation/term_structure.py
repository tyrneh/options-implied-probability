"""2D Term Structure plotting for volatility surfaces.

This module provides a `plot_term_structure` function for visualizing
the ATM (At-The-Money) implied volatility term structure across expirations.
"""

from __future__ import annotations


from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from oipd.presentation.publication import (
    _apply_publication_style,
    _style_publication_axes,
)


def plot_term_structure(
    days_to_expiry: np.ndarray,
    atm_ivs: np.ndarray,
    *,
    title: str = "ATM Implied Volatility Term Structure",
    xlabel: str = "Days to Expiry",
    ylabel: str = "Implied Volatility (%)",
    figsize: tuple[float, float] = (10, 6),
    marker: str = "",
    line_color: Optional[str] = None,
    show_grid: bool = True,
) -> Figure:
    """Plot the ATM implied volatility term structure.

    The term structure shows implied volatility across expirations for
    at-the-money (ATM) strikes. This curve typically slopes upward,
    reflecting higher IV for longer-dated options due to greater
    uncertainty over time.

    Args:
        days_to_expiry: Array of days-to-expiry for each pillar.
        atm_ivs: Array of ATM implied volatilities (in percentage, e.g., 30 for 30%).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height) in inches.
        marker: Marker style for data points (e.g., 'o', 's', '^'). Use "" for no markers.
        line_color: Optional color for the line and markers.
        show_grid: If True, display grid lines.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> days = np.array([7, 14, 30, 60, 90])
        >>> ivs = np.array([25.0, 26.5, 28.0, 30.0, 32.5])
        >>> fig = plot_term_structure(days, ivs)
        >>> plt.show()
    """
    # -------------------------------------------------------------------------
    # Style Configuration
    # -------------------------------------------------------------------------
    _apply_publication_style(plt)

    bg_color = "white"
    text_color = "#333333"  # Matches publication style
    grid_color = "#CCCCCC"  # Matches publication style
    default_line_color = "#1f77b4"  # Matplotlib default blue

    color = line_color or default_line_color

    # -------------------------------------------------------------------------
    # Create Figure & Axes
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # -------------------------------------------------------------------------
    # Plot Data
    # -------------------------------------------------------------------------
    # Sort by days to ensure proper line connection
    sort_idx = np.argsort(days_to_expiry)
    x = days_to_expiry[sort_idx]
    y = atm_ivs[sort_idx]

    ax.plot(
        x,
        y,
        marker=marker,
        color=color,
        linewidth=2,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
        label="ATM IV",
    )

    # -------------------------------------------------------------------------
    # Helper: Apply standard publication axis styling
    # -------------------------------------------------------------------------

    _style_publication_axes(ax)

    # -------------------------------------------------------------------------
    # Additional Custom Styling
    # -------------------------------------------------------------------------
    ax.set_xlabel(xlabel, fontsize=12, color=text_color)
    ax.set_ylabel(ylabel, fontsize=12, color=text_color)
    ax.set_title(title, fontsize=14, color=text_color, pad=15)

    # Ensure tick colors match the theme (helper sets to grey, we override if needed)
    ax.tick_params(axis="both", colors=text_color)

    # Grid (helper enables it, we can enforce dashed style)
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.5, color=grid_color)
    else:
        ax.grid(False)

    plt.tight_layout()
    return fig


__all__ = ["plot_term_structure"]
