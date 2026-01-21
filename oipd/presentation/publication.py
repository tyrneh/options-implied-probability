"""Styling primitives for publication-grade Matplotlib output."""

from __future__ import annotations

from typing import Any


def _apply_publication_style(plt: Any) -> None:
    """Apply publication-ready style settings to Matplotlib.

    Args:
        plt: Matplotlib pyplot module whose global RC params are updated.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F8F8",
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
            "axes.prop_cycle": plt.cycler(
                color=[
                    "#1f77b4",  # blue
                    "#ff7f0e",  # orange
                    "#2ca02c",  # green
                    "#d62728",  # red
                    "#9467bd",  # purple
                    "#8c564b",  # brown
                    "#e377c2",  # pink
                    "#7f7f7f",  # gray
                    "#17becf",  # cyan (skipping olive/yellow)
                ]
            ),
        }
    )


def _style_publication_axes(ax: Any) -> None:
    """Apply publication styling to a Matplotlib Axes object.

    Args:
        ax: Matplotlib axes instance to style.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.grid(True, alpha=0.3, color="#CCCCCC", linewidth=0.5)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="major", labelsize=10, colors="#666666", length=0)

    ax.margins(x=0.02, y=0.02)
