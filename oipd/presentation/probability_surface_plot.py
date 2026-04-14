"""2D plotting utilities for risk-neutral fan summaries."""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from oipd.core.errors import InvalidInputError


def plot_probability_summary(
    summary_data: pd.DataFrame,
    *,
    figsize: tuple[float, float],
    title: Optional[str],
) -> Figure:
    """Plot a precomputed probability fan summary over time.

    Args:
        summary_data: DataFrame containing ``expiry``, ``is_pillar``,
            ``p10``, ``p20``, ``p30``, ``p40``, ``p50``, ``p60``, ``p70``,
            ``p80``, and ``p90``.
        figsize: Matplotlib figure size in inches ``(width, height)``.
        title: Optional chart title.

    Returns:
        Figure: Figure containing the summary plot.

    Raises:
        ImportError: If Matplotlib is unavailable.
        InvalidInputError: For malformed summary inputs.
    """
    required_cols = {
        "expiry",
        "is_pillar",
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
    }
    missing = required_cols.difference(summary_data.columns)
    if missing:
        raise InvalidInputError(
            "Summary DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if summary_data.empty:
        raise InvalidInputError(
            "No valid probability summary rows available for plotting"
        )

    ordered_summary = summary_data.copy()
    ordered_summary["expiry"] = pd.to_datetime(ordered_summary["expiry"])
    ordered_summary = ordered_summary.sort_values("expiry").reset_index(drop=True)
    dates_sorted = [expiry.to_pydatetime() for expiry in ordered_summary["expiry"]]

    fig, ax = plt.subplots(figsize=figsize)

    band_specs = [
        ("p10", "p90", "#dbe6f6", 1, "10-90p"),
        ("p20", "p80", "#b8d0f0", 2, "20-80p"),
        ("p30", "p70", "#82addf", 3, "30-70p"),
        ("p40", "p60", "#4f86c9", 4, "40-60p"),
    ]
    for lower_column, upper_column, color, zorder, label in band_specs:
        ax.fill_between(
            dates_sorted,
            ordered_summary[lower_column].to_numpy(dtype=float),
            ordered_summary[upper_column].to_numpy(dtype=float),
            color=color,
            alpha=0.8,
            label=label,
            zorder=zorder,
        )

    ax.plot(
        dates_sorted,
        ordered_summary["p50"].to_numpy(dtype=float),
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        label="Implied median",
        zorder=5,
    )

    pillar_rows = ordered_summary.loc[ordered_summary["is_pillar"].astype(bool)]
    ax.scatter(
        [expiry.to_pydatetime() for expiry in pillar_rows["expiry"]],
        pillar_rows["p50"].to_numpy(dtype=float),
        s=42,
        facecolor="white",
        edgecolor="#1f77b4",
        linewidth=1.25,
        label="Option expiry dates",
        zorder=6,
    )

    ax.set_xlabel("Expiry")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.margins(x=0.02)

    if title is None:
        title = "Risk-neutral price distribution over time"
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    legend_order = [
        "Option expiry dates",
        "Implied median",
        "40-60p",
        "30-70p",
        "20-80p",
        "10-90p",
    ]
    label_to_handle = dict(zip(labels, handles, strict=True))
    ordered_handles = [
        label_to_handle[label] for label in legend_order if label in label_to_handle
    ]
    ordered_labels = [
        label for label in legend_order if label in label_to_handle
    ]
    legend = ax.legend(ordered_handles, ordered_labels, loc="best")
    if legend is not None:
        for text in legend.get_texts():
            if text.get_text().endswith("p"):
                text.set_color(ax.xaxis.label.get_color())
                text.set_fontsize("small")
            else:
                text.set_color("black")

    has_intraday_precision = any(
        expiry_ts.time() != expiry_ts.normalize().time()
        for expiry_ts in ordered_summary["expiry"]
    )
    if has_intraday_precision:
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate()

    return fig


__all__ = ["plot_probability_summary"]
