"""2D plotting utilities for risk-neutral density summaries."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from oipd.core.errors import InvalidInputError


def _quantile_from_cdf(strikes: np.ndarray, cdf: np.ndarray, q: float) -> float:
    """Interpolate the strike level corresponding to a CDF quantile.

    Args:
        strikes: Strike grid sorted in ascending order.
        cdf: Monotonic cumulative probabilities aligned with ``strikes``.
        q: Target quantile in ``[0, 1]``.

    Returns:
        float: Strike level associated with the requested quantile.
    """

    finite_mask = np.isfinite(strikes) & np.isfinite(cdf)
    if not finite_mask.any():
        raise InvalidInputError("CDF contains no finite values for quantile extraction")

    strikes_clean = np.asarray(strikes[finite_mask], dtype=float)
    cdf_clean = np.asarray(cdf[finite_mask], dtype=float)

    order = np.argsort(strikes_clean)
    strikes_sorted = strikes_clean[order]
    cdf_sorted = cdf_clean[order]
    cdf_sorted = np.maximum.accumulate(cdf_sorted)

    cdf_min = float(cdf_sorted[0])
    cdf_max = float(cdf_sorted[-1])
    if cdf_max - cdf_min < 1e-6:
        return float(strikes_sorted[-1])

    q_clamped = float(np.clip(q, cdf_min, cdf_max))
    return float(np.interp(q_clamped, cdf_sorted, strikes_sorted))


def plot_probability_summary(
    density_data: pd.DataFrame,
    *,
    lower_percentile: float,
    upper_percentile: float,
    figsize: tuple[float, float],
    title: Optional[str],
) -> "matplotlib.figure.Figure":
    """Plot distribution quantiles over time.

    Args:
        density_data: DataFrame from ``RNDSurface.density_surface(..., as_dataframe=True)``.
        lower_percentile: Lower bound percentile for the shaded confidence band.
        upper_percentile: Upper bound percentile for the shaded confidence band.
        figsize: Matplotlib figure size in inches ``(width, height)``.
        title: Optional chart title.

    Returns:
        matplotlib.figure.Figure: Figure containing the summary plot.

    Raises:
        ImportError: If Matplotlib is unavailable.
        InvalidInputError: For malformed density inputs or percentile configuration.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Matplotlib is required for surface plotting. Install with: pip install matplotlib"
        ) from exc

    for pct in (lower_percentile, upper_percentile):
        if not (0.0 <= pct <= 100.0):
            raise InvalidInputError("Percentile bounds must lie within [0, 100]")
    if not lower_percentile < upper_percentile:
        raise InvalidInputError(
            "lower_percentile must be strictly less than upper_percentile"
        )

    required_cols = {"expiry_date", "strike", "cdf"}
    missing = required_cols.difference(density_data.columns)
    if missing:
        raise InvalidInputError(
            "Density DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )

    grouped = density_data.groupby("expiry_date", sort=True)

    dates: list[pd.Timestamp] = []
    lower: list[float] = []
    upper: list[float] = []
    median: list[float] = []

    lower_q = lower_percentile / 100.0
    upper_q = upper_percentile / 100.0

    for expiry, frame in grouped:
        expiry_ts = pd.to_datetime(expiry)
        strikes = frame["strike"].to_numpy(dtype=float)
        cdf_values = np.clip(frame["cdf"].to_numpy(dtype=float), 0.0, 1.0)

        try:
            q_low = _quantile_from_cdf(strikes, cdf_values, lower_q)
            q_high = _quantile_from_cdf(strikes, cdf_values, upper_q)
            q_med = _quantile_from_cdf(strikes, cdf_values, 0.5)
        except InvalidInputError:
            continue

        dates.append(expiry_ts)
        lower.append(q_low)
        upper.append(q_high)
        median.append(q_med)

    if not dates:
        raise InvalidInputError("No valid probability slices available for plotting")

    order = np.argsort(dates)
    dates_sorted = [dates[i].to_pydatetime() for i in order]
    lower_sorted = np.asarray(lower)[order]
    upper_sorted = np.asarray(upper)[order]
    median_sorted = np.asarray(median)[order]

    fig, ax = plt.subplots(figsize=figsize)

    band_label = f"{lower_percentile:.0f}–{upper_percentile:.0f} percentile band"
    ax.fill_between(
        dates_sorted,
        lower_sorted,
        upper_sorted,
        color="#1f77b4",
        alpha=0.15,
        label=band_label,
    )

    ax.plot(
        dates_sorted,
        median_sorted,
        color="#1f77b4",
        linewidth=2.0,
        label="Implied median",
    )

    ax.set_xlabel("Expiry date")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.margins(x=0.02)

    if title is None:
        title = "Risk-neutral price distribution over time"
    ax.set_title(title)

    legend = ax.legend(loc="best")
    if legend is not None:
        for text in legend.get_texts():
            text.set_color("black")
    fig.autofmt_xdate()

    return fig


__all__ = ["plot_probability_summary"]
