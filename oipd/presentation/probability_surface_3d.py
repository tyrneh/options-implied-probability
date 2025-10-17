"""3D plotting helper for risk-neutral probability surfaces."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from oipd.core.errors import InvalidInputError


def _coerce_density_frame(density_data: pd.DataFrame | object) -> pd.DataFrame:
    """Validate and normalise density surface data to a DataFrame.

    Args:
        density_data: Output of ``RNDSurface.density_surface``. Must be a
            pandas DataFrame containing one row per (maturity, moneyness) pair.

    Returns:
        pandas.DataFrame: Copy of the provided data with required columns.

    Raises:
        InvalidInputError: If the input is not a DataFrame or lacks columns.
    """

    if not isinstance(density_data, pd.DataFrame):
        raise InvalidInputError(
            "density_surface must be called with as_dataframe=True to use plot_probability_3d"
        )

    required = {"maturity", "moneyness", "strike", "pdf", "cdf", "forward"}
    missing = required.difference(density_data.columns)
    if missing:
        raise InvalidInputError(
            "Density DataFrame is missing required columns: " + ", ".join(sorted(missing))
        )

    return density_data.copy()


def plot_probability_3d(
    density_data: pd.DataFrame,
    *,
    value: Literal["pdf", "cdf"] = "pdf",
    figsize: tuple[float, float] = (10.0, 6.0),
    title: Optional[str] = None,
    colorscale: str = "Viridis",
) -> "plotly.graph_objects.Figure":
    """Render a 3D risk-neutral density surface using Plotly.

    Args:
        density_data: Output of :meth:`RNDSurface.density_surface` with
            ``as_dataframe=True``. Each row represents one ``(maturity, moneyness)``
            pair and includes both PDF and CDF evaluations.
        value: Selects whether to visualise the probability density function
            (``"pdf"``) or cumulative distribution function (``"cdf"``). The X
            axis is always rendered in strike space.
        figsize: Tuple defining figure size in inches ``(width, height)``.
        title: Optional chart title. When ``None`` a default is used.
        colorscale: Plotly colorscale name used for the surface shading.

    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure representing the
        requested probability surface.

    Raises:
        InvalidInputError: If the input cannot be converted into a dense grid.
        ImportError: If Plotly is not installed.
        ValueError: If ``value`` is invalid.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Plotly is required for probability surface plotting. Install with: pip install plotly"
        ) from exc

    frame = _coerce_density_frame(density_data)

    value = value.lower()
    if value not in {"pdf", "cdf"}:
        raise ValueError("value must be 'pdf' or 'cdf'")

    pivot = frame.pivot_table(
        index="maturity",
        columns="moneyness",
        values=value,
        aggfunc="mean",
    )

    pivot = pivot.sort_index().sort_index(axis=1)

    if pivot.isna().any().any():
        raise InvalidInputError(
            "Density grid contains missing values; ensure density_surface produced a full grid."
        )

    t_grid = pivot.index.to_numpy(dtype=float)
    m_grid = pivot.columns.to_numpy(dtype=float)
    if t_grid.size == 0 or m_grid.size == 0:
        raise InvalidInputError("Density grid is empty")

    Z = pivot.to_numpy(dtype=float)

    forward_map = frame.groupby("maturity")["forward"].mean()
    forward = forward_map.reindex(pivot.index)
    if forward.isna().any():
        raise InvalidInputError("Forward levels missing for one or more maturities")

    forward_values = forward.to_numpy(dtype=float)
    X = forward_values[:, None] * m_grid[None, :]
    Y = np.tile((t_grid * 365.0)[:, None], (1, m_grid.size))

    width_px = int(figsize[0] * 100)
    height_px = int(figsize[1] * 100)
    colorbar_title = "Probability Density" if value == "pdf" else "Cumulative Probability"

    fig = go.Figure()
    fig.add_surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=colorbar_title),
        name=value.upper(),
    )

    x_title = "Strike"
    z_title = "PDF" if value == "pdf" else "CDF"

    fig.update_layout(
        width=width_px,
        height=height_px,
        template="plotly_white",
        title=title or "Risk-Neutral Probability Surface",
        margin=dict(l=0, r=0, b=0, t=60),
        scene=dict(
            xaxis=dict(
                title=x_title,
                backgroundcolor="#F7F7F7",
                gridcolor="#DDDDDD",
                zerolinecolor="#CCCCCC",
            ),
            yaxis=dict(
                title="Maturity (days)",
                backgroundcolor="#F7F7F7",
                gridcolor="#DDDDDD",
                zerolinecolor="#CCCCCC",
            ),
            zaxis=dict(
                title=z_title,
                backgroundcolor="#F7F7F7",
                gridcolor="#DDDDDD",
                zerolinecolor="#CCCCCC",
            ),
        ),
    )

    return fig


__all__ = ["plot_probability_3d"]
