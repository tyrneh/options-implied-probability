"""Generic 3D surface plotting using Matplotlib.

This module provides a reusable `plot_surface_3d` function for visualizing
any 2D scalar field Z(X, Y) as a 3D surface plot. It is designed to be
called by domain-specific wrappers like `VolSurface.plot_3d()` and
`ProbSurface.plot_3d()`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - Required for 3D projection


def plot_surface_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    title: str = "Surface",
    cmap: str = "coolwarm",
    view_angle: tuple[float, float] = (30, -45),
    figsize: tuple[float, float] = (14, 10),
    show_projections: bool = True,
    dark_mode: bool = True,
    zlim: Optional[tuple[float, float]] = None,
    colorbar_label: Optional[str] = None,
    rstride: int = 2,
    cstride: int = 2,
    projection_type: Literal["ortho", "persp"] = "ortho",
    term_structure_days: Optional[np.ndarray] = None,
    term_structure_ivs: Optional[np.ndarray] = None,
    term_structure_wall: Literal["x_min", "x_max"] = "x_min",
    term_structure_offset_ratio: float = 0.05,
    term_structure_color: Optional[str] = None,
    term_structure_linewidth: float = 1.6,
    term_structure_label: Optional[str] = None,
) -> Figure:
    """Render a 3D surface plot with optional wall projections.

    Args:
        X: 2D meshgrid of x-coordinates (e.g., strikes).
        Y: 2D meshgrid of y-coordinates (e.g., time to expiry in days).
        Z: 2D array of z-values (e.g., implied volatility).
        xlabel: Label for the X-axis.
        ylabel: Label for the Y-axis.
        zlabel: Label for the Z-axis.
        title: Plot title.
        cmap: Matplotlib colormap name (e.g., 'coolwarm', 'viridis', 'plasma').
        view_angle: Tuple of (elevation, azimuth) for camera angle.
        figsize: Figure size in inches (width, height).
        show_projections: If True, draw "shadow" projections on walls.
        dark_mode: If True, use dark background (like the reference image).
        zlim: Optional explicit (min, max) limits for the Z-axis.
        colorbar_label: Optional label for the colorbar.
        rstride: Row stride for surface mesh (lower = finer grid).
        cstride: Column stride for surface mesh.
        projection_type: 'ortho' for orthographic (isometric-like) or 'persp' for perspective.
        term_structure_days: Optional term-structure x-values (days to expiry).
        term_structure_ivs: Optional term-structure y-values (implied vol, same units as Z).
        term_structure_wall: Which wall to place the term structure on ("x_min" or "x_max").
        term_structure_offset_ratio: Fraction of X-range to offset the wall line for visibility.
        term_structure_color: Optional line color for the term-structure projection.
            If None, the line is colored by the surface colormap using term-structure IV.
        term_structure_linewidth: Line width for the term-structure projection.
        term_structure_label: Optional legend label for the term-structure projection.


    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # -------------------------------------------------------------------------
    # Style Configuration
    # -------------------------------------------------------------------------
    if dark_mode:
        plt.style.use("dark_background")
        bg_color = "#0d0d0d"
        grid_color = "#333333"
        text_color = "#cccccc"
        projection_color_skew = "#00BFFF"  # Cyan for skew (back wall)
        projection_color_term = "#FF8C00"  # Orange for term structure (left wall)
    else:
        plt.style.use("default")
        bg_color = "white"
        grid_color = "#dddddd"
        text_color = "black"
        projection_color_skew = "blue"
        projection_color_term = "green"

    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax: Axes3D = fig.add_subplot(111, projection="3d", facecolor=bg_color)  # type: ignore
    ax.set_proj_type(projection_type)

    # -------------------------------------------------------------------------
    # Z-Axis Limits & Normalization
    # -------------------------------------------------------------------------
    z_finite = Z[np.isfinite(Z)]
    if zlim is None:
        z_min = float(np.nanmin(z_finite)) if z_finite.size > 0 else 0.0
        z_max = float(np.nanmax(z_finite)) if z_finite.size > 0 else 1.0
        # Add 5% padding
        z_range = z_max - z_min
        zlim = (z_min - 0.05 * z_range, z_max + 0.05 * z_range)

    # Clip Z for color normalization (outliers don't distort color scale)
    Z_clipped = np.clip(Z, zlim[0], zlim[1])

    # -------------------------------------------------------------------------
    # Main Surface
    # -------------------------------------------------------------------------
    norm = plt.Normalize(vmin=zlim[0], vmax=zlim[1])
    cmap_obj = cm.get_cmap(cmap)
    surf = ax.plot_surface(
        X,
        Y,
        Z_clipped,
        cmap=cmap_obj,
        norm=norm,
        rstride=rstride,
        cstride=cstride,
        edgecolor="none",
        alpha=0.95,
        antialiased=True,
    )

    # -------------------------------------------------------------------------
    # Wall Projections (Shadows) - Simple default projections
    # -------------------------------------------------------------------------
    if show_projections:
        x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
        y_min, y_max = float(np.nanmin(Y)), float(np.nanmax(Y))

        # Left/Right Wall: Show term structure projection if provided
        if term_structure_days is not None and term_structure_ivs is not None:
            term_days = np.asarray(term_structure_days, dtype=float)
            term_ivs = np.asarray(term_structure_ivs, dtype=float)
            valid = np.isfinite(term_days) & np.isfinite(term_ivs)
            term_days = term_days[valid]
            term_ivs = term_ivs[valid]

            if term_days.size > 1:
                sort_idx = np.argsort(term_days)
                term_days = term_days[sort_idx]
                term_ivs = term_ivs[sort_idx]
                x_span = x_max - x_min
                offset = max(term_structure_offset_ratio, 0.0) * x_span
                wall_x = x_min - offset if term_structure_wall == "x_min" else x_max + offset
                wall_xs = np.full_like(term_days, wall_x)
                if term_structure_color is None:
                    # Color the term structure by the same IV colormap as the surface.
                    seg_colors = cmap_obj(norm(term_ivs[:-1]))
                    segments = [
                        (
                            (wall_xs[i], term_days[i], term_ivs[i]),
                            (wall_xs[i + 1], term_days[i + 1], term_ivs[i + 1]),
                        )
                        for i in range(term_days.size - 1)
                    ]
                    line_collection = Line3DCollection(
                        segments,
                        colors=seg_colors,
                        linewidths=term_structure_linewidth,
                        alpha=0.95,
                    )
                    line_collection.set_clip_on(False)
                    ax.add_collection3d(line_collection)
                else:
                    line = ax.plot(
                        wall_xs,
                        term_days,
                        term_ivs,
                        color=term_structure_color,
                        linewidth=term_structure_linewidth,
                        alpha=0.9,
                        label=term_structure_label or "ATM Term Structure",
                    )[0]
                    line.set_clip_on(False)
        else:
            # Fallback: use lowest strike slice as term structure
            term_y = Y[:, 0]
            term_z = Z_clipped[:, 0]
            ax.plot(
                np.full_like(term_y, x_min),
                term_y,
                term_z,
                color=projection_color_term,
                linewidth=1.5,
                alpha=0.8,
                label="Term Structure (Low Strike)",
            )

        # Back Wall (Y = Y_max): Show the longest-dated smile (Skew)
        skew_x = X[-1, :]
        skew_z = Z_clipped[-1, :]
        ax.plot(
            skew_x,
            np.full_like(skew_x, y_max),
            skew_z,
            color=projection_color_skew,
            linewidth=1.5,
            alpha=0.8,
            label="Skew (Longest Expiry)",
        )

    # -------------------------------------------------------------------------
    # Axes & Labels
    # -------------------------------------------------------------------------
    ax.set_xlabel(xlabel, fontsize=11, color=text_color, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, color=text_color, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=11, color=text_color, labelpad=10)
    ax.set_title(title, fontsize=14, color=text_color, pad=20)

    ax.set_zlim(zlim)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Reverse Y-axis so 0 days is at the back (furthest), max days toward viewer
    ax.invert_yaxis()

    # Grid styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(grid_color)
    ax.yaxis.pane.set_edgecolor(grid_color)
    ax.zaxis.pane.set_edgecolor(grid_color)
    ax.xaxis._axinfo["grid"]["color"] = grid_color
    ax.yaxis._axinfo["grid"]["color"] = grid_color
    ax.zaxis._axinfo["grid"]["color"] = grid_color

    # Tick label color
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)
    ax.tick_params(axis="z", colors=text_color)

    # -------------------------------------------------------------------------
    # Colorbar
    # -------------------------------------------------------------------------
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.ax.tick_params(colors=text_color)
    if colorbar_label:
        cbar.set_label(colorbar_label, color=text_color, fontsize=10)

    plt.tight_layout()
    return fig


__all__ = ["plot_surface_3d"]
