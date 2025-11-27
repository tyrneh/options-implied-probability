"""Plotting helper for 3D implied volatility surfaces."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, Dict, Any

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError


def _resolve_log_moneyness_domain(
    observations: Sequence[Any],
    *,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> tuple[float, float]:
    """Return a robust log-moneyness interval covering observed quotes."""

    if not observations:
        raise CalculationError("Surface calibration has no observations to plot")

    all_k = np.concatenate(
        [np.asarray(obs.log_moneyness, dtype=float) for obs in observations]
    )
    finite_mask = np.isfinite(all_k)
    if not finite_mask.any():
        raise CalculationError("Unable to determine log-moneyness domain")

    filtered = all_k[finite_mask]
    if filtered.size < 5:
        k_min = float(np.min(filtered))
        k_max = float(np.max(filtered))
    else:
        k_min, k_max = np.nanquantile(
            filtered,
            [lower_quantile, upper_quantile],
        )

    if not np.isfinite(k_min) or not np.isfinite(k_max):
        raise CalculationError("Invalid log-moneyness range for plotting")

    if k_min == k_max:
        padding = max(abs(k_min) * 0.1, 0.1)
        return k_min - padding, k_max + padding

    padding = 0.05 * (k_max - k_min)
    return k_min - padding, k_max + padding


def plot_iv_surface_3d(
    observations: Sequence[Any],
    *,
    total_variance: Callable[[np.ndarray, float], np.ndarray],
    infer_forward: Callable[[float], float],
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str | None = None,
    x_axis: str = "log_moneyness",
    show_observed: bool = False,
    markets_by_maturity: Mapping[float, Any] | None = None,
    observed_iv_by_maturity: Mapping[float, Dict[str, pd.DataFrame]] | None = None,
) -> "plotly.graph_objects.Figure":
    """Render the fitted implied-volatility surface as a 3D Plotly figure."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Plotly is required for 3D surface plotting. Install with: pip install plotly"
        ) from exc

    if not observations:
        raise CalculationError("Surface calibration has no observations to plot")

    maturities = np.array([float(obs.maturity) for obs in observations], dtype=float)
    if maturities.size == 0:
        raise CalculationError("Unable to determine maturity range for plotting")

    t_min = float(np.min(maturities))
    t_max = float(np.max(maturities))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min <= 0 or t_min == t_max:
        padding = max(t_max * 0.02, 1e-6)
        t_min = max(t_min - padding, 1e-6)
        t_max = t_max + padding

    num_time_samples = max(30, min(90, len(observations) * 3))
    num_k_samples = 120

    k_min, k_max = _resolve_log_moneyness_domain(observations)
    k_grid = np.linspace(k_min, k_max, num_k_samples)
    t_grid = np.linspace(t_min, t_max, num_time_samples)

    x_axis_mode = x_axis.lower()
    if x_axis_mode not in {"log_moneyness", "strike"}:
        raise ValueError("x_axis must be 'log_moneyness' or 'strike'")

    X_values = np.zeros((num_time_samples, num_k_samples), dtype=float)
    Z_values = np.zeros_like(X_values)

    for idx, t in enumerate(t_grid):
        total_var = np.asarray(total_variance(k_grid, float(t)), dtype=float)
        iv = np.sqrt(np.maximum(total_var / max(t, 1e-8), 1e-12))
        Z_values[idx, :] = iv

        if x_axis_mode == "log_moneyness":
            X_values[idx, :] = k_grid
        else:
            forward = float(infer_forward(float(t)))
            if forward <= 0 or not np.isfinite(forward):
                raise CalculationError("Positive forward required for strike axis")
            X_values[idx, :] = forward * np.exp(k_grid)

    maturity_days = t_grid * 365.0

    fig = go.Figure()
    fig.add_surface(
        x=X_values,
        y=maturity_days,
        z=Z_values,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Implied Volatility"),
        name="Fitted IV",
    )

    if show_observed and observed_iv_by_maturity:
        scatter_x: list[float] = []
        scatter_y: list[float] = []
        scatter_z: list[float] = []

        tolerance = 5e-3
        for maturity, payload in observed_iv_by_maturity.items():
            if not payload:
                continue

            forward = float(infer_forward(float(maturity)))
            if forward <= 0 or not np.isfinite(forward):
                continue

            combined = []
            for key in ("bid", "ask", "last"):
                df = payload.get(key)
                if df is not None and not df.empty:
                    combined.append(df.loc[:, ["strike", "iv"]])
            if not combined:
                continue
            concat = pd.concat(combined, axis=0, ignore_index=True)
            concat = concat.dropna(subset=["strike", "iv"])
            if concat.empty:
                continue

            strikes = concat["strike"].to_numpy(dtype=float)
            iv_values = concat["iv"].to_numpy(dtype=float)
            finite_mask = np.isfinite(strikes) & np.isfinite(iv_values)
            if not finite_mask.any():
                continue

            strikes = strikes[finite_mask]
            iv_values = iv_values[finite_mask]

            if x_axis_mode == "log_moneyness":
                x_vals = np.log(strikes / forward)
            else:
                x_vals = strikes

            scatter_x.extend(x_vals.tolist())
            scatter_y.extend([float(maturity) * 365.0] * x_vals.size)
            scatter_z.extend(iv_values.tolist())

        if scatter_x and scatter_y and scatter_z:
            fig.add_scatter3d(
                x=scatter_x,
                y=scatter_y,
                z=scatter_z,
                mode="markers",
                marker=dict(size=4, color="crimson", opacity=0.7),
                name="Observed IV",
            )

    width_px = int(figsize[0] * 100)
    height_px = int(figsize[1] * 100)

    fig.update_layout(
        width=width_px,
        height=height_px,
        template="plotly_white",
        title=title or "Implied Volatility Surface",
        margin=dict(l=0, r=0, b=0, t=60),
        scene=dict(
            xaxis=dict(
                title=(
                    "Log Moneyness (ln(K/F))"
                    if x_axis_mode == "log_moneyness"
                    else "Strike"
                ),
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
                title="Implied Volatility",
                backgroundcolor="#F7F7F7",
                gridcolor="#DDDDDD",
                zerolinecolor="#CCCCCC",
            ),
        ),
    )

    return fig


__all__ = ["plot_iv_surface_3d"]
