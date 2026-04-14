"""
Visualization Smoke Tests
=========================
Ensures all plotting methods execute without exception and return Figure objects.

These tests do NOT verify visual correctness (that requires manual review).
They protect against runtime errors in plot generation.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure


def _build_subday_fitted_surface():
    """Fit a surface whose first expiry is sub-day from the valuation time."""
    from oipd import MarketInputs, VolSurface

    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    valuation_timestamp = pd.Timestamp("2025-01-01 09:30:00")
    same_day_expiry = pd.Timestamp("2025-01-01 16:00:00")
    later_expiry = pd.Timestamp("2025-01-05 16:00:00")

    calls_same_day = pd.DataFrame(
        {
            "expiry": [same_day_expiry] * len(strikes),
            "strike": strikes,
            "bid": [20.5, 15.8, 11.0, 6.5, 2.8, 0.9, 0.3, 0.1, 0.05],
            "ask": [21.0, 16.2, 11.5, 7.0, 3.2, 1.2, 0.5, 0.2, 0.1],
            "last_price": [20.75, 16.0, 11.25, 6.75, 3.0, 1.05, 0.4, 0.15, 0.08],
            "option_type": ["call"] * len(strikes),
        }
    )
    calls_later = calls_same_day.copy()
    calls_later["expiry"] = later_expiry
    calls_later["bid"] = np.asarray(calls_same_day["bid"], dtype=float) * 1.2
    calls_later["ask"] = np.asarray(calls_same_day["ask"], dtype=float) * 1.2
    calls_later["last_price"] = (
        np.asarray(calls_same_day["last_price"], dtype=float) * 1.2
    )

    def _puts_from_calls(calls: pd.DataFrame) -> pd.DataFrame:
        time_to_expiry = (
            pd.to_datetime(calls["expiry"]) - valuation_timestamp
        ).dt.total_seconds().to_numpy(dtype=float) / (365.0 * 24.0 * 60.0 * 60.0)
        discount_factor = np.exp(-0.05 * time_to_expiry)
        puts = calls.copy()
        puts["option_type"] = "put"
        puts["last_price"] = (
            calls["last_price"] - 100.0 + calls["strike"] * discount_factor
        ).abs()
        puts["bid"] = (calls["bid"] - 100.0 + calls["strike"] * discount_factor).abs()
        puts["ask"] = (calls["ask"] - 100.0 + calls["strike"] * discount_factor).abs()
        return puts

    chain = pd.concat(
        [
            calls_same_day,
            _puts_from_calls(calls_same_day),
            calls_later,
            _puts_from_calls(calls_later),
        ],
        ignore_index=True,
    )

    market = MarketInputs(
        valuation_date=valuation_timestamp,
        underlying_price=100.0,
        risk_free_rate=0.05,
    )
    return VolSurface().fit(chain, market)


# =============================================================================
# VolCurve Visualization Tests
# =============================================================================


class TestVolCurveVisualization:
    """Smoke tests for VolCurve plotting methods."""

    def test_plot_returns_figure(self, fitted_vol_curve):
        """plot() should return a matplotlib Figure."""
        fig = fitted_vol_curve.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_with_options(self, fitted_vol_curve):
        """plot() should accept optional arguments without error."""
        fig = fitted_vol_curve.plot(
            include_observed=True,
            figsize=(12, 6),
            title="Custom Title",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


# =============================================================================
# VolSurface Visualization Tests
# =============================================================================


class TestVolSurfaceVisualization:
    """Smoke tests for VolSurface plotting methods."""

    def test_plot_returns_figure(self, fitted_vol_surface):
        """plot() should return a matplotlib Figure."""
        fig = fitted_vol_surface.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_3d_returns_figure(self, fitted_vol_surface):
        """plot_3d() should return a matplotlib Figure."""
        fig = fitted_vol_surface.plot_3d()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_3d_with_options(self, fitted_vol_surface):
        """plot_3d() should accept optional arguments without error."""
        fig = fitted_vol_surface.plot_3d(
            figsize=(12, 8),
            view_angle=(30, -45),
            cmap="viridis",
            projection_type="persp",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_term_structure_returns_figure(self, fitted_vol_surface):
        """plot_term_structure() should return a matplotlib Figure."""
        fig = fitted_vol_surface.plot_term_structure()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_term_structure_with_options(self, fitted_vol_surface):
        """plot_term_structure() should accept optional arguments without error."""
        fig = fitted_vol_surface.plot_term_structure(
            at_money="forward",
            figsize=(12, 6),
            title="Custom Term Structure",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_term_structure_preserves_subday_maturity(self):
        """Term-structure plots should not floor a sub-day expiry to 1 day."""
        fitted_surface = _build_subday_fitted_surface()

        fig = fitted_surface.plot_term_structure()
        ax = fig.axes[0]
        x_data = ax.lines[0].get_xdata()

        assert float(np.min(x_data)) == pytest.approx(6.5 / 24.0)
        plt.close(fig)

    def test_plot_surface_day_labels_preserve_subday_precision(self):
        """Surface legends should show continuous day labels for sub-day expiries."""
        fitted_surface = _build_subday_fitted_surface()

        fig = fitted_surface.plot(label_format="days")
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_text = [text.get_text() for text in legend.get_texts()]

        assert "0.27d" in legend_text
        plt.close(fig)

    def test_plot_3d_preserves_subday_expiry_axis_range(self):
        """3D surface expiry axis should include sub-day maturities below 1 day."""
        fitted_surface = _build_subday_fitted_surface()

        fig = fitted_surface.plot_3d()
        ax = fig.axes[0]
        y_limits = tuple(float(value) for value in ax.get_ylim())

        assert min(y_limits) < 1.0
        plt.close(fig)


# =============================================================================
# ProbCurve Visualization Tests
# =============================================================================


class TestProbCurveVisualization:
    """Smoke tests for ProbCurve plotting methods."""

    def test_plot_returns_figure(self, prob_curve):
        """plot() should return a matplotlib Figure."""
        fig = prob_curve.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_pdf(self, prob_curve):
        """plot(kind='pdf') should work."""
        fig = prob_curve.plot(kind="pdf")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_cdf(self, prob_curve):
        """plot(kind='cdf') should work."""
        fig = prob_curve.plot(kind="cdf")
        assert isinstance(fig, Figure)
        plt.close(fig)


# =============================================================================
# ProbSurface Visualization Tests
# =============================================================================


class TestProbSurfaceVisualization:
    """Smoke tests for ProbSurface plotting methods."""

    def test_plot_fan_returns_figure(self, prob_surface):
        """plot_fan() should return a matplotlib Figure."""
        fig = prob_surface.plot_fan()
        assert isinstance(fig, Figure)
        plt.close(fig)

    @pytest.mark.parametrize(
        ("removed_kwarg", "value"),
        [("lower_percentile", 10.0), ("upper_percentile", 90.0)],
    )
    def test_plot_fan_rejects_removed_percentile_kwargs(
        self, prob_surface, removed_kwarg, value
    ):
        """plot_fan() should reject each removed percentile-bound kwarg."""
        with pytest.raises(TypeError, match=removed_kwarg):
            prob_surface.plot_fan(**{removed_kwarg: value})
