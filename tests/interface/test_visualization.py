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
from matplotlib.figure import Figure


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

    def test_plot_fan_with_quantiles(self, prob_surface):
        """plot_fan() should accept custom percentile bounds."""
        fig = prob_surface.plot_fan(lower_percentile=10.0, upper_percentile=90.0)
        assert isinstance(fig, Figure)
        plt.close(fig)
