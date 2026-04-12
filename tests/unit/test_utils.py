"""Unit tests for oipd.core.utils module."""

import math
import pytest

from oipd.core.utils import resolve_risk_free_rate


class TestResolveRiskFreeRate:
    """Test suite for resolve_risk_free_rate utility."""

    def test_continuous_mode_returns_input(self):
        """Continuous mode should return the input rate unchanged.

        Args:
            None.

        Returns:
            None.
        """
        rate = resolve_risk_free_rate(0.05, "continuous", 0.5)
        assert rate == 0.05

    def test_annualized_mode_converts(self):
        """Annualized mode should convert to continuous compounding.

        Args:
            None.

        Returns:
            None.
        """
        t = 0.75
        expected = math.log1p(0.05 * t) / t
        rate = resolve_risk_free_rate(0.05, "annualized", t)
        assert rate == expected

    def test_invalid_mode_raises(self):
        """Unknown modes should raise ValueError.

        Args:
            None.

        Returns:
            None.
        """
        with pytest.raises(ValueError):
            resolve_risk_free_rate(0.05, "simple", 1.0)
