"""Unit tests for oipd.core.utils module."""

import math
import pytest
from datetime import date
import pandas as pd

from oipd.core.utils import calculate_days_to_expiry, resolve_risk_free_rate


class TestCalculateDaysToExpiry:
    """Test suite for calculate_days_to_expiry utility."""

    def test_basic_date(self):
        """Standard date input."""
        result = calculate_days_to_expiry(date(2025, 3, 21), date(2025, 1, 27))
        assert result == 53

    def test_timestamp(self):
        """pandas.Timestamp input."""
        result = calculate_days_to_expiry(pd.Timestamp("2025-03-21"), date(2025, 1, 27))
        assert result == 53

    def test_string(self):
        """ISO string input."""
        result = calculate_days_to_expiry("2025-03-21", date(2025, 1, 27))
        assert result == 53

    def test_expired(self):
        """Expiry in the past should return 0."""
        result = calculate_days_to_expiry(date(2025, 1, 1), date(2025, 1, 27))
        assert result == 0

    def test_same_day(self):
        """Same day should return 0."""
        result = calculate_days_to_expiry(date(2025, 1, 27), date(2025, 1, 27))
        assert result == 0

    def test_timezone_aware_timestamp(self):
        """Timezone-aware Timestamp should be handled correctly.

        Args:
            None.

        Returns:
            None.
        """
        ts = pd.Timestamp("2025-03-21", tz="America/New_York")
        result = calculate_days_to_expiry(ts, date(2025, 1, 27))
        assert result == 53


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
