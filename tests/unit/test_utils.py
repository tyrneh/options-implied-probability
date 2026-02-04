"""Unit tests for oipd.core.utils module."""

import pytest
from datetime import date
import pandas as pd

from oipd.core.utils import calculate_days_to_expiry


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
        """Timezone-aware Timestamp should be handled correctly."""
        ts = pd.Timestamp("2025-03-21", tz="America/New_York")
        result = calculate_days_to_expiry(ts, date(2025, 1, 27))
        assert result == 53
