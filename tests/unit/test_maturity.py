"""Unit tests for canonical maturity helpers."""

from datetime import date

import pandas as pd
import pytest

from oipd.core.maturity import (
    build_maturity_metadata,
    calculate_calendar_days_to_expiry,
    calculate_time_to_expiry,
    calculate_time_to_expiry_days,
    format_timestamp_for_display,
    normalize_datetime_like,
    resolve_maturity,
)


class TestNormalizeDatetimeLike:
    """Tests for canonical datetime normalization policy."""

    def test_plain_date_maps_to_midnight(self):
        """Plain date inputs should normalize to midnight."""
        normalized = normalize_datetime_like(date(2025, 1, 15))
        assert normalized == pd.Timestamp("2025-01-15 00:00:00")

    def test_timezone_aware_inputs_convert_to_utc_then_naive(self):
        """Timezone-aware inputs should preserve absolute instant in UTC."""
        normalized = normalize_datetime_like("2025-01-15T09:30:00-05:00")
        assert normalized == pd.Timestamp("2025-01-15 14:30:00")


class TestResolveMaturity:
    """Tests for resolved maturity semantics and compatibility fields."""

    def test_intraday_precision_is_preserved(self):
        """Resolved maturity should keep intraday precision in year-fraction."""
        valuation_timestamp = pd.Timestamp("2025-01-15 09:30:00")
        expiry_timestamp = pd.Timestamp("2025-01-15 16:00:00")

        resolved = resolve_maturity(expiry_timestamp, valuation_timestamp)

        expected_years = 6.5 / (24.0 * 365.0)
        assert resolved.time_to_expiry_years == pytest.approx(expected_years)
        assert resolved.time_to_expiry_days == pytest.approx(6.5 / 24.0)
        assert resolved.calendar_days_to_expiry == 0

    def test_negative_maturity_is_signed_by_default(self):
        """Canonical resolver should preserve signed maturities unless floored."""
        valuation_timestamp = pd.Timestamp("2025-01-15 12:00:00")
        expiry_timestamp = pd.Timestamp("2025-01-15 10:00:00")

        resolved = resolve_maturity(expiry_timestamp, valuation_timestamp)

        assert resolved.time_to_expiry_years < 0.0
        assert resolved.calendar_days_to_expiry == 0

    def test_floor_at_zero_option(self):
        """Optional floor_at_zero should clamp negative maturities."""
        valuation_timestamp = pd.Timestamp("2025-01-15 12:00:00")
        expiry_timestamp = pd.Timestamp("2025-01-15 10:00:00")

        resolved = resolve_maturity(
            expiry_timestamp,
            valuation_timestamp,
            floor_at_zero=True,
        )

        assert resolved.time_to_expiry_years == 0.0


class TestCompatibilityHelpers:
    """Tests for maturity reporting helpers."""

    def test_calculate_calendar_days_to_expiry_uses_calendar_days(self):
        """Calendar-day helper should preserve date-bucket semantics."""
        valuation_timestamp = pd.Timestamp("2025-01-15 23:00:00")
        expiry_timestamp = pd.Timestamp("2025-01-16 00:30:00")

        result = calculate_calendar_days_to_expiry(
            expiry_timestamp, valuation_timestamp
        )
        assert result == 1

    def test_build_maturity_metadata_exposes_calendar_days(self):
        """Metadata should expose calendar-day buckets explicitly."""
        resolved = resolve_maturity(
            pd.Timestamp("2025-01-16 00:30:00"),
            pd.Timestamp("2025-01-15 23:00:00"),
        )

        metadata = build_maturity_metadata(resolved)
        assert metadata["expiry"] == resolved.expiry
        assert metadata["time_to_expiry_days"] == pytest.approx(1.5 / 24.0)
        assert metadata["calendar_days_to_expiry"] == 1

    def test_calculate_time_to_expiry_days_reports_continuous_days(self):
        """Continuous day reporting should preserve sub-day precision."""
        valuation_timestamp = pd.Timestamp("2025-01-15 09:30:00")
        expiry_timestamp = pd.Timestamp("2025-01-15 16:00:00")

        result = calculate_time_to_expiry_days(expiry_timestamp, valuation_timestamp)
        assert result == pytest.approx(6.5 / 24.0)

    def test_calculate_time_to_expiry_floors_negative_values(self):
        """Compatibility time-to-expiry helper should floor at zero."""
        valuation_timestamp = pd.Timestamp("2025-01-15 12:00:00")
        expiry_timestamp = pd.Timestamp("2025-01-15 10:00:00")

        result = calculate_time_to_expiry(expiry_timestamp, valuation_timestamp)
        assert result == 0.0

    def test_calculate_time_to_expiry_days_floors_negative_values(self):
        """Continuous day reporting should floor expired maturities at zero."""
        valuation_timestamp = pd.Timestamp("2025-01-15 12:00:00")
        expiry_timestamp = pd.Timestamp("2025-01-15 10:00:00")

        result = calculate_time_to_expiry_days(expiry_timestamp, valuation_timestamp)
        assert result == 0.0


class TestFormatTimestampForDisplay:
    """Tests for user-facing timestamp formatting."""

    def test_midnight_timestamp_formats_as_date_only(self):
        """Midnight timestamps should keep the old date-only display style."""
        label = format_timestamp_for_display(pd.Timestamp("2025-03-21 00:00:00"))
        assert label == "Mar 21, 2025"

    def test_intraday_timestamp_formats_with_time_of_day(self):
        """Intraday timestamps should show the clock time."""
        label = format_timestamp_for_display(pd.Timestamp("2025-03-21 12:45:00"))
        assert label == "Mar 21, 2025 12:45"

    def test_second_precision_is_preserved_in_display(self):
        """Explicit seconds should remain visible in labels."""
        label = format_timestamp_for_display(pd.Timestamp("2025-03-21 12:45:09"))
        assert label == "Mar 21, 2025 12:45:09"
