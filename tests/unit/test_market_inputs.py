"""Unit tests for market input timestamp normalization contracts."""

from datetime import date

import pandas as pd

from oipd.market_inputs import MarketInputs, resolve_market


class TestMarketInputsObjectModel:
    """Tests for the timestamp-preserving market input contract."""

    def test_market_inputs_keep_intraday_precision_on_valuation_date(self):
        """MarketInputs should store the normalized timestamp on valuation_date."""
        valuation_timestamp = pd.Timestamp("2025-01-15 09:30:00")

        inputs = MarketInputs(
            valuation_date=valuation_timestamp,
            underlying_price=100.0,
            risk_free_rate=0.05,
        )

        assert inputs.valuation_date == valuation_timestamp
        assert inputs.valuation_calendar_date == date(2025, 1, 15)
        assert inputs.valuation_timestamp == valuation_timestamp
        assert inputs.valuation_timestamp == inputs.valuation_date

    def test_resolved_market_keeps_intraday_precision_on_valuation_date(self):
        """ResolvedMarket should preserve the same canonical timestamp contract."""
        valuation_timestamp = pd.Timestamp("2025-01-15 09:30:00")

        resolved = resolve_market(
            MarketInputs(
                valuation_date=valuation_timestamp,
                underlying_price=100.0,
                risk_free_rate=0.05,
            )
        )

        assert resolved.valuation_date == valuation_timestamp
        assert resolved.valuation_calendar_date == date(2025, 1, 15)
        assert resolved.valuation_timestamp == valuation_timestamp
        assert resolved.valuation_timestamp == resolved.valuation_date
