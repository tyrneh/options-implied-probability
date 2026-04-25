"""Unit tests for market input timestamp normalization contracts."""

from datetime import date

import pandas as pd
import pytest

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

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("dividend_yield", 0.02),
            (
                "dividend_schedule",
                pd.DataFrame(
                    {"ex_date": [pd.Timestamp("2025-01-20")], "amount": [1.0]}
                ),
            ),
        ],
    )
    def test_market_inputs_reject_public_dividend_arguments(self, keyword, value):
        """Explicit dividend inputs should not be accepted on the public path."""
        kwargs = {
            "valuation_date": pd.Timestamp("2025-01-15 09:30:00"),
            "underlying_price": 100.0,
            "risk_free_rate": 0.05,
            keyword: value,
        }

        with pytest.raises(TypeError, match=keyword):
            MarketInputs(**kwargs)

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
        assert resolved.dividend_yield is None
        assert resolved.dividend_schedule is None
        assert resolved.provenance.dividends == "none"
