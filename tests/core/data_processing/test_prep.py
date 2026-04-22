from datetime import date, timedelta
import warnings

import numpy as np
import pandas as pd

from oipd.core.data_processing.selection import (
    filter_stale_options,
    select_price_column,
)


def test_filter_stale_options_drops_entire_strike_when_any_leg_is_stale():
    valuation = date(2024, 1, 5)
    data = pd.DataFrame(
        [
            {
                "strike": 100.0,
                "option_type": "C",
                "last_trade_date": pd.Timestamp(valuation - timedelta(days=10)),
            },
            {
                "strike": 100.0,
                "option_type": "P",
                "last_trade_date": pd.Timestamp(valuation),
            },
            {
                "strike": 105.0,
                "option_type": "C",
                "last_trade_date": pd.Timestamp(valuation),
            },
            {
                "strike": 105.0,
                "option_type": "P",
                "last_trade_date": pd.Timestamp(valuation),
            },
        ]
    )

    result = filter_stale_options(
        data,
        valuation_date=valuation,
        max_staleness_days=3,
        emit_warning=False,
    )
    filtered = result[0]

    assert set(filtered["strike"].unique()) == {105.0}
    assert filtered.shape[0] == 2


def test_select_price_column_counts_only_successful_partial_bid_ask_fallbacks():
    """Partial bid/ask fallback count excludes invalid last_price rows."""
    data = pd.DataFrame(
        {
            "bid": [1.0, np.nan, 3.0, np.nan],
            "ask": [1.2, 2.2, np.nan, 4.2],
            "last_price": [1.1, 2.1, np.nan, 0.0],
        }
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        priced, filled_count = select_price_column(data, "mid", emit_warning=True)

    assert filled_count == 1
    assert len(recorded_warnings) == 1
    assert "Filled 1 missing mid prices" in str(recorded_warnings[0].message)
    assert priced["price"].tolist() == [1.1, 2.1]


def test_select_price_column_counts_only_successful_all_bid_ask_unavailable_fallbacks():
    """All-unavailable bid/ask fallback count excludes invalid last_price rows."""
    data = pd.DataFrame(
        {
            "bid": [np.nan, np.nan, np.nan, np.nan],
            "ask": [np.nan, np.nan, np.nan, np.nan],
            "last_price": [1.1, np.nan, -0.5, 2.1],
        }
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        priced, filled_count = select_price_column(data, "mid", emit_warning=True)

    assert filled_count == 2
    assert len(recorded_warnings) == 1
    assert "Filled 2 missing mid prices" in str(recorded_warnings[0].message)
    assert priced["price"].tolist() == [1.1, 2.1]


def test_select_price_column_precomputed_mid_invalid_last_price_counts_zero():
    """Precomputed mid fallback count is zero when last_price cannot survive."""
    data = pd.DataFrame(
        {
            "mid": [1.0, np.nan, 3.0],
            "last_price": [1.1, 0.0, 3.1],
        }
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        priced, filled_count = select_price_column(data, "mid", emit_warning=True)

    assert filled_count == 0
    assert recorded_warnings == []
    assert priced["price"].tolist() == [1.0, 3.0]


def test_select_price_column_last_method_does_not_report_fallback_fills():
    """price_method='last' reports zero fills even with missing invalid last prices."""
    data = pd.DataFrame({"last_price": [1.1, np.nan, 0.0, 2.1]})

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        priced, filled_count = select_price_column(data, "last", emit_warning=True)

    assert filled_count == 0
    assert recorded_warnings == []
    assert priced["price"].tolist() == [1.1, 2.1]
