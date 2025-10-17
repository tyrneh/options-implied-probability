from datetime import date, timedelta

import pandas as pd

from oipd.core.data_processing.selection import filter_stale_options


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

    filtered = filter_stale_options(
        data,
        valuation_date=valuation,
        max_staleness_days=3,
        emit_warning=False,
    )

    assert set(filtered["strike"].unique()) == {105.0}
    assert filtered.shape[0] == 2
