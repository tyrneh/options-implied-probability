from __future__ import annotations

import pickle
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from oipd.data_access.vendors.yfinance import reader as yfinance_reader


def _option_frame(option_type: str) -> pd.DataFrame:
    """Build a minimal yfinance-style option chain side for tests.

    Args:
        option_type: Option type marker used only to vary prices.

    Returns:
        DataFrame with enough strikes for the yfinance reader.
    """

    offset = 0.0 if option_type == "call" else 0.5
    return pd.DataFrame(
        {
            "strike": [95.0, 100.0, 105.0, 110.0, 115.0],
            "lastPrice": [6.0 + offset, 4.0 + offset, 2.5 + offset, 1.2, 0.6],
        }
    )


def test_yfinance_history_fallback_requests_raw_close(monkeypatch) -> None:
    """Underlying fallback history must explicitly request unadjusted prices."""

    history_calls: list[dict[str, object]] = []

    class FakeTicker:
        info: dict[str, object] = {}

        def history(self, **kwargs: object) -> pd.DataFrame:
            history_calls.append(kwargs)
            return pd.DataFrame({"Close": [500.0]})

        def option_chain(self, expiry: str) -> SimpleNamespace:
            return SimpleNamespace(
                calls=_option_frame("call"),
                puts=_option_frame("put"),
            )

    monkeypatch.setattr(
        yfinance_reader,
        "yf",
        SimpleNamespace(Ticker=lambda ticker: FakeTicker()),
    )

    result = yfinance_reader.Reader(cache_enabled=False)._ingest_data("SPY:2025-04-17")

    assert history_calls == [
        {"period": "1d", "auto_adjust": False, "back_adjust": False, "actions": False}
    ]
    assert result.attrs["underlying_price"] == 500.0
    assert result.attrs["underlying_price_adjustment"] == "raw"


def test_yfinance_cache_rejects_non_raw_underlying_price(tmp_path) -> None:
    """Cached adjusted prices should not be reused for option parity inputs."""

    cache = yfinance_reader._YFinanceCache(cache_dir=str(tmp_path))
    cache_path = cache._path("SPY", "2025-04-17")
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "timestamp": datetime.now(),
                "options_data": _option_frame("call"),
                "underlying_price": 495.0,
                "underlying_price_adjustment": "all",
            },
            handle,
        )

    assert cache.get("SPY", "2025-04-17") is None
