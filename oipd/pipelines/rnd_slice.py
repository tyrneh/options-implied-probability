"""Single-expiry risk-neutral density pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional, Sequence, Mapping, Any
import warnings

import numpy as np
import pandas as pd

from oipd.core.vol_surface_fitting.shared.vol_model import VolModel
from oipd.data_access.vendors import get_reader
from oipd.pipelines.estimator import (
    ModelParams,
    RNDResult,
    CSVSource,
    DataFrameSource,
    TickerSource,
    DataSource,
    _estimate,
    _resolve_slice_vol_model,
)
from oipd.pipelines.market_inputs import (
    MarketInputs,
    VendorSnapshot,
    ResolvedMarket,
    resolve_market,
    FillMode,
)


@contextmanager
def _suppress_oipd_warnings(suppress: bool):
    """Optionally silence UserWarnings emitted from the ``oipd`` namespace."""

    if not suppress:
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"oipd(\.|$)",
        )
        yield


def _fetch_vendor_snapshot(
    ticker: str,
    expiry_str: str,
    *,
    vendor: str,
    cache_enabled: bool,
    cache_ttl_minutes: int,
) -> tuple[pd.DataFrame, VendorSnapshot]:
    column_mapping = {
        "lastPrice": "last_price",
        "lastTradeDate": "last_trade_date",
    }

    source = TickerSource(
        ticker=ticker,
        expiry=expiry_str,
        vendor=vendor,
        column_mapping=column_mapping,
        cache_enabled=cache_enabled,
        cache_ttl_minutes=cache_ttl_minutes,
    )

    chain = source.load()

    snapshot = VendorSnapshot(
        asof=datetime.now(),
        vendor=vendor,
        underlying_price=source.underlying_price,
        dividend_yield=source.dividend_yield,
        dividend_schedule=source.dividend_schedule,
    )
    return chain, snapshot


def _run_estimation(
    options_data: pd.DataFrame,
    resolved_market: ResolvedMarket,
    *,
    model: ModelParams | None,
    vol: VolModel | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    base_model = model or ModelParams()
    effective_model, resolved_vol, requested_method = _resolve_slice_vol_model(
        base_model, vol
    )
    prices, pdf, cdf, meta = _estimate(options_data, resolved_market, effective_model)
    meta.setdefault("vol_model", resolved_vol)
    meta.setdefault("vol_model_method", requested_method)
    return prices, pdf, cdf, meta


def from_dataframe(
    df: pd.DataFrame,
    market: MarketInputs,
    *,
    model: ModelParams | None = None,
    vol: VolModel | None = None,
    column_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> RNDResult:
    """Estimate an RND from an in-memory DataFrame."""

    source = DataFrameSource(df, column_mapping=column_mapping)
    with _suppress_oipd_warnings(suppress=not verbose):
        chain = source.load()
        resolved = resolve_market(market, vendor=None, mode="strict")
        prices, pdf, cdf, meta = _run_estimation(chain, resolved, model=model, vol=vol)
    return RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=resolved, meta=meta)


def from_csv(
    path: str,
    market: MarketInputs,
    *,
    model: ModelParams | None = None,
    vol: VolModel | None = None,
    column_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> RNDResult:
    """Estimate an RND from a CSV file on disk."""

    source = CSVSource(path, column_mapping=column_mapping)
    with _suppress_oipd_warnings(suppress=not verbose):
        chain = source.load()
        resolved = resolve_market(market, vendor=None, mode="strict")
        prices, pdf, cdf, meta = _run_estimation(chain, resolved, model=model, vol=vol)
    return RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=resolved, meta=meta)


def list_expiry_dates(ticker: str, vendor: str = "yfinance") -> list[str]:
    """Return available option expiries for the given ticker and vendor."""

    reader_cls = get_reader(vendor)
    if not hasattr(reader_cls, "list_expiry_dates"):
        raise NotImplementedError(
            f"Vendor '{vendor}' does not expose an expiry listing API."
        )
    return reader_cls.list_expiry_dates(ticker)


def from_ticker(
    ticker: str,
    market: MarketInputs,
    *,
    model: ModelParams | None = None,
    vol: VolModel | None = None,
    vendor: str = "yfinance",
    fill: FillMode = "missing",
    echo: Optional[bool] = None,
    verbose: bool = True,
    cache_enabled: bool = True,
    cache_ttl_minutes: int = 15,
) -> RNDResult:
    """Fetch an option chain from a vendor and estimate the RND."""

    if market.expiry_date is None:
        raise ValueError("MarketInputs.expiry_date is required for vendor fetches")

    expiry = market.expiry_date.strftime("%Y-%m-%d")

    with _suppress_oipd_warnings(suppress=not verbose):
        chain, snapshot = _fetch_vendor_snapshot(
            ticker,
            expiry,
            vendor=vendor,
            cache_enabled=cache_enabled,
            cache_ttl_minutes=cache_ttl_minutes,
        )

        resolved = resolve_market(market, snapshot, mode=fill)

        base_model = (
            model
            if model is not None
            else (
                ModelParams(price_method="last")
                if vendor == "yfinance"
                else ModelParams()
            )
        )

        prices, pdf, cdf, meta = _run_estimation(
            chain, resolved, model=base_model, vol=vol
        )

    meta.update(
        {
            "ticker": ticker,
            "vendor": snapshot.vendor,
            "asof": snapshot.asof.isoformat(),
        }
    )
    result = RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=resolved, meta=meta)

    if echo if echo is not None else verbose:
        print(result.summary())

    return result


class RND:
    """Convenience façade resembling the legacy class-based API.

    .. deprecated:: 0.1.0
       Use ``VolCurve`` / ``VolSurface`` and ``Distribution`` / ``DistributionSurface``
       from ``oipd.interface``. This class remains for transition only.
    """

    def __init__(
        self,
        model: ModelParams | None = None,
        vol: VolModel | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        warnings.warn(
            "RND is deprecated and will be removed in a future version. "
            "Use oipd.VolCurve and oipd.Distribution instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model = model or ModelParams()
        self.vol = vol
        self.verbose = verbose
        self._result: RNDResult | None = None

    def fit(self, source: DataSource, resolved_market: ResolvedMarket) -> "RND":
        with _suppress_oipd_warnings(suppress=not self.verbose):
            chain = source.load()
            prices, pdf, cdf, meta = _run_estimation(
                chain, resolved_market, model=self.model, vol=self.vol
            )
        self._result = RNDResult(
            prices=prices,
            pdf=pdf,
            cdf=cdf,
            market=resolved_market,
            meta=meta,
        )
        return self

    @property
    def result(self) -> RNDResult:
        if self._result is None:
            raise ValueError("Call `fit` before accessing the result")
        return self._result

    # Convenience exporters -------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        return self.result.to_frame()

    def to_csv(self, path: str, **kwargs) -> None:
        self.result.to_csv(path, **kwargs)

    def prob_at_or_above(self, price: float) -> float:
        return self.result.prob_at_or_above(price)

    def prob_below(self, price: float) -> float:
        return self.result.prob_below(price)

    def plot(self, **kwargs):
        return self.result.plot(**kwargs)

    def svi_params(self) -> Dict[str, float]:
        """Return calibrated SVI parameters for the fitted slice.

        Returns:
            dict[str, float]: Dictionary of raw SVI parameters ``(a, b, rho, m, sigma)``.

        Raises:
            ValueError: If SVI calibration was not used for the smile fit.
        """

        return self.result.svi_params()

    def iv_smile(
        self,
        strikes: Sequence[float] | np.ndarray | None = None,
        *,
        num_points: int = 200,
    ) -> pd.DataFrame:
        return self.result.iv_smile(strikes, num_points=num_points)

    # Class-style constructors ----------------------------------------------

    @classmethod
    def from_csv(cls, *args, **kwargs) -> RNDResult:
        return from_csv(*args, **kwargs)

    @classmethod
    def from_dataframe(cls, *args, **kwargs) -> RNDResult:
        return from_dataframe(*args, **kwargs)

    @classmethod
    def from_ticker(cls, *args, **kwargs) -> RNDResult:
        return from_ticker(*args, **kwargs)

    @classmethod
    def list_expiry_dates(cls, ticker: str, vendor: str = "yfinance") -> list[str]:
        return list_expiry_dates(ticker, vendor)


__all__ = [
    "RND",
    "from_csv",
    "from_dataframe",
    "from_ticker",
    "list_expiry_dates",
]
