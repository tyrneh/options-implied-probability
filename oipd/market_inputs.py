from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

from oipd.core.maturity import DateTimeLike, normalize_datetime_like


@dataclass(frozen=True)
class MarketInputs:
    """
    User-provided market parameters for the public fitting workflow.

    Terminology
    ----------
    `underlying_price` is the current tradable price of the instrument the
    option is written on. In the public workflow it is the reference price used
    for diagnostics while option-implied forwards are inferred from parity.
    """

    # Required fields first
    risk_free_rate: float
    valuation_date: DateTimeLike
    # How the provided risk-free rate is quoted
    # 'annualized' means simple annualized nominal rate on ACT/365 for the horizon
    # 'continuous' means continuously compounded rate
    risk_free_rate_mode: Literal["annualized", "continuous"] = "annualized"
    underlying_price: Optional[float] = None

    def __post_init__(self) -> None:
        """Normalize and persist valuation timestamp with intraday precision.

        Args:
            None.
        """
        valuation_timestamp = normalize_datetime_like(self.valuation_date)
        object.__setattr__(self, "valuation_date", valuation_timestamp)

    @property
    def valuation_timestamp(self) -> pd.Timestamp:
        """Return canonical valuation timestamp used internally.

        Returns:
            pd.Timestamp: Canonical timestamp stored on ``valuation_date``.
        """
        return self.valuation_date

    @property
    def valuation_calendar_date(self) -> date:
        """Return the date-only valuation view for calendar arithmetic.

        Returns:
            date: Calendar date associated with ``valuation_date``.
        """
        return self.valuation_date.date()

    # Convenience alias used in some UIs
    @property
    def current_price(
        self,
    ) -> Optional[float]:  # pragma: no cover - trivial alias
        return self.underlying_price


@dataclass(frozen=True)
class VendorSnapshot:
    """Market data fetched from a vendor at a specific point in time."""

    asof: datetime
    vendor: str
    underlying_price: Optional[float] = None
    dividend_yield: Optional[float] = None
    dividend_schedule: Optional[pd.DataFrame] = None

    # Read-only aliases to reduce terminology confusion
    @property
    def current_price(
        self,
    ) -> Optional[float]:  # pragma: no cover - trivial alias
        return self.underlying_price

    @property
    def date(self) -> date:
        """Return the calendar date of the snapshot timestamp.

        Returns:
            date: Calendar date associated with the snapshot ``asof`` timestamp.
        """
        return self.asof.date()


@dataclass(frozen=True)
class Provenance:
    """Tracks the source of each resolved market parameter."""

    price: Literal["user", "vendor", "none"]
    dividends: Literal[
        "user_schedule",
        "user_yield",
        "vendor_schedule",
        "vendor_yield",
        "none",
    ]


@dataclass(frozen=True)
class ResolvedMarket:
    """
    Immutable, fully-resolved market parameters used in calculations.

    The attribute `underlying_price` holds the current price of the instrument
    the option is written on. For futures options (Black‑76), this is the
    current futures price F for the selected contract (not a cash spot).

    A convenience alias `current_price` is provided for readability.
    """

    risk_free_rate: float
    underlying_price: float
    valuation_date: DateTimeLike
    dividend_yield: Optional[float]
    dividend_schedule: Optional[pd.DataFrame]
    provenance: Provenance
    source_meta: Dict[str, Any]  # e.g. {"vendor":"yfinance","asof": "..."} etc.

    def __post_init__(self) -> None:
        """Normalize and persist valuation timestamp with intraday precision.

        Args:
            None.
        """
        valuation_timestamp = normalize_datetime_like(self.valuation_date)
        object.__setattr__(self, "valuation_date", valuation_timestamp)

    def summary(self) -> str:
        """Return a one-line summary of resolved parameters and their sources."""
        div_desc = self.provenance.dividends.replace("_", " ")
        return (
            f"Used underlying {self.underlying_price:.4f} "
            f"(source: {self.provenance.price}); "
            f"dividends: {div_desc}; "
            f"r={self.risk_free_rate}"
        )

    # Read-only aliases ----------------------------------------------------------
    @property
    def current_price(self) -> float:  # pragma: no cover - trivial alias
        return self.underlying_price

    @property
    def valuation_timestamp(self) -> pd.Timestamp:
        """Return canonical valuation timestamp used in calculations.

        Returns:
            pd.Timestamp: Canonical timestamp stored on ``valuation_date``.
        """
        return self.valuation_date

    @property
    def valuation_calendar_date(self) -> date:
        """Return the date-only valuation view for calendar arithmetic.

        Returns:
            date: Calendar date associated with ``valuation_date``.
        """
        return self.valuation_date.date()


def _normalize_dividend_schedule(dividend_schedule: pd.DataFrame) -> pd.DataFrame:
    """Normalize a user-supplied dividend schedule to canonical types.

    Args:
        dividend_schedule: DataFrame expected to contain ``ex_date`` and
            ``amount`` columns.

    Returns:
        pd.DataFrame: Copy of ``dividend_schedule`` with canonical timestamped
        ``ex_date`` values and numeric ``amount`` values.

    Raises:
        TypeError: If ``dividend_schedule`` is not a DataFrame.
        ValueError: If required columns are missing or values cannot be parsed.
    """
    if not isinstance(dividend_schedule, pd.DataFrame):
        raise TypeError("dividend_schedule must be a pandas DataFrame.")

    required_columns = {"ex_date", "amount"}
    if not required_columns.issubset(dividend_schedule.columns):
        raise ValueError("dividend_schedule must contain 'ex_date' and 'amount'.")

    normalized_schedule = dividend_schedule.copy()
    try:
        normalized_schedule["ex_date"] = normalized_schedule["ex_date"].map(
            normalize_datetime_like
        )
    except Exception as exc:
        raise ValueError("Could not parse dividend_schedule ex_date values.") from exc

    try:
        normalized_schedule["amount"] = pd.to_numeric(
            normalized_schedule["amount"],
            errors="raise",
        ).astype(float)
    except Exception as exc:
        raise ValueError("Could not parse dividend_schedule amount values.") from exc

    if not np.isfinite(normalized_schedule["amount"]).all():
        raise ValueError("dividend_schedule amount values must be finite.")

    return normalized_schedule


def resolve_market(
    inputs: MarketInputs,
) -> ResolvedMarket:
    """
    Resolve market parameters from explicit user inputs.

    Parameters
    ----------
    inputs : MarketInputs
        User-provided market parameters. Must contain underlying_price,
        valuation_date, and risk_free_rate.

    Returns
    -------
    ResolvedMarket
        Immutable, fully-resolved market parameters with provenance tracking.

    Raises
    ------
    ValueError
        If required parameters (like underlying_price) are missing.
    """
    # 1) Resolve valuation timestamp
    valuation_timestamp = inputs.valuation_timestamp

    # 2) Resolve underlying price
    if inputs.underlying_price is None:
        raise ValueError(
            "underlying_price must be provided in MarketInputs. "
            "Please explicitly populate this field."
        )
    price = inputs.underlying_price
    price_source = "user"

    # 3) Public market inputs do not carry explicit dividends.
    prov = Provenance(price=price_source, dividends="none")

    # 4) Return resolved market
    return ResolvedMarket(
        risk_free_rate=float(inputs.risk_free_rate),
        underlying_price=price,
        valuation_date=valuation_timestamp,
        dividend_yield=None,
        dividend_schedule=None,
        provenance=prov,
        source_meta={
            "risk_free_rate_mode": inputs.risk_free_rate_mode,
            "risk_free_rate_input": inputs.risk_free_rate,
        },
    )


def _resolve_market_with_dividends(
    inputs: MarketInputs,
    *,
    dividend_yield: Optional[float] = None,
    dividend_schedule: Optional[pd.DataFrame] = None,
) -> ResolvedMarket:
    """Resolve market state with legacy/internal explicit dividends.

    This is intentionally not part of the public constructor path. It preserves
    dividend-bearing ``ResolvedMarket`` coverage for retained Black-Scholes and
    dividend utility tests while ``MarketInputs`` stays small for users.
    """
    valuation_timestamp = inputs.valuation_timestamp

    if inputs.underlying_price is None:
        raise ValueError(
            "underlying_price must be provided in MarketInputs. "
            "Please explicitly populate this field."
        )
    price = inputs.underlying_price
    price_source = "user"

    # Preserve the historical schedule-priority behavior for internal callers.
    yld: Optional[float] = None
    sched: Optional[pd.DataFrame] = None
    div_source: Literal[
        "user_schedule", "user_yield", "vendor_schedule", "vendor_yield", "none"
    ] = "none"

    if dividend_schedule is not None:
        sched = _normalize_dividend_schedule(dividend_schedule)
        div_source = "user_schedule"
    elif dividend_yield is not None:
        yld = dividend_yield
        div_source = "user_yield"

    prov = Provenance(price=price_source, dividends=div_source)

    return ResolvedMarket(
        risk_free_rate=float(inputs.risk_free_rate),
        underlying_price=price,
        valuation_date=valuation_timestamp,
        dividend_yield=yld,
        dividend_schedule=sched,
        provenance=prov,
        source_meta={
            "risk_free_rate_mode": inputs.risk_free_rate_mode,
            "risk_free_rate_input": inputs.risk_free_rate,
        },
    )
