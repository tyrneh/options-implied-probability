from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta


@dataclass(frozen=True)
class MarketInputs:
    """
    User-provided market parameters (may have missing fields for vendor mode).

    Terminology
    ----------
    `underlying_price` is the current tradable price of the instrument the
    option is written on:
    - For equity/ETF/FX spot options (Black–Scholes), this is the cash spot S.
    - For options on futures (Black‑76), this is the current futures price F
      of the relevant contract.
    """

    # Required fields first
    risk_free_rate: float
    valuation_date: date
    # How the provided risk-free rate is quoted
    # 'annualized' means simple annualized nominal rate on ACT/365 for the horizon
    # 'continuous' means continuously compounded rate
    risk_free_rate_mode: Literal["annualized", "continuous"] = "annualized"

    def __post_init__(self):
        # Allow string input for convenience, normalize to date
        if isinstance(self.valuation_date, str):
            # Bypass frozen dataclass constraint using object.__setattr__
            object.__setattr__(
                self, "valuation_date", date.fromisoformat(self.valuation_date)
            )
        elif isinstance(self.valuation_date, datetime):
            object.__setattr__(self, "valuation_date", self.valuation_date.date())


    # Optional fields second
    underlying_price: Optional[float] = None
    dividend_yield: Optional[float] = None
    dividend_schedule: Optional[pd.DataFrame] = None  # columns: ex_date, amount

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
    valuation_date: date  # Valuation date for dividend calculations
    dividend_yield: Optional[float]
    dividend_schedule: Optional[pd.DataFrame]
    provenance: Provenance
    source_meta: Dict[str, Any]  # e.g. {"vendor":"yfinance","asof": "..."} etc.

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
    # 1) Resolve valuation_date
    valuation_date = inputs.valuation_date
    if isinstance(valuation_date, datetime):
        valuation_date = valuation_date.date()
    if valuation_date is None:
        raise ValueError("Valuation date must be provided explicitly.")

    # 2) Resolve underlying price
    if inputs.underlying_price is None:
        raise ValueError(
            "underlying_price must be provided in MarketInputs. "
            "Please explicitly populate this field."
        )
    price = inputs.underlying_price
    price_source = "user"

    # 3) Resolve dividends (yield takes priority over schedule)
    yld: Optional[float] = None
    sched: Optional[pd.DataFrame] = None
    div_source: Literal[
        "user_schedule", "user_yield", "vendor_schedule", "vendor_yield", "none"
    ] = "none"

    if inputs.dividend_schedule is not None:
        sched = inputs.dividend_schedule
        div_source = "user_schedule"
    elif inputs.dividend_yield is not None:
        yld = inputs.dividend_yield
        div_source = "user_yield"

    prov = Provenance(price=price_source, dividends=div_source)

    # 4) Return resolved market
    return ResolvedMarket(
        risk_free_rate=float(inputs.risk_free_rate),
        underlying_price=price,
        valuation_date=valuation_date,
        dividend_yield=yld,
        dividend_schedule=sched,
        provenance=prov,
        source_meta={
            "risk_free_rate_mode": inputs.risk_free_rate_mode,
            "risk_free_rate_input": inputs.risk_free_rate,
        },
    )

