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

    # Optional fields second
    days_to_expiry: Optional[int] = None
    underlying_price: Optional[float] = None
    dividend_yield: Optional[float] = None
    dividend_schedule: Optional[pd.DataFrame] = None  # columns: ex_date, amount
    expiry_date: Optional[date] = None

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
    days_to_expiry: int
    underlying_price: float
    valuation_date: date  # Valuation date for dividend calculations
    expiry_date: date  # Option expiry date
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
            f"days_to_expiry={self.days_to_expiry}; r={self.risk_free_rate}"
        )

    # Read-only aliases ----------------------------------------------------------
    @property
    def current_price(self) -> float:  # pragma: no cover - trivial alias
        return self.underlying_price


FillMode = Literal["missing", "vendor_only", "strict"]


def resolve_market(
    inputs: MarketInputs,
    vendor: Optional[VendorSnapshot],
    *,
    mode: FillMode = "missing",
) -> ResolvedMarket:
    """
    Resolve market parameters by merging user inputs with vendor data.

    Parameters
    ----------
    inputs : MarketInputs
        User-provided market parameters
    vendor : Optional[VendorSnapshot]
        Vendor-fetched market data (if available)
    mode : FillMode
        - "strict": Require all fields from user (no vendor filling)
        - "vendor_only": Use only vendor data, ignore user inputs
        - "missing": Use user values when available, fill missing from vendor

    Returns
    -------
    ResolvedMarket
        Immutable, fully-resolved market parameters with provenance tracking

    Raises
    ------
    ValueError
        If required parameters cannot be resolved
    """
    # 1) Resolve time specification with new logic

    valuation_date = inputs.valuation_date
    if isinstance(valuation_date, datetime):
        valuation_date = valuation_date.date()

    # Default valuation_date to today if not provided
    if valuation_date is None:
        valuation_date = date.today()

    # Resolve time horizon using priority: dates > days_to_expiry
    if inputs.expiry_date is not None:
        expiry = inputs.expiry_date
        if isinstance(expiry, datetime):
            expiry = expiry.date()

        days_to_expiry = (expiry - valuation_date).days
        if days_to_expiry <= 0:
            raise ValueError(
                "expiry_date must be in the future relative to valuation_date"
            )

        # Warn if both days_to_expiry and dates provided
        if inputs.days_to_expiry is not None:
            warnings.warn(
                f"Both days_to_expiry ({inputs.days_to_expiry}) and expiry_date provided. "
                f"Using dates (calculated days_to_expiry={days_to_expiry}), ignoring days_to_expiry parameter.",
                UserWarning,
            )
    elif inputs.days_to_expiry is not None:
        days_to_expiry = inputs.days_to_expiry
        # Calculate expiry_date from valuation_date + days_to_expiry
        expiry = valuation_date + timedelta(days=days_to_expiry)
    else:
        raise ValueError(
            "Must provide either days_to_expiry or expiry_date (or both). "
            "valuation_date defaults to today if not provided."
        )

    # Helper function to pick value based on mode
    def pick(field_user, field_vendor, name: str):
        if mode == "strict":
            if field_user is None:
                raise ValueError(f"{name} required in strict mode")
            return field_user, "user"

        if mode == "vendor_only":
            if vendor and field_vendor is not None:
                return field_vendor, "vendor"
            return None, "none"

        # mode == "missing" (default)
        if field_user is not None:
            return field_user, "user"
        if vendor and field_vendor is not None:
            return field_vendor, "vendor"
        return None, "none"

    # 2) Resolve current/underlying price
    price, price_src = pick(
        inputs.underlying_price,
        vendor.underlying_price if vendor else None,
        "underlying_price",
    )
    if price is None or price <= 0:
        raise ValueError("No valid underlying_price available (user or vendor)")

    # 3) Resolve dividends with precedence:
    # user schedule > user yield > vendor schedule > vendor yield > none
    if inputs.dividend_schedule is not None and mode != "vendor_only":
        sched, yld, div_src = inputs.dividend_schedule, None, "user_schedule"
    elif inputs.dividend_yield is not None and mode != "vendor_only":
        sched, yld, div_src = None, inputs.dividend_yield, "user_yield"
    elif vendor:
        if vendor.dividend_schedule is not None:
            sched, yld, div_src = (
                vendor.dividend_schedule,
                None,
                "vendor_schedule",
            )
        elif vendor.dividend_yield is not None:
            sched, yld, div_src = None, vendor.dividend_yield, "vendor_yield"
        else:
            sched, yld, div_src = None, None, "none"
    else:
        sched, yld, div_src = None, None, "none"

    # Build provenance
    prov = Provenance(price=price_src, dividends=div_src)

    # Build metadata
    meta = {
        "vendor": vendor.vendor if vendor else None,
        "asof": vendor.asof.isoformat() if vendor else None,
        "fill_mode": mode,
    }

    # 4) Resolve and normalize risk-free rate to continuous compounding
    #    The rest of the codebase expects a continuous rate used in exp(-r * T)
    y = float(inputs.risk_free_rate)
    if inputs.risk_free_rate_mode == "annualized":
        T = days_to_expiry / 365.0
        # Simple ACT/365 money-market style annualization → convert to continuous
        # DF_simple = 1 / (1 + y * T)  ⇒  r_cont = -ln(DF_simple) / T = ln(1 + y*T)/T
        r_cont = float(np.log1p(y * T) / T)
    else:
        r_cont = y

    return ResolvedMarket(
        risk_free_rate=r_cont,
        days_to_expiry=days_to_expiry,
        underlying_price=price,
        valuation_date=valuation_date,
        expiry_date=expiry,
        dividend_yield=yld,
        dividend_schedule=sched,
        provenance=prov,
        source_meta={
            **meta,
            "risk_free_rate_mode": inputs.risk_free_rate_mode,
            "risk_free_rate_input": inputs.risk_free_rate,
            "risk_free_rate_continuous": r_cont,
        },
    )
