# OIPD - Options Implied Probability Distribution
"""Generate probability distributions for future stock prices using options data."""

from oipd.core import (
    calculate_pdf,
    calculate_cdf,
    calculate_quartiles,
    OIPDError,
    InvalidInputError,
    CalculationError,
)

from oipd.io import CSVReader, DataFrameReader, CSVReadError

# Optional: explicitly re-export vendor-specific errors later if needed

from oipd.estimator import RND, ModelParams, RNDResult
from oipd.market_inputs import (
    MarketInputs, VendorSnapshot, ResolvedMarket, 
    FillMode, resolve_market
)

__version__ = "0.1.0"

__all__ = [
    # Core functions
    "calculate_pdf",
    "calculate_cdf",
    "calculate_quartiles",
    # Exceptions
    "OIPDError",
    "InvalidInputError",
    "CalculationError",
    "CSVReadError",
    # Readers
    "CSVReader",
    "DataFrameReader",
    # High-level API
    "RND",
    "MarketInputs",  # Immutable input class
    "ResolvedMarket",  # Resolved parameters with provenance
    "VendorSnapshot",  # Vendor data snapshot
    "FillMode",  # Fill mode literal type
    "ModelParams",
    "RNDResult",
]
