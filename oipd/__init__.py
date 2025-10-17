# OIPD - Options Implied Probability Distribution
"""Generate probability distributions for future stock prices using options data."""

from oipd.core import (
    calculate_quartiles,
    OIPDError,
    InvalidInputError,
    CalculationError,
)

from oipd.data_access.readers import CSVReader, DataFrameReader
from oipd.data_access.readers.csv_reader import CSVReadError

# Optional: explicitly re-export vendor-specific errors later if needed

from oipd.pipelines.estimator import ModelParams, RNDResult
from oipd.pipelines.rnd_slice import RND
from oipd.core.vol_surface_fitting.shared.vol_model import VolModel
from oipd.pipelines.rnd_surface import RNDSurface
from oipd.pipelines.market_inputs import (
    MarketInputs,
    VendorSnapshot,
    ResolvedMarket,
    FillMode,
    resolve_market,
)

__version__ = "0.1.0"

__all__ = [
    # Core functions
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
    "VolModel",
    "RNDSurface",
]
