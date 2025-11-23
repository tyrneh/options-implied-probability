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
from oipd.data_access import sources

from oipd.interface import VolCurve, VolSurface, Distribution, DistributionSurface

# Optional: explicitly re-export vendor-specific errors later if needed

from oipd.pipelines.estimator import ModelParams, RNDResult
from oipd.core.vol_surface_fitting.shared.vol_model import VolModel
from oipd.pipelines.reconstruct import (
    RebuiltSlice,
    RebuiltSurface,
    rebuild_slice_from_svi,
    rebuild_surface_from_ssvi,
)
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
    "sources",
    # High-level API
    "VolCurve",
    "VolSurface",
    "Distribution",
    "DistributionSurface",
    "MarketInputs",  # Immutable input class
    "ResolvedMarket",  # Resolved parameters with provenance
    "VendorSnapshot",  # Vendor data snapshot
    "FillMode",  # Fill mode literal type
    "ModelParams",
    "RNDResult",
    "VolModel",
    "RebuiltSlice",
    "RebuiltSurface",
    "rebuild_slice_from_svi",
    "rebuild_surface_from_ssvi",
]
