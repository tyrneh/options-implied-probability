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

from oipd.interface import VolCurve, VolSurface, ProbCurve, ProbSurface

# Optional: explicitly re-export vendor-specific errors later if needed


from oipd.core.vol_surface_fitting.shared.vol_model import VolModel
from oipd.pipelines._internal.reconstruct import (
    RebuiltSlice,
    RebuiltSurface,
    rebuild_slice_from_svi,
    rebuild_surface_from_ssvi,
)
from oipd.market_inputs import (
    MarketInputs,
    VendorSnapshot,
    ResolvedMarket,
    resolve_market,
)

__version__ = "2.0.4"

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
    "ProbCurve",
    "ProbSurface",
    "MarketInputs",  # Immutable public input class
    "ResolvedMarket",  # Lower-level resolved parameters with provenance
    "VendorSnapshot",  # Vendor data snapshot
    "resolve_market",  # Lower-level market resolver retained for compatibility
    "VolModel",
    "RebuiltSlice",
    "RebuiltSurface",
    "rebuild_slice_from_svi",
    "rebuild_surface_from_ssvi",
]
