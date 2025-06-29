# OIPD - Options Implied Probability Distribution
"""Generate probability distributions for future stock prices using options data."""

from oipd.core import (
    calculate_pdf,
    calculate_cdf,
    calculate_quartiles,
    fit_kde,
    OIPDError,
    InvalidInputError,
    CalculationError,
)

from oipd.io import CSVReader, DataFrameReader, CSVReadError

__version__ = "0.0.5"

__all__ = [
    # Core functions
    "calculate_pdf",
    "calculate_cdf",
    "calculate_quartiles",
    "fit_kde",
    # Exceptions
    "OIPDError",
    "InvalidInputError",
    "CalculationError",
    "CSVReadError",
    # Readers
    "CSVReader",
    "DataFrameReader",
]
