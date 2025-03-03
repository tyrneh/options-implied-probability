import pandas as pd
from pandas import DataFrame
from oipd.io.reader import AbstractReader


class CSVReader(AbstractReader):
    """Reader implementation for pre-formatted CSV files containing options data."""

    def _ingest_data(self, filepath: str) -> DataFrame:
        """Read CSV file from the given filepath."""
        return pd.read_csv(filepath)

    # Inherits _clean_data() from AbstractReader.

    def _transform_data(self, cleaned_data: DataFrame) -> DataFrame:
        """Default transformation implementation. Returns the cleaned DataFrame."""
        return cleaned_data
