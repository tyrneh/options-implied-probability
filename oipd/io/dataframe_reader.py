from pandas import DataFrame

from oipd.io.reader import AbstractReader


class DataFrameReader(AbstractReader):
    """Reader implementation for DataFrame containing options data."""

    def _ingest_data(self, df: DataFrame) -> DataFrame:
        """Ingest raw data from a DataFrame

        Arguments:
            df: a DataFrame containing the raw options data
        """
        # Return a copy to avoid modifying the original DataFrame
        return df.copy()

    # Inherits _clean_data() from AbstractReader.

    def _transform_data(self, cleaned_data: DataFrame) -> DataFrame:
        """Default transformation implementation. Returns the cleaned DataFrame."""
        return cleaned_data
