from pandas import DataFrame

from oipd.io.reader import AbstractReader


class DataFrameReader(AbstractReader):
    """Reader implementation for pandas DataFrames containing options data."""

    def _ingest_data(self, data: DataFrame) -> DataFrame:
        """Return the DataFrame as-is since it's already in memory.

        Arguments:
            data: The input DataFrame

        Returns:
            The same DataFrame

        Raises:
            TypeError: If data is not a DataFrame
            ValueError: If DataFrame is empty
        """
        if not isinstance(data, DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data)}")

        if data.empty:
            raise ValueError("Input DataFrame is empty")

        return data

    # Inherits _clean_data() from AbstractReader.

    def _transform_data(self, cleaned_data: DataFrame) -> DataFrame:
        """Apply DataFrame-specific transformations.

        Arguments:
            cleaned_data: The validated DataFrame

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If insufficient data points
        
        Note:
            Common validation (NaN checks, negative prices, etc.) is handled 
            by the base class _validate_data() method.
        """
        # Ensure we have enough data points
        if len(cleaned_data) < 5:
            raise ValueError(
                f"Insufficient options data: need at least 5 strike prices, got {len(cleaned_data)}"
            )

        return cleaned_data.reset_index(drop=True)
