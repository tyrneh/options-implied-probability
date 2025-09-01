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
        """Apply validation and transformation to the DataFrame.

        Arguments:
            cleaned_data: The cleaned DataFrame

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If data contains invalid values
        """
        # Check for NaN values in required columns only
        required_columns = ["strike", "last_price"]
        for col in required_columns:
            if cleaned_data[col].isna().any():
                raise ValueError(f"Options data contains NaN values in required column '{col}'")

        # Check for negative prices (only for non-NaN values)
        if (cleaned_data["last_price"] < 0).any():
            raise ValueError("Options data contains negative prices")

        # Check for zero or negative strikes
        if (cleaned_data["strike"] <= 0).any():
            raise ValueError("Options data contains non-positive strike prices")

        # Sort by strike price for consistency
        cleaned_data = cleaned_data.sort_values("strike")

        # Ensure we have enough data points
        if len(cleaned_data) < 5:
            raise ValueError(
                f"Insufficient options data: need at least 5 strike prices, got {len(cleaned_data)}"
            )

        return cleaned_data.reset_index(drop=True)
