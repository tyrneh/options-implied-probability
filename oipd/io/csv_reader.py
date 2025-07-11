import os
from pathlib import Path
from pandas import DataFrame, read_csv
from oipd.io.reader import AbstractReader


class CSVReadError(Exception):
    """Exception raised when CSV reading fails"""

    pass


class CSVReader(AbstractReader):
    """Reader implementation for pre-formatted CSV files containing options data."""

    def _ingest_data(self, url: str) -> DataFrame:
        """Ingest raw data from a CSV file at the given URL

        Arguments:
            url: the filepath to retrieve the raw csv data from

        Returns:
            DataFrame containing the CSV data

        Raises:
            FileNotFoundError: If the file doesn't exist
            CSVReadError: If the CSV file cannot be read
            ValueError: If the file is empty
        """
        # Check if file exists
        if not os.path.exists(url):
            raise FileNotFoundError(f"CSV file not found: {url}")

        # Check if it's a file (not a directory)
        if not os.path.isfile(url):
            raise CSVReadError(f"Path is not a file: {url}")

        # Check file size
        file_size = os.path.getsize(url)
        if file_size == 0:
            raise ValueError(f"CSV file is empty: {url}")

        try:
            df = read_csv(url)
        except UnicodeDecodeError as e:
            raise CSVReadError(
                f"Unable to decode CSV file (check encoding): {url}. Error: {str(e)}"
            )
        except Exception as e:
            raise CSVReadError(f"Failed to read CSV file: {url}. Error: {str(e)}")

        if df.empty:
            raise ValueError(f"CSV file contains no data: {url}")

        return df

    # Inherits _clean_data() from AbstractReader.

    def _transform_data(self, cleaned_data: DataFrame) -> DataFrame:
        """Apply any processing steps needed to get the data in a `DataFrame` of
        cleaned, ingested data into the correct units, type, fp accuracy etc.

        Arguments:
            cleaned_data: the raw ingested data in a DataFrame

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If data contains invalid values
        """
        # Check for negative prices
        if (cleaned_data["last_price"] < 0).any():
            raise ValueError("Options data contains negative prices")

        # Check for zero or negative strikes
        if (cleaned_data["strike"] <= 0).any():
            raise ValueError("Options data contains non-positive strike prices")

        # Sort by strike price for consistency
        cleaned_data = cleaned_data.sort_values("strike")

        return cleaned_data
