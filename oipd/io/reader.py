from abc import ABC, abstractmethod
from typing import Union, Optional, Dict

from pandas import DataFrame
import pandas as pd
import numpy as np


class AbstractReader(ABC):
    """Abstract class for readers -- ingest data from a source
    and return the cleaned, transformed result as a DataFrame

    Methods:
        read
        _ingest_data
        _clean_data
        _transform_data
    """

    def read(
        self,
        input_data: Union[str, DataFrame],
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> DataFrame:
        """Abstract class for readers -- ingest data from a source
        and return the cleaned, transformed result as a DataFrame.

        Methods:
            read, _ingest_data, _apply_column_mapping, _clean_data, _transform_data
        """

        if isinstance(input_data, DataFrame):
            if input_data.empty:
                raise ValueError("Input DataFrame contains no data")
        elif input_data == "":
            raise ValueError("Input filepath is empty")
        elif input_data is None:
            raise ValueError("Either an input filepath or DataFrame must be specified")

        if isinstance(input_data, str):
            input_data = self._ingest_data(input_data)

        if column_mapping:
            input_data = input_data.rename(columns=column_mapping)

        cleaned_data = self._clean_data(input_data)
        validated_data = self._validate_data(cleaned_data)
        transformed_cleaned_data = self._transform_data(validated_data)
        return transformed_cleaned_data

    @abstractmethod
    def _ingest_data(self, url: str) -> DataFrame:
        """Ingest raw data from a data source at the given URL

        Arguments:
            url: the url to retrieve the raw data from
        """
        raise NotImplementedError("`_ingest_data` method has not been implemented")

    def _clean_data(self, raw_data: DataFrame) -> DataFrame:
        """Default cleaning implementation.

        It verifies that required columns exist and converts them to proper numeric types.
        Optional columns (bid, ask) are handled gracefully if missing.
        Handles common data issues like missing values (dashes, empty strings) and
        comma-separated numbers (e.g., "1,200").

        Arguments:
            raw_data: the raw data ingested from the data source.

        Returns:
            A DataFrame with required columns cleaned and optional columns added if missing.
        """
        import pandas as pd
        import numpy as np
        import warnings

        # Define required vs optional columns
        required_columns = {"strike", "last_price"}
        optional_columns = {"bid", "ask", "last_trade_date"}
        all_expected = required_columns | optional_columns

        # Check for required columns only
        missing_required = required_columns - set(raw_data.columns)
        if missing_required:
            raise ValueError(f"Data is missing required columns: {missing_required}")

        # Check which optional columns are present
        present_optional = optional_columns & set(raw_data.columns)
        missing_optional = optional_columns - set(raw_data.columns)

        if missing_optional:
            warnings.warn(
                f"Optional columns not present: {missing_optional}.", UserWarning
            )

        # Create a copy to avoid modifying the original data
        raw_data = raw_data.copy()

        # Clean all present columns (required + present optional)
        columns_to_clean = required_columns | present_optional

        for col in columns_to_clean:
            # Handle string columns that might have commas in numbers
            if raw_data[col].dtype == "object":
                # Remove commas from numeric strings (e.g., "1,200" -> "1200")
                raw_data[col] = (
                    raw_data[col].astype(str).str.replace(",", "", regex=False)
                )

            # Convert to numeric, coercing invalid values (dashes, empty strings, etc.) to NaN
            raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

        # Add placeholder NaN columns for missing optional columns
        # This ensures consistent DataFrame structure
        for col in missing_optional:
            raw_data[col] = np.nan

        # Normalize option_type column to use consistent "C"/"P" format
        if 'option_type' in raw_data.columns:
            raw_data['option_type'] = raw_data['option_type'].astype(str).str.upper()
            raw_data['option_type'] = raw_data['option_type'].replace({
                'CALL': 'C',
                'PUT': 'P'
            })

        return raw_data

    def _validate_data(self, cleaned_data: DataFrame) -> DataFrame:
        """Common validation logic for all readers.
        
        Validates that data meets minimum requirements for options processing:
        - Strike prices must not contain NaN values
        - Strike prices must be positive
        - Last prices must not be negative (NaN values allowed, filtered downstream)
        
        Arguments:
            cleaned_data: DataFrame after column cleaning
            
        Returns:
            Validated and sorted DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        # Check for NaN values in strike column (required)
        if cleaned_data["strike"].isna().any():
            raise ValueError("Options data contains NaN values in strike column")
        
        # Check for zero or negative strikes
        if (cleaned_data["strike"] <= 0).any():
            raise ValueError("Options data contains non-positive strike prices")
        
        # Check for negative prices (only for valid numeric values)
        # NaN and non-numeric values are allowed and will be filtered downstream
        if "last_price" in cleaned_data.columns:
            # Convert to numeric, treating non-numeric values as NaN
            numeric_prices = pd.to_numeric(cleaned_data["last_price"], errors='coerce')
            valid_prices = numeric_prices.notna()
            if valid_prices.any() and (numeric_prices.loc[valid_prices] < 0).any():
                raise ValueError("Options data contains negative prices")
        
        # Sort by strike price for consistency
        cleaned_data = cleaned_data.sort_values("strike")
        
        return cleaned_data

    @abstractmethod
    def _transform_data(self, cleaned_data: DataFrame):
        """Apply any processing steps needed to get the data in a `DataFrame` of
        cleaned, ingested data into the correct units, type, fp accuracy etc.

        Arguments:
            cleaned_data: the raw ingested data in a DataFrame
        """
        raise NotImplementedError("_transform_data method has not been implemented")
