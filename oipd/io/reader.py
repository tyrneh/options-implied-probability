from abc import ABC, abstractmethod
from typing import Union, Optional, Dict

from pandas import DataFrame
import pandas as pd
import numpy as np
import warnings


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
        comma-separated numbers (e.g., "1,200"). It also supports strike strings that
        end with an option designator (e.g., "70.00C" or "21.00p") by stripping the
        trailing letter and, if missing, deriving an `option_type` column.

        Arguments:
            raw_data: the raw data ingested from the data source.

        Returns:
            A DataFrame with required columns cleaned and optional columns added if missing.
        """
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

        # If strike values come with trailing C/P (case-insensitive), strip and optionally derive option_type
        if "strike" in raw_data.columns:
            strike_as_string = raw_data["strike"].astype(str).str.strip()
            strike_suffix = strike_as_string.str[-1].str.upper()
            has_suffix = strike_suffix.isin({"C", "P"})

            # Derive option_type if not provided
            if has_suffix.any():
                if "option_type" not in raw_data.columns:
                    # Initialize with NaN, then fill where suffix exists
                    raw_data["option_type"] = np.nan
                    raw_data.loc[has_suffix, "option_type"] = strike_suffix.loc[
                        has_suffix
                    ]
                else:
                    # Normalize existing option_type for comparison
                    normalized_option_type = (
                        raw_data["option_type"]
                        .astype(str)
                        .str.upper()
                        .replace({"CALL": "C", "PUT": "P"})
                    )
                    conflict_mask = (
                        has_suffix
                        & normalized_option_type.notna()
                        & (normalized_option_type != strike_suffix)
                    )
                    if conflict_mask.any():
                        conflict_rows = list(raw_data.index[conflict_mask])
                        raise ValueError(
                            f"Conflict between strike suffix and option_type in rows: {conflict_rows}"
                        )
                    # Where option_type is missing but suffix exists, backfill from suffix
                    needs_fill = has_suffix & (
                        normalized_option_type.isna()
                        | (normalized_option_type == "NAN")
                    )
                    if needs_fill.any():
                        raw_data.loc[needs_fill, "option_type"] = strike_suffix.loc[
                            needs_fill
                        ]

                # Remove trailing C/P from strike strings
                strike_as_string = strike_as_string.where(
                    ~has_suffix, strike_as_string.str[:-1]
                )
                raw_data["strike"] = strike_as_string

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
        if "option_type" in raw_data.columns:
            raw_data["option_type"] = raw_data["option_type"].astype(str).str.upper()
            raw_data["option_type"] = raw_data["option_type"].replace(
                {"CALL": "C", "PUT": "P"}
            )

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
            numeric_prices = pd.to_numeric(cleaned_data["last_price"], errors="coerce")
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
