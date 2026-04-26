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
        """Clean raw options data into canonical numeric columns.

        Args:
            raw_data: Raw options data ingested from a source.

        Returns:
            DataFrame with required columns validated, numeric quote columns
            coerced, optional missing bid/ask metadata filled with ``NaN``, and
            ``option_type`` normalized when present.

        Raises:
            ValueError: If ``strike`` is missing or no supported quote shape is
                present. Supported quote shapes are ``last_price``, bid/ask, or
                explicit wide ``call_price``/``put_price`` columns.
        """
        required_columns = {"strike"}
        optional_columns = {"bid", "ask", "last_trade_date"}

        missing_required = required_columns - set(raw_data.columns)
        if missing_required:
            raise ValueError(f"Data is missing required columns: {missing_required}")

        if not self._has_supported_price_columns(raw_data):
            raise ValueError(
                "Data is missing required price columns: provide last_price, "
                "bid/ask, or call_price/put_price."
            )

        # Check which optional columns are present
        present_optional = optional_columns & set(raw_data.columns)
        missing_optional = optional_columns - set(raw_data.columns)

        if missing_optional:
            warnings.warn(
                f"Optional columns not present: {missing_optional}.",
                UserWarning,
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

        price_columns = {"last_price", "mid", "bid", "ask"}
        present_price_columns = price_columns & set(raw_data.columns)
        # Clean numeric columns only. ``last_trade_date`` is date-like metadata
        # used downstream for staleness filtering and must remain parseable.
        columns_to_clean = (
            required_columns | present_optional | present_price_columns
        ) - {"last_trade_date"}

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

        if missing_optional:
            existing = raw_data.attrs.get("_oipd_missing_optional_columns")
            if existing is None:
                raw_data.attrs["_oipd_missing_optional_columns"] = set(missing_optional)
            else:
                raw_data.attrs["_oipd_missing_optional_columns"].update(
                    missing_optional
                )

        # Normalize option_type column to use consistent "C"/"P" format
        if "option_type" in raw_data.columns:
            raw_data["option_type"] = raw_data["option_type"].astype(str).str.upper()
            raw_data["option_type"] = raw_data["option_type"].replace(
                {"CALL": "C", "PUT": "P"}
            )

        return raw_data

    def _has_supported_price_columns(self, raw_data: DataFrame) -> bool:
        """Check whether a DataFrame has one supported quote-price shape.

        Args:
            raw_data: Raw options data to inspect.

        Returns:
            ``True`` when the data has ``last_price``, both bid/ask columns, or
            explicit wide ``call_price``/``put_price`` columns.
        """
        columns = set(raw_data.columns)
        return (
            "last_price" in columns
            or {"bid", "ask"}.issubset(columns)
            or {"call_price", "put_price"}.issubset(columns)
        )

    def _validate_data(self, cleaned_data: DataFrame) -> DataFrame:
        """Validate canonical option data after cleaning.

        Args:
            cleaned_data: DataFrame after column cleaning.

        Returns:
            Validated and strike-sorted DataFrame.

        Raises:
            ValueError: If strikes are missing/non-positive or present quote
                prices are negative.
        """
        # Check for NaN values in strike column (required)
        if cleaned_data["strike"].isna().any():
            raise ValueError("Options data contains NaN values in strike column")

        # Check for zero or negative strikes
        if (cleaned_data["strike"] <= 0).any():
            raise ValueError("Options data contains non-positive strike prices")

        price_columns = ["last_price", "mid", "bid", "ask", "call_price", "put_price"]
        for price_column in price_columns:
            if price_column not in cleaned_data.columns:
                continue
            numeric_prices = pd.to_numeric(cleaned_data[price_column], errors="coerce")
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
