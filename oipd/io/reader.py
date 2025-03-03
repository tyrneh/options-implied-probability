from abc import ABC, abstractmethod
from typing import Union, Optional, Dict

from pandas import DataFrame
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
        transformed_cleaned_data = self._transform_data(cleaned_data)
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

        It verifies that the required columns exist and casts them to the proper type.

        Arguments:
            raw_data: the raw data ingested from the data source.

        Returns:
            A DataFrame with required columns cleaned.
        """
        required_columns = {"strike", "last_price", "bid", "ask"}
        missing_columns = required_columns - set(raw_data.columns)
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")

        raw_data["strike"] = raw_data["strike"].astype(np.float64)
        raw_data["last_price"] = raw_data["last_price"].astype(np.float64)
        raw_data["bid"] = raw_data["bid"].astype(np.float64)
        raw_data["ask"] = raw_data["ask"].astype(np.float64)
        return raw_data

    @abstractmethod
    def _transform_data(self, cleaned_data: DataFrame):
        """Apply any processing steps needed to get the data in a `DataFrame` of
        cleaned, ingested data into the correct units, type, fp accuracy etc.

        Arguments:
            cleaned_data: the raw ingested data in a DataFrame
        """
        raise NotImplementedError("_transform_data method has not been implemented")
