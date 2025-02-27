import numpy as np
import pandas as pd
from pandas import DataFrame

from oipd.io.reader import AbstractReader


class DataFrameReader(AbstractReader):
    """Reader implementation for pre-formatted DataFrame containing options data

    Methods:
        _ingest_data
        _clean_data
        _transform_data
    """

    def _ingest_data(self, df: DataFrame) -> DataFrame:
        """Ingest raw data from a DataFrame

        Arguments:
            df: a DataFrame containing the raw options data
        """
        # Return a copy to avoid modifying the original DataFrame
        return df.copy()

    def _clean_data(self, raw_data: DataFrame) -> DataFrame:
        """Apply cleaning steps to raw, ingested data

        Arguments:
            raw_data: the raw data ingested from the DataFrame
        """
        raw_data["strike"] = raw_data["strike"].astype(np.float64)
        raw_data["last_price"] = raw_data["last_price"].astype(np.float64)
        raw_data["bid"] = raw_data["bid"].astype(np.float64)
        raw_data["ask"] = raw_data["ask"].astype(np.float64)
        return raw_data

    def _transform_data(self, cleaned_data: DataFrame):
        """Apply any processing steps needed to convert the cleaned DataFrame into
        the desired format (e.g., unit conversion, precision adjustments)

        Arguments:
            cleaned_data: the DataFrame after cleaning steps have been applied
        """
        return cleaned_data
