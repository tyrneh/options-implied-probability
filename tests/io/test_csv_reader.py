from pathlib import Path

import numpy as np
import pytest

from probabilistic.io import CSVReader


@pytest.fixture
def sample_url():
    sample_url_path = Path(__file__).parent.resolve() / Path("resources/sample.csv")
    return str(sample_url_path)


class TestCSVReader:
    """Test the implemetation of CSVReader"""

    def test_read_sample_file(self, sample_url):
        """Test that CSVReader reads a sample file correctly"""
        reader = CSVReader()
        result = reader.read(sample_url)
        assert result["strike"][2] == 325
        assert result["ask"][3] == 49.55
        assert len(result) == 78
        assert isinstance(result["bid"][0], np.float64)
