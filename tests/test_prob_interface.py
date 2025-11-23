"""
Test the Probability Interfaces (Distribution, DistributionSurface).
"""

import pandas as pd
import pytest
import numpy as np
from datetime import date

from oipd.interface.volatility import VolCurve, VolSurface
from oipd.interface.probability import Distribution, DistributionSurface
    vc.fit(df_slice, market, column_mapping=column_mapping)
    
    # 3. Derive Distribution
    dist = vc.implied_distribution()
    
    assert isinstance(dist, Distribution)
    assert dist.pdf is not None
    
    # Check consistency
    ev = dist.expected_value()
    assert np.isclose(ev, 259.0, atol=5.0)

if __name__ == "__main__":
    test_distribution_interface()
    test_vol_curve_to_distribution()
    print("\nTest Passed!")
