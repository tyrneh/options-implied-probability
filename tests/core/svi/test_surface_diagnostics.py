from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from oipd.core.vol_surface_fitting.shared.ssvi import (
    check_ssvi_calendar,
    check_ssvi_constraints,
)
from oipd.core.vol_surface_fitting.shared.svi import SVIParameters, check_butterfly

# Legacy RND/ModelParams tests removed


def test_check_butterfly_reports_min_margin():
    params = SVIParameters(a=0.02, b=0.15, rho=-0.2, m=0.0, sigma=0.25)
    grid = np.linspace(-0.6, 0.6, 41)
    diagnostics = check_butterfly(params, grid)
    assert "min_margin" in diagnostics
    assert np.isscalar(diagnostics["min_margin"])


def test_check_ssvi_constraints_and_calendar():
    theta = [0.05, 0.08, 0.11]
    rho = -0.3
    eta = 0.9
    gamma = 0.3
    inequality = check_ssvi_constraints(theta, rho, eta, gamma)
    assert inequality["min_theta_phi_margin"] >= -1e-8
    assert len(inequality["theta_phi_margins"]) == len(theta)

    calendar = check_ssvi_calendar(
        theta, rho, eta, gamma, k_grid=np.linspace(-0.5, 0.5, 41)
    )
    assert calendar["min_margin"] >= -1e-8
    assert len(calendar["margins"]) == len(theta) - 1
