from __future__ import annotations

import numpy as np

from oipd.core.vol_surface_fitting.algorithms.ssvi import (
    SSVISliceObservations,
    calibrate_ssvi_surface,
)
from oipd.core.vol_surface_fitting.shared.ssvi import ssvi_total_variance
from oipd.core.vol_surface_fitting.shared.vol_model import VolModel


def make_observations(theta_values, maturities, rho, eta, gamma):
    observations = []
    k_grid = np.linspace(-0.8, 0.8, 17)
    for theta, maturity in zip(theta_values, maturities):
        w = ssvi_total_variance(k_grid, theta, rho, eta, gamma)
        obs = SSVISliceObservations(
            maturity=maturity,
            log_moneyness=k_grid,
            total_variance=w,
            weights=np.ones_like(w),
        )
        observations.append(obs)
    return observations


def test_calibrate_ssvi_surface_recovers_parameters():
    maturities = np.array([0.25, 0.5, 1.0])
    theta_true = np.array([0.05, 0.07, 0.11])
    rho_true = -0.35
    eta_true = 1.2
    gamma_true = 0.3

    observations = make_observations(
        theta_true, maturities, rho_true, eta_true, gamma_true
    )
    vol_model = VolModel(method="ssvi", strict_no_arbitrage=True)

    fit = calibrate_ssvi_surface(observations, vol_model)

    assert fit.params.theta.shape == theta_true.shape
    assert np.allclose(fit.params.theta, theta_true, atol=5e-3)
    assert abs(fit.params.rho - rho_true) < 5e-2
    assert abs(fit.params.eta - eta_true) < 5e-2
    assert abs(fit.params.gamma - gamma_true) < 5e-2

    w_model = ssvi_total_variance(
        observations[1].log_moneyness,
        fit.params.theta[1],
        fit.params.rho,
        fit.params.eta,
        fit.params.gamma,
    )
    assert np.allclose(w_model, observations[1].total_variance, atol=1e-3)
