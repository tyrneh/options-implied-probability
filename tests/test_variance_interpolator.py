"""Unit tests for TotalVarianceInterpolator."""

import numpy as np
from oipd.core.vol_surface_fitting.variance_interpolator import TotalVarianceInterpolator
from oipd.core.vol_surface_fitting.forward_interpolator import ForwardInterpolator


def test_variance_interpolator():
    print("\n--- Testing TotalVarianceInterpolator ---")

    # Mock pillar curves: flat smile with sigma = 0.2
    # w(k, T) = sigma^2 * T = 0.04 * T
    sigma = 0.2

    def make_pillar(t):
        return lambda k: sigma**2 * t

    pillars = [
        (0.1, make_pillar(0.1)),
        (0.5, make_pillar(0.5)),
        (1.0, make_pillar(1.0)),
    ]
    forward_interp = ForwardInterpolator([(0.1, 100.0), (0.5, 100.0), (1.0, 100.0)])
    interp = TotalVarianceInterpolator(pillars, forward_interp)

    K = 100.0  # ATM, so k = ln(1) = 0

    # Test 1: Exact pillar
    print("Testing exact pillar values...")
    w_01 = interp(K, 0.1)
    expected_01 = sigma**2 * 0.1
    assert abs(w_01 - expected_01) < 1e-9, f"Expected {expected_01}, got {w_01}"
    print(f"  w(K, t=0.1) = {w_01:.6f} PASS")

    # Test 2: Midpoint interpolation
    print("Testing midpoint interpolation...")
    w_03 = interp(K, 0.3)
    # Between T1=0.1 (w=0.004) and T2=0.5 (w=0.02)
    # alpha = (0.5 - 0.3) / (0.5 - 0.1) = 0.5
    # w = 0.5 * 0.004 + 0.5 * 0.02 = 0.012
    expected_03 = 0.5 * (sigma**2 * 0.1) + 0.5 * (sigma**2 * 0.5)
    assert abs(w_03 - expected_03) < 1e-9, f"Expected {expected_03}, got {w_03}"
    print(f"  w(K, t=0.3) = {w_03:.6f} PASS")

    # Test 3: Short-end extrapolation (t < T_first)
    print("Testing short-end extrapolation...")
    w_005 = interp(K, 0.05)
    # Linear from (0, 0) to (0.1, 0.004)
    expected_005 = (sigma**2 * 0.1) * (0.05 / 0.1)
    assert abs(w_005 - expected_005) < 1e-9, f"Expected {expected_005}, got {w_005}"
    print(f"  w(K, t=0.05) = {w_005:.6f} PASS")

    # Test 4: Long-end extrapolation (t > T_last)
    print("Testing long-end extrapolation...")
    w_15 = interp(K, 1.5)
    # Constant vol: sigma^2 * t
    expected_15 = sigma**2 * 1.5
    assert abs(w_15 - expected_15) < 1e-9, f"Expected {expected_15}, got {w_15}"
    print(f"  w(K, t=1.5) = {w_15:.6f} PASS")

    # Test 5: implied_vol method
    print("Testing implied_vol...")
    iv_03 = interp.implied_vol(K, 0.3)
    expected_iv = np.sqrt(expected_03 / 0.3)
    assert abs(iv_03 - expected_iv) < 1e-9, f"Expected {expected_iv}, got {iv_03}"
    print(f"  sigma(K, t=0.3) = {iv_03:.4f} PASS")

    # Test 6: Arbitrage clamping
    print("Testing arbitrage clamping...")

    def crossing_pillar_early(k):
        return 0.05  # Higher variance at earlier time

    def crossing_pillar_late(k):
        return 0.03  # Lower variance at later time (arbitrage!)

    crossing_pillars = [
        (0.1, crossing_pillar_early),
        (0.5, crossing_pillar_late),
    ]
    interp_arb = TotalVarianceInterpolator(
        crossing_pillars, forward_interp, check_arbitrage=True
    )
    w_mid = interp_arb(K, 0.3)
    # With clamping, w_2 should be clamped to w_1 = 0.05
    # alpha = (0.5 - 0.3) / 0.4 = 0.5
    # w = 0.5 * 0.05 + 0.5 * 0.05 = 0.05
    assert w_mid == 0.05, f"Expected 0.05 (clamped), got {w_mid}"
    print(f"  w(K, t=0.3) with clamping = {w_mid} PASS")

    print("\nTotalVarianceInterpolator: ALL TESTS PASSED")


if __name__ == "__main__":
    test_variance_interpolator()
