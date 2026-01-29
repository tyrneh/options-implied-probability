"""Unit tests for build_surface_interpolator pipeline."""

import numpy as np
from oipd.pipelines.vol_surface.interpolator import build_surface_interpolator


def test_pipeline_interpolator():
    print("\n--- Testing build_surface_interpolator Pipeline ---")

    # Mock VolCurve: callable(K) -> sigma
    # Flat vol at 20%
    sigma = 0.2

    class MockVolCurve:
        def __call__(self, K):
            return sigma

    slices = {
        0.1: MockVolCurve(),
        0.5: MockVolCurve(),
        1.0: MockVolCurve(),
    }
    forwards = {0.1: 100.0, 0.5: 100.0, 1.0: 100.0}

    interp = build_surface_interpolator(slices, forwards)

    K = 100.0  # ATM

    # Test 1: Exact pillar
    print("Testing exact pillar via pipeline...")
    w_01 = interp(K, 0.1)
    expected = sigma**2 * 0.1
    assert abs(w_01 - expected) < 1e-9, f"Expected {expected}, got {w_01}"
    print(f"  w(K, t=0.1) = {w_01:.6f} PASS")

    # Test 2: Midpoint
    print("Testing midpoint via pipeline...")
    w_03 = interp(K, 0.3)
    # Linear interp: alpha = 0.5
    expected_03 = 0.5 * (sigma**2 * 0.1) + 0.5 * (sigma**2 * 0.5)
    assert abs(w_03 - expected_03) < 1e-9, f"Expected {expected_03}, got {w_03}"
    print(f"  w(K, t=0.3) = {w_03:.6f} PASS")

    # Test 3: implied_vol
    print("Testing implied_vol via pipeline...")
    iv = interp.implied_vol(K, 0.3)
    assert abs(iv - sigma) < 1e-9, f"Expected {sigma}, got {iv}"
    print(f"  sigma(K, t=0.3) = {iv:.4f} PASS")

    print("\nbuild_surface_interpolator Pipeline: ALL TESTS PASSED")


if __name__ == "__main__":
    test_pipeline_interpolator()
