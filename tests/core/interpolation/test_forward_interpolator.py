"""Unit tests for ForwardInterpolator."""

from oipd.core.vol_surface_fitting.forward_interpolator import ForwardInterpolator


def test_forward_interpolator():
    print("\n--- Testing ForwardInterpolator ---")

    interp = ForwardInterpolator([(0.1, 100.0), (0.5, 105.0), (1.0, 110.0)])

    # Test 1: Exact pillar values
    print("Testing exact pillar values...")
    assert interp(0.1) == 100.0, f"Expected 100.0, got {interp(0.1)}"
    assert interp(0.5) == 105.0, f"Expected 105.0, got {interp(0.5)}"
    assert interp(1.0) == 110.0, f"Expected 110.0, got {interp(1.0)}"
    print("  Exact pillars: PASS")

    # Test 2: Midpoint interpolation
    print("Testing midpoint interpolation...")
    # Between 0.1 (100) and 0.5 (105), at t=0.3
    # Linear: 100 + (105-100) * (0.3-0.1)/(0.5-0.1) = 100 + 5 * 0.5 = 102.5
    mid_val = interp(0.3)
    assert abs(mid_val - 102.5) < 1e-9, f"Expected 102.5, got {mid_val}"
    print(f"  Midpoint (t=0.3): {mid_val} PASS")

    # Test 3: Short-end extrapolation (t < t_min)
    print("Testing short-end extrapolation...")
    short_val = interp(0.05)
    assert short_val == 100.0, f"Expected 100.0 (flat), got {short_val}"
    print(f"  Short-end (t=0.05): {short_val} PASS")

    # Test 4: Long-end extrapolation (t > t_max)
    print("Testing long-end extrapolation...")
    long_val = interp(2.0)
    assert long_val == 110.0, f"Expected 110.0 (flat), got {long_val}"
    print(f"  Long-end (t=2.0): {long_val} PASS")

    # Test 5: Single pillar
    print("Testing single pillar...")
    single = ForwardInterpolator([(0.5, 100.0)])
    assert single(0.1) == 100.0
    assert single(1.0) == 100.0
    print("  Single pillar: PASS")

    print("\nForwardInterpolator: ALL TESTS PASSED")


if __name__ == "__main__":
    test_forward_interpolator()
