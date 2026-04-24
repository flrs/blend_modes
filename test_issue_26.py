import numpy as np
from blend_modes import multiply

print("Test 1: RGBA input (should already work)...")
try:
    rgba_bg = np.random.rand(10, 10, 4) * 255.0
    rgba_fg = np.random.rand(10, 10, 4) * 255.0
    result = multiply(rgba_bg, rgba_fg, 0.5)

    assert result.shape == (10, 10, 4)
    assert result.min() >= 0.0 and result.max() <= 255.0

    print(f"  PASSED - shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"  FAILED - {e}")

print("\nTest 2: RGB input (issue #26 - was crashing)...")
try:
    rgb_bg = np.random.rand(10, 10, 3) * 255.0
    rgb_fg = np.random.rand(10, 10, 3) * 255.0
    result = multiply(rgb_bg, rgb_fg, 0.5)

    assert result.shape == (10, 10, 4)
    assert result.min() >= 0.0 and result.max() <= 255.0

    print(f"  PASSED - shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"  FAILED - {e}")

print("\nTest 3: Output always RGBA...")
try:
    rgb_bg = np.random.rand(10, 10, 3) * 255.0
    rgb_fg = np.random.rand(10, 10, 3) * 255.0
    result = multiply(rgb_bg, rgb_fg, 0.5)

    assert result.shape[2] == 4

    print("  PASSED")
except Exception as e:
    print(f"  FAILED - {e}")

print("\n--- Done ---")