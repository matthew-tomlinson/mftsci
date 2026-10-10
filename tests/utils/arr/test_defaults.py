"""
Unit test module for mftsci.utils.arr.defaults.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages
import pytest
import numpy as np

# Internal packages
from mftsci.utils.arr.defaults import *


# =====================================
# UNIT TESTS
# =====================================


def test_a():
    # Test with no input (should return empty array)
    result = a()
    expected = np.array([], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)

    # Test with single integer
    result = convert(5)
    expected = np.array(5)
    np.testing.assert_array_equal(result, expected)

    # Test with list of integers
    result = convert([1, 2, 3])
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


# =====================================
# MAIN
# =====================================


# Run the tests
if __name__ == "__main__":
    pytest.main()
