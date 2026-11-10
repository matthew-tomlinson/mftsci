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
from mftsci.utils.arr.core import *


# =====================================
# UNIT TESTS
# =====================================


def test_convert():
    # Test with no input (should return empty array)
    result = convert()
    expected = np.array([], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)

    # Test with single integer, conserve_scalar=True
    result = convert(5, conserve_scalar=True)
    expected = np.array(5)
    np.testing.assert_array_equal(result, expected)

    # Test with single integer, conserve_scalar=False
    result = convert(5, conserve_scalar=False)
    expected = np.array([5])
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
