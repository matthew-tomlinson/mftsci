"""
Unit test module for mftsci.utils.xcpt.funcs.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages
import pytest
import numpy as np

# Internal packages
from mftsci.utils.xcpt import *


# =====================================
# FUNCTIONS
# =====================================


def sample_function(x):
    return x * 2


def raise_exception(x):
    raise ValueError("An error occurred.")


# =====================================
# UNIT TESTS
# =====================================


def test_try_f():
    # Test case 1: Function succeeds with numeric input
    success, result = try_f(sample_function, 3)
    expected = (True, 6)
    assert (success, result) == expected

    # Test case 2: Function succeeds with string input
    success, result = try_f(sample_function, 'a')
    expected = (True, 'aa')
    assert (success, result) == expected

    # Test case 3: Function raises an exception
    success, result = try_f(raise_exception, 3)
    expected = (False, None)
    assert (success, result) == expected


# =====================================
# MAIN
# =====================================


# Run the tests
if __name__ == "__main__":
    pytest.main()
