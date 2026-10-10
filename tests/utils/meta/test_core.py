"""
Unit test module for mftsci.utils.magic.core.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages
import pytest
import numpy as np

# Internal packages
from mftsci.utils.meta.core import *


# =====================================
# UNIT TESTS
# =====================================

def test_return_x():
    # Test case 1: Default
    assert return_x() is None
    # Test case 2: int
    assert return_x(10) == 10
    # Test case 3: str
    assert return_x('hello') == 'hello'


def test_set_default():
    # Test case 1: None as value, default_call
    assert set_default(None, 10, None) == 10
    # Test case 2: None as value
    assert set_default(None, 10, lambda: 20) == 20
    # Test case 3: Non-None value
    assert set_default(30, 10, lambda: 20) == 30
    # Test case 4: None as value, default, and default_call
    assert set_default(None, None) is None


def test_astype():
    # Test case 1: int to float
    assert astype(1, float) == 1.0
    # Test case 2: float to int
    assert astype(1.1, int) == 1


# =====================================
# MAIN
# =====================================


# Run the tests
if __name__ == "__main__":
    pytest.main()
