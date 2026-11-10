"""
Unit test module for mftsci.utils.magic.defaults.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages
import pytest
import numpy as np

# Internal packages
from mftsci.utils.meta.defaults import *


# =====================================
# UNIT TESTS
# =====================================

def test_int():
    # Test case 1: Default
    assert int() == 0
    # Test case 2: int
    assert int(1) == 1


def test_float():
    # Test case 1: Default
    assert float() == 0.0
    # Test case 2: float
    assert float(1.0) == 1.0


def test_call():
    # Test case 1: Default
    assert call()(1) == 1
    # Test case 2: doubling
    assert call(lambda x: 2*x)(1) == 2


def test_reset():
    # Test case 1: True as value
    assert reset(True) is True
    # Test case 2: None as value
    assert reset(None) is False


# =====================================
# MAIN
# =====================================


# Run the tests
if __name__ == "__main__":
    pytest.main()
