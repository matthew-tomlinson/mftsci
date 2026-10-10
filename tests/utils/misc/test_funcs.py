"""
Unit test module for mftsci.utils.misc.funcs.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages
import pytest
import numpy as np

# Internal packages
from mftsci.utils.misc import *
from mftsci.utils.misc import _multiple_round_OLD_scalar


# =====================================
# UNIT TESTS
# =====================================


def test_multiple_round_OLD_scalar():
    # Test case 1: Default rounding
    result = _multiple_round_OLD_scalar(val=5.6, factor=2)
    expected = 6
    assert result == expected

    # Test case 2: Floor rounding
    result = _multiple_round_OLD_scalar(val=5.6, factor=2, round_type="floor")
    expected = 4
    assert result == expected

    # Test case 3: Rounding with buffer
    result = _multiple_round_OLD_scalar(val=5.6, factor=2, buffer=0.7)
    expected = 8
    assert result == expected


def test_multiple_round():
    # Test case 1: Rounding an array
    result = multiple_round(val=np.array([5.6, 3.4]), factor=2)
    expected = np.array([6, 4])
    np.testing.assert_array_equal(result, expected)

    # Test case 2: Floor rounding a scalar
    result = multiple_round(val=5.6, factor=2, round_type="floor")
    expected = 4
    assert result == expected

    # Test case 3: Rounding an array with buffer
    result = multiple_round(val=[5.6, 3.4], factor=2, buffer=0.7)
    expected = np.array([8, 4])
    np.testing.assert_array_equal(result, expected)


def test_calc_val_1_0():
    # Test case 1: app_val_1 is True
    result = calc_val_1_0(app_val_1=True, val_1=1, val_0=0)
    expected = 1
    assert result == expected

    # Test case 2: app_val_1 is False
    result = calc_val_1_0(val_1="Y", val_0="N", app_val_1=False)
    expected = "N"
    assert result == expected

    # Test case 3: Default app_val_1
    result = calc_val_1_0(val_1=1, val_0=0)
    expected = 1
    assert result == expected


# =====================================
# MAIN
# =====================================


# Run the tests
if __name__ == "__main__":
    pytest.main()
