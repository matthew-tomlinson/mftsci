"""
Unit test module for mftsci.utils.arr.funcs.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages
import pytest
import numpy as np

# Internal packages
from mftsci.utils.arr import *


# =====================================
# UNIT TESTS
# =====================================


def test_attr():
    # Test with numpy array of complex numbers and "real" attribute
    a = np.array([1. - 1.j, 2., 3. + 1.j])
    result = attr(a, attr="real")
    expected = np.array([1., 2., 3.])
    np.testing.assert_array_equal(result, expected)


def test_fattr():
    # Test with numpy array of strings and "capitalize" attribute
    a = np.array(["apple", "banana"])
    result = fattr(a, attr="capitalize")
    expected = np.array(["Apple", "Banana"])
    np.testing.assert_array_equal(result, expected)


def test_calc_shape():
    # Test with no input (should return shape of empty array)
    result = calc_shape()
    expected = np.array([0])
    np.testing.assert_array_equal(result, expected)

    # Test with empty array
    result = calc_shape([])
    expected = np.array([0])
    np.testing.assert_array_equal(result, expected)

    # Test with single integer
    result = calc_shape(5)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)

    # Test with list of integers
    result = calc_shape([1, 2, 3])
    expected = np.array([3])
    np.testing.assert_array_equal(result, expected)


def test_calc_axes():
    # Test with no input (should return axes of empty array)
    result = calc_axes()
    expected = np.array([0])
    np.testing.assert_array_equal(result, expected)

    # Test with empty array
    result = calc_axes([])
    expected = np.array([0])
    np.testing.assert_array_equal(result, expected)

    # Test with single integer
    result = calc_axes(5)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)

    # Test with list of integers
    result = calc_axes([1, 2, 3])
    expected = np.array([0])
    np.testing.assert_array_equal(result, expected)


def test_calc_length():
    # Test with no input (should return length of empty array)
    result = calc_length()
    expected = 0
    assert result == expected

    # Test with empty array
    result = calc_length([])
    expected = 0
    assert result == expected

    # Test with single integer
    result = calc_length(5)
    expected = 0
    assert result == expected

    # Test with list of integers
    result = calc_length([1, 2, 3])
    expected = 3
    assert result == expected


def test_unique():
    # Test with list containing duplicates
    result = unique([1, 3, 3, 3, 2, 2])
    expected = np.array([1, 3, 2])
    np.testing.assert_array_equal(result, expected)

    # Test with list containing duplicates and appear_order=True
    result = unique([1, 3, 3, 3, 2, 2], appear_order=True)
    expected = np.array([1, 3, 2])
    np.testing.assert_array_equal(result, expected)

    # Test with list containing duplicates and appear_order=False
    result = unique([1, 3, 3, 3, 2, 2], appear_order=False)
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_expand_dims():
    # Test with list and axis=0
    result = expand_dims([1, 2, 3], axis=0)
    expected = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(result, expected)

    # Test with single integer and axis=1
    result = expand_dims(5, axis=1)
    expected = np.array([[5]])
    np.testing.assert_array_equal(result, expected)


def test_pad():
    # Test with list of unequal length lists
    result = pad([[1, 2, 3], [4, 5]])
    expected = [np.array([1, 2, 3]), np.array([4, 5, 0])]
    for res, exp in zip(result, expected):
        np.testing.assert_array_equal(res, exp)

    # Test with single integer and master_shape=(3,)
    result = pad([5], master_shape=(3,))
    expected = [np.array([5, 0, 0])]
    for res, exp in zip(result, expected):
        np.testing.assert_array_equal(res, exp)


def test_moveaxis_destination():
    # Test with source and no destination
    result = moveaxis_destination([0, 1, 2])
    expected = np.array([-3, -2, -1])
    np.testing.assert_array_equal(result, expected)

    # Test with source and destination
    result = moveaxis_destination([0, 1, 2], [2, 0, 1])
    expected = np.array([2, 0, 1])
    np.testing.assert_array_equal(result, expected)


def test_moveaxes():
    # Test with 2D array, moving axes
    result = moveaxes([[1, 2, 3], [4, 5, 6]], source=[0, 1], destination=[1, 0])
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(result, expected)

    # Test with 1D array, moving single axis
    result = moveaxes([1, 2, 3], source=[0], destination=[-1])
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_reshape_iter_pre():
    # Test with 1D array and axes=[0]
    result, N_i, shape_m = reshape_iter_pre([1, 2, 3], axes=[0])
    expected_array = np.array([[1, 2, 3]])
    expected_N_i = 1
    expected_shape_m = [np.array([], dtype=int), np.array([3])]
    np.testing.assert_array_equal(result, expected_array)
    assert N_i == expected_N_i
    for res, exp in zip(shape_m, expected_shape_m):
        np.testing.assert_array_equal(res, exp)

    # Test with 2D array and axes=None
    result, N_i, shape_m = reshape_iter_pre([[1, 2], [3, 4]])
    expected_array = np.array([1, 2, 3, 4])
    expected_N_i = 4
    expected_shape_m = [np.array([2, 2]), np.array([], dtype=int)]
    np.testing.assert_array_equal(result, expected_array)
    assert N_i == expected_N_i
    for res, exp in zip(shape_m, expected_shape_m):
        np.testing.assert_array_equal(res, exp)

    # Test with 2D array and axes=[1]
    result, N_i, shape_m = reshape_iter_pre([[1, 2], [3, 4]], axes=[1])
    expected_array = np.array([[1, 2], [3, 4]])
    expected_N_i = 2
    expected_shape_m = [np.array([2]), np.array([2])]
    np.testing.assert_array_equal(result, expected_array)
    assert N_i == expected_N_i
    for res, exp in zip(shape_m, expected_shape_m):
        np.testing.assert_array_equal(res, exp)


def test_reshape_iter_post():
    # Test with 2D array and shape_m_0=[3]
    result = reshape_iter_post([[1], [2], [3]], shape_m_0=[3], reshape=True)
    expected = np.array([[1], [2], [3]])
    np.testing.assert_array_equal(result, expected)

    # Test with 1D array and shape_m_0=[1, 3]
    result = reshape_iter_post([1, 2, 3], shape_m_0=[1, 3], reshape=True)
    expected = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(result, expected)


def test_fmap():
    # Test applying np.mean to 1D array
    result = fmap([1, 2, 3], f=np.mean, axes=[0])
    expected = np.array([2.])
    np.testing.assert_array_equal(result, expected)

    # Test applying np.sum to 2D array over axis 1
    result = fmap([[1, 2], [3, 4]], f=np.sum, axes=[1])
    expected = np.array([3, 7])
    np.testing.assert_array_equal(result, expected)


def test_f2_axes():
    # Test with 1D arrays and default np.multiply
    result = f2_axes([1, 2, 3], [4, 5, 6])
    expected = np.array([4, 10, 18])
    np.testing.assert_array_equal(result, expected)

    # Test with 2D and 1D arrays, aligning along axis 1
    result = f2_axes([[1, 2], [3, 4]], [5, 6], caxes=[1])
    expected = np.array([[5, 12], [15, 24]])
    np.testing.assert_array_equal(result, expected)


def test_iterarr():
    # Test input scalar float
    result = iterarr(1)
    expected = np.array([1])
    np.testing.assert_array_equal(result, expected)

    # Test input scalar string
    result = iterarr("test")
    expected = np.array(["test"])
    np.testing.assert_array_equal(result, expected)

    # Test input 1D list
    result = iterarr([1, 2, 3])
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)

    # Test input 2D array
    result = iterarr([[1, 2, 3],
                      [4, 5, 6]])
    expected = np.array([1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(result, expected)

    # Test input 1D list with dtype specified
    result = iterarr([1, 2, 3], dtype=np.float32)
    expected = np.array([1., 2., 3.], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

    # Test input 1D array and converting to list
    result = iterarr(np.array([1, 2, 3]), as_list=True)
    expected = [1, 2, 3]
    assert result == expected


def test_append():
    # Test appending two lists
    result = append([1, 2, 3], [4, 5, 6])
    expected = np.array([1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(result, expected)

    # Test appending two integers
    result = append(1, 2)
    expected = np.array([1, 2])
    np.testing.assert_array_equal(result, expected)

    # Test appending empty list and non-empty list
    result = append([], [1, 2, 3])
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_filter():
    # Test filtering with inclusion and exclusion criteria
    result = filter([1, 2, 3, 4, 5], a_in=[1, 2, 3], a_rm=[2])
    expected = np.array([1, 3])
    np.testing.assert_array_equal(result, expected)

    # Test filtering with inclusion criteria only
    result = filter([1, 2, 3, 4, 5], a_in=[2, 4])
    expected = np.array([2, 4])
    np.testing.assert_array_equal(result, expected)

    # Test filtering with exclusion criteria only
    result = filter([1, 2, 3, 4, 5], a_rm=[2, 4])
    expected = np.array([1, 3, 5])
    np.testing.assert_array_equal(result, expected)


def test_index_N():
    # Test with specified N and reshape=True
    result = index_N([2, 3], N=4)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(result, expected)

    # Test with specified N and reshape=False
    result = index_N([2, 3], N=4, reshape=False)
    expected = np.array([[1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_N_index():
    # Test with specified index array
    result = N_index([2, 3], index=[0, 1])
    expected = 1
    assert result == expected

    # Test with specified index array
    result = N_index([2, 3], index=[1, 2])
    expected = 5
    assert result == expected

    # Test with no index array
    result = N_index([2, 3])
    expected = 0
    assert result == expected


def test_index_I_max():
    # Test with specified i and I_max
    result = index_I_max(5, I_max=10)
    expected = 5
    assert result == expected

    # Test with i greater than I_max
    result = index_I_max(11, I_max=10)
    expected = 1
    assert result == expected

    # Test with specified i, I_max and flip=True
    result = index_I_max(5, I_max=10, flip=True)
    expected = 4
    assert result == expected

    # Test with array of i, I_max and circle=False
    result = index_I_max([1, 2, 3], I_max=5, circle=False)
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_calc_val_1_0_N():
    # Test case 1
    result = calc_val_1_0_N(app_val_1=[True, False, True], val_1=1, val_0=0, N=3)
    expected = np.array([1, 0, 1])
    np.testing.assert_array_equal(result, expected)

    # Test case 2
    result = calc_val_1_0_N(val_1="Y", val_0="N", N=2)
    expected = np.array(["Y", "Y"])
    np.testing.assert_array_equal(result, expected)


def test_buffer_add():
    # Test case 1
    result = buffer_add(b=0.5, x=np.array([1, 2]))
    expected = np.array([0.5, 2.5])
    np.testing.assert_array_equal(result, expected)

    # Test case 2
    result = buffer_add()
    expected = np.array([0., 1.])
    np.testing.assert_array_equal(result, expected)


def test_buffer():
    # Test case 1
    result = buffer(b=1.2, x=np.array([1, 2]))
    expected = np.array([0.9, 2.1])
    np.testing.assert_array_equal(result, expected)

    # Test case 2
    result = buffer()
    expected = np.array([0, 1])
    np.testing.assert_array_equal(result, expected)


# =====================================
# MAIN
# =====================================


# Run the tests
if __name__ == "__main__":
    pytest.main()
