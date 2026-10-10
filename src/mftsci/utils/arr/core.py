"""
Core module for arr utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union


# External packages
import numpy as np


# Internal packages


# =====================================
# FUNCTIONS
# =====================================


def convert(
            a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            conserve_scalar: Optional[bool] = None,
            ) -> np.ndarray:
    """
    Converts the input to a NumPy array.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,], optional
        The input to be converted to a NumPy array. If None, an empty list is used.
    conserve_scalar : bool, optional
        If True, converts scalar values to 1D arrays. If None, set to False.

    Returns
    -------
    np.ndarray
        The input converted to a NumPy array.

    Examples
    --------
    >>> default()
    array([], dtype=float64)

    >>> default(5, True)
    array(5)

    >>> default(5, False)
    array([5])

    >>> default([1, 2, 3])
    array([1, 2, 3])
    """

    if conserve_scalar is None:
        conserve_scalar = True

    if a is None:
        a = []
    if not conserve_scalar:
        if np.isscalar(a):
            a = [a]
    a = np.array(a)

    return a
