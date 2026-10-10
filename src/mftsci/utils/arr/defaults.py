"""
Defaults module for arr utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable

# External packages
import numpy as np


# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.arr.core import convert


# =====================================
# DEFAULTS
# =====================================


def a(
            a: object = None,
            ) -> np.ndarray:
    """
    Returns the default array if None is given.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,], optional
        The input to be converted to a NumPy array. If None, an empty list is used.

    Returns
    -------
    np.ndarray
        The input converted to a NumPy array.

    Examples
    --------
    >>> default()
    array([], dtype=float64)

    >>> default(5)
    array(5)

    >>> default([1, 2, 3])
    array([1, 2, 3])
    """

    return convert(a=a, conserve_scalar=True)


def appear_order(
        appear_order: Optional[bool] = None,
        ) -> bool:

    return set_default(val=appear_order, default=True)


def expand_back(
        expand_back: Optional[bool] = None,
        ) -> bool:

    return set_default(val=expand_back, default=False)


def col_iter(
        col_iter: Optional[bool] = None,
        ) -> bool:

    return set_default(val=col_iter, default=True)


def reshape(
        reshape: Optional[bool] = None,
        ) -> bool:

    return set_default(val=reshape, default=True)


def f2(
        f2: Optional[Callable] = None,
        ) -> Callable:

    return set_default(val=f2, default=np.multiply)


def cend(
        cend: Optional[bool] = None,
        ) -> bool:

    return set_default(val=cend, default=False)


def as_list(
        as_list: Optional[bool] = None,
        ) -> bool:

    return set_default(val=as_list, default=False)


def N(
        N: Optional[int] = None,
        ) -> int:

    return set_default(val=N, default=0)


def flip(
        flip: Optional[bool] = None,
        ) -> bool:

    return set_default(val=flip, default=False)


def circle(
        circle: Optional[bool] = None,
        ) -> bool:

    return set_default(val=circle, default=True)


def app_val_1_default(
        app_val_1_default: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_val_1_default, default=True)


def signs(
        signs: Optional[Union[np.ndarray, list,]] = None,
        ) -> Union[np.ndarray, list,]:

    return set_default(val=x, default=np.array([-1, 1]))


def x(
        x: Optional[Union[np.ndarray, list,]] = None,
        ) -> Union[np.ndarray, list,]:

    return set_default(val=x, default=np.array([0, 1]))
