"""
Module for bounds utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from __future__ import annotations
from typing import Optional, Union

# External packages
import numpy as np

# Internal packages
from mftsci.utils import magic
from mftsci.utils.dict import keys


# =====================================
# CLASSES
# =====================================

class Bounds(magic.MagicProp):

    def __init__(
                    self,
                    bounds: Optional[Union[Bounds, np.ndarray, list]] = None,
                    a: Optional[Union[float, int]] = None,
                    b: Optional[Union[float, int]] = None,
                    Delta: Optional[Union[float, int]] = None,
                    dtype: Optional[np.dtype] = None,
                    ):

        self.set_bounds(bounds=bounds, a=a, b=b, Delta=Delta, dtype=dtype)

    def __repr__(
                    self,
                    ):

        return f"Bounds:\n{self.__getmprop__().__repr__()}"

    def set_bounds(
                    self,
                    bounds: Optional[Union[Bounds, np.ndarray, list]] = None,
                    a: Optional[Union[float, int]] = None,
                    b: Optional[Union[float, int]] = None,
                    Delta: Optional[Union[float, int]] = None,
                    dtype: Optional[np.dtype] = None,
                    ):

        self.bounds = calc_bounds(bounds=bounds, a=a, b=b, Delta=Delta, dtype=dtype)

    def Delta(
                self,
                ):

        return np.diff(self.bounds, axis=-1)[..., 0]

    def linspace(
                    self,
                    num: Optional[int] = None,
                    ):

        return np.linspace(*self.bounds, num=num)

    def xs(
            self,
            num: Optional[int] = None,
            delta: Optional[Union[float, int]] = None,
            ):

        if num is None:
            if delta is None:
                num = 1001
            else:
                num = 1 + int(self.Delta()//delta)

        return self.linspace(num=num)


magic._setmprop(
                _class=Bounds, mprop="bounds", noval_rm="__repr__",
                ret_class_val=True, ret_class_val_bool=False,
                )


# =====================================
# FUNCTIONS
# =====================================


def calc_bounds(
                    bounds: Optional[Union[Bounds, np.ndarray, list]] = None,
                    a: Optional[Union[float, int]] = None,
                    b: Optional[Union[float, int]] = None,
                    Delta: Optional[Union[float, int]] = None,
                    dtype=None
                    ):

    if bounds is None:
        if a is None:
            a = 0
        if b is None:
            if Delta is not None:
                b = a + Delta
            else:
                b = 1
        _bounds = np.moveaxis(np.array([a, b]), source=0, destination=-1)
    elif isinstance(bounds, Bounds):
        _bounds = bounds.bounds
    else:
        _bounds = bounds

    return np.array(_bounds, dtype=dtype)


def stack_bounds(
                    bounds_list: Optional[Union[np.ndarray, list]] = None
                    ):

    return np.array([b.bounds for b in bounds_list])


def outer_merge(
                bounds_list: Optional[Union[np.ndarray, list]] = None
                ):

    stack = stack_bounds(bounds_list)

    return Bounds(bounds=np.array([np.min(stack[:, 0]), np.max(stack[:, 1])]))


# Defaults
# -------------------------------------


def default_bounds_mode(
                        bounds_mode: Optional[str] = None,
                        ):

    if bounds_mode is None:
        bounds_mode = "unit"
    return bounds_mode


def default_bounds(
                    bounds_mode: Optional[str] = None,
                    ):

    bounds_mode = default_bounds_mode(bounds_mode=bounds_mode)

    if bounds_mode.lower()[:4] == "real":
        bounds = np.array([-np.inf, np.inf])
    else:
        bounds = np.array([0, 1])

    uscore = bounds.find("_")
    if uscore >= 0:
        postscore = bounds[1+uscore:]

        sign_str = {
                        "-": ["-", "neg"],
                        "+": ["+", "pos"],
                    }
        signs = keys(sign_str)

        for s in range(signs.size):
            _issign = np.any([  # noqa: F841
                                postscore.lower()[:3] == sign_str[1-s]
                                    ])

        if postscore.lower()[:3] == "pos" or \
                postscore.lower()[:6] == "nonneg" or \
                postscore.lower()[1:4] == "neg":
            bounds[0] = 0
        elif postscore.lower()[:8] == "negative" or \
                postscore.lower()[:11] == "nonpositive" or \
                postscore.lower()[1:9] == "positive":
            bounds[1] = 0


def default_num(
                num: Optional[int] = None,
                ):

    if num is None:
        num = 1001
    return num
