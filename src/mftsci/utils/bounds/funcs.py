"""
Functions module for bounds utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from __future__ import annotations
from typing import Optional, Union

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils import magic
from mftsci.utils.arr import pad as _pad
from mftsci.utils.pd import arr_to_srs, srs_to_arr
from mftsci.utils.unit import Unit, time_symb_dict
from mftsci.utils.bounds import defaults    # noqa: F401


# =====================================
# CLASSES
# =====================================

class Bounds(magic.MagicProp):
    """
    Local bounds.
    """

    def __init__(
                    self,
                    bounds: Optional[Union[Bounds, np.ndarray, list]] = None,
                    a: Optional[Union[float, int]] = None,
                    b: Optional[Union[float, int]] = None,
                    Delta: Optional[Union[float, int]] = None,
                    dtype: Optional[np.dtype] = None,
                    unit: Optional[Unit] = None,
                    ):

        self.set_bounds(bounds=bounds, a=a, b=b, Delta=Delta, dtype=dtype, unit=unit)

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
                    unit: Optional[Unit] = None,
                    ):

        self.bounds = calc_bounds(bounds=bounds, a=a, b=b, Delta=Delta, dtype=dtype)
        self.dtype = pd.Series(next(iter(self.bounds))).dtype
        if unit is None:
            if isinstance(bounds, Bounds):
                unit = bounds.unit
        self.set_unit(unit=unit, reset=True)

    def set_unit(
                    self,
                    unit: Optional[Unit] = None,
                    reset: Optional[bool] = None,
                    ):
        """
        Set dimension unit.
        """
        if unit is not None:
            self.unit = unit
        elif defaults.reset(reset):
            loc = np.min(self.bounds[..., 0])
            if pd.api.types.is_datetime64_any_dtype(self.dtype) \
               or pd.api.types.is_timedelta64_dtype(self.dtype):
                Delta_min = np.min(self.Delta(dedim=False))
                max_m = 1200
                S = pd.Series(time_symb_dict()).to_frame()
                S["m"] = Delta_min/S.index
                invalid = S["m"] > max_m
                if not invalid.all():
                    S = S[~invalid]
                scale = S.index[S["m"].argmax()]
            else:
                scale = 1

            self.unit = Unit(loc=loc, scale=scale)

    def get_bounds(
                    self,
                    dedim: Optional[bool] = None,
                    ):

        bounds = self.bounds
        if defaults.dedim(dedim):
            if pd.api.types.is_datetime64_any_dtype(self.dtype) \
               or pd.api.types.is_timedelta64_dtype(self.dtype):
                bounds = srs_to_arr(self.unit.dedim(arr_to_srs(bounds)))
            else:
                bounds = self.unit.dedim(bounds)

        return bounds

    def Delta(
                self,
                dedim: Optional[bool] = None,
                ):

        Delta = np.diff(self.get_bounds(dedim=False), axis=-1)[..., 0]
        if pd.api.types.is_datetime64_any_dtype(self.dtype) \
           or pd.api.types.is_timedelta64_dtype(self.dtype):
            Delta = Delta.astype(np.timedelta64)
        if defaults.dedim(dedim):
            Delta = self.unit.dedim(Delta, skip_loc=True)

        return Delta

    def linspace(
                    self,
                    num: Optional[int] = None,
                    dedim: Optional[bool] = None,
                    ):

        dedim_ls = np.linspace(*self.get_bounds(dedim=True), num=num)

        if defaults.dedim(dedim):
            return dedim_ls
        else:
            return self.unit.redim(dedim_ls)

    def xs(
            self,
            num: Optional[int] = None,
            delta: Optional[Union[float, int]] = None,
            dedim: Optional[bool] = None,
            ):

        dedim = defaults.dedim(dedim)
        if num is None:
            if delta is None:
                num = defaults.num()
            else:
                num = 1 + int(self.Delta(dedim=dedim)//delta)

        return self.linspace(num=num, dedim=dedim)


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
        a = defaults.a(a)
        if b is None:
            if Delta is not None:
                b = a + Delta
            else:
                b = defaults.b()
        _bounds = np.moveaxis(np.array(_pad([a, b])), source=0, destination=-1)
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
