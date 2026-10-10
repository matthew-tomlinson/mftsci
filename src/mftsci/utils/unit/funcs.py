"""
Functions module for unit utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from __future__ import annotations
from typing import Optional

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils.pd import arr_to_srs, srs_to_arr
from mftsci.utils.unit import defaults


# =====================================
# CLASSES
# =====================================

class Unit():
    """
    Class for dimension unit
    """

    def __init__(
                    self,
                    loc: Optional[object] = None,
                    scale: Optional[object] = None,
                    symbol: Optional[str] = None,
                    ):

        self.set_loc(loc=loc, reset=True)
        self.set_scale(scale=scale, reset=True)
        self.set_symbol(symbol=symbol, reset=True)

    def __repr__(
                    self,
                    ) -> str:

        return f"{self.symbol}: loc={repr(self.loc)}, scale={repr(self.scale)}"

    def set_loc(
                    self,
                    loc: Optional[object] = None,
                    reset: Optional[bool] = None,
                    ):

        if loc is not None:
            self.loc = loc
        elif defaults.reset(reset=reset):
            self.loc = defaults.loc()

    def set_scale(
                    self,
                    scale: Optional[object] = None,
                    reset: Optional[bool] = None,
                    ):

        if scale is not None:
            self.scale = scale
        elif defaults.reset(reset=reset):
            if pd.api.types.is_datetime64_any_dtype(pd.Series(self.loc).dtype):
                self.scale = pd.Timedelta(seconds=1)
            else:
                self.scale = defaults.scale()

    def set_symbol(
                    self,
                    symbol: Optional[str] = None,
                    reset: Optional[bool] = None,
                    ):

        if symbol is not None:
            self.symbol = symbol
        elif defaults.reset(reset=reset):
            self.symbol = defaults.symbol(time_symb_dict().get(self.scale, None))

    def dedim(
                self,
                x: Optional[object] = None,
                skip_loc: Optional[bool] = None,
                loc: Optional[object] = None,
                scale: Optional[object] = None,
                do: Optional[bool] = None,
                ) -> object:

        if not defaults.do(do):
            x_out = x
        else:
            loc = defaults.set_default(val=loc, default=self.loc)
            scale = defaults.set_default(val=scale, default=self.scale)

            x = defaults.set_default(val=x, default=self.loc)
            is_arr = is_array_or_list(x=x)
            if is_arr:
                is_scalar = len(np.shape(x)) == 0
                x = arr_to_srs(x)

            if not defaults.skip_loc(skip_loc):
                x_out = (x - loc)/scale
            else:
                x_out = x/scale

            if is_arr:
                x_out = srs_to_arr(x_out)
                if is_scalar and x_out.size == 1:
                    x_out = x_out[0]

        return x_out

    def redim(
                self,
                x: Optional[object] = None,
                skip_loc: Optional[bool] = None,
                loc: Optional[object] = None,
                scale: Optional[object] = None,
                do: Optional[bool] = None,
                ) -> object:

        if not defaults.do(do):
            x_out = x
        else:
            loc = defaults.set_default(val=loc, default=self.loc)
            scale = defaults.set_default(val=scale, default=self.scale)

            x = defaults.set_default(val=x, default=0)

            if not defaults.skip_loc(skip_loc):
                x_out = loc + scale * x
            else:
                x_out = scale * x

        return x_out

    def transform(
                    self,
                    x: object,
                    unit: Optional[Unit] = None,
                    skip_loc: Optional[bool] = None,
                    loc: Optional[object] = None,
                    scale: Optional[object] = None,
                    unit_loc: Optional[object] = None,
                    unit_scale: Optional[object] = None,
                    do: Optional[bool] = None,
                    ):

        x_prime = self.dedim(x, skip_loc=skip_loc, loc=loc, scale=scale, do=do)

        if unit is None:
            return x_prime
        else:
            return unit.redim(x_prime, skip_loc=skip_loc,
                              loc=unit_loc, scale=unit_scale, do=do)

    def get_symbol_unitsuff(
                    self,
                    inv: Optional[bool] = None,
                    inv_op: Optional[bool] = None,
                    app_unit_suff: Optional[bool] = None,
                    ):

        op = ""
        post = ""
        s = ""

        if defaults.app_unit_suff(app_unit_suff):
            inv = defaults.inv(inv)
            inv_op = defaults.inv_op(inv_op)
            s = self.symbol

            if len(s) > 0:
                if inv_op and inv:
                    op = " \\: \\cdot \\: "
                else:
                    op = " \\: / \\: "
                    if inv:
                        s = f"{{{s}}}"
                        post = "^{-1}"

        return f"{op}{s}{post}"


# =====================================
# FUNCTIONS
# =====================================


def time_symb_dict():

    d = {
            pd.Timedelta(milliseconds=1E-6): "ns",
            pd.Timedelta(milliseconds=1E-3): "\\mu{}s",
            pd.Timedelta(milliseconds=1): "ms",
            pd.Timedelta(seconds=1): "s",
            pd.Timedelta(minutes=1): "m",
            pd.Timedelta(hours=1): "h",
            pd.Timedelta(days=1): "d",
            }
    return d


def is_array_or_list(
                        x: object,
                    ):

    return isinstance(x, np.ndarray) or isinstance(x, list)
