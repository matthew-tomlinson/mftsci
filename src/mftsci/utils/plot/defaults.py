"""
Defaults module for plot utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union


# External packages
import numpy as np

# Internal packages
from mftsci.utils.meta import set_default, set_default_kwargs   # noqa: F401
from mftsci.utils.arr import iterarr
from mftsci.utils.str.defaults import use_tex   # noqa: F401


# =====================================
# DEFAULTS
# =====================================


def fname(
        fname: Optional[str] = None,
        ) -> str:

    return set_default(val=fname, default="figure")


def dpi(
        dpi: Optional[Union[float, int]] = None,
        ) -> Union[float, int]:

    return set_default(val=dpi, default=600)


def minor_visible(
        minor_visible: Optional[bool] = None,
        ) -> bool:

    return set_default(val=minor_visible, default=False)


def font_family(
        font_family: Optional[bool] = None,
        ) -> str:

    # return set_default(val=font_family, default="serif")
    return set_default(val=font_family, default="sans-serif")


def supress_warnings(
        supress_warnings: Optional[bool] = None,
        ) -> bool:

    return set_default(val=supress_warnings, default=True)


def xys(
        xys: Optional[Union[np.ndarray, list, str,]] = None,
        ) -> np.ndarray:

    return iterarr(set_default(val=xys, default=["x", "y"]))


def maj_offset(
        maj_offset: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return iterarr(set_default(val=maj_offset, default=0.0))


def nrows(
        nrows: Optional[int] = None,
        ) -> int:

    return set_default(val=nrows, default=1)


def ncols(
        ncols: Optional[int] = None,
        ) -> int:

    return set_default(val=ncols, default=1)
